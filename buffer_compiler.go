package wagogpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	wago "github.com/wago-org/wago"
	"sort"
	"strings"
)

type loweredKernel struct {
	shader        string
	reads, writes map[uint32]bool
	// Accepted bodies are straight-line and every access uses the invocation
	// index. A read before the first store is the only need for old contents
	// within that invocation. Partial dispatches must still preserve the tail.
	readBeforeWrite [8]bool
	float           bool
}
type valueRecord struct {
	text   string
	typ    byte
	kind   byte
	number uint32
}

const (
	valueNumber byte = iota
	valueConstant
	valueIndex
	valueHandle
)

// CompileBufferWGSL translates the selected kernel, without creating a device.
// Every failure is a *CompileError. This decoder is bounded; Wago remains the
// validator for the whole module before any runtime pipeline can be attached.
func CompileBufferWGSL(source []byte, kernel KernelConfig) (string, error) {
	if e := validateKernel(kernel, 8); e != nil {
		return "", e
	}
	m, e := decodeBufferModule(source)
	if e != nil {
		return "", e
	}
	body, e := m.kernelBody(kernel)
	if e != nil {
		return "", e
	}
	l, e := lowerBufferBody(m, body, kernel)
	return l.shader, e
}
func lowerBufferBody(m *bufferModule, body []byte, k KernelConfig) (loweredKernel, error) {
	out := loweredKernel{reads: map[uint32]bool{}, writes: map[uint32]bool{}}
	fail := func(kind CompileErrorKind, msg string) (loweredKernel, error) {
		return out, compileError(kind, "%s", msg)
	}
	if len(body) > maxKernelBytes {
		return fail(CompileLimit, "kernel byte limit")
	}
	bindings := map[uint32]BindingConfig{}
	for _, b := range k.Bindings {
		bindings[b.Slot] = b
		if b.Type == TypeF16 || b.Type == TypeF32 || b.Type == TypeF64 {
			out.float = true
		}
		if b.Type >= TypeI64 {
			return fail(CompileUnsupported, "64-bit GPU buffers are unsupported")
		}
	}
	r := reader{b: body}
	locals := []valueRecord{{text: "index", typ: 0x7f, kind: valueIndex}}
	for n := r.count(); n > 0 && r.err == nil; n-- {
		count := r.u32()
		t := r.byte()
		if count > 4096-uint32(len(locals)) {
			return fail(CompileLimit, "expanded local limit")
		}
		if t != 0x7f && t != 0x7d {
			return fail(CompileUnsupported, "local type is unsupported")
		}
		if t == 0x7d {
			out.float = true
		}
		for j := uint32(0); j < count; j++ {
			locals = append(locals, valueRecord{text: "0u", typ: t, kind: valueConstant})
		}
	}
	var code strings.Builder
	stack := make([]valueRecord, 0, 16)
	values := 0
	var lowerErr error
	pop := func(t byte) valueRecord {
		if len(stack) == 0 {
			lowerErr = compileError(CompileInvalidContract, "operand stack underflow")
			return valueRecord{}
		}
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if v.typ != t {
			lowerErr = compileError(CompileInvalidContract, "operand type mismatch")
		}
		return v
	}
	emit := func(expr string, t byte) valueRecord {
		v := valueRecord{text: fmt.Sprintf("v%d", values), typ: t}
		fmt.Fprintf(&code, "  let %s: u32 = %s;\n", v.text, expr)
		values++
		return v
	}
	done := false
	for pc := 0; len(r.b) > 0 && r.err == nil && lowerErr == nil; pc++ {
		if pc >= 4096 || values >= 8192 || len(stack) > 4096 || code.Len() > 1<<20 {
			return fail(CompileLimit, "kernel workspace limit")
		}
		op := r.byte()
		switch op {
		case 0x0b:
			if len(stack) != 0 || len(r.b) != 0 {
				return fail(CompileInvalidContract, "nonempty final stack or trailing code")
			}
			done = true
		case 0x0f:
			if len(stack) != 0 || len(r.b) != 1 || r.byte() != 0x0b {
				return fail(CompileUnsupported, "return must directly precede final end")
			}
			done = true
		case 0x20, 0x21, 0x22:
			idx := r.u32()
			if idx >= uint32(len(locals)) {
				return fail(CompileInvalidContract, "local index out of range")
			}
			if op == 0x20 {
				stack = append(stack, locals[idx])
			} else {
				v := pop(locals[idx].typ)
				locals[idx] = v
				if op == 0x22 {
					stack = append(stack, v)
				}
			}
		case 0x41:
			n := uint32(r.leb(32, true))
			stack = append(stack, valueRecord{fmt.Sprintf("0x%08xu", n), 0x7f, valueConstant, n})
		case 0x43:
			b := r.take(4)
			if r.err != nil {
				break
			}
			out.float = true
			stack = append(stack, valueRecord{fmt.Sprintf("0x%08xu", binary.LittleEndian.Uint32(b)), 0x7d, valueConstant, 0})
		case 0x92, 0x93, 0x94:
			right, left := pop(0x7d), pop(0x7d)
			out.float = true
			stack = append(stack, emit(fmt.Sprintf("bitcast<u32>(bitcast<f32>(%s) %c bitcast<f32>(%s))", left.text, "+-*"[op-0x92], right.text), 0x7d))
		case 0x10:
			index := r.u32()
			if index >= uint32(len(m.imports)) {
				return fail(CompileUnsupported, "only buffer intrinsic calls are supported")
			}
			imp := m.imports[index]
			if imp.module != "wago_gpu_v1" {
				return fail(CompileUnsupported, "call is not a buffer intrinsic")
			}
			var descriptor *bufferImport
			for i := range bufferImports {
				if bufferImports[i].name == imp.name {
					descriptor = &bufferImports[i]
					break
				}
			}
			sig, ok := m.signature(index)
			if !ok || descriptor == nil {
				return fail(CompileUnsupported, "unknown intrinsic")
			}
			// Wasm binary encodings of public scalar types are fixed here.
			if len(sig.params) != len(descriptor.params) || len(sig.results) != len(descriptor.results) {
				return fail(CompileInvalidContract, "intrinsic signature mismatch")
			}
			for i, t := range descriptor.params {
				if sig.params[i] != wasmType(t) {
					return fail(CompileInvalidContract, "intrinsic parameter mismatch")
				}
			}
			for i, t := range descriptor.results {
				if sig.results[i] != wasmType(t) {
					return fail(CompileInvalidContract, "intrinsic result mismatch")
				}
			}
			if imp.name == "getBuffer" {
				slot := pop(0x7f)
				if slot.kind != valueConstant {
					return fail(CompileUnsupported, "slot must be constant")
				}
				if _, ok := bindings[slot.number]; !ok {
					return fail(CompileInvalidContract, "undeclared slot")
				}
				stack = append(stack, valueRecord{typ: 0x7f, kind: valueHandle, number: slot.number})
				break
			}
			write := strings.HasPrefix(imp.name, "writeBuffer")
			read := strings.HasPrefix(imp.name, "readBuffer")
			if !read && !write {
				return fail(CompileUnsupported, "management call in kernel")
			}
			prefix := "readBuffer"
			if write {
				prefix = "writeBuffer"
			}
			suffix := strings.TrimPrefix(imp.name, prefix)
			var typ ElementType
			for i, e := range elementSpecs {
				if e.suffix == suffix {
					typ = ElementType(i)
				}
			}
			if typ == 0 || typ >= TypeI64 {
				return fail(CompileUnsupported, "intrinsic type unsupported on GPU")
			}
			var data valueRecord
			if write {
				data = pop(wasmType(typ.spec().scalar))
				if data.kind == valueHandle {
					return fail(CompileInvalidContract, "buffer handle cannot be stored as data")
				}
			}
			element, handle := pop(0x7f), pop(0x7f)
			if handle.kind != valueHandle {
				return fail(CompileUnsupported, "handle is not a selected buffer slot")
			}
			if element.kind != valueIndex {
				return fail(CompileUnsupported, "element must be the original invocation index")
			}
			b, ok := bindings[handle.number]
			if !ok || b.Type != typ {
				return fail(CompileInvalidContract, "intrinsic type does not match declared slot")
			}
			if write && b.Access == AccessRead || read && b.Access == AccessWrite {
				return fail(CompileInvalidContract, "intrinsic violates slot access")
			}
			access := fmt.Sprintf("slot%d[index]", b.Slot)
			if write {
				out.writes[b.Slot] = true
				expr := data.text
				switch typ {
				case TypeI8, TypeU8:
					expr = "(" + expr + " & 255u)"
				case TypeI16, TypeU16:
					expr = "(" + expr + " & 65535u)"
				case TypeF16:
					expr = "f32_to_half(" + expr + ")"
				}
				fmt.Fprintf(&code, "  %s = %s;\n", access, expr)
			} else {
				if !out.writes[b.Slot] {
					out.readBeforeWrite[b.Slot] = true
				}
				out.reads[b.Slot] = true
				expr := access
				switch typ {
				case TypeI8:
					expr = "bitcast<u32>(bitcast<i32>(" + expr + " << 24u) >> 24u)"
				case TypeU8:
					expr = "(" + expr + " & 255u)"
				case TypeI16:
					expr = "bitcast<u32>(bitcast<i32>(" + expr + " << 16u) >> 16u)"
				case TypeU16:
					expr = "(" + expr + " & 65535u)"
				case TypeF16:
					expr = "half_to_f32(" + expr + ")"
				}
				stack = append(stack, emit(expr, wasmType(typ.spec().scalar)))
			}
		default:
			return fail(CompileUnsupported, fmt.Sprintf("unsupported opcode 0x%02x", op))
		}
		if done {
			break
		}
	}
	if r.err != nil {
		var typed *CompileError
		if errors.As(r.err, &typed) {
			return out, typed
		}
		return fail(CompileInvalidContract, r.err.Error())
	}
	if lowerErr != nil {
		return out, lowerErr
	}
	if !done {
		return fail(CompileInvalidContract, "missing function end")
	}
	var shader strings.Builder
	shader.WriteString("struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }\n@group(0) @binding(0) var<uniform> params: Params;\n")
	slots := make([]int, 0, len(bindings))
	for slot := range bindings {
		if out.reads[slot] || out.writes[slot] {
			slots = append(slots, int(slot))
		}
	}
	sort.Ints(slots)
	for i, slot := range slots {
		mode := "read"
		if out.writes[uint32(slot)] {
			mode = "read_write"
		}
		fmt.Fprintf(&shader, "@group(0) @binding(%d) var<storage, %s> slot%d: array<u32>;\n", i+1, mode, slot)
	}
	for _, b := range bindings {
		if b.Type == TypeF16 {
			shader.WriteString(halfWGSL)
			break
		}
	}
	shader.WriteString("@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {\n  let index = id.x;\n  if (index >= params.count) { return; }\n")
	shader.WriteString(code.String())
	shader.WriteString("}\n")
	if shader.Len() > 512<<10 {
		return fail(CompileLimit, "WGSL byte limit")
	}
	out.shader = shader.String()
	return out, nil
}

func wasmType(t wago.ValType) byte {
	switch t {
	case wago.ValI32:
		return 0x7f
	case wago.ValI64:
		return 0x7e
	case wago.ValF32:
		return 0x7d
	case wago.ValF64:
		return 0x7c
	case wago.ValAnyRef:
		return 0x6e
	}
	return 0
}
