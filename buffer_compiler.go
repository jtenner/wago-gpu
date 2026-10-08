package wagogpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	wago "github.com/wago-org/wago"
	"strings"
)

type loweredKernel struct {
	shader        string
	reads, writes [8]bool
	// Accepted bodies are straight-line and every access uses the invocation
	// index. A read before the first store is the only need for old contents
	// within that invocation. Partial dispatches must still preserve the tail.
	readBeforeWrite [8]bool
	float           bool
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
	var scratch lowerScratch
	return lowerBufferBodyScratch(m, body, k, &scratch)
}
func lowerBufferBodyScratch(m *bufferModule, body []byte, k KernelConfig, scratch *lowerScratch) (loweredKernel, error) {
	scratch.reset()
	out := loweredKernel{}
	fail := func(kind CompileErrorKind, msg string) (loweredKernel, error) {
		return out, compileError(kind, "%s", msg)
	}
	if len(body) > maxKernelBytes {
		return fail(CompileLimit, "kernel byte limit")
	}
	var bindings [8]BindingConfig
	var declared [8]bool
	for _, b := range k.Bindings {
		if b.Slot >= 8 {
			return fail(CompileInvalidContract, "invalid slot")
		}
		bindings[b.Slot] = b
		declared[b.Slot] = true
		if b.Type == TypeF16 || b.Type == TypeF32 || b.Type == TypeF64 {
			out.float = true
		}
		if b.Type >= TypeI64 {
			return fail(CompileUnsupported, "64-bit GPU buffers are unsupported")
		}
	}
	r := reader{b: body}
	scratch.locals = append(scratch.locals, valueRecord{typ: 0x7f, kind: valueIndex})
	for n := r.count(); n > 0 && r.err == nil; n-- {
		count := r.u32()
		t := r.byte()
		if count > 4096-uint32(len(scratch.locals)) {
			return fail(CompileLimit, "expanded local limit")
		}
		if t != 0x7f && t != 0x7d {
			return fail(CompileUnsupported, "local type is unsupported")
		}
		if t == 0x7d {
			out.float = true
		}
		for j := uint32(0); j < count; j++ {
			scratch.locals = append(scratch.locals, valueRecord{typ: t, kind: valueConstant})
		}
	}

	var lowerErr error
	pop := func(t byte) valueRecord {
		if len(scratch.stack) == 0 {
			lowerErr = compileError(CompileInvalidContract, "operand stack underflow")
			return valueRecord{}
		}
		v := scratch.stack[len(scratch.stack)-1]
		scratch.stack = scratch.stack[:len(scratch.stack)-1]
		if v.typ != t {
			lowerErr = compileError(CompileInvalidContract, "operand type mismatch")
		}
		return v
	}
	done := false
	for pc := 0; len(r.b) > 0 && r.err == nil && lowerErr == nil; pc++ {
		if pc >= 4096 || len(scratch.nodes) >= 8192 || len(scratch.stack) > 4096 || len(scratch.code) > 1<<20 {
			return fail(CompileLimit, "kernel workspace limit")
		}
		op := r.byte()
		switch op {
		case 0x0b:
			if len(scratch.stack) != 0 || len(r.b) != 0 {
				return fail(CompileInvalidContract, "nonempty final stack or trailing code")
			}
			done = true
		case 0x0f:
			if len(scratch.stack) != 0 || len(r.b) != 1 || r.byte() != 0x0b {
				return fail(CompileUnsupported, "return must directly precede final end")
			}
			done = true
		case 0x20, 0x21, 0x22:
			idx := r.u32()
			if idx >= uint32(len(scratch.locals)) {
				return fail(CompileInvalidContract, "local index out of range")
			}
			if op == 0x20 {
				scratch.stack = append(scratch.stack, scratch.locals[idx])
			} else {
				v := pop(scratch.locals[idx].typ)
				scratch.locals[idx] = v
				if op == 0x22 {
					scratch.stack = append(scratch.stack, v)
				}
			}
		case 0x41:
			n := uint32(r.leb(32, true))
			scratch.stack = append(scratch.stack, valueRecord{number: n, typ: 0x7f, kind: valueConstant})
		case 0x43:
			b := r.take(4)
			if r.err != nil {
				break
			}
			out.float = true
			scratch.stack = append(scratch.stack, valueRecord{number: binary.LittleEndian.Uint32(b), typ: 0x7d, kind: valueConstant})
		case 0x92, 0x93, 0x94:
			right, left := pop(0x7d), pop(0x7d)
			out.float = true
			if lowerErr == nil {
				scratch.stack = append(scratch.stack, scratch.arithmetic(op, left, right))
			}
		case 0x10:
			index := r.u32()
			if index >= uint32(len(m.imports)) {
				return fail(CompileUnsupported, "only buffer intrinsic calls are supported")
			}
			imp := m.imports[index]
			if imp.flags&intrinsicNamespace == 0 {
				return fail(CompileUnsupported, "call is not a buffer intrinsic")
			}
			if imp.intrinsic == 0 {
				return fail(CompileUnsupported, "unknown intrinsic")
			}
			if imp.flags&intrinsicBadSignature != 0 {
				return fail(CompileInvalidContract, "intrinsic signature mismatch")
			}
			if imp.flags&intrinsicGet != 0 {
				slot := pop(0x7f)
				if slot.kind != valueConstant {
					return fail(CompileUnsupported, "slot must be constant")
				}
				if slot.number >= 8 || !declared[slot.number] {
					return fail(CompileInvalidContract, "undeclared slot")
				}
				scratch.stack = append(scratch.stack, valueRecord{typ: 0x7f, kind: valueHandle, number: slot.number})
				break
			}
			write := imp.flags&intrinsicWrite != 0
			read := imp.flags&intrinsicRead != 0
			if !read && !write {
				return fail(CompileUnsupported, "management call in kernel")
			}
			typ := ElementType(imp.element)
			if typ == 0 || typ >= TypeI64 {
				return fail(CompileUnsupported, "intrinsic type unsupported on GPU")
			}
			var data valueRecord
			if write {
				data = pop(imp.scalar)
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
			if lowerErr != nil {
				break
			}
			b := bindings[handle.number]
			if !declared[handle.number] || b.Type != typ {
				return fail(CompileInvalidContract, "intrinsic type does not match declared slot")
			}
			if write && b.Access == AccessRead || read && b.Access == AccessWrite {
				return fail(CompileInvalidContract, "intrinsic violates slot access")
			}
			if write {
				out.writes[b.Slot] = true
				scratch.store(b.Slot, typ, data)
			} else {
				if !out.writes[b.Slot] {
					out.readBeforeWrite[b.Slot] = true
				}
				out.reads[b.Slot] = true
				scratch.stack = append(scratch.stack, scratch.load(b.Slot, typ))
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
	helper := ""
	for _, b := range bindings {
		if b.Type == TypeF16 {
			helper = halfWGSL
			break
		}
	}
	shader.Grow(len(scratch.code) + len(helper) + 1024)
	shader.WriteString("struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }\n@group(0) @binding(0) var<uniform> params: Params;\n")
	physical := 1
	for slot := 0; slot < 8; slot++ {
		if !out.reads[slot] && !out.writes[slot] {
			continue
		}
		mode := "read"
		if out.writes[slot] {
			mode = "read_write"
		}
		shader.WriteString("@group(0) @binding(")
		writeNumber(&shader, uint32(physical))
		shader.WriteString(") var<storage, ")
		shader.WriteString(mode)
		shader.WriteString("> slot")
		writeNumber(&shader, uint32(slot))
		shader.WriteString(": array<u32>;\n")
		physical++
	}
	shader.WriteString(helper)
	shader.WriteString("@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {\n  let index = id.x;\n  if (index >= params.count) { return; }\n")
	shader.Write(scratch.code)
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
