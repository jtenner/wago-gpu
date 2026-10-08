// Package wagogpu provides an optional Wago plugin for small f32 GPU kernels.
package wagogpu

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
)

const (
	maxModuleBytes  = 4 << 20
	maxEntries      = 16384
	maxKernelBytes  = 16384
	maxInstructions = 4096
)

// CompileWGSL translates one explicitly named, defined Wasm function export.
// This is a bounded subset decoder, not a general Wasm validator. Wago validates
// the entire module before the plugin creates a GPU pipeline.
func CompileWGSL(source []byte, export string) (string, error) {
	if export == "" || len(source) > maxModuleBytes || len(source) < 8 || !bytes.Equal(source[:8], []byte{0, 97, 115, 109, 1, 0, 0, 0}) {
		return "", fmt.Errorf("invalid module header, size, or kernel name")
	}
	r := reader{b: source[8:]}
	var types []bool
	var funcs []uint32
	var imports uint32
	var bodies [][]byte
	var target uint32
	found := false
	seen := uint16(0)
	last := byte(0)
	for len(r.b) > 0 && r.err == nil {
		id := r.byte()
		s := reader{b: r.take(r.u32())}
		if id == 0 {
			continue
		}
		// This first version accepts MVP section order only (no data-count).
		if id > 11 || id <= last || seen&(1<<id) != 0 {
			return "", fmt.Errorf("unsupported or repeated section %d", id)
		}
		last = id
		seen |= 1 << id
		switch id {
		case 1:
			n := s.count()
			types = make([]bool, n)
			for i := range types {
				if s.byte() != 0x60 {
					return "", fmt.Errorf("only function types are supported")
				}
				p := s.take(s.count())
				v := s.take(s.count())
				types[i] = len(p) == 1 && p[0] == 0x7d && len(v) == 1 && v[0] == 0x7d
			}
		case 2:
			for n := s.count(); n > 0 && s.err == nil; n-- {
				s.name()
				s.name()
				if s.byte() != 0 {
					return "", fmt.Errorf("only function imports are supported")
				}
				if s.u32() >= uint32(len(types)) {
					return "", fmt.Errorf("invalid imported function type")
				}
				imports++
			}
		case 3:
			funcs = make([]uint32, s.count())
			for i := range funcs {
				funcs[i] = s.u32()
				if funcs[i] >= uint32(len(types)) {
					return "", fmt.Errorf("invalid function type")
				}
			}
		case 5:
			if s.count() != 1 {
				return "", fmt.Errorf("expected one unshared Memory32")
			}
			flags := s.u32()
			if flags > 1 {
				return "", fmt.Errorf("shared memory and Memory64 are unsupported")
			}
			s.u32()
			if flags == 1 {
				s.u32()
			}
		case 7:
			for n := s.count(); n > 0 && s.err == nil; n-- {
				name := s.name()
				kind := s.byte()
				index := s.u32()
				if name == export {
					if found || kind != 0 {
						return "", fmt.Errorf("kernel export is not a unique function")
					}
					target = index
					found = true
				}
			}
		case 10:
			bodies = make([][]byte, s.count())
			for i := range bodies {
				bodies[i] = s.take(s.u32())
			}
		default:
			// Unselected CPU bodies and other sections remain Wago's responsibility.
			s.b = nil
		}
		if s.err != nil {
			return "", s.err
		}
		if len(s.b) != 0 {
			return "", fmt.Errorf("trailing bytes in section %d", id)
		}
	}
	if r.err != nil {
		return "", r.err
	}
	if !found || target < imports || uint64(target-imports) >= uint64(len(funcs)) || len(funcs) != len(bodies) {
		return "", fmt.Errorf("missing or invalid defined kernel %q", export)
	}
	index := target - imports
	if !types[funcs[index]] {
		return "", fmt.Errorf("kernel must have type (f32) -> f32")
	}
	return compileBody(bodies[index])
}

func compileBody(body []byte) (string, error) {
	if len(body) > maxKernelBytes {
		return "", fmt.Errorf("kernel is too large")
	}
	r := reader{b: body}
	if r.u32() != 0 {
		return "", fmt.Errorf("kernel locals are unsupported; use only parameter 0")
	}
	var out strings.Builder
	out.Grow(min(len(body)*16+256, 16<<10))
	out.WriteString("fn kernel(x: f32) -> f32 {\n")
	stack := make([]int, 0, 16)
	for pc := 0; len(r.b) > 0 && r.err == nil; pc++ {
		if pc >= maxInstructions {
			return "", fmt.Errorf("too many kernel instructions")
		}
		op := r.byte()
		if op == 0x0b {
			if len(stack) != 1 || len(r.b) != 0 {
				return "", fmt.Errorf("invalid final kernel stack or trailing code")
			}
			out.WriteString("  return t")
			writeNumber(&out, uint32(stack[0]))
			out.WriteString(";\n}\n")
			out.WriteString("@group(0) @binding(0) var<storage, read_write> values: array<f32>;\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {\n  if (id.x < arrayLength(&values)) { values[id.x] = kernel(values[id.x]); }\n}\n")
			return out.String(), nil
		}
		switch op {
		case 0x20:
			if r.u32() != 0 {
				return "", fmt.Errorf("only local.get 0 is supported")
			}
			out.WriteString("  let t")
			writeNumber(&out, uint32(pc))
			out.WriteString(": f32 = x;\n")
		case 0x43:
			b := r.take(4)
			if r.err != nil {
				return "", r.err
			}
			out.WriteString("  let t")
			writeNumber(&out, uint32(pc))
			out.WriteString(": f32 = bitcast<f32>(")
			writeBits(&out, binary.LittleEndian.Uint32(b))
			out.WriteString(");\n")
		case 0x92, 0x93, 0x94:
			if len(stack) < 2 {
				return "", fmt.Errorf("kernel stack underflow")
			}
			l, h := stack[len(stack)-2], stack[len(stack)-1]
			stack = stack[:len(stack)-2]
			out.WriteString("  let t")
			writeNumber(&out, uint32(pc))
			out.WriteString(": f32 = t")
			writeNumber(&out, uint32(l))
			out.WriteByte(' ')
			out.WriteByte("+-*"[op-0x92])
			out.WriteString(" t")
			writeNumber(&out, uint32(h))
			out.WriteString(";\n")
		default:
			return "", fmt.Errorf("unsupported kernel opcode 0x%02x", op)
		}
		stack = append(stack, pc)
	}
	if r.err != nil {
		return "", r.err
	}
	return "", fmt.Errorf("missing kernel end")
}

type reader struct {
	b   []byte
	err error
}

func (r *reader) take(n uint32) []byte {
	if r.err != nil {
		return nil
	}
	if uint64(n) > uint64(len(r.b)) {
		r.err = fmt.Errorf("truncated Wasm")
		return nil
	}
	v := r.b[:n]
	r.b = r.b[n:]
	return v
}
func (r *reader) byte() byte {
	b := r.take(1)
	if len(b) == 0 {
		return 0
	}
	return b[0]
}
func (r *reader) u32() uint32 {
	var n uint32
	for i := 0; i < 5; i++ {
		b := r.byte()
		if r.err != nil {
			return 0
		}
		if i == 4 && b > 15 {
			r.err = fmt.Errorf("invalid u32 LEB128")
			return 0
		}
		n |= uint32(b&127) << uint(7*i)
		if b < 128 {
			return n
		}
	}
	r.err = fmt.Errorf("invalid u32 LEB128")
	return 0
}
func (r *reader) count() uint32 {
	n := r.u32()
	if n > maxEntries {
		r.err = compileError(CompileLimit, "section entry limit exceeded")
		return 0
	}
	return n
}
func (r *reader) name() string {
	b := r.take(r.u32())
	if string(b) == "wago_gpu_v1" {
		return "wago_gpu_v1"
	}
	if id := bufferIntrinsicNames[string(b)]; id != 0 {
		return bufferImports[id-1].name
	}
	return string(b)
}
