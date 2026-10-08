package wagogpu

import (
	"bytes"
	"errors"
	"fmt"
)

type wasmSignature struct{ params, results []byte }
type wasmImport struct {
	module, name string
	signature    uint32
	intrinsic    uint8
}
type memoryDeclaration struct{ wide, shared bool }
type bufferModule struct {
	types     []wasmSignature
	imports   []wasmImport
	functions []uint32
	bodies    [][]byte // Borrows the source only during compilation.
	exports   map[string]uint32
	memories  []memoryDeclaration
}

// Metadata is indexed once per module. Known ABI names share immutable strings.
// An intrinsic index fits the existing import record's alignment padding.
var bufferIntrinsicNames = func() map[string]uint8 {
	names := make(map[string]uint8, len(bufferImports))
	for i := range bufferImports {
		names[bufferImports[i].name] = uint8(i + 1)
	}
	return names
}()

var wasmSectionOrder = [14]byte{1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 13: 6, 6: 7, 7: 8, 8: 9, 9: 10, 12: 11, 10: 12, 11: 13}

func (r *reader) leb(bits int, signed bool) uint64 {
	var value uint64
	n := (bits + 6) / 7
	for i := 0; i < n; i++ {
		b := r.byte()
		if r.err != nil {
			return 0
		}
		payload := uint64(b & 127)
		if i == n-1 {
			used := bits - 7*i
			mask := byte(127<<used) & 127
			if !signed {
				if b&mask != 0 {
					r.err = fmt.Errorf("LEB128 overflow")
					return 0
				}
			} else {
				sign := byte(1 << (used - 1))
				if b&sign == 0 && b&mask != 0 || b&sign != 0 && b&mask != mask {
					r.err = fmt.Errorf("signed LEB128 overflow")
					return 0
				}
			}
		}
		value |= payload << uint(7*i)
		if b&128 == 0 {
			if signed && b&64 != 0 && 7*(i+1) < 64 {
				value |= ^uint64(0) << uint(7*(i+1))
			}
			return value
		}
	}
	r.err = fmt.Errorf("unterminated LEB128")
	return 0
}
func (r *reader) valueType() byte {
	t := r.byte()
	switch t {
	case 0x63, 0x64:
		r.leb(33, true)
	case 0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x70, 0x6f, 0x6e, 0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x68, 0x67, 0x66, 0x65, 0x78, 0x77:
	default:
		r.err = fmt.Errorf("unsupported value type 0x%x", t)
	}
	return t
}
func (r *reader) valueTypes() []byte {
	n := r.count()
	start := r.b
	wide := false
	for i := uint32(0); i < n && r.err == nil; i++ {
		before := len(r.b)
		r.valueType()
		wide = wide || before-len(r.b) != 1
	}
	if r.err != nil {
		return nil
	}
	if !wide {
		return start[:n]
	} // Borrow scalar signature bytes during this compilation.
	v := make([]byte, n)
	view := reader{b: start[:len(start)-len(r.b)]}
	for i := range v {
		v[i] = view.valueType()
	}
	return v
}
func readLimits(r *reader) memoryDeclaration {
	flags := r.u32()
	if flags & ^uint32(7) != 0 {
		r.err = fmt.Errorf("unsupported memory limits")
	}
	m := memoryDeclaration{wide: flags&4 != 0, shared: flags&2 != 0}
	bits := 32
	if m.wide {
		bits = 64
	}
	r.leb(bits, false)
	if flags&1 != 0 {
		r.leb(bits, false)
	}
	return m
}
func decodeBufferModule(source []byte) (*bufferModule, error) {
	if len(source) > maxModuleBytes {
		return nil, compileError(CompileLimit, "module exceeds decoder byte limit")
	}
	if len(source) < 8 || !bytes.Equal(source[:8], []byte{0, 97, 115, 109, 1, 0, 0, 0}) {
		return nil, compileError(CompileInvalidContract, "invalid Wasm header")
	}
	m := &bufferModule{exports: map[string]uint32{}}
	r := reader{b: source[8:]}
	seen := uint32(0)
	last := 0
	for len(r.b) > 0 && r.err == nil {
		id := r.byte()
		s := reader{b: r.take(r.u32())}
		if id == 0 {
			continue
		}

		// Explicit order avoids treating data-count as a code-section successor.
		order := 0
		if int(id) < len(wasmSectionOrder) {
			order = int(wasmSectionOrder[id])
		}
		if order == 0 || order <= last || seen&(1<<id) != 0 {
			return nil, compileError(CompileUnsupported, "unsupported section order")
		}
		last = order
		seen |= 1 << id
		switch id {
		case 1:
			for n := s.count(); n > 0 && s.err == nil; n-- {
				groups := uint32(1)
				tag := s.byte()
				if tag == 0x4e {
					groups = s.count()
					tag = 0
				}
				for j := uint32(0); j < groups && s.err == nil; j++ {
					if len(m.types) >= maxEntries {
						return nil, compileError(CompileLimit, "expanded type limit")
					}
					t := tag
					if t == 0 {
						t = s.byte()
					}
					if t == 0x50 || t == 0x4f {
						for n := s.count(); n > 0; n-- {
							s.u32()
						}
						t = s.byte()
					}
					sig := wasmSignature{}
					switch t {
					case 0x60:
						sig = wasmSignature{s.valueTypes(), s.valueTypes()}
					case 0x5e:
						s.valueType()
						s.byte()
					case 0x5f:
						for n := s.count(); n > 0 && s.err == nil; n-- {
							s.valueType()
							s.byte()
						}
					default:
						s.err = fmt.Errorf("unsupported composite type")
					}
					m.types = append(m.types, sig)
				}
			}
		case 2:
			for n := s.count(); n > 0 && s.err == nil; n-- {
				module, name := s.name(), s.name()
				kind := s.byte()
				switch kind {
				case 0:
					imp := wasmImport{module: module, name: name, signature: s.u32()}
					if module == "wago_gpu_v1" {
						imp.intrinsic = bufferIntrinsicNames[name]
					}
					m.imports = append(m.imports, imp)
				case 1:
					s.valueType()
					readLimits(&s)
				case 2:
					m.memories = append(m.memories, readLimits(&s))
				case 3:
					s.valueType()
					s.byte()
				case 4:
					s.byte()
					s.u32()
				default:
					s.err = fmt.Errorf("unsupported import kind")
				}
			}
		case 3:
			m.functions = make([]uint32, s.count())
			for i := range m.functions {
				m.functions[i] = s.u32()
			}
		case 5:
			for n := s.count(); n > 0 && s.err == nil; n-- {
				m.memories = append(m.memories, readLimits(&s))
			}
		case 7:
			for n := s.count(); n > 0 && s.err == nil; n-- {
				name := s.name()
				kind := s.byte()
				idx := s.u32()
				if kind == 0 {
					if _, ok := m.exports[name]; ok {
						s.err = fmt.Errorf("duplicate export")
					}
					m.exports[name] = idx
				}
			}
		case 10:
			m.bodies = make([][]byte, s.count())
			for i := range m.bodies {
				m.bodies[i] = s.take(s.u32())
			}
		default:
			s.b = nil // Wago validates unrelated sections before attaching metadata.
		}
		if s.err != nil {
			var ce *CompileError
			if errors.As(s.err, &ce) {
				return nil, ce
			}
			return nil, compileError(CompileInvalidContract, "bounded module decoder: %v", s.err)
		}
		if len(s.b) != 0 {
			return nil, compileError(CompileInvalidContract, "trailing section bytes")
		}
	}
	if r.err != nil {
		return nil, compileError(CompileInvalidContract, "%v", r.err)
	}
	if len(m.bodies) != len(m.functions) {
		return nil, compileError(CompileInvalidContract, "function/code count mismatch")
	}
	return m, nil
}
func (m *bufferModule) signature(index uint32) (wasmSignature, bool) {
	var t uint32
	if index < uint32(len(m.imports)) {
		t = m.imports[index].signature
	} else {
		index -= uint32(len(m.imports))
		if index >= uint32(len(m.functions)) {
			return wasmSignature{}, false
		}
		t = m.functions[index]
	}
	if t >= uint32(len(m.types)) {
		return wasmSignature{}, false
	}
	return m.types[t], true
}
func (m *bufferModule) kernelBody(k KernelConfig) ([]byte, error) {
	for _, name := range []string{k.Export, k.CPUExport} {
		index, ok := m.exports[name]
		sig, valid := m.signature(index)
		if !ok || !valid || !bytes.Equal(sig.params, []byte{0x7f}) || len(sig.results) != 0 || index < uint32(len(m.imports)) {
			return nil, compileError(CompileInvalidContract, "%s must be a defined (i32) -> () export", name)
		}
	}
	return m.bodies[m.exports[k.Export]-uint32(len(m.imports))], nil
}
