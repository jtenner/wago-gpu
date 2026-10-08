package wagogpu

import (
	"strconv"
	"strings"
)

// Values are immutable symbolic snapshots. Locals and the operand stack copy
// these eight-byte records, never an expression string or a mutable local read.
type valueRecord struct {
	number    uint32
	typ, kind byte
}

type shaderNode struct {
	left, right valueRecord
	op, depth   byte
	emitted     bool
}

// This is a bounded, single-pass Valent-style workspace. Pure math is deferred
// until consumed. Loads are materialized immediately, before a later store can
// change their value. Nodes emit at most once, even when several locals use them.
// Scratch belongs to one compilation and is reset between selected functions.
type lowerScratch struct {
	locals, stack []valueRecord
	nodes         []shaderNode
	code          []byte
	localSpace    [16]valueRecord
	stackSpace    [16]valueRecord
	nodeSpace     [32]shaderNode
	codeSpace     [1024]byte
}

func (s *lowerScratch) reset() {
	if s.locals == nil {
		s.locals = s.localSpace[:]
	}
	if s.stack == nil {
		s.stack = s.stackSpace[:]
	}
	if s.nodes == nil {
		s.nodes = s.nodeSpace[:]
	}
	if s.code == nil {
		s.code = s.codeSpace[:]
	}
	s.locals = s.locals[:0]
	s.stack = s.stack[:0]
	s.nodes = s.nodes[:0]
	s.code = s.code[:0]
}

func appendNumber(dst []byte, n uint32) []byte { return strconv.AppendUint(dst, uint64(n), 10) }
func appendBits(dst []byte, n uint32) []byte {
	dst = append(dst, '0', 'x')
	const hex = "0123456789abcdef"
	for shift := 28; shift >= 0; shift -= 4 {
		dst = append(dst, hex[(n>>uint(shift))&15])
	}
	return append(dst, 'u')
}
func writeNumber(dst *strings.Builder, n uint32) {
	var bytes [10]byte
	dst.Write(appendNumber(bytes[:0], n))
}
func writeBits(dst *strings.Builder, n uint32) {
	var bytes [11]byte
	dst.Write(appendBits(bytes[:0], n))
}
func (s *lowerScratch) text(v string) { s.code = append(s.code, v...) }
func (s *lowerScratch) name(n uint32) { s.code = appendNumber(append(s.code, 'v'), n) }
func (s *lowerScratch) access(slot uint32) {
	s.text("slot")
	s.code = appendNumber(s.code, slot)
	s.text("[index]")
}

func (s *lowerScratch) arithmetic(op byte, left, right valueRecord) valueRecord {
	depth := byte(1)
	for _, v := range [2]valueRecord{left, right} {
		if v.kind == valueNumber {
			n := &s.nodes[v.number]
			if n.depth >= 6 {
				s.materialize(v)
			}
			if !n.emitted {
				depth = max(depth, n.depth+1)
			}
		}
	}
	v := valueRecord{number: uint32(len(s.nodes)), typ: 0x7d, kind: valueNumber}
	s.nodes = append(s.nodes, shaderNode{left: left, right: right, op: op, depth: depth})
	return v
}

func (s *lowerScratch) materialize(v valueRecord) {
	if v.kind != valueNumber {
		return
	}
	// Take a copy: materializing a dependency changes its emitted flag.
	n := s.nodes[v.number]
	if n.emitted {
		return
	}
	s.materialize(n.left)
	s.materialize(n.right)
	s.text("  let ")
	s.name(v.number)
	s.text(": f32 = ")
	s.use(n.left, false)
	s.text(" ")
	s.code = append(s.code, "+-*"[n.op-0x92])
	s.text(" ")
	s.use(n.right, false)
	s.text(";\n")
	s.nodes[v.number].emitted = true
}

// Float arithmetic uses f32 temporaries. Storage/copies use integer bits. A
// float use of a saved exact load never changes that load's original bit record.
func (s *lowerScratch) use(v valueRecord, bits bool) {
	s.materialize(v)
	floatNode := v.kind == valueNumber && s.nodes[v.number].op != 0
	cast := ""
	if v.typ == 0x7d && bits != !floatNode {
		if bits {
			cast = "bitcast<u32>("
		} else {
			cast = "bitcast<f32>("
		}
		s.text(cast)
	}
	switch v.kind {
	case valueNumber:
		s.name(v.number)
	case valueIndex:
		s.text("index")
	default:
		s.code = appendBits(s.code, v.number)
	}
	if cast != "" {
		s.text(")")
	}
}

func (s *lowerScratch) load(slot uint32, typ ElementType) valueRecord {
	v := valueRecord{number: uint32(len(s.nodes)), typ: wasmType(typ.spec().scalar), kind: valueNumber}
	s.nodes = append(s.nodes, shaderNode{emitted: true})
	s.text("  let ")
	s.name(v.number)
	s.text(": u32 = ")
	before, after := "", ""
	switch typ {
	case TypeI8:
		before, after = "bitcast<u32>(bitcast<i32>(", " << 24u) >> 24u)"
	case TypeI16:
		before, after = "bitcast<u32>(bitcast<i32>(", " << 16u) >> 16u)"
	case TypeU8:
		before, after = "(", " & 255u)"
	case TypeU16:
		before, after = "(", " & 65535u)"
	case TypeF16:
		before, after = "half_to_f32(", ")"
	}
	s.text(before)
	s.access(slot)
	s.text(after)
	s.text(";\n")
	return v
}

func (s *lowerScratch) store(slot uint32, typ ElementType, v valueRecord) {
	// Emit pure dependencies before starting the store's assignment text.
	s.materialize(v)
	s.text("  ")
	s.access(slot)
	s.text(" = ")
	before, after := "", ""
	switch typ {
	case TypeI8, TypeU8:
		before, after = "(", " & 255u)"
	case TypeI16, TypeU16:
		before, after = "(", " & 65535u)"
	case TypeF16:
		before, after = "f32_to_half(", ")"
	}
	s.text(before)
	s.use(v, true)
	s.text(after)
	s.text(";\n")
}
