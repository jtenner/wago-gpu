package wagogpu

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"github.com/jtenner/wago-gpu/internal/fixtures"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func bufferConfig() Config {
	slots := []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessWrite}}
	return Config{Disabled: true, Kernels: []KernelConfig{{ID: 1, Export: "wago_gpu.kernel.double", CPUExport: "wago_gpu.cpu.double", Bindings: slots, RelaxedFloat: true}, {ID: 2, Export: "wago_gpu.kernel.addOne", CPUExport: "wago_gpu.cpu.addOne", Bindings: slots, RelaxedFloat: true}}}
}
func bufferFixture(t testing.TB) []byte {
	t.Helper()
	b, e := os.ReadFile("internal/fixtures/buffers.wasm")
	if e != nil {
		t.Fatal(e)
	}
	return b
}
func wat(t testing.TB, source string) []byte {
	t.Helper()
	tool, e := exec.LookPath("wasm-tools")
	if e != nil {
		t.Skip("wasm-tools required for generated Wasm test cases")
	}
	dir := t.TempDir()
	in, out := filepath.Join(dir, "test.wat"), filepath.Join(dir, "test.wasm")
	if e = os.WriteFile(in, []byte(source), 0600); e != nil {
		t.Fatal(e)
	}
	if msg, e := exec.Command(tool, "parse", in, "-o", out).CombinedOutput(); e != nil {
		t.Fatalf("wat: %v: %s", e, msg)
	}
	b, e := os.ReadFile(out)
	if e != nil {
		t.Fatal(e)
	}
	return b
}
func TestBufferStartCPU(t *testing.T) {
	rt, p := setup(t, bufferConfig(), nil)
	_, i := instance(t, rt, bufferFixture(t))
	for j, want := range []float32{3, 5, 7, 9} {
		v, ok := i.ReadFloat32Le(uint32(16 + 4*j))
		if !ok || v != want {
			t.Fatalf("output %d: %v", j, v)
		}
	}
	s := p.BufferSnapshot()
	if s.RuntimeBufferBytes != 0 || s.Totals.GuestSetBytes != 16 || s.Totals.GuestCopyBytes != 16 {
		t.Fatalf("snapshot: %+v", s)
	}
}
func TestBufferTranslation(t *testing.T) {
	for _, k := range bufferConfig().Kernels {
		s, e := CompileBufferWGSL(bufferFixture(t), k)
		if e != nil {
			t.Fatal(e)
		}
		if !strings.Contains(s, "slot0[index]") || !strings.Contains(s, "slot1[index] =") || !strings.Contains(s, "binding(0) var<uniform>") {
			t.Fatal(s)
		}
	}
}
func TestHalfAllPatterns(t *testing.T) {
	for n := 0; n < 65536; n++ {
		h := uint16(n)
		bits := halfToFloat(h)
		want := h
		if h&0x7c00 == 0x7c00 && h&1023 != 0 {
			want = 0x7e00
		}
		if got := floatToHalf(bits); got != want {
			t.Fatalf("%04x -> %08x -> %04x want %04x", h, bits, got, want)
		}
	}
	for _, c := range []struct {
		bits uint32
		want uint16
	}{{0x33000000, 0}, {0x33000001, 1}, {0x3f801000, 0x3c00}, {0x3f803000, 0x3c02}, {0x477ff000, 0x7c00}, {0xffc12345, 0x7e00}} {
		if got := floatToHalf(c.bits); got != c.want {
			t.Fatalf("round %08x: %04x", c.bits, got)
		}
	}
}
func testModule(body string) string {
	return `(module
(import "wago_gpu_v1" "createBufferPacked" (func $create (param i32 i32) (result i64)))
(import "wago_gpu_v1" "bindBuffer" (func $bind (param i32 i32) (result i32)))
(import "wago_gpu_v1" "getBuffer" (func $get (param i32) (result i32)))
(import "wago_gpu_v1" "readBufferF32" (func $read (param i32 i32) (result f32)))
(import "wago_gpu_v1" "writeBufferF32" (func $write (param i32 i32 f32)))
(import "wago_gpu_v1" "dispatch" (func $dispatch (param i32 i32) (result i32)))
(import "wago_gpu_v1" "setBuffer32" (func $set (param i32 i32 i32 i32 i32) (result i32)))
(import "wago_gpu_v1" "copyBuffer32" (func $copy (param i32 i32 i32 i32 i32) (result i32)))
(memory (export "memory") 1)
(export "create" (func $create)) (export "bind" (func $bind)) (export "dispatch" (func $dispatch))
(export "set" (func $set)) (export "copy" (func $copy))
(func (export "firstWord") (result i32) (i32.load (i32.const 0)))
(func (export "wago_gpu.kernel.double") (param $i i32) ` + body + `)
(func (export "wago_gpu.cpu.double") (param i32))
(func (export "scalarLoop") (param i32)
 (call $write (call $get (i32.const 1)) (i32.const 0)
 (call $read (call $get (i32.const 0)) (i32.const 0)) ))
)`
}
func TestBufferFallbackLastAndRanges(t *testing.T) {
	c := bufferConfig()
	c.Kernels = c.Kernels[:1]
	rt, p := setup(t, c, nil)
	_, i := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`)))
	create := func() uint64 {
		r, e := i.Invoke("create", uint64(TypeF32), 4)
		if e != nil || r[0]>>32 != 0 {
			t.Fatalf("create: %v %v", r, e)
		}
		return uint64(uint32(r[0]))
	}
	a, b := create(), create()
	invoke(t, i, "bind", 0, a)
	invoke(t, i, "bind", 1, b)
	if got := invoke(t, i, "dispatch", 1, 4); got != V1CPUFallback {
		t.Fatal(got)
	}
	before := p.BufferSnapshot().Last
	invoke(t, i, "scalarLoop", 4)
	after := p.BufferSnapshot().Last
	if before != after {
		t.Fatalf("scalar replaced Last: %+v", after)
	}
	if got := invoke(t, i, "set", a, 0, 0, 65535, 4); got != V1InvalidRange {
		t.Fatal(got)
	}
	invoke(t, i, "bind", 1, a)
	for _, n := range []uint64{0, 4} {
		if got := invoke(t, i, "dispatch", 1, n); got != V1InvalidBinding {
			t.Fatal(got)
		}
	}
	if err := i.Close(); err != nil {
		t.Fatal(err)
	}
	if p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("instance leaked buffers")
	}
}
func TestPendingContractEviction(t *testing.T) {
	rt, p := setup(t, bufferConfig(), nil)
	source := bufferFixture(t)
	for n := 0; n < 70; n++ {
		prep, e := rt.PrepareCompile(source)
		if e != nil {
			t.Fatal(e)
		}
		prep.Close()
	}
	if p.BufferSnapshot().PendingContracts > 64 {
		t.Fatal("unbounded pending metadata")
	}
	_, i := instance(t, rt, source)
	v, _ := i.ReadUint32Le(32)
	if v != 1 {
		t.Fatal(v)
	}
	if err := rt.CloseContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	s := p.BufferSnapshot()
	if len(s.Instances) != 0 || s.PendingContracts != 0 || s.RuntimeBufferBytes != 0 {
		t.Fatalf("cleanup: %+v", s)
	}
}
func TestCompilerSavedLocal(t *testing.T) {
	c := bufferConfig().Kernels[0]
	src := wat(t, testModule(`(local $x f32)
(local.set $x (f32.const 1))
(call $write (call $get (i32.const 1)) (local.get $i)
(f32.add (local.get $x) (local.tee $x (f32.const 2))))`))
	shader, e := CompileBufferWGSL(src, c)
	if e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(shader, "bitcast<f32>(0x3f800000u) + bitcast<f32>(0x40000000u)") {
		t.Fatal(shader)
	}
}
func TestDecoderDoesNotAllocateExpandedLocals(t *testing.T) {
	m, e := decodeBufferModule(bufferFixture(t))
	if e != nil {
		t.Fatal(e)
	}
	body := []byte{1, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x7f, 0x0b}
	_, e = lowerBufferBody(m, body, bufferConfig().Kernels[0])
	ce, ok := e.(*CompileError)
	if !ok || ce.Kind != CompileLimit {
		t.Fatalf("%v", e)
	}
}
func TestFloatBits(t *testing.T) {
	data := make([]byte, 4)
	for _, bits := range []uint32{0x7fc12345, 0x80000000, 1, 0xff800000} {
		binary.LittleEndian.PutUint32(data, math.Float32bits(math.Float32frombits(bits)))
		if !bytes.Equal(data, []byte{byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24)}) {
			t.Fatal("bits changed")
		}
	}
}

func TestBufferMemory64AndGC(t *testing.T) {
	c := bufferConfig()
	c.Kernels = c.Kernels[:1]
	for _, tc := range []struct{ name, imports, memory, body string }{
		{"memory64", `(import "wago_gpu_v1" "setBuffer64" (func $set (param i32 i32 i32 i64 i32) (result i32))) (import "wago_gpu_v1" "copyBuffer64" (func $copy (param i32 i32 i32 i64 i32) (result i32)))`, `(memory i64 1)`, `(f32.store (i64.const 0) (f32.const 7)) (call $set (local.get $h) (i32.const 0) (i32.const 0) (i64.const 0) (i32.const 1)) (if (then unreachable)) (call $copy (local.get $h) (i32.const 0) (i32.const 0) (i64.const 4) (i32.const 1)) (if (then unreachable)) (f32.load (i64.const 4))`},
		{"gc", `(type $array (array (mut f32))) (import "wago_gpu_v1" "setBufferGC" (func $set (param i32 i32 anyref i32 i32) (result i32))) (import "wago_gpu_v1" "copyBufferGC" (func $copy (param i32 i32 anyref i32 i32) (result i32)))`, ``, `(local.set $a (array.new_fixed $array 1 (f32.const 7))) (call $set (local.get $h) (i32.const 0) (local.get $a) (i32.const 0) (i32.const 1)) (if (then unreachable)) (array.set $array (local.get $a) (i32.const 0) (f32.const 0)) (call $copy (local.get $h) (i32.const 0) (local.get $a) (i32.const 0) (i32.const 1)) (if (then unreachable)) (array.get $array (local.get $a) (i32.const 0))`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			local := ""
			if tc.name == "gc" {
				local = "(local $a (ref null $array))"
			}
			source := `(module ` + tc.imports + ` (import "wago_gpu_v1" "createBufferPacked" (func $create (param i32 i32) (result i64))) ` + tc.memory + ` (func (export "wago_gpu.kernel.double") (param i32)) (func (export "wago_gpu.cpu.double") (param i32)) (func (export "test") (result f32) (local $h i32) ` + local + ` (local.set $h (i32.wrap_i64 (call $create (i32.const 8) (i32.const 1)))) ` + tc.body + `))`
			rt, _ := setup(t, c, nil)
			_, i := instance(t, rt, wat(t, source))
			v, e := i.Invoke("test")
			if e != nil || len(v) != 1 || math.Float32frombits(uint32(v[0])) != 7 {
				t.Fatal(v, e)
			}
		})
	}
}
func TestSharedMemoryRejectedWithGPUDisabled(t *testing.T) {
	c := bufferConfig()
	c.Kernels = c.Kernels[:1]
	source := strings.Replace(testModule(""), `(memory (export "memory") 1)`, `(import "env" "memory" (memory 1 1 shared))`, 1)
	raw := wat(t, source)
	metadata, e := decodeBufferModule(raw)
	if e != nil || len(metadata.memories) != 1 || !metadata.memories[0].shared {
		t.Fatal("shared metadata", e)
	}
	rt, p := setup(t, c, nil)
	if m, e := rt.Compile(raw); e == nil {
		m.Close()
		t.Fatal("pinned Wago unexpectedly accepted shared memory with host imports")
	}
	// Exercise the plugin's own declaration check independently of Wago's
	// earlier shared-memory restriction.
	_, i := instance(t, rt, wat(t, testModule("")))
	p.mu.Lock()
	for _, state := range p.buffers.instances {
		state.module.memories[0].shared = true
	}
	p.mu.Unlock()
	r, e := i.Invoke("create", 8, 1)
	if e != nil {
		t.Fatal(e)
	}
	if got := invoke(t, i, "set", uint64(uint32(r[0])), 0, 0, 0, 1); got != V1UnsupportedStorage {
		t.Fatal(got)
	}
}

func TestInvalidContractSurvivesEviction(t *testing.T) {
	c := bufferConfig()
	c.Kernels = c.Kernels[:1]
	c.Kernels[0].Bindings[1].Access = AccessRead
	rt, _ := setup(t, c, nil)
	source := wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (f32.const 2))`))
	prep, e := rt.PrepareCompile(source)
	if e != nil {
		t.Fatal(e)
	}
	defer prep.Close()
	for n := 0; n < 20; n++ {
		other, e := rt.PrepareCompile(bufferFixture(t))
		if e != nil {
			t.Fatal(e)
		}
		other.Close()
	}
	m, e := prep.Compile()
	if e != nil {
		t.Fatal(e)
	}
	defer m.Close()
	i, e := rt.Instantiate(context.Background(), m)
	if e != nil {
		t.Fatal(e)
	}
	defer i.Close()
	if got := invoke(t, i, "dispatch", 1, 0); got != V1InvalidKernel {
		t.Fatal(got)
	}
}

func TestCPUScalarTypes(t *testing.T) {
	for typ := TypeI8; typ <= TypeF64; typ++ {
		t.Run(typ.spec().suffix, func(t *testing.T) {
			s := typ.spec()
			valueType := map[byte]string{0x7f: "i32", 0x7e: "i64", 0x7d: "f32", 0x7c: "f64"}[wasmType(s.scalar)]
			source := fmt.Sprintf(`(module
 (import "wago_gpu_v1" "createBufferPacked" (func $create (param i32 i32) (result i64)))
 (import "wago_gpu_v1" "readBuffer%s" (func $read (param i32 i32) (result %s)))
 (import "wago_gpu_v1" "writeBuffer%s" (func $write (param i32 i32 %s)))
 (func (export "wago_gpu.kernel.double") (param i32)) (func (export "wago_gpu.cpu.double") (param i32))
 (func (export "roundtrip") (param $v %s) (result %s) (local $h i32)
 (local.set $h (i32.wrap_i64 (call $create (i32.const %d) (i32.const 1))))
 (call $write (local.get $h) (i32.const 0) (local.get $v))
 (call $read (local.get $h) (i32.const 0))))`, s.suffix, valueType, s.suffix, valueType, valueType, valueType, typ)
			c := bufferConfig()
			c.Kernels = c.Kernels[:1]
			rt, _ := setup(t, c, nil)
			_, in := instance(t, rt, wat(t, source))
			for _, bits := range []uint64{0, 0xffffffffffffffff, 0x80000000, 0x7fc12345, 0x3f801000} {
				want := bits
				switch typ {
				case TypeI8:
					want = uint64(uint32(int32(int8(bits))))
				case TypeU8:
					want = uint64(uint8(bits))
				case TypeI16:
					want = uint64(uint32(int32(int16(bits))))
				case TypeU16:
					want = uint64(uint16(bits))
				case TypeI32, TypeU32, TypeF32:
					want = uint64(uint32(bits))
				case TypeF16:
					want = uint64(halfToFloat(floatToHalf(uint32(bits))))
				}
				v, e := in.Invoke("roundtrip", bits)
				if e != nil || v[0] != want {
					t.Fatalf("bits %016x: %v %v, want %016x", bits, v, e, want)
				}
			}
		})
	}
}
func TestImmutableGCTransferBudget(t *testing.T) {
	source := `(module (type $a (array f32))
 (import "wago_gpu_v1" "createBufferPacked" (func $create (param i32 i32) (result i64)))
 (import "wago_gpu_v1" "setBufferGC" (func $set (param i32 i32 anyref i32 i32) (result i32)))
 (func (export "wago_gpu.kernel.double") (param i32)) (func (export "wago_gpu.cpu.double") (param i32))
 (func (export "test") (param $n i32) (result i32) (local $h i32)
 (local.set $h (i32.wrap_i64 (call $create (i32.const 8) (i32.const 1))))
 (call $set (local.get $h) (i32.const 0) (array.new $a (f32.const 7) (i32.const 1000000)) (i32.const 0) (local.get $n))))`
	c := bufferConfig()
	c.Kernels = c.Kernels[:1]
	c.MaxInstanceBufferBytes = 64
	rt, p := setup(t, c, nil)
	_, i := instance(t, rt, wat(t, source))
	if got := invoke(t, i, "test", 1); got != V1LimitExceeded {
		t.Fatal(got)
	}
	if got := invoke(t, i, "test", 0); got != 0 {
		t.Fatal(got)
	}
	if p.BufferSnapshot().RuntimeBufferBytes != 8 {
		t.Fatal("temporary copy charged after failure")
	}
}
func TestFailedStartReleasesBuffers(t *testing.T) {
	source, e := os.ReadFile("internal/fixtures/buffers.wat")
	if e != nil {
		t.Fatal(e)
	}
	raw := wat(t, strings.Replace(string(source), `(call $check (call $free (global.get $b)))`, `unreachable`, 1))
	rt, p := setup(t, bufferConfig(), nil)
	m, e := rt.Compile(raw)
	if e != nil {
		t.Fatal(e)
	}
	defer m.Close()
	if i, e := rt.Instantiate(context.Background(), m); e == nil {
		i.Close()
		t.Fatal("start should trap")
	}
	s := p.BufferSnapshot()
	if len(s.Instances) != 0 || s.RuntimeBufferBytes != 0 {
		t.Fatal("failed start leak", s)
	}
}

func TestBufferCPUAllocationBound(t *testing.T) {
	c := bufferConfig()
	c.Kernels = []KernelConfig{{ID: 1, Export: "wago_gpu.kernel.twice", CPUExport: "wago_gpu.cpu.twice", Bindings: c.Kernels[0].Bindings, RelaxedFloat: true}}
	rt, _ := setup(t, c, nil)
	_, i := instance(t, rt, fixtures.BufferWork)
	invoke(t, i, "setup", 1024)
	run := func(n uint64) float64 {
		return testing.AllocsPerRun(3, func() {
			if _, e := i.Invoke("wago_gpu.cpu.twice", n); e != nil {
				panic(e)
			}
		})
	}
	small, large := run(16), run(1024)
	if large > small+16 {
		t.Fatalf("per-element heap allocations: 16=%g 1024=%g", small, large)
	}
}
