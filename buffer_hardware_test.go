//go:build webgpu && cgo && linux

package wagogpu

import (
	"context"
	"encoding/binary"
	"github.com/oliverbestmann/webgpu/wgpu"
	"math"
	"runtime"
	"strings"
	"testing"
)

func TestBufferHardwareSeedRules(t *testing.T) {
	requireHardware(t)
	testBufferSeedRules(t, true)
}

func TestBufferHardwareParameterReuse(t *testing.T) {
	requireHardware(t)
	testBufferParameterReuse(t, true)
}

func TestBufferHardwareStart(t *testing.T) {
	requireHardware(t)
	for _, paths := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
		t.Run(map[bool]string{true: "cpu", false: "gpu"}[paths[0]]+"_"+map[bool]string{true: "cpu", false: "gpu"}[paths[1]], func(t *testing.T) {
			c := bufferConfig()
			c.Disabled = false
			for i, cpu := range paths {
				if cpu {
					c.Kernels[i].MinElements = 5
				}
			}
			rt, p := setup(t, c, nil)
			_, i := instance(t, rt, bufferFixture(t))
			for j, want := range []float32{3, 5, 7, 9} {
				got, _ := i.ReadFloat32Le(uint32(16 + j*4))
				if got != want {
					t.Fatalf("value %d: %v", j, got)
				}
			}
			for j, cpu := range paths {
				got, _ := i.ReadUint32Le(uint32(32 + j*4))
				want := uint32(0)
				if cpu {
					want = 1
				}
				if got != want {
					t.Fatalf("kernel %d status %d: %+v", j, got, p.BufferSnapshot())
				}
			}
		})
	}
}
func TestBindingRejectsInvalidCommand(t *testing.T) {
	requireHardware(t)
	backend, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	b := backend.(*gpuBackend)
	defer b.Close()
	r, e := b.AllocateBuffer(16)
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	e = b.submitBuffers(context.Background(), func(encoder *wgpu.CommandEncoder) error {
		return encoder.TryCopyBufferToBuffer(r.(*nativeBuffer).buffer, 0, r.(*nativeBuffer).buffer, 0, 16)
	})
	if e == nil {
		t.Fatal("invalid self-copy was accepted")
	}
	// A later successful queue notification cannot erase the checked error.
	if e = b.waitIdle(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestBindingDelayedCallbacksGC(t *testing.T) {
	requireHardware(t)
	backend, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	b := backend.(*gpuBackend)
	defer b.Close()
	for i := 0; i < 100; i++ {
		done := make(chan wgpu.QueueWorkDoneStatus, 1)
		b.queue.OnSubmittedWorkDone(func(s wgpu.QueueWorkDoneStatus) { done <- s })
		runtime.GC()
		b.device.Poll(false, nil)
		select {
		case s := <-done:
			if s != wgpu.QueueWorkDoneStatusSuccess {
				t.Fatal(s)
			}
		default:
			t.Fatal("callback missing")
		}
	}
}

func TestBindingSubmissionError(t *testing.T) {
	requireHardware(t)
	backend, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	b := backend.(*gpuBackend)
	defer b.Close()
	source, e := b.AllocateBuffer(16)
	if e != nil {
		t.Fatal(e)
	}
	defer source.Close()
	destination, e := b.AllocateBuffer(16)
	if e != nil {
		t.Fatal(e)
	}
	defer destination.Close()
	encoder, e := b.device.TryCreateCommandEncoder(nil)
	if e != nil {
		t.Fatal(e)
	}
	defer encoder.Release()
	if e = encoder.TryCopyBufferToBuffer(source.(*nativeBuffer).buffer, 0, destination.(*nativeBuffer).buffer, 0, 16); e != nil {
		t.Fatal(e)
	}
	command, e := encoder.TryFinish(nil)
	if e != nil {
		t.Fatal(e)
	}
	defer command.Release()
	// Encoding was valid. Destruction makes the submitted resource invalid.
	source.(*nativeBuffer).buffer.Destroy()
	if e = b.queue.TrySubmit(command); e == nil {
		t.Fatal("submission error was not reported")
	}
	if e = b.waitIdle(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestBindingInvalidMappedRange(t *testing.T) {
	requireHardware(t)
	backend, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	b := backend.(*gpuBackend)
	defer b.Close()
	r, e := b.AllocateBuffer(16)
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	var data []byte
	e = b.device.Check(func() error { data = r.(*nativeBuffer).buffer.GetMappedRange(0, 16); return nil })
	if e == nil || data != nil {
		t.Fatal("invalid map did not return an error", e)
	}
}

func TestBufferHardwareF16Patterns(t *testing.T) {
	requireHardware(t)
	for _, widen := range []bool{false, true} {
		t.Run(map[bool]string{false: "roundtrip", true: "widen"}[widen], func(t *testing.T) {
			c := bufferConfig()
			c.Disabled = false
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Bindings[0].Type = TypeF16
			outputType := TypeF16
			outputSize := uint32(2)
			if widen {
				outputType = TypeF32
				outputSize = 4
			}
			c.Kernels[0].Bindings[1].Type = outputType
			source := testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`)
			source = strings.Replace(source, "readBufferF32", "readBufferF16", 1)
			if !widen {
				source = strings.Replace(source, "writeBufferF32", "writeBufferF16", 1)
			}
			source = strings.Replace(source, `(memory (export "memory") 1)`, `(memory (export "memory") 8)`, 1)
			rt, p := setup(t, c, nil)
			_, i := instance(t, rt, wat(t, source))
			for slot, typ := range []ElementType{TypeF16, outputType} {
				r, e := i.Invoke("create", uint64(typ), 65536)
				if e != nil || r[0]>>32 != 0 {
					t.Fatal(r, e)
				}
				if s := invoke(t, i, "bind", uint64(slot), uint64(uint32(r[0]))); s != 0 {
					t.Fatal(s)
				}
			}
			data := make([]byte, 65536*2)
			for n := 0; n < 65536; n++ {
				binary.LittleEndian.PutUint16(data[n*2:], uint16(n))
			}
			i.Write(0, data)
			if s := invoke(t, i, "set", 1, 0, 0, 0, 65536); s != 0 {
				t.Fatal(s)
			}
			if s := invoke(t, i, "dispatch", 1, 65536); s != 0 {
				t.Fatal(s, p.BufferSnapshot())
			}
			if s := invoke(t, i, "copy", 2, 0, 0, 131072, 65536); s != 0 {
				t.Fatal(s)
			}
			result, _ := i.Read(131072, 65536*outputSize)
			for n := 0; n < 65536; n++ {
				if widen {
					got := binary.LittleEndian.Uint32(result[n*4:])
					if got != halfToFloat(uint16(n)) {
						t.Fatalf("%04x -> %08x", n, got)
					}
				} else {
					got := binary.LittleEndian.Uint16(result[n*2:])
					want := uint16(n)
					if want&0x7c00 == 0x7c00 && want&1023 != 0 {
						want = 0x7e00
					}
					if got != want {
						t.Fatalf("%04x -> %04x, want %04x", n, got, want)
					}
				}
			}
		})
	}
}

func TestBufferHardwareInPlaceSeed(t *testing.T) {
	requireHardware(t)
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	c.Kernels[0].Bindings = []BindingConfig{{0, TypeF32, AccessReadWrite}}
	rt, p := setup(t, c, nil)
	_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 0)) (local.get $i) (f32.add (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 1)))`)))
	values, e := in.Invoke("create", uint64(TypeF32), 65)
	if e != nil {
		t.Fatal(e)
	}
	handle := uint64(uint32(values[0]))
	invoke(t, in, "bind", 0, handle)
	for n := uint32(0); n < 65; n++ {
		in.WriteFloat32Le(n*4, float32(n+10))
	}
	invoke(t, in, "set", handle, 0, 0, 0, 65)
	// Reuse old scratch three times. Its contents differ from the committed input.
	for pass := 0; pass < 3; pass++ {
		if got := invoke(t, in, "dispatch", 1, 65); got != V1OK {
			t.Fatal(got, p.BufferSnapshot())
		}
	}
	if got := invoke(t, in, "copy", handle, 0, 0, 0, 65); got != V1OK {
		t.Fatal(got)
	}
	for n := uint32(0); n < 65; n++ {
		got, _ := in.ReadFloat32Le(n * 4)
		if got != float32(n+13) {
			t.Fatal(n, got)
		}
	}
}

func TestBufferHardwareSavedFloatBits(t *testing.T) {
	requireHardware(t)
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	c.Kernels[0].Bindings = append(c.Kernels[0].Bindings, BindingConfig{2, TypeF32, AccessWrite})
	body := `(local $saved f32)
 (local.set $saved (call $read (call $get (i32.const 0)) (local.get $i)))
 (call $write (call $get (i32.const 1)) (local.get $i) (f32.add (local.get $saved) (f32.const 1)))
 (call $write (call $get (i32.const 2)) (local.get $i) (local.get $saved))`
	rt, p := setup(t, c, nil)
	_, in := instance(t, rt, wat(t, testModule(body)))
	bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc01234, 0xffa00001, 0x3f800000}
	for slot := uint64(0); slot < 3; slot++ {
		r, e := in.Invoke("create", uint64(TypeF32), uint64(len(bits)))
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(r[0])))
	}
	for j, v := range bits {
		in.WriteUint32Le(uint32(j*4), v)
	}
	invoke(t, in, "set", 1, 0, 0, 0, uint64(len(bits)))
	if got := invoke(t, in, "dispatch", 1, uint64(len(bits))); got != V1OK {
		t.Fatal(got, p.BufferSnapshot())
	}
	invoke(t, in, "copy", 3, 0, 0, 128, uint64(len(bits)))
	for j, want := range bits {
		got, _ := in.ReadUint32Le(uint32(128 + j*4))
		if got != want {
			t.Fatalf("saved float %08x -> %08x", want, got)
		}
	}
}

func TestBufferHardwareDeferredValues(t *testing.T) {
	requireHardware(t)
	for _, tc := range []struct {
		name, body string
		want       float32
	}{
		{"localSnapshot", `(local $x f32)
 (local.set $x (f32.const 1))
 (call $write (call $get (i32.const 1)) (local.get $i)
   (f32.add (local.get $x) (local.tee $x (f32.const 2))))`, 3},
		{"loadBeforeStore", `(local $saved f32)
 (local.set $saved (f32.add (call $read (call $get (i32.const 1)) (local.get $i)) (f32.const 1)))
 (call $write (call $get (i32.const 1)) (local.get $i) (f32.const 3))
 (call $write (call $get (i32.const 1)) (local.get $i) (local.get $saved))`, 10},
		{"zeroLocal", `(local $x f32)
 (call $write (call $get (i32.const 1)) (local.get $i) (f32.sub (local.get $x) (f32.const 1)))`, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := bufferConfig()
			c.Disabled = false
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Bindings[1].Access = AccessReadWrite
			rt, p := setup(t, c, nil)
			_, in := instance(t, rt, wat(t, testModule(tc.body)))
			for slot := uint64(0); slot < 2; slot++ {
				r, err := in.Invoke("create", uint64(TypeF32), 65)
				if err != nil {
					t.Fatal(err)
				}
				if got := invoke(t, in, "bind", slot, uint64(uint32(r[0]))); got != V1OK {
					t.Fatal(got)
				}
			}
			for j := uint32(0); j < 65; j++ {
				in.WriteFloat32Le(j*4, 9)
			}
			if got := invoke(t, in, "set", 2, 0, 0, 0, 65); got != V1OK {
				t.Fatal(got)
			}
			if got := invoke(t, in, "dispatch", 1, 65); got != V1OK {
				t.Fatal(got, p.BufferSnapshot())
			}
			if got := invoke(t, in, "copy", 2, 0, 0, 0, 65); got != V1OK {
				t.Fatal(got)
			}
			for j := uint32(0); j < 65; j++ {
				got, ok := in.ReadFloat32Le(j * 4)
				if !ok || got != tc.want {
					t.Fatalf("element %d: %v, want %v", j, got, tc.want)
				}
			}
		})
	}
}

func TestBufferHardwareIntegerData(t *testing.T) {
	requireHardware(t)
	for _, typ := range []ElementType{TypeI8, TypeU8, TypeI16, TypeU16, TypeI32, TypeU32} {
		t.Run(typ.spec().suffix, func(t *testing.T) {
			c := bufferConfig()
			c.Disabled = false
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Bindings = []BindingConfig{{0, typ, AccessWrite}, {1, typ, AccessWrite}}
			// Save the original index, overwrite it, restore it, then use it as data.
			source := testModule(`(local $saved i32)
    (local.set $saved (local.get $i)) (local.set $i (i32.const 0)) (local.set $i (local.get $saved))
    (call $write (call $get (i32.const 0)) (local.get $i) (i32.const -1))
    (call $write (call $get (i32.const 1)) (local.get $i) (local.get $saved))`)
			source = strings.ReplaceAll(source, "readBufferF32", "readBuffer"+typ.spec().suffix)
			source = strings.ReplaceAll(source, "writeBufferF32", "writeBuffer"+typ.spec().suffix)
			source = strings.ReplaceAll(source, "f32", "i32")
			rt, p := setup(t, c, nil)
			_, in := instance(t, rt, wat(t, source))
			const count = 257
			for slot := uint64(0); slot < 2; slot++ {
				r, e := in.Invoke("create", uint64(typ), count)
				if e != nil {
					t.Fatal(e)
				}
				invoke(t, in, "bind", slot, uint64(uint32(r[0])))
			}
			if got := invoke(t, in, "dispatch", 1, count); got != V1OK {
				t.Fatal(got, p.BufferSnapshot())
			}
			size := typ.spec().size
			for handle := uint64(1); handle <= 2; handle++ {
				if got := invoke(t, in, "copy", handle, 0, 0, 0, count); got != V1OK {
					t.Fatal(got)
				}
				data, _ := in.Read(0, uint32(count*size))
				for n := uint32(0); n < count; n++ {
					want := uint32(0xffffffff)
					if handle == 2 {
						want = n
					}
					var got uint32
					offset := uint64(n) * size
					switch size {
					case 1:
						got = uint32(data[offset])
						want &= 255
					case 2:
						got = uint32(binary.LittleEndian.Uint16(data[offset:]))
						want &= 65535
					case 4:
						got = binary.LittleEndian.Uint32(data[offset:])
					}
					if got != want {
						t.Fatal(n, got, want)
					}
				}
			}
		})
	}
}

func TestBindingPendingMapLifetime(t *testing.T) {
	requireHardware(t)
	backend, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	b := backend.(*gpuBackend)
	defer b.Close()
	for _, abort := range []bool{false, true} {
		for n := 0; n < 10; n++ {
			buffer, e := b.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: 16, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
			if e != nil {
				t.Fatal(e)
			}
			done := make(chan wgpu.MapAsyncStatus, 1)
			if e = buffer.TryMapAsync(wgpu.MapModeRead, 0, 16, func(s wgpu.MapAsyncStatus) { done <- s }); e != nil {
				buffer.Release()
				t.Fatal(e)
			}
			// The pinned backend defers map completion until polling or destruction.
			select {
			case <-done:
				buffer.Release()
				t.Fatal("map callback was not delayed; lifetime test needs a pending callback")
			default:
			}
			runtime.GC()
			pressure := make([]byte, 1<<20)
			pressure[len(pressure)-1] = 1
			if abort {
				buffer.Destroy()
			} else {
				b.device.Poll(true, nil)
			}
			runtime.KeepAlive(pressure)
			select {
			case status := <-done:
				if !abort && status != wgpu.MapAsyncStatusSuccess {
					buffer.Release()
					t.Fatal(status)
				}
				if abort && status == wgpu.MapAsyncStatusSuccess {
					buffer.Release()
					t.Fatal("destroyed pending map succeeded")
				}
			default:
				buffer.Release()
				t.Fatal("pending callback was not delivered")
			}
			if !abort {
				if e = buffer.TryUnmap(); e != nil {
					buffer.Release()
					t.Fatal(e)
				}
			}
			buffer.Release()
		}
	}
}

func TestBufferHardwareBindingCacheLifetime(t *testing.T) {
	requireHardware(t)
	backend, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	b := backend.(*gpuBackend)
	defer b.Close()
	shader := `struct Params { count: u32, a:u32,b:u32,c:u32 }
 @group(0) @binding(0) var<uniform> p:Params;
 @group(0) @binding(1) var<storage,read> input:array<f32>;
 @group(0) @binding(2) var<storage,read_write> output:array<f32>;
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 if (id.x<p.count) {output[id.x]=input[id.x]*2.0;}}
 `
	pipeline, e := b.BuildBuffer(shader)
	if e != nil {
		t.Fatal(e)
	}
	defer func() {
		if pipeline != nil {
			pipeline.Close()
		}
	}()
	uniform, e := b.AllocateUniform(16)
	if e != nil {
		t.Fatal(e)
	}
	defer uniform.Close()
	var input deviceBuffer
	input, e = b.AllocateBuffer(16)
	if e != nil {
		t.Fatal(e)
	}
	defer func() { input.Close() }()
	var outputs [2]deviceBuffer
	for n := range outputs {
		outputs[n], e = b.AllocateBuffer(16)
		if e != nil {
			t.Fatal(e)
		}
		defer outputs[n].Close()
	}
	ctx := context.Background()
	for n := 0; n < 8; n++ {
		op := BufferOperation{}
		if e = b.ExecuteBuffers(ctx, pipeline, uniform, []deviceBuffer{input, outputs[n%2]}, nil, 4, n == 0, &op); e != nil {
			t.Fatal(e)
		}
	}
	hits, misses := b.bindingCounts()
	if hits != 6 || misses != 2 {
		t.Fatal(hits, misses)
	}
	input.Close()
	for _, entry := range b.groups {
		if entry.group != nil {
			t.Fatal("closed input retained by cache")
		}
	}
	input, e = b.AllocateBuffer(16)
	if e != nil {
		t.Fatal(e)
	}
	var data [16]byte
	for n := 0; n < 4; n++ {
		binary.LittleEndian.PutUint32(data[n*4:], math.Float32bits(float32(n+5)))
	}
	if e = b.UploadBuffer(ctx, input, data[:]); e != nil {
		t.Fatal(e)
	}
	if e = b.ExecuteBuffers(ctx, pipeline, uniform, []deviceBuffer{input, outputs[0]}, nil, 3, true, &BufferOperation{}); e != nil {
		t.Fatal(e)
	}
	staging, e := b.AllocateReadback(16)
	if e != nil {
		t.Fatal(e)
	}
	defer staging.Close()
	if e = b.ReadBuffer(ctx, outputs[0], staging, 16, func(result []byte) {
		for n := 0; n < 3; n++ {
			got := math.Float32frombits(binary.LittleEndian.Uint32(result[n*4:]))
			if got != float32(n+5)*2 {
				t.Fatal(n, got)
			}
		}
	}); e != nil {
		t.Fatal(e)
	}
	hits, misses = b.bindingCounts()
	if hits != 6 || misses != 3 {
		t.Fatal(hits, misses)
	}
	// A parameter-only change can reuse the same group without stale resources.
	if e = b.ExecuteBuffers(ctx, pipeline, uniform, []deviceBuffer{input, outputs[0]}, nil, 4, true, &BufferOperation{}); e != nil {
		t.Fatal(e)
	}
	hits, misses = b.bindingCounts()
	if hits != 7 || misses != 3 {
		t.Fatal(hits, misses)
	}
	// Close layout/pipeline only after all matching groups have been released.
	pipeline.Close()
	pipeline = nil
	for _, entry := range b.groups {
		if entry.group != nil {
			t.Fatal("pipeline group retained")
		}
	}
}

func TestBufferHardwareTailBoundaries(t *testing.T) {
	requireHardware(t)
	testBufferTailBoundaries(t, true)
}
