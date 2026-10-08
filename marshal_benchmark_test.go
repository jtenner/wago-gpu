package wagogpu

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/jtenner/wago-gpu/internal/fixtures"
)

// Transfers, dispatch, guest copies, and host-import overhead are all timed.
// Runtime, module, buffers, and pipelines stay live between operations.
func BenchmarkBufferMarshal(b *testing.B) {
	if os.Getenv("WAGO_GPU_TEST") != "1" {
		b.Skip("requires a real GPU and -tags webgpu")
	}
	for _, count := range []uint32{1024, 16384, 1000000, 10000000} {
		for _, profile := range []bool{false, true} {
			b.Run(fmt.Sprintf("Fresh/%d/profile=%v", count, profile), func(b *testing.B) {
				c := bufferConfig()
				c.Disabled, c.ProfileStages = false, profile
				c.Kernels = c.Kernels[:1]
				c.Kernels[0].Export, c.Kernels[0].CPUExport = "wago_gpu.kernel.twice", "wago_gpu.cpu.twice"
				rt, p := setup(b, c, nil)
				_, in := instance(b, rt, fixtures.BufferWork)
				if err := fixtures.Prepare(in, count); err != nil {
					b.Fatal(err)
				}
				call := func(name string, args ...uint64) {
					if r, err := in.Invoke(name, args...); err != nil || len(r) > 0 && r[0] != 0 {
						b.Fatalf("%s: %v %v", name, r, err)
					}
				}
				call("setup", uint64(count))
				call("direct.twice", uint64(count))
				step := func() {
					call("upload")
					call("dispatch", 1, uint64(count))
					call("download")
				}
				step()
				step()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					step()
				}
				b.StopTimer()
				if _, err := fixtures.VerifyKernel(in, count, "twice", 1); err != nil {
					b.Fatal(err)
				}
				if p.BufferSnapshot().Last.Status != V1OK {
					b.Fatal("copy failed")
				}
			})
		}
	}
	for _, count := range []uint32{1000000, 10000000} {
		b.Run(fmt.Sprintf("Patch/%d", count), func(b *testing.B) {
			c := bufferConfig()
			c.Disabled = false
			c.Kernels = c.Kernels[:1]
			rt, p := setup(b, c, nil)
			_, in := instance(b, rt, wat(b, testModule(`(call $write (call $get (i32.const 1)) (local.get $i)
 (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)))
			for slot, size := range []uint64{uint64(count), 1} {
				r, err := in.Invoke("create", uint64(TypeF32), size)
				if err != nil || r[0]>>32 != 0 {
					b.Fatal(r, err)
				}
				invoke(b, in, "bind", uint64(slot), uint64(uint32(r[0])))
			}
			step := func(value float32) {
				in.WriteFloat32Le(0, value)
				if invoke(b, in, "set", 1, 0, 0, 0, 1) != V1OK ||
					invoke(b, in, "dispatch", 1, 1) != V1OK ||
					invoke(b, in, "copy", 2, 0, 0, 4, 1) != V1OK {
					b.Fatal(p.BufferSnapshot())
				}
			}
			step(1)
			step(1)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				step(float32(1 + i%2))
			}
			b.StopTimer()
			got, _ := in.ReadFloat32Le(4)
			if got != float32(2*(1+(b.N-1)%2)) {
				b.Fatal(got)
			}
		})
	}
	for _, count := range []uint32{1024, 1000000} {
		b.Run(fmt.Sprintf("FourInputs/%d", count), func(b *testing.B) {
			c := bufferConfig()
			c.Disabled = false
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Bindings = []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessRead}, {2, TypeF32, AccessRead}, {3, TypeF32, AccessReadWrite}}
			source := testModule(`(call $write (call $get (i32.const 3)) (local.get $i)
 (f32.add (f32.add (call $read (call $get (i32.const 0)) (local.get $i))
 (call $read (call $get (i32.const 1)) (local.get $i)))
 (f32.add (call $read (call $get (i32.const 2)) (local.get $i))
 (call $read (call $get (i32.const 3)) (local.get $i)))))`)
			source = strings.Replace(source, `(memory (export "memory") 1)`, fmt.Sprintf(`(memory (export "memory") %d)`, (uint64(count)*8+65535)/65536), 1)
			rt, p := setup(b, c, nil)
			_, in := instance(b, rt, wat(b, source))
			in.WriteFloat32Le(0, 1)
			for slot := uint64(0); slot < 4; slot++ {
				r, err := in.Invoke("create", uint64(TypeF32), uint64(count))
				if err != nil || r[0]>>32 != 0 {
					b.Fatal(r, err)
				}
				invoke(b, in, "bind", slot, uint64(uint32(r[0])))
			}
			step := func() {
				for handle := uint64(1); handle <= 4; handle++ {
					if invoke(b, in, "set", handle, 0, 0, 0, uint64(count)) != V1OK {
						b.Fatal("set failed")
					}
				}
				if invoke(b, in, "dispatch", 1, uint64(count)) != V1OK ||
					invoke(b, in, "copy", 4, 0, 0, uint64(count)*4, uint64(count)) != V1OK {
					b.Fatal(p.BufferSnapshot())
				}
			}
			step()
			step()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				step()
			}
			b.StopTimer()
			for j := uint32(0); j < count; j++ {
				got, _ := in.ReadFloat32Le(count*4 + j*4)
				want := float32(0)
				if j == 0 {
					want = 4
				}
				if got != want {
					b.Fatal(j, got, want)
				}
			}
		})
	}
}
