package wagogpu

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
)

func TestBufferSeedRules(t *testing.T) { testBufferSeedRules(t, false) }

// Run the same cases with fake storage and with actual compiled shaders.
func testBufferSeedRules(t *testing.T, hardware bool) {
	double := `(call $write (call $get (i32.const 1)) (local.get $i)
 (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`
	for _, tc := range []struct {
		name, body, mode string
		count            uint32
		seed             bool
		want             [4]float32
	}{
		{"full", double, "", 4, false, [4]float32{2, 4, 6, 8}},
		{"prefix", double, "", 3, true, [4]float32{2, 4, 6, 9}},
		{"read-before-write", `(call $write (call $get (i32.const 1)) (local.get $i)
 (f32.add (call $read (call $get (i32.const 1)) (local.get $i)) (f32.const 1)))`, "increment", 4, true, [4]float32{10, 10, 10, 10}},
		{"store-first", `(call $write (call $get (i32.const 1)) (local.get $i) (f32.const 3))
 (call $write (call $get (i32.const 1)) (local.get $i)
 (f32.add (call $read (call $get (i32.const 1)) (local.get $i)) (f32.const 1)))`, "store-first", 4, false, [4]float32{4, 4, 4, 4}},
		{"saved-old-value", `(local $old f32)
 (local.set $old (call $read (call $get (i32.const 1)) (local.get $i)))
 (call $write (call $get (i32.const 1)) (local.get $i) (f32.const 3))
 (call $write (call $get (i32.const 1)) (local.get $i) (f32.add (local.get $old) (f32.const 1)))`, "increment", 4, true, [4]float32{10, 10, 10, 10}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := bufferConfig()
			c.Disabled = false
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Bindings[1].Access = AccessReadWrite
			var factory func() (backend, error)
			if !hardware {
				factory = func() (backend, error) { return &fakeBufferDevice{mode: tc.mode}, nil }
			}
			rt, p := setup(t, c, factory)
			_, in := instance(t, rt, wat(t, testModule(tc.body)))
			var handles [2]uint64
			for slot := uint64(0); slot < 2; slot++ {
				v, e := in.Invoke("create", uint64(TypeF32), 4)
				if e != nil || v[0]>>32 != 0 {
					t.Fatal(v, e)
				}
				handles[slot] = uint64(uint32(v[0]))
				invoke(t, in, "bind", slot, handles[slot])
			}
			for j := uint32(0); j < 4; j++ {
				in.WriteFloat32Le(j*4, float32(j+1))
				in.WriteFloat32Le(16+j*4, 9)
			}
			invoke(t, in, "set", handles[0], 0, 0, 0, 4)
			invoke(t, in, "set", handles[1], 0, 0, 16, 4)
			// A reused output contains unrelated data. Coverage and read order,
			// rather than fresh zero-filled device storage, must make it safe.
			func() {
				p.mu.Lock()
				defer p.mu.Unlock()
				for _, state := range p.buffers.instances {
					r, status := p.takeBufferScratch(state, 16)
					if status != V1OK {
						t.Fatal(status)
					}
					var dirty [16]byte
					for n := 0; n < 16; n += 4 {
						binary.LittleEndian.PutUint32(dirty[n:], math.Float32bits(1234))
					}
					if e := p.bufferDevice().UploadBuffer(context.Background(), r, dirty[:]); e != nil {
						t.Fatal(e)
					}
					p.keepBufferScratch(state, r, 16)
				}
			}()
			if got := invoke(t, in, "dispatch", 1, uint64(tc.count)); got != V1OK {
				t.Fatal(got, p.BufferSnapshot())
			}
			op := p.BufferSnapshot().Last
			seedBytes := uint64(op.DeviceCopyCount) * 16
			if tc.name == "prefix" {
				seedBytes = 4
			}
			if (op.DeviceCopyCount != 0) != tc.seed || op.DeviceCopyBytes != seedBytes {
				t.Fatal("wrong seed traffic", op)
			}
			var inputBytes uint64
			if tc.mode == "" {
				inputBytes = 16
			}
			if !tc.seed && op.LogicalUploadBytes != inputBytes {
				t.Fatal("old output was uploaded", op)
			}
			if got := invoke(t, in, "copy", handles[1], 0, 0, 32, 4); got != V1OK {
				t.Fatal(got)
			}
			for j, want := range tc.want {
				v, ok := in.ReadFloat32Le(uint32(32 + j*4))
				if !ok || v != want {
					t.Fatal("dirty scratch or old value leaked", j, v, want)
				}
			}
			if e := rt.CloseContext(context.Background()); e != nil {
				t.Fatal(e)
			}
			if p.BufferSnapshot().RuntimeBufferBytes != 0 {
				t.Fatal("allocation leak")
			}
		})
	}
}

func TestBufferTailBoundaries(t *testing.T) { testBufferTailBoundaries(t, false) }
func testBufferTailBoundaries(t *testing.T, hardware bool) {
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	var factory func() (backend, error)
	if !hardware {
		factory = func() (backend, error) { return &fakeBufferDevice{}, nil }
	}
	rt, p := setup(t, c, factory)
	_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)))
	const total = 513
	for slot := uint64(0); slot < 2; slot++ {
		r, e := in.Invoke("create", uint64(TypeF32), total)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(r[0])))
	}
	for n := uint32(0); n < total; n++ {
		in.WriteFloat32Le(n*4, float32(n+1))
		in.WriteFloat32Le(4096+n*4, 9)
	}
	invoke(t, in, "set", 1, 0, 0, 0, total)
	for _, count := range []uint32{1, 255, 256, 257, 512, 513, 256} {
		invoke(t, in, "set", 2, 0, 0, 4096, total)
		if status := invoke(t, in, "dispatch", 1, uint64(count)); status != V1OK {
			t.Fatal(status, p.BufferSnapshot())
		}
		op := p.BufferSnapshot().Last
		if op.DeviceCopyBytes != uint64(total-count)*4 {
			t.Fatal("seed contains overwritten prefix", count, op.DeviceCopyBytes)
		}
		invoke(t, in, "copy", 2, 0, 0, 8192, total)
		for n := uint32(0); n < total; n++ {
			want := float32(9)
			if n < count {
				want = float32(n+1) * 2
			}
			got, _ := in.ReadFloat32Le(8192 + n*4)
			if got != want {
				t.Fatal(count, n, got, want)
			}
		}
	}
}
