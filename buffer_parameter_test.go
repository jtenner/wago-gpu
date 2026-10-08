package wagogpu

import (
	"context"
	"testing"
)

func TestBufferParameterReuse(t *testing.T) { testBufferParameterReuse(t, false) }

func testBufferParameterReuse(t *testing.T, hardware bool) {
	for _, profile := range []bool{false, true} {
		t.Run(map[bool]string{false: "plain", true: "profile"}[profile], func(t *testing.T) {
			c := bufferConfig()
			c.Disabled, c.ProfileStages = false, profile
			c.Kernels = c.Kernels[:1]
			var factory func() (backend, error)
			if !hardware {
				factory = func() (backend, error) { return &fakeBufferDevice{}, nil }
			}
			rt, p := setup(t, c, factory)
			_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i)
 (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)))
			for slot := uint64(0); slot < 2; slot++ {
				values, e := in.Invoke("create", uint64(TypeF32), 4)
				if e != nil || values[0]>>32 != 0 {
					t.Fatal(values, e)
				}
				invoke(t, in, "bind", slot, uint64(uint32(values[0])))
			}
			var tail float32
			for pass, count := range []uint32{4, 4, 3, 3, 4, 4} {
				if pass == 5 {
					// The count stays the same, but its buffer was evicted. The
					// replacement must receive parameters before any dispatch.
					p.mu.Lock()
					for _, state := range p.buffers.instances {
						p.evictBufferPool(state)
					}
					p.mu.Unlock()
				}
				value := float32(pass + 1)
				for j := uint32(0); j < 4; j++ {
					in.WriteFloat32Le(j*4, value)
				}
				invoke(t, in, "set", 1, 0, 0, 0, 4)
				if got := invoke(t, in, "dispatch", 1, uint64(count)); got != V1OK {
					t.Fatal(got, p.BufferSnapshot())
				}
				op := p.BufferSnapshot().Last
				wantUpload := pass == 0 || pass == 2 || pass == 4 || pass == 5
				if (op.ParameterUploadCount == 1) != wantUpload || op.ParameterUploadBytes != uint64(op.ParameterUploadCount)*16 {
					t.Fatal("wrong parameter traffic", pass, op)
				}
				if profile && count == 4 && op.DeviceCopy != 0 {
					t.Fatal("profiled a seed submission with no seeds", op)
				}
				invoke(t, in, "copy", 2, 0, 0, 32, 4)
				if count == 4 {
					tail = value * 2
				}
				for j := uint32(0); j < 4; j++ {
					want := value * 2
					if j >= count {
						want = tail
					}
					actual, _ := in.ReadFloat32Le(32 + j*4)
					if actual != want {
						t.Fatal("stale count or damaged prefix tail", pass, j, actual, want)
					}
				}
			}
			if p.BufferSnapshot().Totals.ParameterUploadCount != 4 {
				t.Fatal("parameter totals")
			}
			if e := rt.CloseContext(context.Background()); e != nil {
				t.Fatal(e)
			}
			if p.BufferSnapshot().RuntimeBufferBytes != 0 {
				t.Fatal("cleanup leak")
			}
		})
	}
}
