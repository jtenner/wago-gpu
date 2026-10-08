package wagogpu

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/jtenner/wago-gpu/internal/fixtures"
)

func TestBulkCPURunners(t *testing.T) {
	for _, name := range []string{"twice", "square", "copy", "polynomial"} {
		t.Run(name, func(t *testing.T) {
			c := bufferConfig()
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Export = "wago_gpu.kernel." + name
			c.Kernels[0].CPUExport = "wago_gpu.cpu." + name
			rt, p := setup(t, c, nil)
			_, in := instance(t, rt, fixtures.BufferWork)
			const count = 257
			if e := fixtures.Prepare(in, count); e != nil {
				t.Fatal(e)
			}
			invoke(t, in, "setup", count)
			invoke(t, in, "upload")
			invoke(t, in, "direct."+name, count)
			if status := invoke(t, in, "dispatch", 1, count); status != V1CPUFallback {
				t.Fatal(status)
			}
			sequence := p.BufferSnapshot().Last.Sequence
			invoke(t, in, c.Kernels[0].CPUExport, count)
			if _, e := fixtures.VerifyKernel(in, count, name, 1); e != nil {
				t.Fatal(e)
			}
			// Independent values detect errors shared by the generated GPU
			// kernel and its bulk CPU math helper.
			for j := uint32(0); j < count; j++ {
				want := float32(int32(j%2049)-1024) / 32
				switch name {
				case "twice":
					want *= 2
				case "square":
					want = want*want + 1
				case "polynomial":
					for step := 0; step < 32; step++ {
						want = float32(want * 0.875)
						want = float32(want + 0.25)
					}
				}
				got, _ := in.ReadFloat32Le(count*8 + j*4)
				if got != want {
					t.Fatal("CPU oracle", j, got, want)
				}
			}

			totals := p.BufferSnapshot().Totals
			if totals.GuestCopyBytes != count*4 || totals.GuestSetBytes != count*8 {
				t.Fatal("missing bulk transfers", totals)
			}
			if p.BufferSnapshot().Last.Sequence != sequence+2 {
				t.Fatal("runner did not use exactly two bulk imports")
			}
			invoke(t, in, "download")
			if _, e := fixtures.VerifyKernel(in, count, name, 1); e != nil {
				t.Fatal(e)
			}
			// Prefix writes must preserve the output tail, including its last element.
			tail, _ := in.ReadFloat32Le(count*8 + (count-1)*4)
			in.WriteFloat32Le(0, 7)
			invoke(t, in, "upload")
			invoke(t, in, c.Kernels[0].CPUExport, 256)
			invoke(t, in, "download")
			got, _ := in.ReadFloat32Le(count*8 + (count-1)*4)
			if got != tail {
				t.Fatal("prefix changed tail", got, tail)
			}
		})
	}
}

func TestCPUBulkDeadlineIncludesLockWait(t *testing.T) {
	for _, tc := range []struct {
		name          string
		offset, count uint64
		want          int32
	}{
		{"expired-set", 0, 4, V1Cancelled}, {"invalid-set", 65536, 4, V1InvalidRange}, {"zero-set", 0, 0, V1OK},
		{"expired-copy", 0, 4, V1Cancelled}, {"invalid-copy", 65536, 4, V1InvalidRange}, {"zero-copy", 0, 0, V1OK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt, p := setup(t, bufferConfig(), nil)
			_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`)))
			invoke(t, in, "create", uint64(TypeF32), 4)
			in.WriteFloat32Le(0, 42)
			invoke(t, in, "set", 1, 0, 0, 0, 4)
			export := "set"
			if tc.name[len(tc.name)-4:] == "copy" {
				export = "copy"
			}
			var version uint64
			for _, state := range p.buffers.instances {
				version = state.buffers[1].version
			}
			p.mu.Lock()
			p.config.RunTimeout = 20 * time.Millisecond
			started, done := make(chan struct{}), make(chan error, 1)
			go func() {
				close(started)
				v, e := in.Invoke(export, 1, 0, 0, tc.offset, tc.count)
				if e == nil && (len(v) != 1 || int32(v[0]) != tc.want) {
					e = fmt.Errorf("status %v want %d", v, tc.want)
				}
				done <- e
			}()
			<-started
			time.Sleep(100 * time.Millisecond)
			p.mu.Unlock()
			if e := <-done; e != nil {
				t.Fatal(e)
			}
			for _, state := range p.buffers.instances {
				if state.buffers[1].version != version {
					t.Fatal("failed/no-op transfer changed version")
				}
			}
			v, _ := in.ReadFloat32Le(0)
			if v != 42 {
				t.Fatal("failed transfer changed guest memory", v)
			}
			if e := rt.CloseContext(context.Background()); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestCPUFallbackDeadlineIncludesLockWait(t *testing.T) {
	for _, tc := range []struct {
		count uint64
		want  int32
	}{{4, V1Cancelled}, {5, V1InvalidRange}, {0, V1OK}} {
		t.Run(fmt.Sprint(tc.count), func(t *testing.T) {
			c := bufferConfig()
			c.Kernels = c.Kernels[:1]
			rt, p := setup(t, c, nil)
			_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`)))
			for slot := uint64(0); slot < 2; slot++ {
				r, e := in.Invoke("create", uint64(TypeF32), 4)
				if e != nil {
					t.Fatal(e)
				}
				invoke(t, in, "bind", slot, uint64(uint32(r[0])))
			}
			p.mu.Lock()
			p.config.RunTimeout = 20 * time.Millisecond
			started, done := make(chan struct{}), make(chan error, 1)
			go func() {
				close(started)
				v, e := in.Invoke("dispatch", 1, tc.count)
				if e == nil && (len(v) != 1 || int32(v[0]) != tc.want) {
					e = fmt.Errorf("status %v want %d", v, tc.want)
				}
				done <- e
			}()
			<-started
			time.Sleep(100 * time.Millisecond)
			p.mu.Unlock()
			if e := <-done; e != nil {
				t.Fatal(e)
			}
			for _, state := range p.buffers.instances {
				for _, b := range state.buffers {
					if b.version != 1 || !b.cpuCurrent {
						t.Fatal("expired/no-op dispatch changed contents")
					}
				}
			}
		})
	}
}

type timeoutReadback struct {
	*fakeBufferDevice
	deadlineSeen bool
}

func (d *timeoutReadback) ReadBuffer(ctx context.Context, source, staging deviceBuffer, size uint64, commit func([]byte)) error {
	_, d.deadlineSeen = ctx.Deadline()
	if !d.deadlineSeen {
		return fmt.Errorf("GPU fallback readback has no deadline")
	}
	<-ctx.Done()
	return ctx.Err()
}
func TestCPUFallbackReadbackKeepsTimer(t *testing.T) {
	d := &timeoutReadback{fakeBufferDevice: &fakeBufferDevice{}}
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)))
	for slot := uint64(0); slot < 2; slot++ {
		r, e := in.Invoke("create", uint64(TypeF32), 4)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(r[0])))
	}
	if status := invoke(t, in, "dispatch", 1, 4); status != V1OK {
		t.Fatal(status)
	}
	p.mu.Lock()
	p.config.Disabled = true
	p.config.RunTimeout = 20 * time.Millisecond
	p.mu.Unlock()
	if status := invoke(t, in, "dispatch", 1, 4); status != V1Cancelled || !d.deadlineSeen {
		t.Fatal(status, d.deadlineSeen)
	}
	for _, state := range p.buffers.instances {
		output := state.buffers[2]
		if output.version != 2 || output.cpuCurrent || !output.lost {
			t.Fatal("failed readback changed committed version or enabled stale CPU contents")
		}
	}
}
