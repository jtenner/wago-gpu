//go:build webgpu && cgo && linux

package wagogpu

import (
	"os"
	"sync"
	"testing"

	"github.com/jtenner/wago-gpu/internal/fixtures"
)

func requireHardware(t testing.TB) {
	t.Helper()
	if os.Getenv("WAGO_GPU_TEST") != "1" {
		t.Skip("set WAGO_GPU_TEST=1 to require a hardware GPU")
	}
}

func TestHardware(t *testing.T) {
	requireHardware(t)
	rt, p := setup(t, config(), nil)
	if s := p.Snapshot(); s.Device == "" {
		t.Fatalf("hardware GPU required: %s", s.Reason)
	} else {
		t.Log(s.Device)
	}
	for _, tc := range []struct {
		name string
		wasm []byte
	}{{"twice", fixtures.Twice}, {"square", fixtures.Square}} {
		t.Run(tc.name, func(t *testing.T) {
			m, i := instance(t, rt, tc.wasm)
			if e := fixtures.Prepare(i, 1_000_000); e != nil {
				t.Fatal(e)
			}
			for _, n := range []uint32{1, 257, 1024, 16384, 1_000_000, 1024, 1} {
				if e := fixtures.Fill(i, n); e != nil {
					t.Fatal(e)
				}
				for repeat := 0; repeat < 3; repeat++ {
					// Change data without changing buffer size or pipeline. This
					// checks that every call uploads fresh input and reads new output.
					if !i.WriteFloat32Le((n/2)*4, float32(repeat+17)) {
						t.Fatal("write changed input")
					}
					invoke(t, i, "cpu", 0, uint64(n)*4, uint64(n))
					if got := invoke(t, i, "gpu", 0, uint64(n)*8, uint64(n)); got != Success {
						t.Fatalf("status %d: %+v", got, p.Snapshot())
					}
					if e := fixtures.Verify(i, n); e != nil {
						t.Fatal(e)
					}
				}
			}
			m.Close()
			i.Close()
			if s := p.Snapshot(); s.Pipelines != 0 || s.BufferBytes != 0 {
				t.Fatal(s)
			}
		})
	}
}

func TestHardwareConcurrentInstances(t *testing.T) {
	requireHardware(t)
	rt, p := setup(t, config(), nil)
	if p.Snapshot().Device == "" {
		t.Fatal(p.Snapshot().Reason)
	}
	var wg sync.WaitGroup
	for j := 0; j < 4; j++ {
		_, i := instance(t, rt, fixtures.Square)
		if e := fixtures.Prepare(i, 1024); e != nil {
			t.Fatal(e)
		}
		invoke(t, i, "cpu_batch", 0, 4096, 1024, 3)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < 3; k++ {
				v, e := i.Invoke("gpu_batch", 0, 8192, 1024, 3)
				if e != nil || len(v) != 1 || int32(v[0]) != Success {
					t.Errorf("%v %v", v, e)
					return
				}
			}
			if _, e := fixtures.VerifyKernel(i, 1024, "square", 3); e != nil {
				t.Error(e)
			}
		}()
	}
	wg.Wait()
	if p.Snapshot().Successes != 12 {
		t.Fatal(p.Snapshot())
	}
}

func TestHardwareShaderFailure(t *testing.T) {
	requireHardware(t)
	b, e := openBackend()
	if e != nil {
		t.Fatal(e)
	}
	defer b.Close()
	if p, e := b.Compile("this is not WGSL"); e == nil {
		p.Close()
		t.Fatal("invalid shader accepted")
	}
}

func TestHardwareBatch(t *testing.T) {
	requireHardware(t)
	cfg := config()
	cfg.ProfileStages = true
	rt, p := setup(t, cfg, nil)
	for _, tc := range []struct {
		name   string
		wasm   []byte
		passes uint64
	}{{"twice", fixtures.Twice, 8}, {"square", fixtures.Square, 3}, {"heavy", fixtures.Heavy, 8}, {"heavy", fixtures.Heavy, 64}} {
		m, i := instance(t, rt, tc.wasm)
		if e := fixtures.Prepare(i, 16384); e != nil {
			t.Fatal(e)
		}
		for _, n := range []uint64{257, 16384, 1024} {
			if e := fixtures.Fill(i, uint32(n)); e != nil {
				t.Fatal(e)
			}
			invoke(t, i, "cpu_batch", 0, n*4, n, tc.passes)
			for repeat := 0; repeat < 2; repeat++ {
				if got := invoke(t, i, "gpu_batch", 0, n*8, n, tc.passes); got != Success {
					t.Fatal(got, p.Snapshot())
				}
				if delta, e := fixtures.VerifyKernel(i, uint32(n), tc.name, uint32(tc.passes)); e != nil {
					t.Fatal(e)
				} else {
					t.Log(tc.name, n, tc.passes, "max absolute error", delta)
				}
				last := p.Snapshot().Last
				if last.Passes != uint32(tc.passes) || last.UploadBytes != n*4 || last.DownloadBytes != n*4 || last.Compute <= 0 {
					t.Fatal(last)
				}
			}
			// A batch with one transfer per pass gives the same result.
			if got := invoke(t, i, "gpu_separate", 0, n*8, n, tc.passes); got != Success {
				t.Fatal(got, p.Snapshot())
			}
			if _, e := fixtures.VerifyKernel(i, uint32(n), tc.name, uint32(tc.passes)); e != nil {
				t.Fatal(e)
			}
			if got := invoke(t, i, "gpu_batch", 0, 0, n, tc.passes); got != Success {
				t.Fatal(got, p.Snapshot())
			}
			data, ok := i.Read(0, uint32(n)*4)
			if !ok || !i.Write(uint32(n)*8, data) {
				t.Fatal("copy in-place result")
			}
			if _, e := fixtures.VerifyKernel(i, uint32(n), tc.name, uint32(tc.passes)); e != nil {
				t.Fatal(e)
			}

		}
		i.Close()
		m.Close()
		if s := p.Snapshot(); s.BufferBytes != 0 || s.Pipelines != 0 {
			t.Fatal(s)
		}
	}
}

// Destroy only this test's readback buffer to inject a real WebGPU validation
// failure. This tests fail-closed behavior, not physical GPU removal.
func TestHardwareReadbackFailure(t *testing.T) {
	requireHardware(t)
	rt, p := setup(t, config(), nil)
	_, i := instance(t, rt, fixtures.Twice)
	if e := fixtures.Prepare(i, 257); e != nil {
		t.Fatal(e)
	}
	invoke(t, i, "cpu_batch", 0, 1028, 257, 8)
	if got := invoke(t, i, "gpu_batch", 0, 2056, 257, 8); got != Success {
		t.Fatal(got, p.Snapshot())
	}
	p.mu.Lock()
	for _, cached := range p.cache {
		cached.program.(*gpuProgram).readback.Destroy()
	}
	p.mu.Unlock()
	marker := []byte{9, 8, 7, 6}
	i.Write(2056, marker)
	if got := invoke(t, i, "gpu_batch", 0, 2056, 257, 8); got != Fallback {
		t.Fatal(got)
	}
	data, _ := i.Read(2056, 4)
	if string(data) != string(marker) {
		t.Fatal("failed GPU call changed memory")
	}
	if !p.Snapshot().GPUFailed {
		t.Fatal("failed GPU remained enabled")
	}
	if got := invoke(t, i, "run_batch", 0, 2056, 257, 8); got != Fallback {
		t.Fatal(got)
	}
	if e := fixtures.Verify(i, 257); e != nil {
		t.Fatal(e)
	}
}
