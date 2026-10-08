package wagogpu

import (
	"bytes"
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"wago-gpu/internal/fixtures"
)

func TestBatchAndFallback(t *testing.T) {
	for _, disabled := range []bool{false, true} {
		cfg := config()
		cfg.Disabled = disabled
		b := &fakeBackend{}
		rt, p := setup(t, cfg, func() (backend, error) { return b, nil })
		_, i := instance(t, rt, fixtures.Twice)
		for _, passes := range []uint64{1, 2, 8, 64} {
			if e := fixtures.Prepare(i, 257); e != nil {
				t.Fatal(e)
			}
			invoke(t, i, "cpu_batch", 0, 1028, 257, passes)
			want := Success
			if disabled {
				want = Fallback
			}
			if got := invoke(t, i, "run_batch", 0, 2056, 257, passes); got != want {
				t.Fatal(got, p.Snapshot())
			}
			if e := fixtures.Verify(i, 257); e != nil {
				t.Fatal(e)
			}
			// Also apply the entire batch in place.
			i.WriteFloat32Le(0, 1)
			if got := invoke(t, i, "run_batch", 0, 0, 1, passes); got != want {
				t.Fatal(got)
			}
			x, _ := i.ReadFloat32Le(0)
			if x != float32(math.Ldexp(1, int(passes))) {
				t.Fatal(x)
			}
		}
	}
}
func TestBatchInvalidAndAtomicFailure(t *testing.T) {
	b := &fakeBackend{}
	rt, p := setup(t, config(), func() (backend, error) { return b, nil })
	_, i := instance(t, rt, fixtures.Twice)
	i.WriteFloat32Le(0, 3)
	marker := []byte{1, 2, 3, 4}
	i.Write(16, marker)
	for _, passes := range []uint64{0, 65, 0xffffffff} {
		if got := invoke(t, i, "run_batch", 0, 16, 1, passes); got != InvalidBatch {
			t.Fatal(got)
		}
	}
	if got := invoke(t, i, "gpu_batch", 65535, 16, 1, 2); got != InvalidRange {
		t.Fatal(got)
	}
	if b.runs != 0 {
		t.Fatal("invalid request reached device")
	}
	b.runErr = errors.New("injected device failure")
	if got := invoke(t, i, "gpu_batch", 0, 16, 1, 8); got != Fallback {
		t.Fatal(got)
	}
	data, _ := i.Read(16, 4)
	if !bytes.Equal(data, marker) {
		t.Fatal("partial commit")
	}
	b.runErr = nil
	// The device remains disabled after a failure, including for another module.
	_, other := instance(t, rt, fixtures.Twice)
	invoke(t, other, "run_batch", 0, 16, 1, 2)
	if !p.Snapshot().GPUFailed || b.runs != 1 || b.builds != 1 {
		t.Fatal(p.Snapshot(), b)
	}
	if got := invoke(t, i, "run_batch", 0, 16, 1, 8); got != Fallback {
		t.Fatal(got)
	}
	x, _ := i.ReadFloat32Le(16)
	if x != 768 {
		t.Fatal("full CPU fallback failed", x)
	}
}
func TestMinimumElements(t *testing.T) {
	cfg := config()
	cfg.MinElements = 32
	b := &fakeBackend{}
	rt, p := setup(t, cfg, func() (backend, error) { return b, nil })
	_, i := instance(t, rt, fixtures.Twice)
	for _, n := range []uint64{31, 32} {
		if e := fixtures.Fill(i, uint32(n)); e != nil {
			t.Fatal(e)
		}
		invoke(t, i, "cpu_batch", 0, n*4, n, 8)
		got := invoke(t, i, "run_batch", 0, n*8, n, 8)
		want := Success
		if n < 32 {
			want = Fallback
		}
		if got != want {
			t.Fatal(got, p.Snapshot())
		}
		if e := fixtures.Verify(i, uint32(n)); e != nil {
			t.Fatal(e)
		}
	}
	if b.runs != 1 || p.Snapshot().GPUFailed {
		t.Fatal(b, p.Snapshot())
	}
	for _, cfg := range []Config{{KernelExport: "kernel", RunTimeout: -time.Second}, {KernelExport: "kernel", MinElements: 33, MaxElements: 32}} {
		if _, e := New(cfg); e == nil {
			t.Fatal("invalid configuration accepted")
		}
	}
}
func TestRunTimeoutDisablesFurtherGPUCalls(t *testing.T) {
	cfg := config()
	cfg.RunTimeout = 2 * time.Millisecond
	b := &fakeBackend{wait: true}
	rt, p := setup(t, cfg, func() (backend, error) { return b, nil })
	_, i := instance(t, rt, fixtures.Twice)
	i.WriteFloat32Le(0, 3)
	if got := invoke(t, i, "run_batch", 0, 16, 1, 8); got != Fallback {
		t.Fatal(got)
	}
	x, _ := i.ReadFloat32Le(16)
	if x != 768 || !p.Snapshot().GPUFailed || b.runs != 1 {
		t.Fatal(x, b, p.Snapshot())
	}
	invoke(t, i, "run", 0, 16, 1)
	if b.runs != 1 {
		t.Fatal("failed device reused")
	}
}
func TestWaitGPU(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	calls := 0
	if e := waitGPU(ctx, func() (bool, error) { calls++; return true, nil }); !errors.Is(e, context.Canceled) || calls != 0 {
		t.Fatal(e, calls)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	calls = 0
	if e := waitGPU(ctx, func() (bool, error) {
		calls++
		if calls == 3 {
			cancel()
		}
		return false, nil
	}); !errors.Is(e, context.Canceled) || calls != 3 {
		t.Fatal(e, calls)
	}
	sentinel := errors.New("device lost")
	if e := waitGPU(context.Background(), func() (bool, error) { return false, sentinel }); !errors.Is(e, sentinel) {
		t.Fatal(e)
	}
	if e := waitGPU(context.Background(), func() (bool, error) { return true, nil }); e != nil {
		t.Fatal(e)
	}
	ctx, cancel = context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e := waitGPU(ctx, func() (bool, error) { return false, nil }); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
}

func TestVerificationRejectsWrongOrNonfiniteResults(t *testing.T) {
	cfg := config()
	cfg.Disabled = true
	rt, _ := setup(t, cfg, nil)
	_, i := instance(t, rt, fixtures.Heavy)
	if e := fixtures.Prepare(i, 32); e != nil {
		t.Fatal(e)
	}
	invoke(t, i, "cpu_batch", 0, 128, 32, 8)
	invoke(t, i, "run_batch", 0, 256, 32, 8)
	if delta, e := fixtures.VerifyKernel(i, 32, "heavy", 8); e != nil || delta != 0 {
		t.Fatal(delta, e)
	}
	for _, v := range []float32{100, float32(math.Inf(1)), float32(math.NaN())} {
		i.WriteFloat32Le(256, v)
		if _, e := fixtures.VerifyKernel(i, 32, "heavy", 8); e == nil {
			t.Fatal("invalid result accepted", v)
		}
	}
}
