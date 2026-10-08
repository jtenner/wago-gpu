package wagogpu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"sync"
	"testing"

	wago "github.com/wago-org/wago"
	"wago-gpu/internal/fixtures"
)

type fakeBackend struct {
	wait                                bool
	compileErr, runErr                  error
	builds, closes, runs, programCloses int
}

func (b *fakeBackend) Info() string { return "TEST DOUBLE; not a GPU" }
func (b *fakeBackend) Close()       { b.closes++ }
func (b *fakeBackend) Compile(string) (program, error) {
	b.builds++
	if b.compileErr != nil {
		return nil, b.compileErr
	}
	return &fakeProgram{b: b}, nil
}

type fakeProgram struct{ b *fakeBackend }

func (p *fakeProgram) Close()              { p.b.programCloses++ }
func (p *fakeProgram) BufferBytes() uint64 { return 0 }
func (p *fakeProgram) Run(ctx context.Context, input []byte, commit func([]byte), opts runOptions) (Timing, error) {
	p.b.runs++
	if p.b.wait {
		<-ctx.Done()
		return Timing{}, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return Timing{}, err
	}
	if p.b.runErr != nil {
		return Timing{}, p.b.runErr
	}
	result := make([]byte, len(input))
	for i := 0; i < len(input); i += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(input[i:]))
		for j := uint32(0); j < opts.passes; j++ {
			v *= 2
		}
		binary.LittleEndian.PutUint32(result[i:], math.Float32bits(v))
	}
	commit(result)
	return Timing{}, nil
}
func setup(t testing.TB, config Config, open func() (backend, error)) (*wago.Runtime, *Plugin) {
	t.Helper()
	p, e := New(config)
	if e != nil {
		t.Fatal(e)
	}
	if open != nil {
		p.open = open
	}
	s, e := p.PluginSet()
	if e != nil {
		t.Fatal(e)
	}
	rt := wago.NewRuntime()
	t.Cleanup(func() { rt.CloseContext(context.Background()) })
	if e = rt.LoadPlugins(context.Background(), s); e != nil {
		t.Fatal(e)
	}
	return rt, p
}
func instance(t testing.TB, rt *wago.Runtime, source []byte) (*wago.Module, *wago.Instance) {
	t.Helper()
	m, e := rt.Compile(source)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { m.Close() })
	i, e := rt.Instantiate(context.Background(), m)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { i.Close() })
	return m, i
}
func config() Config { return Config{KernelExport: "kernel", RelaxedFloat: true} }
func invoke(t testing.TB, i *wago.Instance, export string, args ...uint64) int32 {
	t.Helper()
	v, e := i.Invoke(export, args...)
	if e != nil {
		t.Fatal(e)
	}
	if len(v) == 0 {
		return 0
	}
	return wago.AsI32(v[0])
}

func TestFallback(t *testing.T) {
	for _, mode := range []string{"device missing", "shader failure", "execution failure", "disabled", "strict float", "unsupported"} {
		t.Run(mode, func(t *testing.T) {
			cfg := config()
			b := &fakeBackend{}
			op := func() (backend, error) { return b, nil }
			source := fixtures.Twice
			switch mode {
			case "device missing":
				op = func() (backend, error) { return nil, errors.New("no GPU") }
			case "shader failure":
				b.compileErr = errors.New("shader rejected")
			case "execution failure":
				b.runErr = errors.New("readback failed")
			case "disabled":
				cfg.Disabled = true
			case "strict float":
				cfg.RelaxedFloat = false
			case "unsupported":
				cfg.KernelExport = "cpu"
			}
			rt, p := setup(t, cfg, op)
			_, i := instance(t, rt, source)
			if e := fixtures.Prepare(i, 32); e != nil {
				t.Fatal(e)
			}
			invoke(t, i, "cpu", 0, 128, 32)
			// Failed raw GPU calls must not commit any output.
			i.Write(256, []byte{1, 2, 3, 4})
			if got := invoke(t, i, "gpu", 0, 256, 32); got != Fallback {
				t.Fatalf("status %d", got)
			}
			v, _ := i.Read(256, 4)
			if string(v) != string([]byte{1, 2, 3, 4}) {
				t.Fatal("failed GPU call changed output")
			}
			if got := invoke(t, i, "run", 0, 256, 32); got != Fallback {
				t.Fatalf("status %d", got)
			}
			if e := fixtures.Verify(i, 32); e != nil {
				t.Fatal(e)
			}
			if p.Snapshot().Fallbacks != 2 {
				t.Fatal(p.Snapshot())
			}
		})
	}
}
func TestInvalidRangesAndLimits(t *testing.T) {
	b := &fakeBackend{}
	cfg := config()
	cfg.MaxElements = 32
	rt, _ := setup(t, cfg, func() (backend, error) { return b, nil })
	_, i := instance(t, rt, fixtures.Twice)
	for _, args := range [][]uint64{{65535, 0, 1}, {0, 65535, 1}, {0, 0, 0xffffffff}, {0xffffffff, 0, 1}, {0, 0xffffffff, 0}} {
		if got := invoke(t, i, "gpu", args...); got != InvalidRange {
			t.Fatalf("args %v: status %d", args, got)
		}
	}
	if b.runs != 0 {
		t.Fatal("invalid range reached backend")
	}
	if got := invoke(t, i, "gpu", 0, 256, 33); got != Fallback {
		t.Fatalf("limit status %d", got)
	}
	if got := invoke(t, i, "gpu", 65536, 65536, 0); got != Success {
		t.Fatalf("empty status %d", got)
	}
}
func TestCacheRepeatedExecutionAndCleanup(t *testing.T) {
	b := &fakeBackend{}
	rt, p := setup(t, config(), func() (backend, error) { return b, nil })
	m, i := instance(t, rt, fixtures.Twice)
	m2, i2 := instance(t, rt, fixtures.Twice)
	if b.builds != 1 {
		t.Fatal("pipeline not cached")
	}
	for _, in := range []*wago.Instance{i, i2} {
		if e := fixtures.Prepare(in, 64); e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "cpu", 0, 256, 64)
		for n := 0; n < 4; n++ {
			if invoke(t, in, "run", 0, 512, 64) != Success {
				t.Fatal("GPU failed")
			}
		}
		if e := fixtures.Verify(in, 64); e != nil {
			t.Fatal(e)
		}
	}
	m.Close()
	m2.Close()
	if b.programCloses != 0 {
		t.Fatal("closed pipeline still used by live instances")
	}
	if invoke(t, i, "run", 0, 512, 64) != Success {
		t.Fatal("closed module lost live pipeline")
	}
	i.Close()
	if b.programCloses != 0 {
		t.Fatal("pipeline freed too soon")
	}
	i2.Close()
	i2.Close()
	s := p.Snapshot()
	if b.programCloses != 1 || s.Instances != 0 || s.Modules != 0 || s.Pipelines != 0 || s.Pending != 0 {
		t.Fatalf("leaked resources: %+v, %+v", s, b)
	}
	rt.CloseContext(context.Background())
	if b.closes != 1 {
		t.Fatal("device not closed once")
	}
}
func TestCompileFailureCleanup(t *testing.T) {
	b := &fakeBackend{}
	rt, p := setup(t, config(), func() (backend, error) { return b, nil })
	if _, e := rt.Compile([]byte("bad wasm")); e == nil {
		t.Fatal("invalid module compiled")
	}
	if s := p.Snapshot(); s.Pending != 0 || s.Pipelines != 0 {
		t.Fatal(s)
	}
}

func TestRuntimeCloseWithLiveResources(t *testing.T) {
	b := &fakeBackend{}
	rt, p := setup(t, config(), func() (backend, error) { return b, nil })
	_, i := instance(t, rt, fixtures.Twice)
	if invoke(t, i, "gpu", 0, 16, 1) != Success {
		t.Fatal("GPU failed")
	}
	if e := rt.CloseContext(context.Background()); e != nil {
		t.Fatal(e)
	}
	s := p.Snapshot()
	if s.Instances != 0 || s.Modules != 0 || s.Pipelines != 0 || s.Pending != 0 || b.closes != 1 || b.programCloses != 1 {
		t.Fatalf("resources not closed: %+v %+v", s, b)
	}
}

func TestAbandonedPreparationsAreBounded(t *testing.T) {
	b := &fakeBackend{}
	rt, p := setup(t, config(), func() (backend, error) { return b, nil })
	old, e := rt.PrepareCompile(fixtures.Twice)
	if e != nil {
		t.Fatal(e)
	}
	defer old.Close()
	for j := 0; j < 100; j++ {
		prepared, e := rt.PrepareCompile(fixtures.Twice)
		if e != nil {
			t.Fatal(e)
		}
		if e = prepared.Close(); e != nil {
			t.Fatal(e)
		}
	}
	if s := p.Snapshot(); s.Pending > 16 || s.Pipelines != 0 {
		t.Fatal(s)
	}
	m, e := old.Compile()
	if e != nil {
		t.Fatal(e)
	}
	defer m.Close()
	i, e := rt.Instantiate(context.Background(), m)
	if e != nil {
		t.Fatal(e)
	}
	defer i.Close()
	if got := invoke(t, i, "gpu", 0, 16, 1); got != Fallback {
		t.Fatalf("evicted preparation status %d", got)
	}
	_, fresh := instance(t, rt, fixtures.Twice)
	if got := invoke(t, fresh, "gpu", 0, 16, 1); got != Success {
		t.Fatalf("new preparation status %d", got)
	}
}

func TestOverlap(t *testing.T) {
	b := &fakeBackend{}
	rt, _ := setup(t, config(), func() (backend, error) { return b, nil })
	_, i := instance(t, rt, fixtures.Twice)
	i.WriteFloat32Le(0, 1)
	i.WriteFloat32Le(4, 3)
	if invoke(t, i, "run", 0, 4, 2) != Fallback {
		t.Fatal("partial overlap did not use CPU")
	}
	a, _ := i.ReadFloat32Le(4)
	z, _ := i.ReadFloat32Le(8)
	if a != 2 || z != 4 || b.runs != 0 {
		t.Fatalf("%v %v; runs %d", a, z, b.runs)
	}
	if invoke(t, i, "run", 4, 4, 2) != Success {
		t.Fatal("in-place call failed")
	}
	a, _ = i.ReadFloat32Le(4)
	z, _ = i.ReadFloat32Le(8)
	if a != 4 || z != 8 {
		t.Fatalf("%v %v", a, z)
	}
}
func TestConcurrentInstances(t *testing.T) {
	b := &fakeBackend{}
	rt, p := setup(t, config(), func() (backend, error) { return b, nil })
	var instances []*wago.Instance
	for j := 0; j < 8; j++ {
		_, i := instance(t, rt, fixtures.Twice)
		if e := fixtures.Prepare(i, 257); e != nil {
			t.Fatal(e)
		}
		invoke(t, i, "cpu", 0, 1028, 257)
		instances = append(instances, i)
	}
	var wg sync.WaitGroup
	for _, i := range instances {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				v, e := i.Invoke("run", 0, 2056, 257)
				if e != nil || len(v) != 1 || wago.AsI32(v[0]) != Success {
					t.Errorf("%v %v", v, e)
					return
				}
			}
			if e := fixtures.Verify(i, 257); e != nil {
				t.Error(e)
			}
		}()
	}
	wg.Wait()
	if p.Snapshot().Successes != 80 {
		t.Fatal(p.Snapshot())
	}
}
