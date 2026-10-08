package wagogpu

import (
	"context"
	"encoding/binary"
	"errors"
	wago "github.com/wago-org/wago"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeBufferDevice struct {
	fakeBackend
	allocations, readbackAllocations, uniformAllocations, executions int
	readErrorAfterCommit                                             bool
	buffers                                                          int
	pipelineCount                                                    int
	fail, errorAfterCompletion                                       bool
	lose                                                             bool
	beforeReturn                                                     func()
	mode                                                             string
	parameterUploads                                                 int
}
type fakeDeviceBuffer struct {
	owner  *fakeBufferDevice
	bytes  []byte
	closed bool
}

func (b *fakeDeviceBuffer) Close() {
	if b.closed {
		panic("double buffer close")
	}
	b.closed = true
	b.owner.buffers--
}

type fakeBufferPipeline struct{ owner *fakeBufferDevice }

func (p *fakeBufferPipeline) Close() { p.owner.pipelineCount-- }
func (d *fakeBufferDevice) BuildBuffer(string) (bufferPipeline, error) {
	d.pipelineCount++
	return &fakeBufferPipeline{d}, nil
}
func (d *fakeBufferDevice) AllocateBuffer(n uint64) (deviceBuffer, error) {
	d.buffers++
	d.allocations++
	return &fakeDeviceBuffer{owner: d, bytes: make([]byte, n)}, nil
}
func (d *fakeBufferDevice) AllocateReadback(n uint64) (deviceBuffer, error) {
	d.readbackAllocations++
	return d.AllocateBuffer(n)
}
func (d *fakeBufferDevice) AllocateUniform(n uint64) (deviceBuffer, error) {
	d.uniformAllocations++
	return d.AllocateBuffer(n)
}
func (d *fakeBufferDevice) UploadBuffer(ctx context.Context, b deviceBuffer, data []byte) error {
	copy(b.(*fakeDeviceBuffer).bytes, data)
	return ctx.Err()
}
func (d *fakeBufferDevice) ReadBuffer(ctx context.Context, b, staging deviceBuffer, n uint64, commit func([]byte)) error {
	if d.fail {
		return errors.New("readback failed")
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	copy(staging.(*fakeDeviceBuffer).bytes, b.(*fakeDeviceBuffer).bytes[:n])
	commit(staging.(*fakeDeviceBuffer).bytes[:n])
	if d.readErrorAfterCommit {
		return errors.New("unmap failed after copy")
	}
	return nil
}
func (d *fakeBufferDevice) Lost() bool { return d.lose }
func (d *fakeBufferDevice) ExecuteBuffers(ctx context.Context, _ bufferPipeline, parameter deviceBuffer, buffers []deviceBuffer, seeds []bufferSeed, count uint32, parameterUpload bool, op *BufferOperation) error {
	d.executions++
	if parameterUpload {
		binary.LittleEndian.PutUint32(parameter.(*fakeDeviceBuffer).bytes, count)
		d.parameterUploads++
	}
	if binary.LittleEndian.Uint32(parameter.(*fakeDeviceBuffer).bytes) != count {
		return errors.New("stale dispatch parameters")
	}
	for _, s := range seeds {
		copy(s.destination.(*fakeDeviceBuffer).bytes, s.source.(*fakeDeviceBuffer).bytes)
	}
	if d.mode == "mandelbrot" {
		cr, ci := buffers[0].(*fakeDeviceBuffer).bytes, buffers[1].(*fakeDeviceBuffer).bytes
		x, y := buffers[2].(*fakeDeviceBuffer).bytes, buffers[3].(*fakeDeviceBuffer).bytes
		for j := uint32(0); j < count; j++ {
			r, im := math.Float32frombits(binary.LittleEndian.Uint32(x[j*4:])), math.Float32frombits(binary.LittleEndian.Uint32(y[j*4:]))
			a, b := math.Float32frombits(binary.LittleEndian.Uint32(cr[j*4:])), math.Float32frombits(binary.LittleEndian.Uint32(ci[j*4:]))
			binary.LittleEndian.PutUint32(x[j*4:], math.Float32bits(r*r-im*im+a))
			binary.LittleEndian.PutUint32(y[j*4:], math.Float32bits(2*r*im+b))
		}
		if d.executions == 2 {
			return errors.New("injected failure on second GPU pass")
		}
	} else if d.mode == "increment" || d.mode == "store-first" {
		dst := buffers[0].(*fakeDeviceBuffer).bytes
		for j := uint32(0); j < count; j++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(dst[j*4:]))
			if d.mode == "store-first" {
				v = 3
			}
			binary.LittleEndian.PutUint32(dst[j*4:], math.Float32bits(v+1))
		}
	} else if len(buffers) >= 2 {
		src, dst := buffers[0].(*fakeDeviceBuffer).bytes, buffers[1].(*fakeDeviceBuffer).bytes
		for j := uint32(0); j < count; j++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(src[j*4:]))
			binary.LittleEndian.PutUint32(dst[j*4:], math.Float32bits(v*2))
		}
	}
	if d.beforeReturn != nil {
		d.beforeReturn()
	}
	if d.fail || d.errorAfterCompletion {
		return errors.New("validation error after completion")
	}
	return ctx.Err()
}
func fakeModule(t *testing.T, d *fakeBufferDevice) (*Plugin, func(string, ...uint64) int32, func(), context.CancelFunc) {
	t.Helper()
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (f32.mul (call $read (call $get (i32.const 0)) (local.get $i)) (f32.const 2)))`)))
	for slot := uint64(0); slot < 2; slot++ {
		r, e := in.Invoke("create", uint64(TypeF32), 4)
		if e != nil || r[0]>>32 != 0 {
			t.Fatal(r, e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(r[0])))
	}
	for j := uint32(0); j < 4; j++ {
		in.WriteFloat32Le(j*4, float32(j+1))
	}
	invoke(t, in, "set", 1, 0, 0, 0, 4)
	return p, func(name string, args ...uint64) int32 { return invoke(t, in, name, args...) }, func() {
		if e := rt.CloseContext(context.Background()); e != nil {
			t.Fatal(e)
		}
	}, func() {}
}
func TestBufferFakeResidentAndFallback(t *testing.T) {
	d := &fakeBufferDevice{}
	p, call, close, _ := fakeModule(t, d)
	if got := call("dispatch", 1, 4); got != V1OK {
		t.Fatal(got, p.BufferSnapshot())
	}
	p.mu.Lock()
	var instance *bufferInstance
	for _, i := range p.buffers.instances {
		instance = i
	}
	output := instance.buffers[2]
	if output.cpuCurrent || !output.gpuCurrent || output.version != 2 {
		t.Fatal("output not GPU current")
	}
	p.config.Disabled = true
	p.config.ProfileStages = true
	p.mu.Unlock()
	if got := call("dispatch", 1, 4); got != V1CPUFallback {
		t.Fatal(got)
	}
	if !output.cpuCurrent || binary.LittleEndian.Uint32(output.cpu) != math.Float32bits(2) {
		t.Fatal("fallback did not prepare current CPU data")
	}
	s := p.BufferSnapshot()
	if s.Last.Outcome != "CPU_FALLBACK_READY" || s.Last.DownloadCount != 1 || !s.Last.ProfileValid || s.Last.Download <= 0 {
		t.Fatalf("diagnostics: %+v", s.Last)
	}
	close()
	if d.buffers != 0 || d.pipelineCount != 0 || p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("cleanup leak", d.buffers, d.pipelineCount, p.BufferSnapshot())
	}
}
func TestBufferFakeRejectedCompletion(t *testing.T) {
	d := &fakeBufferDevice{errorAfterCompletion: true}
	p, call, close, _ := fakeModule(t, d)
	if got := call("dispatch", 1, 4); got != V1CPUFallback {
		t.Fatal(got)
	}
	p.mu.Lock()
	for _, i := range p.buffers.instances {
		if i.buffers[2].version != 1 || !i.buffers[2].cpuCurrent {
			t.Fatal("failed output was published")
		}
	}
	p.mu.Unlock()
	if p.BufferSnapshot().Last.Outcome != "CPU_FALLBACK_READY" {
		t.Fatal("reported success")
	}
	close()
	if d.buffers != 0 {
		t.Fatal("retired resources leaked")
	}
}
func TestBufferFakeLossAndRepair(t *testing.T) {
	d := &fakeBufferDevice{}
	p, call, close, _ := fakeModule(t, d)
	call("dispatch", 1, 4)
	d.lose = true
	if got := call("copy", 2, 0, 0, 16, 4); got != V1ContentsLost {
		t.Fatal(got)
	}
	if got := call("set", 2, 0, 0, 0, 0); got != V1OK {
		t.Fatal(got)
	}
	if got := call("copy", 2, 0, 0, 16, 4); got != V1ContentsLost {
		t.Fatal("no-op repaired contents", got)
	}
	if got := call("set", 2, 0, 0, 0, 4); got != V1OK {
		t.Fatal(got)
	}
	if got := call("copy", 2, 0, 0, 16, 4); got != V1OK {
		t.Fatal(got)
	}
	close()
	if p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("budget leak")
	}
}
func TestBufferConcurrentInstances(t *testing.T) {
	rt, p := setup(t, bufferConfig(), nil)
	m, e := rt.Compile(bufferFixture(t))
	if e != nil {
		t.Fatal(e)
	}
	defer m.Close()
	var wg sync.WaitGroup
	for n := 0; n < 8; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			i, e := rt.Instantiate(context.Background(), m)
			if e != nil {
				t.Error(e)
				return
			}
			if e = i.Close(); e != nil {
				t.Error(e)
			}
		}()
	}
	wg.Wait()
	if p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("concurrent instance leak")
	}
}

func (d *fakeBufferDevice) RetainedBytes() uint64 { return 0 }

func TestActualWriteSetAndPool(t *testing.T) {
	d := &fakeBufferDevice{}
	p, call, close, _ := fakeModule(t, d)
	for n := 0; n < 4; n++ {
		if got := call("dispatch", 1, 4); got != 0 {
			t.Fatal(got)
		}
	}
	p.mu.Lock()
	for _, i := range p.buffers.instances {
		if i.buffers[1].version != 2 || !i.buffers[1].cpuCurrent {
			t.Fatal("read-only input changed version/state")
		}
		if i.poolCount == 0 || i.poolBytes != 16 {
			t.Fatal("scratch was not reused")
		}
		if i.buffers[2].version != 5 {
			t.Fatal("store did not advance version")
		}
	}
	p.mu.Unlock()
	if d.buffers != 4 {
		t.Fatal("unbounded scratch allocations", d.buffers)
	}
	close()
}
func TestBufferBudgetRollback(t *testing.T) {
	d := &fakeBufferDevice{}
	p, call, close, _ := fakeModule(t, d)
	p.mu.Lock()
	p.config.MaxInstanceBufferBytes = 64
	p.mu.Unlock()
	if got := call("dispatch", 1, 4); got != V1LimitExceeded {
		t.Fatal(got)
	}
	p.mu.Lock()
	for _, i := range p.buffers.instances {
		if i.buffers[2].version != 1 || !i.buffers[2].cpuCurrent {
			t.Fatal("budget failure published output")
		}
		if i.bytes > 64 {
			t.Fatal("budget exceeded", i.bytes)
		}
	}
	p.mu.Unlock()
	close()
	if d.buffers != 0 {
		t.Fatal("resource leak")
	}
}
func TestBufferCancelledBeforeCommit(t *testing.T) {
	d := &fakeBufferDevice{}
	p, _, close, _ := fakeModule(t, d)
	ctx, cancel := context.WithCancel(context.Background())
	d.beforeReturn = cancel
	p.mu.Lock()
	for _, i := range p.buffers.instances {
		op := BufferOperation{}
		status := p.dispatchBuffers(ctx, i, 1, 4, &op)
		if status != V1Cancelled {
			t.Fatal(status)
		}
		if i.buffers[2].version != 1 || !i.buffers[2].cpuCurrent {
			t.Fatal("cancelled output was published")
		}
	}
	p.mu.Unlock()
	close()
}

func TestOuterCancellationAfterCommit(t *testing.T) {
	d := &fakeBufferDevice{}
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	source := testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`)
	source = strings.Replace(source, "(module", "(module (import \"test\" \"cancel\" (func $cancel))", 1)
	source = strings.TrimSuffix(source, ")") + `(func (export "cancelAfter") (drop (call $dispatch (i32.const 1) (i32.const 4))) (call $cancel)))`
	module, e := rt.Compile(wat(t, source))
	if e != nil {
		t.Fatal(e)
	}
	defer module.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	imports := wago.NewImports()
	imports.HostFunc("test", "cancel", func() { cancel() })
	i, e := rt.Instantiate(context.Background(), module, wago.WithImports(imports))
	if e != nil {
		t.Fatal(e)
	}
	defer i.Close()
	for slot := uint64(0); slot < 2; slot++ {
		v, e := i.Invoke("create", 8, 4)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, i, "bind", slot, uint64(uint32(v[0])))
	}
	if _, e = i.InvokeContext(ctx, "cancelAfter"); !errors.Is(e, context.Canceled) {
		t.Fatalf("outer error: %v", e)
	}
	if op := p.BufferSnapshot().Last; op.Status != V1OK || op.Outcome != "GPU" {
		t.Fatalf("committed import changed after outer cancellation: %+v", op)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, state := range p.buffers.instances {
		if state.buffers[2].version != 2 || !state.buffers[2].gpuCurrent {
			t.Fatal("outer error rolled back commit")
		}
	}
}
func TestShutdownWithBlockedDispatch(t *testing.T) {
	d := &fakeBufferDevice{}
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	_, i := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (f32.const 1))`)))
	for slot := uint64(0); slot < 2; slot++ {
		v, e := i.Invoke("create", 8, 4)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, i, "bind", slot, uint64(uint32(v[0])))
	}
	entered, release := make(chan struct{}), make(chan struct{})
	d.beforeReturn = func() { close(entered); <-release }
	callDone := make(chan error, 1)
	go func() { _, e := i.Invoke("dispatch", 1, 4); callDone <- e }()
	<-entered
	closeDone := make(chan error, 1)
	go func() { closeDone <- rt.CloseContext(context.Background()) }()
	close(release)
	for _, done := range []chan error{callDone, closeDone} {
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("shutdown deadlock")
		}
	}
	s := p.BufferSnapshot()
	if s.RuntimeBufferBytes != 0 || len(s.Instances) != 0 || d.buffers != 0 || d.pipelineCount != 0 {
		t.Fatal("shutdown leaked resources", s)
	}
}

// Declared slots govern validation even when lowering removes unused bindings.
func TestUnusedDeclaredSlotAlias(t *testing.T) {
	for _, disabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "eligible", true: "fallback"}[disabled], func(t *testing.T) {
			c := bufferConfig()
			c.Disabled = disabled
			c.Kernels = c.Kernels[:1]
			c.Kernels[0].Bindings = []BindingConfig{{0, TypeF32, AccessReadWrite}, {7, TypeF32, AccessWrite}}
			rt, _ := setup(t, c, func() (backend, error) { return &fakeBufferDevice{}, nil })
			_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 0)) (local.get $i) (f32.const 7))`)))
			values, e := in.Invoke("create", uint64(TypeF32), 4)
			if e != nil {
				t.Fatal(e)
			}
			handle := uint64(uint32(values[0]))
			invoke(t, in, "bind", 0, handle)
			invoke(t, in, "bind", 7, handle)
			for _, count := range []uint64{0, 4} {
				if got := invoke(t, in, "dispatch", 1, count); got != V1InvalidBinding {
					t.Fatal(count, got)
				}
			}
		})
	}
}

func TestUnusedPermissionsDoNotPublish(t *testing.T) {
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	c.Kernels[0].Bindings = []BindingConfig{{0, TypeF32, AccessReadWrite}, {1, TypeF32, AccessWrite}, {7, TypeF32, AccessWrite}}
	d := &fakeBufferDevice{}
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	_, in := instance(t, rt, wat(t, testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`)))
	for _, slot := range []uint64{0, 1, 7} {
		values, e := in.Invoke("create", uint64(TypeF32), 4)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(values[0])))
	}
	if got := invoke(t, in, "dispatch", 1, 4); got != V1OK {
		t.Fatal(got)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, state := range p.buffers.instances {
		for _, handle := range []uint32{1, 3} {
			b := state.buffers[handle]
			if b.version != 1 || !b.cpuCurrent {
				t.Fatal("unused write permission changed contents", handle, b.version)
			}
		}
		if state.buffers[2].version != 2 {
			t.Fatal("actual store did not publish")
		}
	}
}

func TestMultipleOutputsFailureAtomic(t *testing.T) {
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	c.Kernels[0].Bindings = append(c.Kernels[0].Bindings, BindingConfig{2, TypeF32, AccessWrite})
	d := &fakeBufferDevice{errorAfterCompletion: true}
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	_, in := instance(t, rt, wat(t, testModule(`
 (call $write (call $get (i32.const 1)) (local.get $i) (f32.const 2))
 (call $write (call $get (i32.const 2)) (local.get $i) (f32.const 3))`)))
	for slot := uint64(0); slot < 3; slot++ {
		r, e := in.Invoke("create", uint64(TypeF32), 4)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(r[0])))
	}
	if got := invoke(t, in, "dispatch", 1, 4); got != V1CPUFallback {
		t.Fatal(got)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, state := range p.buffers.instances {
		for _, b := range state.buffers {
			if b.version != 1 || !b.cpuCurrent {
				t.Fatal("failed multi-output transaction published contents")
			}
		}
	}
}
