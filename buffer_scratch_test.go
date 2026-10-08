package wagogpu

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

func TestScratchReusesAllOutputsAndTransfers(t *testing.T) {
	d := &fakeBufferDevice{}
	p, call, close, _ := fakeModule(t, d)
	p.mu.Lock()
	for _, m := range p.buffers.modules {
		m.kernels[1].lowered.writes[0] = true
	}
	p.mu.Unlock()
	var allocations int
	for pass := 0; pass < 8; pass++ {
		if got := call("dispatch", 1, 4); got != V1OK {
			t.Fatal(got)
		}
		for _, handle := range []uint64{1, 2} {
			if got := call("copy", handle, 0, 0, 0, 4); got != V1OK {
				t.Fatal(got)
			}
		}
		// Full covering outputs do not allocate their old GPU contents. The
		// second pass establishes the second side of the output scratch pair.
		if pass == 1 {
			allocations = d.allocations
		}
		if pass > 1 && d.allocations != allocations {
			t.Fatal("repeated pass allocated device storage", pass, d.allocations, allocations)
		}
	}
	if d.readbackAllocations != 1 || d.uniformAllocations != 1 {
		t.Fatal("transfer storage was not reused", d.readbackAllocations, d.uniformAllocations)
	}
	s := p.BufferSnapshot()
	for _, i := range s.Instances {
		if i.ReservedBytes != i.CPUBytes+i.GPUBytes+i.StagingBytes+i.ScratchBytes+i.RetainedPoolBytes {
			t.Fatal("allocation categories do not sum to reservation", i)
		}
	}
	close()
	if d.buffers != 0 || p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("scratch leak")
	}
}

func TestActiveConversionCannotBeEvicted(t *testing.T) {
	d := &fakeBufferDevice{}
	p, _, close, _ := fakeModule(t, d)
	func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		for _, i := range p.buffers.instances {
			b := i.buffers[1]
			b.typ, b.cpu = TypeU8, []byte{1, 2, 3, 4}
			p.releaseBufferBytes(i, 12)
			p.config.MaxInstanceBufferBytes = 64
			if got := p.ensureGPU(context.Background(), i, b, &BufferOperation{}); got != V1LimitExceeded {
				t.Fatal("active conversion was released to admit upload", got)
			}
			if i.bytes != 52 || cap(i.conversion) != 16 {
				t.Fatal("conversion charge lost", i.bytes, cap(i.conversion))
			}
			p.config.MaxInstanceBufferBytes = 80
			if got := p.ensureGPU(context.Background(), i, b, &BufferOperation{}); got != V1OK {
				t.Fatal(got)
			}
			if i.bytes != 52 {
				t.Fatal("scratch was charged twice", i.bytes)
			}
			// Growth admission must account for overlap, including eviction of old idle scratch.
			p.config.MaxInstanceBufferBytes = 68
			if _, got := p.conversionScratch(i, 32); got != V1OK {
				t.Fatal(got)
			}
			if i.bytes != 68 || cap(i.conversion) != 32 {
				t.Fatal("growth/eviction charge wrong", i.bytes)
			}
		}
	}()
	close()
	if d.buffers != 0 || p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("conversion leak")
	}
}

func TestFailedReadbackDoesNotPublishGuestCopy(t *testing.T) {
	d := &fakeBufferDevice{}
	p, call, close, _ := fakeModule(t, d)
	if got := call("dispatch", 1, 4); got != V1OK {
		t.Fatal(got)
	}
	d.readErrorAfterCommit = true
	if got := call("copy", 2, 0, 0, 0, 4); got != V1DeviceError {
		t.Fatal(got)
	}
	func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		for _, i := range p.buffers.instances {
			if i.readback.resource != nil || i.retiredBytes == 0 {
				t.Fatal("uncertain readback returned to idle")
			}
			if !i.buffers[2].cpuCurrent {
				t.Fatal("completed CPU cache copy was discarded")
			}
		}
	}()
	if got := call("firstWord"); got != int32(math.Float32bits(1)) {
		t.Fatal("failed transfer changed guest memory", got)
	}
	// A CPU cache populated before the unmap failure is still valid and does
	// not require reusing the failed staging resource.
	if got := call("copy", 2, 0, 0, 0, 4); got != V1OK {
		t.Fatal(got)
	}
	if d.readbackAllocations != 1 {
		t.Fatal("failed staging was reused")
	}
	close()
	if d.buffers != 0 || p.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("retired staging leak")
	}
}

func TestScalarReadbackDeadlineIncludesLockWait(t *testing.T) {
	d := &fakeBufferDevice{}
	c := bufferConfig()
	c.Disabled = false
	c.Kernels = c.Kernels[:1]
	rt, p := setup(t, c, func() (backend, error) { return d, nil })
	source := strings.TrimSuffix(testModule(`(call $write (call $get (i32.const 1)) (local.get $i) (call $read (call $get (i32.const 0)) (local.get $i)))`), ")") + `(func (export "readDirect") (result f32) (call $read (i32.const 1) (i32.const 0))))`
	_, in := instance(t, rt, wat(t, source))
	for slot := uint64(0); slot < 2; slot++ {
		result, e := in.Invoke("create", uint64(TypeF32), 4)
		if e != nil {
			t.Fatal(e)
		}
		invoke(t, in, "bind", slot, uint64(uint32(result[0])))
	}
	if got := invoke(t, in, "dispatch", 1, 4); got != V1OK {
		t.Fatal(got)
	}
	p.mu.Lock()
	p.config.RunTimeout = 20 * time.Millisecond
	for _, i := range p.buffers.instances {
		i.buffers[1].cpuCurrent = false
	}
	started, done := make(chan struct{}), make(chan error, 1)
	go func() { close(started); _, e := in.Invoke("readDirect"); done <- e }()
	<-started
	time.Sleep(100 * time.Millisecond)
	p.mu.Unlock()
	if e := <-done; e == nil {
		t.Fatal("expired scalar readback succeeded")
	}
	if d.readbackAllocations != 0 {
		t.Fatal("expired readback allocated/submitted work")
	}
}
