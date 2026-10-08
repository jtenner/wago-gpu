//go:build webgpu && cgo

package wagogpu

import (
	"context"
	"fmt"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

type gpuBackend struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	info     string
	limits   wgpu.Limits
}

func openBackend() (backend, error) {
	b := &gpuBackend{}
	// v0.23.0 does not initialize all fields in its optional instance extras.
	// The native default descriptor is complete and avoids that binding defect.
	b.instance = wgpu.CreateInstance(nil)
	a, err := b.instance.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance})
	if err != nil {
		b.Close()
		return nil, err
	}
	b.adapter = a
	info := a.GetInfo()
	// A software Vulkan adapter must never be reported as real GPU execution.
	if info.AdapterType != wgpu.AdapterTypeDiscreteGPU && info.AdapterType != wgpu.AdapterTypeIntegratedGPU {
		b.Close()
		return nil, fmt.Errorf("no hardware GPU: adapter %s (%s)", info.Name, info.AdapterType)
	}
	b.info = fmt.Sprintf("%s; %s; %s; %s", info.Name, info.DriverDescription, info.AdapterType, info.BackendType)
	// The optional device-loss callback also violates cgo pointer rules in this
	// binding version. Use error returns and the readback completion status.
	b.device, err = a.RequestDevice(nil)
	if err != nil {
		b.Close()
		return nil, err
	}
	b.queue = b.device.GetQueue()
	b.limits = b.device.GetLimits().Limits
	return b, nil
}
func (b *gpuBackend) Info() string { return b.info }
func (b *gpuBackend) Close() {
	if b.queue != nil {
		b.queue.Release()
		b.queue = nil
	}
	if b.device != nil {
		b.device.Release()
		b.device = nil
	}
	if b.adapter != nil {
		b.adapter.Release()
		b.adapter = nil
	}
	if b.instance != nil {
		b.instance.Release()
		b.instance = nil
	}
}
func (b *gpuBackend) Compile(source string) (program, error) {
	s, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: source}})
	if err != nil {
		return nil, err
	}
	defer s.Release()
	p, err := b.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Compute: wgpu.ProgrammableStageDescriptor{Module: s, EntryPoint: "main"}})
	if err != nil {
		return nil, err
	}
	return &gpuProgram{backend: b, pipeline: p, layout: p.GetBindGroupLayout(0)}, nil
}

type gpuProgram struct {
	backend           *gpuBackend
	pipeline          *wgpu.ComputePipeline
	layout            *wgpu.BindGroupLayout
	group             *wgpu.BindGroup
	storage, readback *wgpu.Buffer
	capacity, bound   uint64
}

func (p *gpuProgram) BufferBytes() uint64 { return p.capacity * 2 }
func (p *gpuProgram) freeBuffers() {
	if p.group != nil {
		p.group.Release()
		p.group = nil
	}
	if p.storage != nil {
		p.storage.Destroy()
		p.storage.Release()
		p.storage = nil
	}
	if p.readback != nil {
		p.readback.Destroy()
		p.readback.Release()
		p.readback = nil
	}
	p.capacity = 0
	p.bound = 0
}
func (p *gpuProgram) Close() {
	p.freeBuffers()
	if p.layout != nil {
		p.layout.Release()
		p.layout = nil
	}
	if p.pipeline != nil {
		p.pipeline.Release()
		p.pipeline = nil
	}
}
func (p *gpuProgram) buffers(size uint64) error {
	d := p.backend.device
	if size > p.capacity {
		capacity := min(max(size, p.capacity*2), uint64(DefaultMaxElements)*4, p.backend.limits.MaxBufferSize, p.backend.limits.MaxStorageBufferBindingSize)
		p.freeBuffers()
		var err error
		p.storage, err = d.CreateBuffer(&wgpu.BufferDescriptor{Size: capacity, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		if err != nil {
			return err
		}
		p.readback, err = d.CreateBuffer(&wgpu.BufferDescriptor{Size: capacity, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			p.freeBuffers()
			return err
		}
		p.capacity = capacity
	}
	if p.bound != size {
		if p.group != nil {
			p.group.Release()
			p.group = nil
		}
		p.bound = 0
		var err error
		p.group, err = d.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: p.layout, Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: p.storage, Size: size}}})
		if err != nil {
			return err
		}
		p.bound = size
	}
	return nil
}
func (p *gpuProgram) submit(encode func(*wgpu.CommandEncoder) error) error {
	b := p.backend
	e, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer e.Release()
	if err = encode(e); err != nil {
		return err
	}
	c, err := e.Finish(nil)
	if err != nil {
		return err
	}
	defer c.Release()
	b.queue.Submit(c)
	return nil
}
func (p *gpuProgram) compute(e *wgpu.CommandEncoder, n, passes uint32) error {
	for j := uint32(0); j < passes; j++ {
		// Separate passes make writes visible to the next dispatch.
		pass := e.BeginComputePass(nil)
		pass.SetPipeline(p.pipeline)
		pass.SetBindGroup(0, p.group, nil)
		pass.DispatchWorkgroups((n+255)/256, 1, 1)
		err := pass.End()
		pass.Release()
		if err != nil {
			return err
		}
	}
	return nil
}
func (b *gpuBackend) waitIdle(ctx context.Context) error {
	return waitGPU(ctx, func() (bool, error) { return b.device.Poll(false, nil), nil })
}
func (p *gpuProgram) Run(ctx context.Context, input []byte, commit func([]byte), opts runOptions) (t Timing, err error) {
	b := p.backend
	size := uint64(len(input))
	n := uint32(size / 4)
	limits := b.limits
	if err = ctx.Err(); err != nil {
		return t, err
	}
	if size == 0 || size%4 != 0 || size > limits.MaxStorageBufferBindingSize || size > limits.MaxBufferSize || (n+255)/256 > limits.MaxComputeWorkgroupsPerDimension || opts.passes == 0 || opts.passes > MaxBatchPasses {
		return t, fmt.Errorf("GPU buffer, dispatch, or batch limit exceeded")
	}
	if err = p.buffers(size); err != nil {
		return t, err
	}
	t.Passes = opts.passes
	start := time.Now()
	if err = b.queue.WriteBuffer(p.storage, 0, input); err != nil {
		return t, err
	}
	t.UploadBytes = size
	if opts.profile {
		b.queue.Submit()
		if err = b.waitIdle(ctx); err != nil {
			return t, err
		}
		t.Upload = time.Since(start)
		start = time.Now()
		if err = p.submit(func(e *wgpu.CommandEncoder) error { return p.compute(e, n, opts.passes) }); err != nil {
			return t, err
		}
		if err = b.waitIdle(ctx); err != nil {
			return t, err
		}
		t.Compute = time.Since(start)
		start = time.Now()
	}
	if err = p.submit(func(e *wgpu.CommandEncoder) error {
		if !opts.profile {
			if err := p.compute(e, n, opts.passes); err != nil {
				return err
			}
		}
		return e.CopyBufferToBuffer(p.storage, 0, p.readback, 0, size)
	}); err != nil {
		return t, err
	}
	// A buffered channel lets a late cancellation callback finish. It retains
	// neither guest input nor commit, and no polling goroutine outlives this call.
	done := make(chan wgpu.BufferMapAsyncStatus, 1)
	if err = p.readback.MapAsync(wgpu.MapModeRead, 0, size, func(s wgpu.BufferMapAsyncStatus) { done <- s }); err != nil {
		return t, err
	}
	defer p.readback.Unmap() // Also aborts a pending map on timeout.
	err = waitGPU(ctx, func() (bool, error) {
		b.device.Poll(false, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatusSuccess {
				return false, fmt.Errorf("GPU readback: %s", status)
			}
			return true, nil
		default:
			return false, nil
		}
	})
	if err != nil {
		return t, err
	}
	result := p.readback.GetMappedRange(0, uint(size))
	if uint64(len(result)) != size {
		return t, fmt.Errorf("GPU readback length mismatch")
	}
	if err = ctx.Err(); err != nil {
		return t, err
	}
	t.DownloadBytes = size
	if opts.profile {
		t.Download = time.Since(start)
	}
	start = time.Now()
	commit(result)
	t.Commit = time.Since(start)
	return t, nil
}
