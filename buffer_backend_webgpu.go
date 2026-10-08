//go:build webgpu && cgo && linux

package wagogpu

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/oliverbestmann/webgpu/wgpu"
	"time"
)

type nativeBuffer struct{ buffer *wgpu.Buffer }

func (b *nativeBuffer) Close() {
	if b.buffer != nil {
		b.buffer.Destroy()
		b.buffer.Release()
		b.buffer = nil
	}
}

type nativeBufferPipeline struct {
	pipeline *wgpu.ComputePipeline
	layout   *wgpu.BindGroupLayout
}

func (p *nativeBufferPipeline) Close()      { p.layout.Release(); p.pipeline.Release() }
func (b *gpuBackend) RetainedBytes() uint64 { return b.retainedBytes }
func (b *gpuBackend) Lost() bool            { return b.lost.Load() }
func (b *gpuBackend) BuildBuffer(source string) (bufferPipeline, error) {
	if b.lost.Load() {
		return nil, fmt.Errorf("device lost")
	}
	s, e := b.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSLSource: &wgpu.ShaderSourceWGSL{Code: source}})
	if e != nil {
		return nil, e
	}
	defer s.Release()
	pipeline, e := b.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{Compute: wgpu.ProgrammableStageDescriptor{Module: s, EntryPoint: "main"}})
	if e != nil {
		return nil, e
	}
	var layout *wgpu.BindGroupLayout
	e = b.device.Check(func() error { layout = pipeline.GetBindGroupLayout(0); return nil })
	if e != nil {
		pipeline.Release()
		if layout != nil {
			layout.Release()
		}
		return nil, e
	}
	return &nativeBufferPipeline{pipeline, layout}, nil
}
func (b *gpuBackend) AllocateBuffer(size uint64) (deviceBuffer, error) {
	if size == 0 || size > b.limits.MaxBufferSize || size > b.limits.MaxStorageBufferBindingSize {
		return nil, fmt.Errorf("buffer device limit")
	}
	r, e := b.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
	if e != nil {
		return nil, e
	}
	return &nativeBuffer{r}, nil
}
func (b *gpuBackend) UploadBuffer(ctx context.Context, resource deviceBuffer, data []byte) (resultErr error) {
	if e := b.queue.TryWriteBuffer(resource.(*nativeBuffer).buffer, 0, data); e != nil {
		return e
	}
	defer func() {
		if resultErr != nil {
			b.retainedBytes += uint64(len(data))
		}
	}()
	if e := b.queue.TrySubmit(); e != nil {
		return e
	}
	return b.waitIdle(ctx)
}
func (b *gpuBackend) submitBuffers(ctx context.Context, encode func(*wgpu.CommandEncoder) error, operations ...*BufferOperation) error {
	e, err := b.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer e.Release()
	if err = b.device.Check(func() error { return encode(e) }); err != nil {
		return err
	}
	c, err := e.TryFinish(nil)
	if err != nil {
		return err
	}
	defer c.Release()
	for _, op := range operations {
		op.HardwareSubmitted = true
	}
	if err = b.queue.TrySubmit(c); err != nil {
		return err
	}
	return b.waitIdle(ctx)
}
func (b *gpuBackend) AllocateReadback(size uint64) (deviceBuffer, error) {
	r, e := b.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if e != nil {
		return nil, e
	}
	return &nativeBuffer{r}, nil
}
func (b *gpuBackend) AllocateUniform(size uint64) (deviceBuffer, error) {
	r, e := b.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
	if e != nil {
		return nil, e
	}
	return &nativeBuffer{r}, nil
}
func (b *gpuBackend) ReadBuffer(ctx context.Context, resource, staging deviceBuffer, size uint64, commit func([]byte)) (resultErr error) {
	r := staging.(*nativeBuffer).buffer
	var err error
	if err = b.submitBuffers(ctx, func(e *wgpu.CommandEncoder) error {
		return e.TryCopyBufferToBuffer(resource.(*nativeBuffer).buffer, 0, r, 0, size)
	}); err != nil {
		return err
	}
	done := make(chan wgpu.MapAsyncStatus, 1)
	if err = r.TryMapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.MapAsyncStatus) { done <- status }); err != nil {
		return err
	}
	defer func() { resultErr = errors.Join(resultErr, r.TryUnmap()) }()
	err = waitGPU(ctx, func() (bool, error) {
		if err := b.poll(); err != nil {
			return false, err
		}
		select {
		case status := <-done:
			if status != wgpu.MapAsyncStatusSuccess {
				return false, fmt.Errorf("map: %s", status)
			}
			return true, nil
		default:
			return false, nil
		}
	})
	if err != nil {
		return err
	}
	var data []byte
	if err = b.device.Check(func() error { data = r.GetMappedRange(0, uint(size)); return nil }); err != nil {
		return err
	}
	if uint64(len(data)) != size {
		return fmt.Errorf("map length mismatch")
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if b.lost.Load() {
		return fmt.Errorf("device lost")
	}
	commit(data)
	return nil
}
func (b *gpuBackend) ExecuteBuffers(ctx context.Context, pipeline bufferPipeline, parameter deviceBuffer, resources []deviceBuffer, seeds []bufferSeed, count uint32, parameterUpload bool, op *BufferOperation) (resultErr error) {
	if uint32(len(resources)) > b.limits.MaxStorageBuffersPerShaderStage || uint32(len(resources)+1) > b.limits.MaxBindingsPerBindGroup || b.limits.MaxUniformBuffersPerShaderStage < 1 || b.limits.MaxUniformBufferBindingSize < 16 || b.limits.MaxComputeInvocationsPerWorkgroup < 256 || b.limits.MaxComputeWorkgroupSizeX < 256 || (count+255)/256 > b.limits.MaxComputeWorkgroupsPerDimension {
		return fmt.Errorf("dispatch device limit")
	}
	p := pipeline.(*nativeBufferPipeline)
	uniform := parameter.(*nativeBuffer).buffer
	parameterQueued := false
	defer func() {
		if resultErr != nil && parameterQueued {
			b.retainedBytes += 16
		}
	}()
	if parameterUpload {
		var params [16]byte
		binary.LittleEndian.PutUint32(params[:], count)
		if e := b.queue.TryWriteBuffer(uniform, 0, params[:]); e != nil {
			return e
		}
		parameterQueued = true
	}
	var entries [9]wgpu.BindGroupEntry
	entries[0] = wgpu.BindGroupEntry{Binding: 0, Buffer: uniform, Size: 16}
	for i, r := range resources {
		buffer := r.(*nativeBuffer).buffer
		entries[i+1] = wgpu.BindGroupEntry{Binding: uint32(i + 1), Buffer: buffer, Size: buffer.GetSize()}
	}
	group, e := b.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: p.layout, Entries: entries[:len(resources)+1]})
	if e != nil {
		return e
	}
	defer group.Release()
	copySeeds := func(encoder *wgpu.CommandEncoder) error {
		for _, seed := range seeds {
			if e := encoder.TryCopyBufferToBuffer(seed.source.(*nativeBuffer).buffer, 0, seed.destination.(*nativeBuffer).buffer, 0, seed.size); e != nil {
				return e
			}
		}
		return nil
	}
	// Submitted is a conservative fact, never a claim of completed execution.
	if op.ProfileValid && len(seeds) != 0 {
		start := time.Now()
		if e = b.submitBuffers(ctx, copySeeds, op); e != nil {
			return e
		}
		op.DeviceCopy = time.Since(start)
	}
	start := time.Now()
	e = b.submitBuffers(ctx, func(encoder *wgpu.CommandEncoder) error {
		if !op.ProfileValid {
			if e := copySeeds(encoder); e != nil {
				return e
			}
		}
		pass := encoder.BeginComputePass(nil)
		defer pass.Release()
		pass.SetPipeline(p.pipeline)
		pass.SetBindGroup(0, group, nil)
		pass.DispatchWorkgroups((count+255)/256, 1, 1)
		return pass.TryEnd()
	}, op)
	op.Compute = time.Since(start)
	return e
}
