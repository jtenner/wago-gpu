package wagogpu

import (
	"context"
	"encoding/binary"
	"sort"
	"time"
)

// These interfaces make transaction failures testable without a GPU. Resources
// stay private to the plugin; neither native handles nor guest views cross them.
type deviceBuffer interface{ Close() }
type bufferPipeline interface{ Close() }
type bufferDevice interface {
	BuildBuffer(string) (bufferPipeline, error)
	AllocateBuffer(uint64) (deviceBuffer, error)
	UploadBuffer(context.Context, deviceBuffer, []byte) error
	ReadBuffer(context.Context, deviceBuffer, uint64, func([]byte)) error
	ExecuteBuffers(context.Context, bufferPipeline, []deviceBuffer, []bufferSeed, uint32, *BufferOperation) error
	Lost() bool
	RetainedBytes() uint64
}
type bufferSeed struct {
	source, destination deviceBuffer
	size                uint64
}
type bufferPipelineEntry struct {
	pipeline bufferPipeline
	refs     int
}
type retiredBuffer struct {
	resource deviceBuffer
	size     uint64
}

func physicalBytes(b *bufferState) uint64    { return uint64(b.count) * 4 }
func (p *Plugin) bufferDevice() bufferDevice { d, _ := p.device.(bufferDevice); return d }
func (p *Plugin) buildBufferPipelines(m *moduleContract) {
	d := p.bufferDevice()
	if d == nil || p.config.Disabled || p.stats.GPUFailed {
		return
	}
	for _, k := range m.kernels {
		if k.invalid != nil || k.rejection != nil || k.lowered.shader == "" || k.lowered.float && !k.config.RelaxedFloat {
			continue
		}
		key := k.lowered.shader
		if cached := p.buffers.pipelines[key]; cached != nil {
			cached.refs++
			k.pipeline = cached
			p.buffers.stats.PipelineHits++
			continue
		}
		if len(p.buffers.pipelines)+len(p.cache) >= 256 {
			k.rejection = compileError(CompileLimit, "pipeline count limit")
			continue
		}
		p.buffers.stats.PipelineMisses++
		start := time.Now()
		pipeline, e := d.BuildBuffer(key)
		k.pipelineTime = time.Since(start)
		if e != nil {
			k.pipelineError = e
			continue
		}
		entry := &bufferPipelineEntry{pipeline: pipeline, refs: 1}
		p.buffers.pipelines[key] = entry
		k.pipeline = entry
	}
}
func (p *Plugin) releaseBufferModule(m *moduleContract) {
	if m == nil {
		return
	}
	for _, k := range m.kernels {
		if c := k.pipeline; c != nil {
			c.refs--
			if c.refs == 0 {
				c.pipeline.Close()
				delete(p.buffers.pipelines, k.lowered.shader)
			}
			k.pipeline = nil
		}
	}
}
func (p *Plugin) closeBufferResource(i *bufferInstance, r deviceBuffer, size uint64) {
	if r == nil {
		return
	}
	if p.stats.GPUFailed && !p.stopped {
		p.buffers.retired = append(p.buffers.retired, retiredBuffer{r, size})
		i.retiredBytes += size
		return
	}
	r.Close()
	p.releaseBufferBytes(i, size)
}
func (p *Plugin) quarantineBuffers(reason string) {
	p.stats.GPUFailed = true
	p.buffers.stats.Reason = reason
	for _, i := range p.buffers.instances {
		for _, b := range i.buffers {
			if !b.cpuCurrent {
				b.lost = true
			}
			b.gpuCurrent = false
		}
	}
}
func (p *Plugin) ensureCPU(ctx context.Context, i *bufferInstance, b *bufferState, op *BufferOperation) int32 {
	if b.lost {
		return V1ContentsLost
	}
	if b.cpuCurrent {
		return V1OK
	}
	d := p.bufferDevice()
	if d == nil || p.stats.GPUFailed || d.Lost() {
		p.quarantineBuffers("device is unavailable")
		return V1ContentsLost
	}
	size := physicalBytes(b)
	if !p.reserveBuffer(i, size) {
		return V1LimitExceeded
	}
	before := d.RetainedBytes()
	defer func() {
		if d.RetainedBytes() > before {
			i.retiredBytes += size
		} else {
			p.releaseBufferBytes(i, size)
		}
	}()
	start := time.Now()
	var conversion time.Duration
	err := d.ReadBuffer(ctx, b.gpu, size, func(data []byte) {
		conversionStart := time.Now()
		if b.typ.spec().size == 4 {
			copy(b.cpu, data)
		} else {
			stride := b.typ.spec().size
			for j := uint32(0); j < b.count; j++ {
				v := binary.LittleEndian.Uint32(data[uint64(j)*4:])
				if stride == 1 {
					b.cpu[j] = byte(v)
				} else {
					binary.LittleEndian.PutUint16(b.cpu[uint64(j)*2:], uint16(v))
				}
			}
		}
		b.cpuCurrent = true
		conversion = time.Since(conversionStart)
	})
	if err != nil {
		p.markTransferUnknown(op)
		p.quarantineBuffers(err.Error())
		if ctx.Err() != nil {
			return V1Cancelled
		}
		return V1DeviceError
	}
	delta := TransferCounters{LogicalDownloadBytes: uint64(len(b.cpu)), GPUDownloadBytes: size, DownloadCount: 1}
	p.addTransfers(delta, op)
	if op != nil {
		op.Download += time.Since(start) - conversion
		op.Conversion += conversion
	}
	return V1OK
}
func (p *Plugin) addTransfers(d TransferCounters, op *BufferOperation) {
	add := func(v *TransferCounters) {
		v.GuestSetBytes += d.GuestSetBytes
		v.GuestCopyBytes += d.GuestCopyBytes
		v.GuestSetCount += d.GuestSetCount
		v.GuestCopyCount += d.GuestCopyCount
		v.LogicalUploadBytes += d.LogicalUploadBytes
		v.LogicalDownloadBytes += d.LogicalDownloadBytes
		v.GPUUploadBytes += d.GPUUploadBytes
		v.GPUDownloadBytes += d.GPUDownloadBytes
		v.UploadCount += d.UploadCount
		v.DownloadCount += d.DownloadCount
		v.DeviceCopyBytes += d.DeviceCopyBytes
		v.DeviceCopyCount += d.DeviceCopyCount
		v.ParameterUploadBytes += d.ParameterUploadBytes
		v.ParameterUploadCount += d.ParameterUploadCount
	}
	add(&p.buffers.stats.Totals)
	if op != nil {
		add(&op.TransferCounters)
	}
}
func (p *Plugin) ensureGPU(ctx context.Context, i *bufferInstance, b *bufferState, op *BufferOperation) int32 {
	if b.lost {
		return V1ContentsLost
	}
	if b.gpuCurrent {
		return V1OK
	}
	if !b.cpuCurrent {
		return V1ContentsLost
	}
	d := p.bufferDevice()
	size := physicalBytes(b)
	if b.gpu == nil {
		if !p.reserveBuffer(i, size) {
			return V1LimitExceeded
		}
		resource, e := d.AllocateBuffer(size)
		if e != nil {
			p.releaseBufferBytes(i, size)
			return V1DeviceError
		}
		b.gpu = resource
	}
	data := b.cpu
	if b.typ.spec().size != 4 {
		conversionStart := time.Now()
		if !p.reserveBuffer(i, size) {
			return V1LimitExceeded
		}
		defer p.releaseBufferBytes(i, size)
		data = make([]byte, size)
		stride := b.typ.spec().size
		for j := uint32(0); j < b.count; j++ {
			v := uint32(b.cpu[uint64(j)*stride])
			if stride == 2 {
				v = uint32(binary.LittleEndian.Uint16(b.cpu[uint64(j)*2:]))
			}
			binary.LittleEndian.PutUint32(data[uint64(j)*4:], v)
		}
		op.Conversion += time.Since(conversionStart)
	}
	if !p.reserveBuffer(i, size) {
		return V1LimitExceeded
	}
	before := d.RetainedBytes()
	defer func() {
		if d.RetainedBytes() > before {
			i.retiredBytes += size
		} else {
			p.releaseBufferBytes(i, size)
		}
	}()
	if ctx.Err() != nil {
		return V1Cancelled
	}
	start := time.Now()
	if err := d.UploadBuffer(ctx, b.gpu, data); err != nil {
		p.markTransferUnknown(op)
		p.quarantineBuffers(err.Error())
		if ctx.Err() != nil {
			return V1Cancelled
		}
		return V1DeviceError
	}
	b.gpuCurrent = true
	p.addTransfers(TransferCounters{LogicalUploadBytes: uint64(len(b.cpu)), GPUUploadBytes: size, UploadCount: 1}, op)
	op.Upload += time.Since(start)
	return V1OK
}
func (p *Plugin) executeBufferKernel(ctx context.Context, i *bufferInstance, k *kernelContract, count uint32, op *BufferOperation) int32 {
	d := p.bufferDevice()
	if d.Lost() {
		p.quarantineBuffers("device lost")
		return V1ContentsLost
	}
	slots := make([]uint32, 0, len(k.config.Bindings))
	for _, binding := range k.config.Bindings {
		if k.lowered.reads[binding.Slot] || k.lowered.writes[binding.Slot] {
			slots = append(slots, binding.Slot)
		}
	}
	sort.Slice(slots, func(a, b int) bool { return slots[a] < slots[b] })
	resources := make([]deviceBuffer, len(slots))
	outputs := map[uint32]deviceBuffer{}
	seeds := make([]bufferSeed, 0, len(slots))
	defer func() {
		for slot, r := range outputs {
			b := i.buffers[i.bindings[slot]]
			p.closeBufferResource(i, r, physicalBytes(b))
		}
	}()
	for idx, slot := range slots {
		b := i.buffers[i.bindings[slot]]
		if status := p.ensureGPU(ctx, i, b, op); status != V1OK {
			return status
		}
		resources[idx] = b.gpu
		if k.lowered.writes[slot] {
			size := physicalBytes(b)
			r, status := p.takeBufferScratch(i, size)
			if status != V1OK {
				return status
			}
			outputs[slot] = r
			resources[idx] = r
			seeds = append(seeds, bufferSeed{b.gpu, r, size})
		}
	}
	// Charge the uniform parameter block and its upload payload. Bindings and commands have driver
	// overhead, but no unbounded application payload allocation.
	if !p.reserveBuffer(i, 32) {
		return V1LimitExceeded
	}
	before := d.RetainedBytes()
	defer func() {
		if d.RetainedBytes() > before {
			retained := d.RetainedBytes() - before
			if retained > 32 {
				panic("backend exceeded uniform reservation")
			}
			i.retiredBytes += retained
			p.releaseBufferBytes(i, 32-retained)
		} else {
			p.releaseBufferBytes(i, 32)
		}
	}()
	err := d.ExecuteBuffers(ctx, k.pipeline.pipeline, resources, seeds, count, op)
	if err != nil {
		op.TransferCountsComplete = false
		p.buffers.stats.TotalsComplete = false
		p.quarantineBuffers(err.Error())
		op.Reason = err.Error()
		op.ReasonCode = "DEVICE_FAILURE"
		if ctx.Err() != nil {
			return V1Cancelled
		}
		return V1DeviceError
	}
	delta := TransferCounters{ParameterUploadBytes: 16, ParameterUploadCount: 1}
	for _, seed := range seeds {
		delta.DeviceCopyBytes += seed.size
		delta.DeviceCopyCount++
	}
	p.addTransfers(delta, op)
	if ctx.Err() != nil {
		op.ReasonCode = "CANCELLED"
		return V1Cancelled
	}
	if d.Lost() {
		p.quarantineBuffers("device lost before commit")
		return V1DeviceError
	}
	commitStart := time.Now()
	// Single commit decision under Plugin.mu. All fallible GPU checks are over.
	for slot, r := range outputs {
		b := i.buffers[i.bindings[slot]]
		p.keepBufferScratch(i, b.gpu, physicalBytes(b))
		b.gpu = r
		b.gpuCurrent = true
		b.cpuCurrent = false
		b.version++
		delete(outputs, slot)
	}
	op.Commit += time.Since(commitStart)
	op.Outcome = "GPU"
	op.ReasonCode = "NONE"
	return V1OK
}

func (p *Plugin) markTransferUnknown(op *BufferOperation) {
	p.buffers.stats.TotalsComplete = false
	if op != nil {
		op.TransferCountsComplete = false
	}
}

func (p *Plugin) evictBufferPool(i *bufferInstance) {
	if i.pool != nil {
		i.pool.Close()
		p.releaseBufferBytes(i, i.poolBytes)
		i.pool = nil
		i.poolBytes = 0
	}
}
func (p *Plugin) takeBufferScratch(i *bufferInstance, size uint64) (deviceBuffer, int32) {
	if i.pool != nil && i.poolBytes == size {
		r := i.pool
		i.pool = nil
		i.poolBytes = 0
		return r, V1OK
	}
	p.evictBufferPool(i)
	if !p.reserveBuffer(i, size) {
		return nil, V1LimitExceeded
	}
	r, e := p.bufferDevice().AllocateBuffer(size)
	if e != nil {
		p.releaseBufferBytes(i, size)
		return nil, V1DeviceError
	}
	return r, V1OK
}
func (p *Plugin) keepBufferScratch(i *bufferInstance, r deviceBuffer, size uint64) {
	if i.pool != nil {
		p.closeBufferResource(i, r, size)
		return
	}
	i.pool = r
	i.poolBytes = size
}
