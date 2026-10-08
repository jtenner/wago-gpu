package wagogpu

import (
	"context"
	"encoding/binary"
	"time"
)

// These interfaces make transaction failures testable without a GPU. Resources
// stay private to the plugin; neither native handles nor guest views cross them.
type deviceBuffer interface{ Close() }
type bufferPipeline interface{ Close() }
type bufferDevice interface {
	BuildBuffer(string) (bufferPipeline, error)
	AllocateBuffer(uint64) (deviceBuffer, error)
	AllocateReadback(uint64) (deviceBuffer, error)
	AllocateUniform(uint64) (deviceBuffer, error)
	UploadBuffer(context.Context, deviceBuffer, []byte) error
	ReadBuffer(context.Context, deviceBuffer, deviceBuffer, uint64, func([]byte)) error
	ExecuteBuffers(context.Context, bufferPipeline, deviceBuffer, []deviceBuffer, []bufferSeed, uint32, *BufferOperation) error
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
	staging, status := p.takeTransferScratch(i, &i.readback, size, d.AllocateReadback)
	if status != V1OK {
		return status
	}
	defer func() {
		if p.stats.GPUFailed {
			p.closeBufferResource(i, staging, size)
		} else {
			i.readback = retiredBuffer{staging, size}
		}
	}()
	start := time.Now()
	var conversion time.Duration
	err := d.ReadBuffer(ctx, b.gpu, staging, size, func(data []byte) {
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
		var status int32
		data, status = p.conversionScratch(i, size)
		if status != V1OK {
			return status
		}
		// The active payload must not be reclaimed by a later reservation.
		i.conversion = nil
		defer func() { i.conversion = data }()
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
	// The ABI has at most eight slots. Fixed storage preserves binding order
	// without a sort, growing slice, or per-dispatch output map.
	var slots [8]uint32
	var resources [8]deviceBuffer
	var outputs [8]deviceBuffer
	var seeds [8]bufferSeed
	used, seedCount := 0, 0
	for slot := uint32(0); slot < 8; slot++ {
		if k.lowered.reads[slot] || k.lowered.writes[slot] {
			slots[used] = slot
			used++
		}
	}
	defer func() {
		for slot, r := range outputs {
			if r != nil {
				b := i.buffers[i.bindings[slot]]
				p.closeBufferResource(i, r, physicalBytes(b))
			}
		}
	}()
	for idx, slot := range slots[:used] {
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
			seeds[seedCount] = bufferSeed{b.gpu, r, size}
			seedCount++
		}
	}
	uniform, status := p.takeTransferScratch(i, &i.uniform, 16, d.AllocateUniform)
	if status != V1OK {
		return status
	}
	defer func() {
		if p.stats.GPUFailed {
			p.closeBufferResource(i, uniform, 16)
		} else {
			i.uniform = retiredBuffer{uniform, 16}
		}
	}()
	// The persistent parameter buffer has its own charge. Each pending queue
	// write additionally holds a 16-byte payload until completion.
	if !p.reserveBuffer(i, 16) {
		return V1LimitExceeded
	}
	before := d.RetainedBytes()
	defer func() {
		if d.RetainedBytes() > before {
			retained := d.RetainedBytes() - before
			if retained > 16 {
				panic("backend exceeded uniform reservation")
			}
			i.retiredBytes += retained
			p.releaseBufferBytes(i, 16-retained)
		} else {
			p.releaseBufferBytes(i, 16)
		}
	}()
	err := d.ExecuteBuffers(ctx, k.pipeline.pipeline, uniform, resources[:used], seeds[:seedCount], count, op)
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
	for _, seed := range seeds[:seedCount] {
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
		if r == nil {
			continue
		}
		b := i.buffers[i.bindings[slot]]
		p.keepBufferScratch(i, b.gpu, physicalBytes(b))
		b.gpu = r
		b.gpuCurrent = true
		b.cpuCurrent = false
		b.version++
		outputs[slot] = nil
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
	for _, item := range []*retiredBuffer{&i.readback, &i.uniform} {
		if item.resource != nil {
			item.resource.Close()
			p.releaseBufferBytes(i, item.size)
			*item = retiredBuffer{}
		}
	}
	if i.conversion != nil {
		p.releaseBufferBytes(i, uint64(cap(i.conversion)))
		i.conversion = nil
	}
	for n := 0; n < i.poolCount; n++ {
		item := i.pool[n]
		item.resource.Close()
		p.releaseBufferBytes(i, item.size)
		i.pool[n] = retiredBuffer{}
	}
	i.poolCount = 0
	i.poolBytes = 0
}
func (p *Plugin) takeBufferScratch(i *bufferInstance, size uint64) (deviceBuffer, int32) {
	for n := 0; n < i.poolCount; n++ {
		if i.pool[n].size == size {
			r := i.pool[n].resource
			i.poolCount--
			i.pool[n] = i.pool[i.poolCount]
			i.pool[i.poolCount] = retiredBuffer{}
			i.poolBytes -= size
			return r, V1OK
		}
	}
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
	if i.poolCount == len(i.pool) {
		p.closeBufferResource(i, r, size)
		return
	}
	i.pool[i.poolCount] = retiredBuffer{r, size}
	i.poolCount++
	i.poolBytes += size
}

// The caller removes active scratch from idle storage before making further
// reservations. Idle eviction must never release an in-flight operation.
func (p *Plugin) takeTransferScratch(i *bufferInstance, idle *retiredBuffer, size uint64, allocate func(uint64) (deviceBuffer, error)) (deviceBuffer, int32) {
	item := *idle
	*idle = retiredBuffer{}
	if item.resource != nil {
		if item.size == size {
			return item.resource, V1OK
		}
		item.resource.Close()
		p.releaseBufferBytes(i, item.size)
	}
	if !p.reserveBuffer(i, size) {
		return nil, V1LimitExceeded
	}
	r, e := allocate(size)
	if e != nil {
		p.releaseBufferBytes(i, size)
		return nil, V1DeviceError
	}
	return r, V1OK
}
func (p *Plugin) conversionScratch(i *bufferInstance, size uint64) ([]byte, int32) {
	if uint64(cap(i.conversion)) >= size {
		return i.conversion[:size], V1OK
	}
	// Reserve growth while the old allocation is still owned. Admission can
	// reclaim idle scratch, including the old conversion storage.
	if !p.reserveBuffer(i, size) {
		return nil, V1LimitExceeded
	}
	old := uint64(cap(i.conversion))
	i.conversion = make([]byte, size)
	p.releaseBufferBytes(i, old)
	return i.conversion, V1OK
}
