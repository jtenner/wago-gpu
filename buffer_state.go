package wagogpu

import (
	"errors"
	wago "github.com/wago-org/wago"
	"sort"
	"time"
)

type kernelContract struct {
	translateTime, pipelineTime time.Duration
	pipeline                    *bufferPipelineEntry
	pipelineError               error
	config                      KernelConfig
	invalid                     error
	rejection                   error
	lowered                     loweredKernel
}
type moduleContract struct {
	instances int
	closed    bool
	verified  bool
	memories  []memoryDeclaration
	kernels   map[uint32]*kernelContract
}
type pendingContract struct {
	digest wago.ModuleSourceDigest
	module *moduleContract
	bytes  uint64
}
type bufferState struct {
	gpu                          deviceBuffer
	cpuCurrent, gpuCurrent, lost bool
	typ                          ElementType
	count                        uint32
	cpu                          []byte
	version                      uint64
}
type bufferInstance struct {
	peakBytes    uint64
	pool         [8]retiredBuffer
	poolCount    int
	readback     retiredBuffer
	uniform      retiredBuffer
	uniformCount uint32
	conversion   []byte
	poolBytes    uint64
	retiredBytes uint64
	module       *moduleContract
	buffers      map[uint32]*bufferState
	bindings     [8]uint32
	bytes        uint64
}
type bufferEngine struct {
	pipelines  map[string]*bufferPipelineEntry
	retired    []retiredBuffer
	pending    map[wago.CompilationIdentity]pendingContract
	order      [64]wago.CompilationIdentity
	next       int
	modules    map[wago.ModuleIdentity]*moduleContract
	instances  map[wago.InstanceIdentity]*bufferInstance
	nextHandle uint64
	bytes      uint64
	stats      BufferSnapshot
}

func newBufferEngine() *bufferEngine {
	return &bufferEngine{pipelines: map[string]*bufferPipelineEntry{}, pending: map[wago.CompilationIdentity]pendingContract{}, modules: map[wago.ModuleIdentity]*moduleContract{}, instances: map[wago.InstanceIdentity]*bufferInstance{}, stats: BufferSnapshot{TotalsComplete: true}}
}
func (b *bufferEngine) instantiate(id wago.InstanceIdentity, module wago.ModuleIdentity) {
	if m := b.modules[module]; m != nil {
		m.instances++
	}
	b.instances[id] = &bufferInstance{module: b.modules[module], buffers: map[uint32]*bufferState{}}
}
func (p *Plugin) closeBufferInstance(id wago.InstanceIdentity) {
	b := p.buffers
	if s := b.instances[id]; s != nil {
		p.evictBufferPool(s)
		for _, v := range s.buffers {
			p.closeBufferResource(s, v.gpu, physicalBytes(v))
		}
		if m := s.module; m != nil {
			m.instances--
			if m.closed && m.instances == 0 {
				p.releaseBufferModule(m)
			}
		}
		b.bytes -= s.bytes - s.retiredBytes
		delete(b.instances, id)
	}
}
func (p *Plugin) inspectBuffers(ctx wago.ModuleSourceContext, source []byte) {
	if len(p.config.Kernels) == 0 {
		return
	}
	p.mu.Lock()
	if p.stopped {
		p.mu.Unlock()
		return
	}
	p.buffers.stats.ActiveCompilations = 1
	p.mu.Unlock()
	contract := &moduleContract{kernels: map[uint32]*kernelContract{}}
	module, e := decodeBufferModule(source)
	var size uint64 = 128
	if e == nil {
		contract.verified = true
		contract.memories = append([]memoryDeclaration(nil), module.memories...)
		size += uint64(len(module.memories)) * 8
		var translationBytes uint64
		for _, k := range p.config.Kernels {
			c := &kernelContract{config: k}
			contract.kernels[k.ID] = c
			size += 512
			body, err := module.kernelBody(k)
			if err != nil {
				c.invalid = err
				continue
			}
			start := time.Now()
			c.lowered, err = lowerBufferBody(module, body, k)
			c.translateTime = time.Since(start)
			translationBytes += uint64(len(c.lowered.shader))
			if translationBytes > 8<<20 {
				c.lowered.shader = ""
				if err == nil {
					err = compileError(CompileLimit, "per-compilation WGSL limit")
				}
			}
			var ce *CompileError
			if errors.As(err, &ce) && ce.Kind == CompileInvalidContract {
				c.invalid = err
			} else {
				c.rejection = err
			}
		}
	}
	// Source buffers and parser tables die here. Only bounded contract data and
	// bounded WGSL survive. An absent/evicted record means unverified.
	p.mu.Lock()
	defer p.mu.Unlock()
	b := p.buffers
	b.stats.ActiveCompilations = 0
	if p.stopped {
		return
	}
	delete(b.pending, b.order[b.next])
	b.order[b.next] = ctx.Compilation
	b.next = (b.next + 1) % len(b.order)
	if size > 64<<10 {
		return
	}
	b.pending[ctx.Compilation] = pendingContract{wago.DigestModuleSource(source), contract, size}
	// Retain at most 16 translation sets. Contract violations remain durable.
	translationCount := 0
	var wgslBytes uint64
	for j := 1; j <= 64; j++ {
		idx := (b.next - j + 64) % 64
		pending, ok := b.pending[b.order[idx]]
		if !ok {
			continue
		}
		has := false
		for _, k := range pending.module.kernels {
			if k.lowered.shader != "" {
				has = true
				wgslBytes += uint64(len(k.lowered.shader))
			}
		}
		if has {
			translationCount++
		}
		if translationCount > 16 || wgslBytes > 16<<20 {
			for _, k := range pending.module.kernels {
				k.lowered.shader = ""
				if k.invalid == nil && k.rejection == nil {
					k.rejection = compileError(CompileLimit, "pending translation evicted")
				}
			}
		}
	}
}
func (p *Plugin) attachBufferModule(e wago.ModuleCompiledEvent) {
	b := p.buffers
	pending, ok := b.pending[e.Compilation]
	delete(b.pending, e.Compilation)
	if ok && pending.digest == e.SourceDigest {
		b.modules[e.Module.Identity()] = pending.module
		p.buildBufferPipelines(pending.module)
	} else {
		b.modules[e.Module.Identity()] = &moduleContract{}
	}
}
func (p *Plugin) reserveBuffer(s *bufferInstance, n uint64) bool {
	// One lock serializes reservations across instances. Include legacy retained
	// storage in the same runtime limit.
	legacy := uint64(0)
	for _, c := range p.cache {
		legacy += c.program.BufferBytes()
	}
	// Reclaim known-idle capacity before rejecting a reservation. The total
	// work is bounded by configured instances/buffers, not by element count.
	if s.bytes > p.config.MaxInstanceBufferBytes || n > p.config.MaxInstanceBufferBytes-s.bytes {
		p.evictBufferPool(s)
	}
	if p.buffers.bytes+legacy > p.config.MaxRuntimeBufferBytes || n > p.config.MaxRuntimeBufferBytes-p.buffers.bytes-legacy {
		for _, instance := range p.buffers.instances {
			p.evictBufferPool(instance)
		}
	}
	if s.bytes > p.config.MaxInstanceBufferBytes || n > p.config.MaxInstanceBufferBytes-s.bytes || p.buffers.bytes+legacy > p.config.MaxRuntimeBufferBytes || n > p.config.MaxRuntimeBufferBytes-p.buffers.bytes-legacy {
		return false
	}
	s.bytes += n
	p.buffers.bytes += n
	s.peakBytes = max(s.peakBytes, s.bytes)
	p.buffers.stats.PeakRuntimeBufferBytes = max(p.buffers.stats.PeakRuntimeBufferBytes, p.buffers.bytes+legacy)
	return true
}
func (p *Plugin) releaseBufferBytes(s *bufferInstance, n uint64) { s.bytes -= n; p.buffers.bytes -= n }
func (p *Plugin) BufferSnapshot() BufferSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.observeBufferDeviceLoss()
	b := p.buffers
	s := b.stats
	s.RuntimeBufferBytes = b.bytes
	s.State = "UNAVAILABLE"
	s.DeviceState = "UNAVAILABLE"
	s.Reason = p.stats.Reason
	if p.bufferDevice() != nil {
		s.State = "READY"
		s.DeviceState = "USABLE"
		s.Reason = ""
	}
	if p.stats.GPUFailed {
		s.State = "FAILED"
		s.DeviceState = "QUARANTINED"
		s.Reason = b.stats.Reason
	}
	if p.config.Disabled {
		s.State = "DISABLED"
		s.Reason = "GPU disabled"
	}
	if p.stopped {
		s.State = "STOPPED"
	}
	for id, pending := range b.pending {
		_ = id
		s.PendingContracts++
		s.PendingContractBytes += pending.bytes
		has := false
		for _, k := range pending.module.kernels {
			if k.lowered.shader != "" {
				has = true
				s.PendingWGSLBytes += uint64(len(k.lowered.shader))
			}
		}
		if has {
			s.PendingTranslations++
		}
	}
	s.PendingCompilations = s.PendingContracts
	for id, m := range b.modules {
		for _, configured := range p.config.Kernels {
			k := m.kernels[configured.ID]
			d := KernelDiagnostic{Module: id, KernelID: configured.ID, ContractVerified: m.verified, State: "UNVERIFIED_CONTRACT"}
			if k != nil {
				d.Translate = k.translateTime
				d.Pipeline = k.pipelineTime
				d.State = "CPU_ONLY"
				if k.invalid != nil {
					d.State = "INVALID_CONTRACT"
					d.Reason = k.invalid.Error()
				} else if k.pipeline != nil {
					d.State = "PIPELINE_READY"
				} else if k.pipelineError != nil {
					d.State = "PIPELINE_FAILED"
					d.Reason = k.pipelineError.Error()
				} else if k.rejection != nil {
					d.Reason = k.rejection.Error()
				}
			}
			s.Kernels = append(s.Kernels, d)
		}
	}
	for id, i := range b.instances {
		idle := i.poolBytes + i.readback.size + i.uniform.size + uint64(cap(i.conversion))
		d := InstanceDiagnostic{Instance: id, Buffers: uint32(len(i.buffers)), RetainedPoolBytes: idle, ReservedBytes: i.bytes, ScratchBytes: i.retiredBytes, PeakBytes: i.peakBytes}
		s.RuntimePoolBytes += idle
		for _, v := range i.buffers {
			d.LogicalBytes += uint64(v.count) * v.typ.spec().size
			d.CPUBytes += uint64(len(v.cpu))
			if v.gpu != nil {
				d.GPUBytes += physicalBytes(v)
			}
			switch {
			case v.lost:
				d.ContentsLost++
			case v.cpuCurrent && v.gpuCurrent:
				d.BothCurrent++
			case v.cpuCurrent:
				d.CPUCurrent++
			case v.gpuCurrent:
				d.GPUCurrent++
			}
		}
		s.Instances = append(s.Instances, d)
	}
	sort.Slice(s.Kernels, func(i, j int) bool { return s.Kernels[i].KernelID < s.Kernels[j].KernelID })
	for _, c := range p.cache {
		s.LegacyBufferBytes += c.program.BufferBytes()
	}
	s.RuntimeBufferBytes += s.LegacyBufferBytes
	return s
}

func (p *Plugin) observeBufferDeviceLoss() {
	if d := p.bufferDevice(); d != nil && d.Lost() && !p.stats.GPUFailed {
		p.quarantineBuffers("device lost")
	}
}
