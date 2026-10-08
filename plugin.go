package wagogpu

import (
	"context"
	"fmt"
	"sync"
	"time"

	wago "github.com/wago-org/wago"
)

const (
	Success int32 = iota
	Fallback
	InvalidRange
	InvalidBatch
	DefaultMaxElements uint32 = 10_000_000
	MaxBatchPasses     uint32 = 64
)

// Config selects one export per module. RelaxedFloat must be an explicit opt-in:
// GPU f32 can fuse operations and flush subnormals (see README).
type Config struct {
	Kernels                                                          []KernelConfig
	MaxBuffersPerInstance, MaxKernelsPerModule, MaxBindingsPerKernel uint32
	MaxInstanceBufferBytes, MaxRuntimeBufferBytes                    uint64
	KernelExport                                                     string
	RelaxedFloat                                                     bool
	Disabled                                                         bool
	// ProfileStages adds queue waits to measure each stage. Leave false for use.
	ProfileStages bool
	MaxElements   uint32
	// MinElements selects CPU below this size. Zero always attempts GPU execution.
	// The limit applies to a complete batch, not to each pass separately.
	MinElements uint32
	// RunTimeout bounds result polling. Native driver entry points may still block.
	RunTimeout time.Duration
}

// Timing contains host wall times. DeviceCompute and TimestampsValid remain
// zero and false until a supported backend exposes reliable GPU timestamps.
// Stage times are present only with ProfileStages. Total includes guest commit.
type Timing struct {
	Upload, Compute, Download, Commit, Total time.Duration
	DeviceCompute                            time.Duration
	TimestampsValid                          bool
	UploadBytes, DownloadBytes               uint64
	Passes                                   uint32
}
type BuildTiming struct{ Translate, Pipeline time.Duration }
type Snapshot struct {
	Device, Reason                         string
	GPUFailed                              bool
	DeviceInit                             time.Duration
	Build                                  BuildTiming
	Last                                   Timing
	Successes, Fallbacks                   uint64
	Modules, Instances, Pipelines, Pending int
	BufferBytes                            uint64
}

// The private backend commits mapped bytes only after all GPU operations succeed.
// It must not retain input or commit. All its calls are serialized by Plugin.mu.
type backend interface {
	Compile(string) (program, error)
	Info() string
	Close()
}
type program interface {
	Run(context.Context, []byte, func([]byte), runOptions) (Timing, error)
	BufferBytes() uint64
	Close()
}
type runOptions struct {
	passes  uint32
	profile bool
}

type cachedProgram struct {
	program program
	refs    int
	shader  string
}
type moduleState struct {
	cache       *cachedProgram
	instances   int
	closed      bool
	compilation wago.CompilationIdentity
	reason      string
}
type pendingBuild struct {
	shader  string
	digest  wago.ModuleSourceDigest
	elapsed time.Duration
	reason  string
}

// Plugin owns all GPU state. Make a new Plugin for each Wago Runtime.
// Different instances may call it concurrently; GPU work is serialized so that
// cached buffers can be reused. Wago controls calls within a single instance.
type Plugin struct {
	mu                  sync.Mutex
	compileMu           sync.Mutex
	buffers             *bufferEngine
	config              Config
	registered, stopped bool
	open                func() (backend, error)
	device              backend
	resolver            *wago.CallerResolver
	pending             map[wago.CompilationIdentity]pendingBuild
	// Wago has no event for an abandoned PreparedCompile. Bound retained source
	// state; an evicted compilation safely uses CPU fallback if completed later.
	pendingOrder [16]wago.CompilationIdentity
	pendingNext  int
	compiled     map[wago.CompilationIdentity]wago.ModuleIdentity
	modules      map[wago.ModuleIdentity]*moduleState
	instances    map[wago.InstanceIdentity]*moduleState
	cache        map[string]*cachedProgram
	stats        Snapshot
}

func New(config Config) (*Plugin, error) {
	if config.KernelExport == "" && len(config.Kernels) == 0 {
		return nil, fmt.Errorf("KernelExport must select a function")
	}
	if config.MaxElements == 0 {
		config.MaxElements = DefaultMaxElements
	}
	if config.RunTimeout == 0 {
		config.RunTimeout = 5 * time.Second
	}
	if config.RunTimeout < 0 {
		return nil, fmt.Errorf("RunTimeout must be positive")
	}
	if config.MinElements > config.MaxElements {
		return nil, fmt.Errorf("MinElements exceeds MaxElements")
	}
	if config.MaxElements > DefaultMaxElements {
		return nil, fmt.Errorf("MaxElements exceeds %d", DefaultMaxElements)
	}
	if err := validateBufferConfig(&config); err != nil {
		return nil, err
	}
	return &Plugin{buffers: newBufferEngine(), config: config, open: openBackend, pending: make(map[wago.CompilationIdentity]pendingBuild), compiled: make(map[wago.CompilationIdentity]wago.ModuleIdentity), modules: make(map[wago.ModuleIdentity]*moduleState), instances: make(map[wago.InstanceIdentity]*moduleState), cache: make(map[string]*cachedProgram)}, nil
}

func definition() wago.PluginDefinition {
	d := wago.PluginDefinition{ID: "github.com/jtenner/wago-gpu", Name: "Wago GPU", Version: "0.0.0", Description: "Explicit Wasm GPU kernels and typed buffers", Stability: wago.Experimental}
	// The repository owns this experimental plugin identity.
	d.Provenance = wago.PluginProvenance{Repository: "https://github.com/jtenner/wago-gpu", License: "MIT"}
	for _, a := range []wago.Authority{wago.AuthorityHostImportDefine, wago.AuthorityHostCallerIdentify, wago.AuthorityModuleSourceTransform, wago.AuthorityModuleCompileObserve, wago.AuthorityModuleCloseObserve, wago.AuthorityInstanceInstantiateIntercept, wago.AuthorityInstanceCloseObserve} {
		r := wago.AuthorityRequest{Name: a, Mode: wago.AuthorityRequired, Reason: "connect the selected kernel to checked guest calls and release its resources"}
		if a == wago.AuthorityHostImportDefine {
			r.Scope = wago.AuthorityScope{Modules: []string{"wago_gpu", "wago_gpu_v1"}}
		}
		d.Authorities = append(d.Authorities, r)
	}
	return d
}

// PluginSet grants only this experiment's declared authorities. The host opts
// into these grants by passing this set to Runtime.LoadPlugins.
func (p *Plugin) PluginSet() (wago.PluginSet, error) {
	d := definition()
	digest, err := wago.DefinitionDigest(d)
	if err != nil {
		return wago.PluginSet{}, err
	}
	s := wago.PluginSelection{ID: d.ID, DefinitionDigest: digest, Direct: true, Dependencies: map[string]string{}}
	for _, r := range d.Authorities {
		s.Grants = append(s.Grants, wago.AuthorityGrant{Name: r.Name, Scope: r.Scope})
	}
	return wago.PluginSet{Providers: []wago.PluginProvider{{Definition: d, New: func() wago.Plugin { return p }}}, Selections: []wago.PluginSelection{s}}, nil
}

func (p *Plugin) Register(reg *wago.Registrar) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.registered {
		return fmt.Errorf("Plugin already belongs to a runtime")
	}
	p.registered = true
	var err error
	p.resolver, err = reg.HostCallers()
	if err != nil {
		return err
	}
	i, err := reg.HostImports()
	if err != nil {
		return err
	}
	i.HostFunc("wago_gpu", "run", p.run).Params(wago.ValI32, wago.ValI32, wago.ValI32).Results(wago.ValI32).Docs("run(input byte offset, output byte offset, f32 count): 0 success, 1 CPU fallback, 2 invalid range")
	i.HostFunc("wago_gpu", "run_batch", p.runBatch).Params(wago.ValI32, wago.ValI32, wago.ValI32, wago.ValI32).Results(wago.ValI32).Docs("run the selected kernel 1..64 times with one upload and one download")
	if err := p.registerBuffers(reg); err != nil {
		return err
	}
	t, err := reg.ModuleSourceTransformer()
	if err != nil {
		return err
	}
	if err = t.Transform(p.transform); err != nil {
		return err
	}
	c, err := reg.ModuleCompileObserver()
	if err != nil {
		return err
	}
	if err = c.Observe(p.onCompiled); err != nil {
		return err
	}
	if err = c.OnError(p.onCompileError); err != nil {
		return err
	}
	m, err := reg.ModuleCloseObserver()
	if err != nil {
		return err
	}
	if err = m.Observe(func(e wago.ModuleCloseEvent) { p.mu.Lock(); defer p.mu.Unlock(); p.closeModule(e.Module.Identity()) }); err != nil {
		return err
	}
	a, err := reg.InstanceInstantiateInterceptor()
	if err != nil {
		return err
	}
	if err = a.After(func(e wago.InstantiationEvent) error {
		p.mu.Lock()
		defer p.mu.Unlock()
		p.buffers.instantiate(e.Instance, e.Module.Identity())
		if s := p.modules[e.Module.Identity()]; s != nil {
			s.instances++
			p.instances[e.Instance] = s
		}
		return nil
	}); err != nil {
		return err
	}
	x, err := reg.InstanceCloseObserver()
	if err != nil {
		return err
	}
	if err = x.After(func(e wago.InstanceCloseEvent) {
		p.mu.Lock()
		defer p.mu.Unlock()
		p.closeBufferInstance(e.Instance)
		if s := p.instances[e.Instance]; s != nil {
			delete(p.instances, e.Instance)
			s.instances--
			if s.closed && s.instances == 0 {
				p.release(s)
			}
		}
	}); err != nil {
		return err
	}
	return reg.Lifecycle(wago.PluginLifecycle{Start: p.start, Stop: p.stop})
}

func (p *Plugin) start(context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.config.Disabled {
		p.stats.Reason = "GPU disabled"
		return nil
	}
	if !p.config.RelaxedFloat && len(p.config.Kernels) == 0 {
		p.stats.Reason = "relaxed f32 behavior was not accepted"
		return nil
	}
	start := time.Now()
	d, err := p.open()
	p.stats.DeviceInit = time.Since(start)
	if err != nil {
		p.stats.Reason = err.Error()
		return nil
	}
	p.device = d
	p.stats.Device = d.Info()
	return nil
}
func (p *Plugin) transform(ctx wago.ModuleSourceContext, source []byte) ([]byte, error) {
	p.compileMu.Lock()
	defer p.compileMu.Unlock()
	p.mu.Lock()
	stopped := p.stopped
	p.mu.Unlock()
	if stopped {
		return source, nil
	}
	p.inspectBuffers(ctx, source)
	start := time.Now()
	shader, err := CompileWGSL(source, p.config.KernelExport)
	b := pendingBuild{shader: shader, digest: wago.DigestModuleSource(source), elapsed: time.Since(start)}
	if err != nil {
		b.reason = err.Error()
	}
	p.mu.Lock()
	if p.stopped {
		p.mu.Unlock()
		return source, nil
	}
	delete(p.pending, p.pendingOrder[p.pendingNext])
	p.pendingOrder[p.pendingNext] = ctx.Compilation
	p.pendingNext = (p.pendingNext + 1) % len(p.pendingOrder)
	p.pending[ctx.Compilation] = b
	p.mu.Unlock()
	return source, nil // An unsupported GPU kernel must not prevent CPU compilation.
}
func (p *Plugin) onCompiled(e wago.ModuleCompiledEvent) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.attachBufferModule(e)
	b, ok := p.pending[e.Compilation]
	delete(p.pending, e.Compilation)
	s := &moduleState{compilation: e.Compilation, reason: b.reason}
	p.modules[e.Module.Identity()] = s
	p.compiled[e.Compilation] = e.Module.Identity()
	p.stats.Build = BuildTiming{Translate: b.elapsed}
	if !ok || b.digest != e.SourceDigest {
		s.reason = "source unavailable or changed after GPU inspection"
		p.stats.Reason = s.reason
		return
	}
	if b.reason != "" {
		p.stats.Reason = b.reason
		return
	}
	if p.device == nil || p.stats.GPUFailed {
		return
	}
	if c := p.cache[b.shader]; c != nil {
		c.refs++
		s.cache = c
		return
	}
	if len(p.cache)+len(p.buffers.pipelines) >= 256 {
		s.reason = "pipeline count limit"
		return
	}
	start := time.Now()
	program, err := p.device.Compile(b.shader)
	p.stats.Build.Pipeline = time.Since(start)
	if err != nil {
		s.reason = err.Error()
		p.stats.Reason = s.reason
		return
	}
	c := &cachedProgram{program: program, refs: 1, shader: b.shader}
	p.cache[b.shader] = c
	s.cache = c
}
func (p *Plugin) onCompileError(e wago.ModuleCompileErrorEvent) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.pending, e.Compilation)
	delete(p.buffers.pending, e.Compilation)
	if id, ok := p.compiled[e.Compilation]; ok {
		p.closeModule(id)
	}
}
func (p *Plugin) closeModule(id wago.ModuleIdentity) {
	if m := p.buffers.modules[id]; m != nil {
		m.closed = true
		if m.instances == 0 {
			p.releaseBufferModule(m)
		}
		delete(p.buffers.modules, id)
	}
	if s := p.modules[id]; s != nil {
		delete(p.modules, id)
		delete(p.compiled, s.compilation)
		s.closed = true
		if s.instances == 0 {
			p.release(s)
		}
	}
}
func (p *Plugin) release(s *moduleState) {
	if c := s.cache; c != nil {
		s.cache = nil
		c.refs--
		if c.refs == 0 && !p.stats.GPUFailed {
			c.program.Close()
			delete(p.cache, c.shader)
		}
	}
}
func (p *Plugin) stop(context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		return nil
	}
	p.stopped = true
	for id := range p.buffers.instances {
		p.closeBufferInstance(id)
	}
	for _, m := range p.buffers.modules {
		p.releaseBufferModule(m)
	}
	for _, r := range p.buffers.retired {
		r.resource.Close()
		p.buffers.bytes -= r.size
	}
	p.buffers.retired = nil
	clear(p.buffers.modules)
	clear(p.buffers.pending)
	for _, c := range p.cache {
		c.program.Close()
	}
	clear(p.cache)
	clear(p.pending)
	clear(p.pendingOrder[:])
	clear(p.compiled)
	clear(p.modules)
	clear(p.instances)
	if p.device != nil {
		if d := p.bufferDevice(); d != nil {
			p.buffers.bytes -= d.RetainedBytes()
		}
		p.device.Close()
		p.device = nil
	}
	return nil
}

func (p *Plugin) run(caller wago.Caller, call wago.HostCall) { p.runPasses(caller, call, 1) }
func (p *Plugin) runBatch(caller wago.Caller, call wago.HostCall) {
	p.runPasses(caller, call, uint32(call.I32(3)))
}
func (p *Plugin) runPasses(caller wago.Caller, call wago.HostCall, passes uint32) {
	// Capture all parameters before setting a result: HostCall can share slots.
	in, out, n := uint64(uint32(call.I32(0))), uint64(uint32(call.I32(1))), uint64(uint32(call.I32(2)))
	if passes == 0 || passes > MaxBatchPasses {
		call.SetI32(0, InvalidBatch)
		return
	}
	call.SetI32(0, Fallback)
	id, err := p.resolver.Resolve(caller)
	if err != nil {
		return
	}
	ctx, err := p.resolver.InvocationContext(caller)
	if err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, p.config.RunTimeout)
	defer cancel()
	p.mu.Lock()
	defer p.mu.Unlock()
	start := time.Now()
	p.stats.Last = Timing{}
	status := Fallback
	defer func() {
		call.SetI32(0, status)
		if status == Success {
			p.stats.Successes++
		} else if status == Fallback {
			p.stats.Fallbacks++
		}
	}()
	storage, ok := any(caller).(wago.GuestStorageHostModule)
	if !ok {
		return
	}
	err = storage.WithGuestStorage(func(s wago.GuestStorage) error {
		info, e := s.MemoryInfo(0)
		if e != nil {
			return e
		}
		if info.AddressType != wago.GuestMemory32 {
			return fmt.Errorf("Memory32 required")
		}
		input, e := s.MemoryRange(0, in, n*4, wago.GuestStorageRead)
		if e != nil {
			return e
		}
		output, e := s.MemoryRange(0, out, n*4, wago.GuestStorageWrite)
		if e != nil {
			return e
		}
		// The original guest may read and write in a forward loop. Whole-array
		// GPU staging has different semantics for a partial overlap.
		if in != out && in < out+n*4 && out < in+n*4 {
			p.stats.Reason = "partial input/output overlap requires CPU execution"
			return nil
		}
		if n > uint64(p.config.MaxElements) {
			p.stats.Reason = "element limit exceeded"
			return nil
		}
		if n < uint64(p.config.MinElements) {
			p.stats.Reason = "array is below MinElements"
			return nil
		}
		state := p.instances[id]
		if p.stopped || p.stats.GPUFailed || state == nil || state.cache == nil {
			if !p.stats.GPUFailed && state != nil && state.reason != "" {
				p.stats.Reason = state.reason
			}
			return nil
		}
		if n == 0 {
			status = Success
			return nil
		}
		// A call can expire while another instance owns the queue. No GPU
		// work was submitted for this call, so this is not a device failure.
		if e := ctx.Err(); e != nil {
			p.stats.Reason = e.Error()
			return nil
		}
		if budgeted, ok := state.cache.program.(interface{ PeakBufferBytes(uint64) uint64 }); ok {
			used := p.buffers.bytes
			for _, cache := range p.cache {
				if cache != state.cache {
					used += cache.program.BufferBytes()
				}
			}
			peak := budgeted.PeakBufferBytes(n * 4)
			if used > p.config.MaxRuntimeBufferBytes || peak > p.config.MaxRuntimeBufferBytes-used {
				p.stats.Reason = "runtime memory budget exceeded"
				return nil
			}
		}
		retainedBefore := uint64(0)
		if d := p.bufferDevice(); d != nil {
			retainedBefore = d.RetainedBytes()
		}
		timing, e := state.cache.program.Run(ctx, input, func(data []byte) { copy(output, data) }, runOptions{passes: passes, profile: p.config.ProfileStages})
		p.stats.Last = timing
		if d := p.bufferDevice(); d != nil {
			p.buffers.bytes += d.RetainedBytes() - retainedBefore
		}
		if e != nil {
			p.stats.Reason = e.Error()
			p.quarantineBuffers(e.Error()) // One device-wide failure transition for both ABIs.
			return nil
		}
		status = Success
		return nil
	})
	if err != nil {
		status = InvalidRange
		p.stats.Reason = err.Error()
	}
	p.stats.Last.Total = time.Since(start)
}

func (p *Plugin) Snapshot() Snapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	s := p.stats
	s.Modules = len(p.modules)
	s.Instances = len(p.instances)
	s.Pipelines = len(p.cache)
	s.Pending = len(p.pending)
	for _, c := range p.cache {
		s.BufferBytes += c.program.BufferBytes()
	}
	return s
}
