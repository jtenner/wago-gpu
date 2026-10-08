package wagogpu

import (
	"context"
	"fmt"
	wago "github.com/wago-org/wago"
)

// Host is the controlled v1 integration. It exposes source compilation only and
// supplies no per-instance import overrides. It never adopts compiled artifacts.
// Use this type when the host cannot otherwise enforce the source-identity rules.
type Host struct {
	runtime *wago.Runtime
	plugin  *Plugin
}
type Module struct {
	owner  *Host
	module *wago.Module
}
type Preparation struct {
	owner    *Host
	prepared *wago.PreparedCompile
}

func NewHost(ctx context.Context, config Config) (*Host, error) {
	p, e := New(config)
	if e != nil {
		return nil, e
	}
	set, e := p.PluginSet()
	if e != nil {
		return nil, e
	}
	r := wago.NewRuntime()
	if e = r.LoadPlugins(ctx, set); e != nil {
		_ = r.CloseContext(context.Background())
		return nil, e
	}
	return &Host{r, p}, nil
}
func (h *Host) Compile(source []byte) (*Module, error) {
	m, e := h.runtime.Compile(source)
	if e != nil {
		return nil, e
	}
	return &Module{h, m}, nil
}
func (h *Host) Prepare(source []byte) (*Preparation, error) {
	p, e := h.runtime.PrepareCompile(source)
	if e != nil {
		return nil, e
	}
	return &Preparation{h, p}, nil
}
func (p *Preparation) Compile() (*Module, error) {
	m, e := p.prepared.Compile()
	if e != nil {
		return nil, e
	}
	return &Module{p.owner, m}, nil
}
func (p *Preparation) Close() error { return p.prepared.Close() }
func (m *Module) Close() error      { return m.module.Close() }
func (h *Host) Instantiate(ctx context.Context, module *Module) (*wago.Instance, error) {
	if module == nil || module.owner != h {
		return nil, fmt.Errorf("module belongs to a different host")
	}
	return h.runtime.Instantiate(ctx, module.module)
}
func (h *Host) Close(ctx context.Context) error { return h.runtime.CloseContext(ctx) }
func (h *Host) BufferSnapshot() BufferSnapshot  { return h.plugin.BufferSnapshot() }
func (h *Host) Snapshot() Snapshot              { return h.plugin.Snapshot() }
