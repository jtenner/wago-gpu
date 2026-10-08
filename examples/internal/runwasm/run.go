// Package runwasm keeps setup and cleanup out of the example programs.
package runwasm

import (
	"context"
	"errors"
	gpu "github.com/jtenner/wago-gpu"
	wago "github.com/wago-org/wago"
	"github.com/wago-org/wasi/p1"
)

type Options struct {
	GPU   *gpu.Config
	WASI  *p1.Config
	Calls []string
}

type Result struct {
	Instance *wago.Instance
	Values   []uint64
	GPU      *gpu.Plugin
}

// Run compiles source, calls the selected exports, checks the result, and closes
// all resources. WASI imports use their own namespace; GPU intrinsics cannot be
// overridden here. check must not retain the instance after Run returns.
func Run(source []byte, options Options, check func(Result) error) (err error) {
	ctx := context.Background()
	rt := wago.NewRuntime()
	defer func() { err = errors.Join(err, rt.CloseContext(context.Background())) }()
	var plugin *gpu.Plugin
	if options.GPU != nil {
		var e error
		plugin, e = gpu.New(*options.GPU)
		if e != nil {
			return e
		}
		set, e := plugin.PluginSet()
		if e != nil {
			return e
		}
		if e = rt.LoadPlugins(ctx, set); e != nil {
			return e
		}
	}
	module, e := rt.Compile(source)
	if e != nil {
		return e
	}
	defer func() { err = errors.Join(err, module.Close()) }()
	var instantiate []wago.InstantiateOption
	if options.WASI != nil {
		instantiate = append(instantiate, wago.WithImports(p1.Imports(*options.WASI)))
	}
	instance, e := rt.Instantiate(ctx, module, instantiate...)
	if e != nil {
		return e
	}
	defer func() { err = errors.Join(err, instance.Close()) }()
	result := Result{Instance: instance, GPU: plugin}
	for _, name := range options.Calls {
		result.Values, e = instance.Invoke(name)
		if e != nil {
			var exit *wago.ExitError
			if options.WASI == nil || !errors.As(e, &exit) || exit.Code != 0 {
				return e
			}
			// WASI proc_exit ends the command even when its status is zero.
			break
		}
	}
	if check != nil {
		return check(result)
	}
	return nil
}
