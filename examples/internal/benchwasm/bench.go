// Package benchwasm measures complete example commands, not individual kernels.
package benchwasm

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
	wago "github.com/wago-org/wago"
	"github.com/wago-org/wasi/p1"
)

// Run checks every result. GPU cases must also check their dispatch statuses.
// Both timings include instance creation, guest execution, checks, and cleanup.
// ReuseModule retains the module and device, but never a guest instance.
func Run(b *testing.B, source []byte, options runwasm.Options, hardware bool, check func(runwasm.Result) error) {
	b.Helper()
	if hardware && os.Getenv("WAGO_GPU_TEST") != "1" {
		b.Skip("set WAGO_GPU_TEST=1 and build with -tags webgpu to require hardware")
	}
	for _, reuse := range []bool{false, true} {
		name := "SetupAndExecute"
		if reuse {
			name = "ReuseModule"
		}
		b.Run(name, func(b *testing.B) {
			s := prepare(b, source, options)
			b.Cleanup(func() {
				if err := s.close(); err != nil {
					b.Error(err)
				}
			})
			start := time.Now()
			if err := s.invoke(options, check); err != nil {
				b.Fatal(err)
			}
			first := time.Since(start)
			var peak uint64
			if s.plugin != nil {
				peak = s.plugin.BufferSnapshot().PeakRuntimeBufferBytes
			}
			if !reuse {
				if err := s.close(); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				current := s
				if !reuse {
					current = prepare(b, source, options)
				}
				err := current.invoke(options, check)
				if !reuse {
					err = errors.Join(err, current.close())
				}
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			if reuse {
				b.ReportMetric(float64(s.compile)/1e6, "compile-ms")
				b.ReportMetric(float64(first)/1e6, "first-command-ms")
				if s.plugin != nil {
					b.ReportMetric(float64(s.plugin.Snapshot().DeviceInit)/1e6, "device-init-ms")
					peak = s.plugin.BufferSnapshot().PeakRuntimeBufferBytes
					if hardware {
						b.Logf("device: %s", s.plugin.Snapshot().Device)
					}
				}
			}
			b.ReportMetric(float64(peak), "tracked-peak-B")
			if err := s.close(); err != nil {
				b.Fatal(err)
			}
		})
	}
}

type session struct {
	runtime *wago.Runtime
	module  *wago.Module
	plugin  *gpu.Plugin
	compile time.Duration
}

func prepare(b *testing.B, source []byte, options runwasm.Options) *session {
	b.Helper()
	s := &session{runtime: wago.NewRuntime()}
	// Setup errors close resources here. The caller closes successful sessions.
	var err error
	if options.GPU != nil {
		s.plugin, err = gpu.New(*options.GPU)
		if err == nil {
			var set wago.PluginSet
			set, err = s.plugin.PluginSet()
			if err == nil {
				err = s.runtime.LoadPlugins(context.Background(), set)
			}
		}
	}
	if err == nil {
		start := time.Now()
		s.module, err = s.runtime.Compile(source)
		s.compile = time.Since(start)
	}
	if err != nil {
		b.Fatal(errors.Join(err, s.close()))
	}
	return s
}

func (s *session) invoke(options runwasm.Options, check func(runwasm.Result) error) (err error) {
	var instantiate []wago.InstantiateOption
	if options.WASI != nil {
		instantiate = append(instantiate, wago.WithImports(p1.Imports(*options.WASI)))
	}
	instance, err := s.runtime.Instantiate(context.Background(), s.module, instantiate...)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, instance.Close()) }()
	result := runwasm.Result{Instance: instance, GPU: s.plugin}
	for _, name := range options.Calls {
		result.Values, err = instance.Invoke(name)
		if err != nil {
			var exit *wago.ExitError
			if options.WASI == nil || !errors.As(err, &exit) || exit.Code != 0 {
				return err
			}
			break
		}
	}
	return check(result)
}

func (s *session) close() (err error) {
	if s.module != nil {
		err = s.module.Close()
		s.module = nil
	}
	if s.runtime != nil {
		err = errors.Join(err, s.runtime.CloseContext(context.Background()))
		s.runtime = nil
	}
	return err
}
