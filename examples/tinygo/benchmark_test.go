//go:build !tinygo

package main

import (
	"fmt"
	"testing"

	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/benchwasm"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
)

func BenchmarkExample(b *testing.B) {
	for _, mode := range []string{"CPU", "GPU"} {
		b.Run(mode, func(b *testing.B) {
			config := gpu.Config{Disabled: mode == "CPU", Kernels: []gpu.KernelConfig{{
				ID: 1, Export: "wago_gpu.kernel.double", CPUExport: "wago_gpu.cpu.double", RelaxedFloat: true,
				Bindings: []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead},
					{Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessWrite}},
			}}}
			benchwasm.Run(b, guest, runwasm.Options{GPU: &config, Calls: []string{"_initialize", "run"}}, mode == "GPU", func(r runwasm.Result) error {
				want := uint64(0)
				if mode == "CPU" {
					want = 1
				}
				// The guest checks all four output elements before returning status.
				if len(r.Values) != 1 || r.Values[0] != want {
					return fmt.Errorf("guest failed or took the wrong path: %v", r.Values)
				}
				return nil
			})
		})
	}
}
