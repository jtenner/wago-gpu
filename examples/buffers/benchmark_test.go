package main

import (
	"fmt"
	"testing"

	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/benchwasm"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
)

func BenchmarkExample(b *testing.B) {
	for _, mode := range []struct {
		name string
		cpu  [2]bool
	}{{"CPU_CPU", [2]bool{true, true}}, {"GPU_GPU", [2]bool{}},
		{"GPU_CPU", [2]bool{false, true}}, {"CPU_GPU", [2]bool{true, false}}} {
		b.Run(mode.name, func(b *testing.B) {
			slots := []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead},
				{Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessWrite}}
			config := gpu.Config{Disabled: mode.cpu[0] && mode.cpu[1]}
			for i, name := range []string{"double", "addOne"} {
				minimum := uint32(0)
				if mode.cpu[i] {
					minimum = 5
				}
				config.Kernels = append(config.Kernels, gpu.KernelConfig{ID: uint32(i + 1),
					Export: "wago_gpu.kernel." + name, CPUExport: "wago_gpu.cpu." + name,
					Bindings: slots, RelaxedFloat: true, MinElements: minimum})
			}
			benchwasm.Run(b, wasm, runwasm.Options{GPU: &config}, !config.Disabled, func(r runwasm.Result) error {
				for i, want := range [4]float32{3, 5, 7, 9} {
					got, ok := r.Instance.ReadFloat32Le(16 + uint32(i)*4)
					if !ok || got != want {
						return fmt.Errorf("incorrect element %d: %v", i, got)
					}
				}
				for i, cpu := range mode.cpu {
					want := uint32(0)
					if cpu {
						want = 1
					}
					got, ok := r.Instance.ReadUint32Le(32 + uint32(i)*4)
					if !ok || got != want {
						return fmt.Errorf("kernel %d: status %d, require %d", i+1, got, want)
					}
				}
				return nil
			})
		})
	}
}
