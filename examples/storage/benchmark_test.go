package main

import (
	"fmt"
	"math"
	"testing"

	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/benchwasm"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
)

func BenchmarkExample(b *testing.B) {
	for _, kind := range []string{"f16", "memory64", "gc"} {
		modes := []string{"CPU"}
		if kind == "f16" {
			modes = append(modes, "GPU")
		}
		for _, mode := range modes {
			b.Run(kind+"/"+mode, func(b *testing.B) {
				source, err := modules.ReadFile(kind + ".wasm")
				if err != nil {
					b.Fatal(err)
				}
				typ := gpu.TypeF32
				if kind == "f16" {
					typ = gpu.TypeF16
				}
				config := gpu.Config{Disabled: mode == "CPU", Kernels: []gpu.KernelConfig{{
					ID: 1, Export: "wago_gpu.kernel.storage", CPUExport: "wago_gpu.cpu.storage", RelaxedFloat: true,
					Bindings: []gpu.BindingConfig{{Slot: 0, Type: typ, Access: gpu.AccessReadWrite}},
				}}}
				benchwasm.Run(b, source, runwasm.Options{GPU: &config, Calls: []string{"run"}}, mode == "GPU", func(r runwasm.Result) error {
					want := float32(7)
					if kind == "f16" {
						want = 2
					}
					// Invoke results borrow Wago's scratch. Check run's value before
					// invoking status, which can reuse that same result storage.
					if len(r.Values) != 1 || math.Float32frombits(uint32(r.Values[0])) != want {
						return fmt.Errorf("incorrect result: %v", r.Values)
					}
					if kind == "f16" {
						status, err := r.Instance.Invoke("status")
						if err != nil {
							return err
						}
						code := uint64(0)
						if mode == "CPU" {
							code = 1
						}
						if len(status) != 1 || status[0] != code {
							return fmt.Errorf("incorrect execution path: %v", status)
						}
					}
					return nil
				})
			})
		}
	}
}
