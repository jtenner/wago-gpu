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
			config := gpu.Config{KernelExport: "kernel", RelaxedFloat: true, Disabled: mode == "CPU"}
			benchwasm.Run(b, wasm, runwasm.Options{GPU: &config}, mode == "GPU", func(r runwasm.Result) error {
				used, err := r.Instance.Invoke("used_gpu")
				if err != nil {
					return err
				}
				want := uint64(0)
				if mode == "GPU" {
					want = 1
				}
				if len(used) != 1 || used[0] != want {
					return fmt.Errorf("incorrect execution path: %v", used)
				}
				for i := uint32(0); i < 4; i++ {
					input, ok := r.Instance.ReadFloat32Le(i * 4)
					output, valid := r.Instance.ReadFloat32Le(16 + i*4)
					if !ok || !valid || input != float32(i+1) || output != float32((i+1)*2) {
						return fmt.Errorf("incorrect result at %d", i)
					}
				}
				return nil
			})
		})
	}
}
