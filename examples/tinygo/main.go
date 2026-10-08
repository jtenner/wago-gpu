//go:build !tinygo

package main

import (
	_ "embed"
	"flag"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
	"os"
)

//go:embed guest.wasm
var guest []byte

func run(cpu, requireGPU bool) error {
	kernel := gpu.KernelConfig{
		ID: 1, Export: "wago_gpu.kernel.double", CPUExport: "wago_gpu.cpu.double", RelaxedFloat: true,
		Bindings: []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead},
			{Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessWrite}},
	}
	config := gpu.Config{Disabled: cpu, Kernels: []gpu.KernelConfig{kernel}}
	return runwasm.Run(guest, runwasm.Options{GPU: &config, Calls: []string{"_initialize", "run"}}, func(result runwasm.Result) error {
		values := result.Values
		if len(values) != 1 || values[0] > 1 {
			return fmt.Errorf("guest failed: %v", values)
		}
		if requireGPU && values[0] != 0 {
			return fmt.Errorf("hardware required: %+v", result.GPU.BufferSnapshot())
		}
		fmt.Printf("TinyGo result verified: [2 4 6 8]; dispatch status %d\n", values[0])
		return nil
	})
}
func main() {
	cpu := flag.Bool("cpu", false, "disable GPU")
	required := flag.Bool("require-gpu", false, "fail unless GPU runs")
	flag.Parse()
	if e := run(*cpu, *required); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
