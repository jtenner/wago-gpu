package main

import (
	"embed"
	"flag"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
	"math"
	"os"
)

//go:generate wasm-tools parse f16.wat -o f16.wasm
//go:generate wasm-tools parse memory64.wat -o memory64.wasm
//go:generate wasm-tools parse gc.wat -o gc.wasm
//go:embed *.wasm
var modules embed.FS

func run(kind string, cpu, required bool) error {
	source, e := modules.ReadFile(kind + ".wasm")
	if e != nil {
		return e
	}
	if required && (cpu || kind != "f16") {
		return fmt.Errorf("require-gpu needs the F16 example with GPU enabled")
	}
	typ := gpu.TypeF32
	if kind == "f16" {
		typ = gpu.TypeF16
	}
	config := gpu.Config{Disabled: cpu, Kernels: []gpu.KernelConfig{{
		ID: 1, Export: "wago_gpu.kernel.storage", CPUExport: "wago_gpu.cpu.storage", RelaxedFloat: true,
		Bindings: []gpu.BindingConfig{{Slot: 0, Type: typ, Access: gpu.AccessReadWrite}},
	}}}
	return runwasm.Run(source, runwasm.Options{GPU: &config, Calls: []string{"run"}}, func(result runwasm.Result) error {
		values := result.Values
		want := float32(7)
		if kind == "f16" {
			want = 2
		}
		if len(values) != 1 || math.Float32frombits(uint32(values[0])) != want {
			return fmt.Errorf("incorrect result: %v", values)
		}
		if kind == "f16" {
			status, e := result.Instance.Invoke("status")
			if e != nil {
				return e
			}
			if required && status[0] != 0 {
				return fmt.Errorf("GPU required: %+v", result.GPU.BufferSnapshot())
			}
			fmt.Printf("F16 dispatch status %d (0=GPU, 1=CPU)\n", status[0])
		}
		fmt.Printf("%s result verified: %v\n", kind, want)
		return nil
	})
}
func main() {
	kind := flag.String("kind", "f16", "f16, memory64, or gc")
	cpu := flag.Bool("cpu", false, "disable GPU")
	required := flag.Bool("require-gpu", false, "require the F16 GPU dispatch")
	flag.Parse()
	if e := run(*kind, *cpu, *required); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
