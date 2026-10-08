// Command start runs a GPU import from a real Wasm start function.
package main

import (
	_ "embed"
	"flag"
	"fmt"
	"os"

	wagogpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
	wago "github.com/wago-org/wago"
)

//go:generate wat2wasm module.wat -o module.wasm
//go:embed module.wasm
var wasm []byte

func main() {
	cpuOnly := flag.Bool("cpu-only", false, "force CPU fallback inside the start function")
	requireGPU := flag.Bool("require-gpu", false, "fail if startup used CPU fallback")
	flag.Parse()
	if err := run(*cpuOnly, *requireGPU); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(cpuOnly, requireGPU bool) error {
	config := wagogpu.Config{KernelExport: "kernel", RelaxedFloat: true, Disabled: cpuOnly}
	return runwasm.Run(wasm, runwasm.Options{GPU: &config}, func(result runwasm.Result) error {
		instance := result.Instance
		// The computation is complete. Read the flag and memory; do not call a
		// "run" export or call "kernel" from Go.
		flag, err := instance.Invoke("used_gpu")
		if err != nil {
			return err
		}
		usedGPU := len(flag) == 1 && wago.AsI32(flag[0]) == 1
		if requireGPU && !usedGPU {
			return fmt.Errorf("GPU required: %s", result.GPU.Snapshot().Reason)
		}
		var input, output [4]float32
		for j := range input {
			var ok bool
			input[j], ok = instance.ReadFloat32Le(uint32(j) * 4)
			if !ok {
				return fmt.Errorf("cannot read input %d", j)
			}
			output[j], ok = instance.ReadFloat32Le(16 + uint32(j)*4)
			if !ok || input[j] != float32(j+1) || output[j] != float32((j+1)*2) {
				return fmt.Errorf("incorrect startup result at element %d", j)
			}
		}
		mode := "CPU fallback"
		if usedGPU {
			mode = "GPU"
		}
		fmt.Printf("Execution: %s, inside the Wasm start function\n", mode)
		fmt.Printf("Input:  %v\nOutput: %v\n", input, output)
		return nil
	})
}
