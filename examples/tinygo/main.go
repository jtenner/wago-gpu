//go:build !tinygo

package main

import (
	"context"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"os"
)

//go:embed guest.wasm
var guest []byte

func run(cpu, requireGPU bool) (result error) {
	host, e := gpu.NewHost(context.Background(), gpu.Config{Disabled: cpu, Kernels: []gpu.KernelConfig{{ID: 1, Export: "wago_gpu.kernel.double", CPUExport: "wago_gpu.cpu.double", RelaxedFloat: true, Bindings: []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead}, {Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessWrite}}}}})
	if e != nil {
		return e
	}
	defer func() { result = errors.Join(result, host.Close(context.Background())) }()
	module, e := host.Compile(guest)
	if e != nil {
		return e
	}
	defer func() { result = errors.Join(result, module.Close()) }()
	instance, e := host.Instantiate(context.Background(), module)
	if e != nil {
		return e
	}
	defer func() { result = errors.Join(result, instance.Close()) }()
	if _, e = instance.Invoke("_initialize"); e != nil {
		return e
	}
	values, e := instance.Invoke("run")
	if e != nil {
		return e
	}
	if len(values) != 1 || values[0] > 1 {
		return fmt.Errorf("guest failed: %v", values)
	}
	if requireGPU && values[0] != 0 {
		return fmt.Errorf("hardware required: %+v", host.BufferSnapshot())
	}
	fmt.Printf("TinyGo result verified: [2 4 6 8]; dispatch status %d\n", values[0])
	return nil
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
