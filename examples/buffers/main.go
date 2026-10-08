package main

import (
	"context"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	wago "github.com/wago-org/wago"
	wagogpu "wago-gpu"
)

//go:embed module.wasm
var wasm []byte

func main() {
	firstCPU := flag.Bool("first-cpu", false, "force the first kernel to CPU")
	secondCPU := flag.Bool("second-cpu", false, "force the second kernel to CPU")
	requireGPU := flag.Bool("require-gpu", false, "require both hardware dispatches")
	flag.Parse()
	if err := run(*firstCPU, *secondCPU, *requireGPU); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(firstCPU, secondCPU, requireGPU bool) (resultErr error) {
	if requireGPU && (firstCPU || secondCPU) {
		return fmt.Errorf("require-gpu conflicts with a forced CPU kernel")
	}
	threshold := func(cpu bool) uint32 {
		if cpu {
			return 5
		} // This example dispatches four elements.
		return 0
	}
	slots := []wagogpu.BindingConfig{
		{Slot: 0, Type: wagogpu.TypeF32, Access: wagogpu.AccessRead},
		{Slot: 1, Type: wagogpu.TypeF32, Access: wagogpu.AccessWrite},
	}
	p, err := wagogpu.New(wagogpu.Config{
		Kernels: []wagogpu.KernelConfig{
			{ID: 1, Export: "wago_gpu.kernel.double",
				CPUExport: "wago_gpu.cpu.double", Bindings: slots,
				RelaxedFloat: true, MinElements: threshold(firstCPU)},
			{ID: 2, Export: "wago_gpu.kernel.addOne",
				CPUExport: "wago_gpu.cpu.addOne", Bindings: slots,
				RelaxedFloat: true, MinElements: threshold(secondCPU)},
		},
	})
	if err != nil {
		return err
	}
	set, err := p.PluginSet()
	if err != nil {
		return err
	}

	ctx := context.Background()
	rt := wago.NewRuntime()
	// Cleanup must not inherit cancellation from the work context.
	cleanupCtx := context.Background()
	defer func() { resultErr = errors.Join(resultErr, rt.CloseContext(cleanupCtx)) }()
	if err := rt.LoadPlugins(ctx, set); err != nil {
		return err
	}
	module, err := rt.Compile(wasm)
	if err != nil {
		return err
	}
	defer func() { resultErr = errors.Join(resultErr, module.Close()) }()
	// No import overrides are supplied.
	instance, err := rt.Instantiate(ctx, module)
	if err != nil {
		return err
	}
	defer func() { resultErr = errors.Join(resultErr, instance.Close()) }()
	expected := [4]float32{3, 5, 7, 9}
	for i, want := range expected {
		got, ok := instance.ReadFloat32Le(16 + uint32(i)*4)
		if !ok || math.Float32bits(got) != math.Float32bits(want) {
			return fmt.Errorf("incorrect final element %d: %v", i, got)
		}
	}
	forced := [2]bool{firstCPU, secondCPU}
	for i := 0; i < 2; i++ {
		status, ok := instance.ReadUint32Le(32 + uint32(i)*4)
		if !ok || status > 1 {
			return fmt.Errorf("invalid saved status for kernel %d", i+1)
		}
		if forced[i] && status != 1 {
			return fmt.Errorf("kernel %d did not use the requested CPU path", i+1)
		}
		if requireGPU && status != 0 {
			return fmt.Errorf("kernel %d used fallback", i+1)
		}
		fmt.Printf("kernel %d: status %d (0=GPU, 1=CPU fallback)\n", i+1, status)
	}
	fmt.Println("verified:", expected)
	return nil
}
