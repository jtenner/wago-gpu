// Command mandelbrot runs a WASI guest and writes its PGM image to stdout.
package main

import (
	"embed"
	"flag"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/examples/internal/runwasm"
	"github.com/jtenner/wago-gpu/examples/mandelbrot/internal/settings"
	"github.com/wago-org/wasi/p1"
	"io"
	"os"
	"strconv"
	"strings"
)

//go:generate tinygo build -target=wasi -scheduler=none -panic=trap -gc=leaking -opt=2 -no-debug -o cpu.wasm ./guest
//go:generate tinygo build -target=wasi -tags=gpu -scheduler=none -panic=trap -gc=leaking -opt=2 -no-debug -o gpu.wasm ./guest
//go:embed cpu.wasm gpu.wasm
var programs embed.FS

func kernel() gpu.KernelConfig {
	return gpu.KernelConfig{
		ID: 1, Export: "wago_gpu.kernel.step", CPUExport: "wago_gpu.cpu.step", RelaxedFloat: true,
		Bindings: []gpu.BindingConfig{
			{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead},
			{Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessRead},
			{Slot: 2, Type: gpu.TypeF32, Access: gpu.AccessReadWrite},
			{Slot: 3, Type: gpu.TypeF32, Access: gpu.AccessReadWrite},
		},
	}
}

// Use the same kernel and bounded plugin storage in hosts and benchmarks.
func bufferConfig(disabled bool) gpu.Config {
	return gpu.Config{Disabled: disabled, Kernels: []gpu.KernelConfig{kernel()}}
}

func run(program string, cpu, required bool, width, height, iterations uint32, out, stderr io.Writer) error {
	if program != "cpu" && program != "buffers" {
		return fmt.Errorf("program must be cpu or buffers")
	}
	if width == 0 || height == 0 || width > settings.MaxDimension || height > settings.MaxDimension || iterations == 0 || iterations > settings.MaxIterations {
		return fmt.Errorf("width/height must be 1..%d and iterations 1..%d", settings.MaxDimension, settings.MaxIterations)
	}
	if uint64(width)*uint64(height) > settings.MaxPixels {
		return fmt.Errorf("image exceeds %d pixels", settings.MaxPixels)
	}
	if required && (cpu || program != "buffers") {
		return fmt.Errorf("require-gpu needs program=buffers with the GPU enabled")
	}
	file := "cpu.wasm"
	options := runwasm.Options{Calls: []string{"_start"}, WASI: &p1.Config{
		Args:  []string{"mandelbrot", strconv.FormatUint(uint64(width), 10), strconv.FormatUint(uint64(height), 10), strconv.FormatUint(uint64(iterations), 10)},
		Stdin: strings.NewReader(""), Stdout: out, Stderr: stderr,
	}}
	if program == "buffers" {
		file = "gpu.wasm"
		config := bufferConfig(cpu)
		options.GPU = &config
	}
	source, e := programs.ReadFile(file)
	if e != nil {
		return e
	}
	return runwasm.Run(source, options, func(result runwasm.Result) error {
		if program == "cpu" {
			return nil
		}
		passes, e := result.Instance.Invoke("gpu_passes")
		if e != nil {
			return e
		}
		if len(passes) != 1 {
			return fmt.Errorf("invalid GPU pass count")
		}
		// Invoke reuses its result storage. Save this scalar before the next call.
		gpuPasses := passes[0]
		fallback, e := result.Instance.Invoke("cpu_passes")
		if e != nil {
			return e
		}
		if len(fallback) != 1 || gpuPasses+fallback[0] != uint64(iterations) {
			return fmt.Errorf("guest did not complete every pass")
		}
		if required && gpuPasses != uint64(iterations) {
			return fmt.Errorf("GPU required: %+v", result.GPU.BufferSnapshot())
		}
		if cpu && fallback[0] != uint64(iterations) {
			return fmt.Errorf("CPU fallback was not used")
		}
		snapshot := result.GPU.BufferSnapshot()
		if len(snapshot.Instances) != 1 || snapshot.Instances[0].Buffers != 0 {
			return fmt.Errorf("guest did not free its buffers")
		}
		return nil
	})
}

func main() {
	program := flag.String("program", "cpu", "cpu or buffers")
	cpu := flag.Bool("cpu", false, "force buffer CPU fallback")
	required := flag.Bool("require-gpu", false, "require every buffer pass to use the GPU")
	width := flag.Uint("width", settings.DefaultWidth, "image width, 1..4096")
	height := flag.Uint("height", settings.DefaultHeight, "image height, 1..4096")
	iterations := flag.Uint("iterations", settings.DefaultIterations, "iteration limit, 1..128")
	flag.Parse()
	if *width > settings.MaxDimension || *height > settings.MaxDimension || *iterations > settings.MaxIterations {
		fmt.Fprintln(os.Stderr, "dimensions or iteration count exceed their limit")
		os.Exit(1)
	}
	if e := run(*program, *cpu, *required, uint32(*width), uint32(*height), uint32(*iterations), os.Stdout, os.Stderr); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
