package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	gpu "github.com/jtenner/wago-gpu"
	wago "github.com/wago-org/wago"
	"github.com/wago-org/wasi/p1"
)

var benchmarkSizes = [][2]uint32{{96, 64}, {256, 192}, {512, 512}}

// Both cases include a fresh WASI command instance and its cleanup. The second
// case retains the compiled module, device, and pipeline between commands.
func BenchmarkMandelbrotSetupAndRender(b *testing.B) { benchmarkMandelbrot(b, false) }
func BenchmarkMandelbrotReuseModule(b *testing.B)    { benchmarkMandelbrot(b, true) }

func benchmarkMandelbrot(b *testing.B, reuse bool) {
	for _, size := range benchmarkSizes {
		for _, mode := range []string{"cpu", "fallback", "gpu"} {
			b.Run(fmt.Sprintf("%dx%d/%s", size[0], size[1], mode), func(b *testing.B) {
				if mode == "gpu" && os.Getenv("WAGO_GPU_TEST") != "1" {
					b.Skip("set WAGO_GPU_TEST=1 and build with -tags webgpu to require hardware")
				}
				var reference, actual bytes.Buffer
				if e := run("cpu", false, false, size[0], size[1], 32, &reference, io.Discard); e != nil {
					b.Fatal(e)
				}
				program := "cpu"
				if mode != "cpu" {
					program = "buffers"
				}
				if e := run(program, mode == "fallback", mode == "gpu", size[0], size[1], 32, &actual, io.Discard); e != nil {
					b.Fatal(e)
				}
				if actual.Len() != reference.Len() {
					b.Fatal("invalid image length")
				}
				var different int
				for i, value := range reference.Bytes() {
					if value != actual.Bytes()[i] {
						different++
					}
				}
				if mode != "gpu" && different != 0 {
					b.Fatal("CPU images differ")
				}
				invoke := func() error {
					return run(program, mode == "fallback", mode == "gpu", size[0], size[1], 32, io.Discard, io.Discard)
				}
				var plugin *gpu.Plugin
				var compile, first time.Duration
				if reuse {
					invoke, plugin, compile = reusedCommand(b, mode, size)
					start := time.Now()
					if e := invoke(); e != nil {
						b.Fatal(e)
					}
					first = time.Since(start)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if e := invoke(); e != nil {
						b.Fatal(e)
					}
				}
				b.StopTimer()
				b.ReportMetric(float64(different), "different-pixels")
				if reuse {
					b.ReportMetric(float64(compile)/1e6, "compile-ms")
					b.ReportMetric(float64(first)/1e6, "first-render-ms")
				}
				if plugin != nil {
					s := plugin.BufferSnapshot()
					b.ReportMetric(float64(s.PeakRuntimeBufferBytes)/(1<<20), "tracked-peak-MiB")
					// One untimed first command precedes the measured commands.
					commands := float64(b.N + 1)
					b.ReportMetric(float64(s.Totals.GPUUploadBytes)/commands/(1<<20), "upload-MiB/op")
					b.ReportMetric(float64(s.Totals.GPUDownloadBytes)/commands/(1<<20), "readback-MiB/op")
					b.ReportMetric(float64(s.Totals.DeviceCopyBytes)/commands/(1<<20), "seed-copy-MiB/op")
					b.ReportMetric(float64(plugin.Snapshot().DeviceInit)/1e6, "device-init-ms")
					for _, k := range s.Kernels {
						b.ReportMetric(float64(k.Translate)/1e6, "translate-ms")
						b.ReportMetric(float64(k.Pipeline)/1e6, "pipeline-ms")
					}
				}
			})
		}
	}
}

func reusedCommand(b *testing.B, mode string, size [2]uint32) (func() error, *gpu.Plugin, time.Duration) {
	b.Helper()
	rt := wago.NewRuntime()
	b.Cleanup(func() {
		if e := rt.CloseContext(context.Background()); e != nil {
			b.Error(e)
		}
	})
	file := "cpu.wasm"
	var plugin *gpu.Plugin
	if mode != "cpu" {
		file = "gpu.wasm"
		var e error
		plugin, e = gpu.New(gpu.Config{Disabled: mode == "fallback", Kernels: []gpu.KernelConfig{kernel()}})
		if e != nil {
			b.Fatal(e)
		}
		set, e := plugin.PluginSet()
		if e != nil {
			b.Fatal(e)
		}
		if e = rt.LoadPlugins(context.Background(), set); e != nil {
			b.Fatal(e)
		}
	}
	source, e := programs.ReadFile(file)
	if e != nil {
		b.Fatal(e)
	}
	start := time.Now()
	module, e := rt.Compile(source)
	compile := time.Since(start)
	if e != nil {
		b.Fatal(e)
	}
	b.Cleanup(func() {
		if e := module.Close(); e != nil {
			b.Error(e)
		}
	})
	return func() (result error) {
		imports := p1.Imports(p1.Config{Args: []string{"mandelbrot", strconv.Itoa(int(size[0])), strconv.Itoa(int(size[1])), "32"}, Stdin: strings.NewReader(""), Stdout: io.Discard, Stderr: io.Discard})
		instance, e := rt.Instantiate(context.Background(), module, wago.WithImports(imports))
		if e != nil {
			return e
		}
		defer func() { result = errors.Join(result, instance.Close()) }()
		if _, e = instance.Invoke("_start"); e != nil {
			var exit *wago.ExitError
			if !errors.As(e, &exit) || exit.Code != 0 {
				return e
			}
		}
		if mode != "cpu" {
			export := "cpu_passes"
			if mode == "gpu" {
				export = "gpu_passes"
			}
			passes, e := instance.Invoke(export)
			if e != nil {
				return e
			}
			if len(passes) != 1 || passes[0] != 32 {
				return fmt.Errorf("required %s passes did not run: %v", mode, passes)
			}
			s := plugin.BufferSnapshot()
			if len(s.Instances) != 1 || s.Instances[0].Buffers != 0 {
				return fmt.Errorf("guest did not free buffers")
			}
		}
		return nil
	}, plugin, compile
}
