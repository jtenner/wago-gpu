package wagogpu

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	wago "github.com/wago-org/wago"
	"github.com/wago-org/wasi/p1"
)

func TestMandelbrotBulkFallbackAfterGPU(t *testing.T) {
	gpuSource, e := os.ReadFile("examples/mandelbrot/gpu.wasm")
	if e != nil {
		t.Fatal(e)
	}
	cpuSource, e := os.ReadFile("examples/mandelbrot/cpu.wasm")
	if e != nil {
		t.Fatal(e)
	}
	config := Config{Kernels: []KernelConfig{{ID: 1, Export: "wago_gpu.kernel.step", CPUExport: "wago_gpu.cpu.step", RelaxedFloat: true, Bindings: []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessRead}, {2, TypeF32, AccessReadWrite}, {3, TypeF32, AccessReadWrite}}}}}
	d := &fakeBufferDevice{mode: "mandelbrot"}
	rt, plugin := setup(t, config, func() (backend, error) { return d, nil })
	render := func(source []byte) (*wago.Instance, []byte) {
		t.Helper()
		module, e := rt.Compile(source)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() {
			if e := module.Close(); e != nil {
				t.Error(e)
			}
		})
		var image bytes.Buffer
		in, e := rt.Instantiate(context.Background(), module, wago.WithImports(p1.Imports(p1.Config{Args: []string{"mandelbrot", "19", "17", "12"}, Stdin: strings.NewReader(""), Stdout: &image, Stderr: io.Discard})))
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() {
			if e := in.Close(); e != nil {
				t.Error(e)
			}
		})
		if _, e := in.Invoke("_start"); e != nil {
			var exit *wago.ExitError
			if !errors.As(e, &exit) || exit.Code != 0 {
				t.Fatal(e)
			}
		}
		return in, image.Bytes()
	}
	in, actual := render(gpuSource)
	_, reference := render(cpuSource)
	if !bytes.Equal(actual, reference) {
		t.Fatal("bulk fallback did not use the last committed GPU orbit")
	}
	values, e := in.Invoke("gpu_passes")
	if e != nil || len(values) != 1 || values[0] != 1 {
		t.Fatal(values, e)
	}
	values, e = in.Invoke("cpu_passes")
	if e != nil || len(values) != 1 || values[0] != 11 {
		t.Fatal(values, e)
	}
	if e := rt.CloseContext(context.Background()); e != nil {
		t.Fatal(e)
	}
	if d.buffers != 0 || plugin.BufferSnapshot().RuntimeBufferBytes != 0 {
		t.Fatal("mixed fallback leaked resources")
	}
}
