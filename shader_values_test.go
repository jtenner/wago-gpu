package wagogpu

import (
	"strings"
	"testing"

	"github.com/jtenner/wago-gpu/internal/fixtures"
)

func TestCompilerScratchReuse(t *testing.T) {
	m, err := decodeBufferModule(fixtures.BufferWork)
	if err != nil {
		t.Fatal(err)
	}
	var scratch lowerScratch
	var saved string
	slots := []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessWrite}}
	for repeat := 0; repeat < 3; repeat++ {
		for _, name := range []string{"polynomial", "twice", "square", "copy"} {
			k := KernelConfig{ID: 1, Export: "wago_gpu.kernel." + name,
				CPUExport: "wago_gpu.cpu." + name, Bindings: slots, RelaxedFloat: true}
			body, err := m.kernelBody(k)
			if err != nil {
				t.Fatal(err)
			}
			out, err := lowerBufferBodyScratch(m, body, k, &scratch)
			if err != nil {
				t.Fatal(err)
			}
			want, err := CompileBufferWGSL(fixtures.BufferWork, k)
			if err != nil || out.shader != want {
				t.Fatalf("%s: reused workspace differs: %v", name, err)
			}
			if repeat == 0 && name == "polynomial" {
				saved = out.shader
			}
		}
	}
	k := KernelConfig{ID: 1, Export: "wago_gpu.kernel.polynomial", CPUExport: "wago_gpu.cpu.polynomial", Bindings: slots, RelaxedFloat: true}
	want, err := CompileBufferWGSL(fixtures.BufferWork, k)
	if err != nil || saved != want {
		t.Fatal("reset changed an earlier shader", err)
	}
}

func TestCompilerSharedValuesStayLinear(t *testing.T) {
	// A local used twice at every step forms an exponentially large expression
	// tree if its text is copied. The compiler must emit each value only once.
	const steps = 256
	body := `(local $x f32)
(local.set $x (call $read (call $get (i32.const 0)) (local.get $i)))` +
		strings.Repeat(`
(local.set $x (f32.add (local.get $x) (local.get $x)))`, steps) + `
(call $write (call $get (i32.const 1)) (local.get $i) (local.get $x))`
	shader, err := CompileBufferWGSL(wat(t, testModule(body)), bufferConfig().Kernels[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(shader, " + ") != steps || len(shader) > 16384 {
		t.Fatalf("nonlinear emission: %d additions, %d bytes", strings.Count(shader, " + "), len(shader))
	}
}
