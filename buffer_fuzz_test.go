package wagogpu

import (
	"testing"
	"wago-gpu/internal/fixtures"
)

func FuzzBufferCompiler(f *testing.F) {
	f.Add(fixtures.BufferWork)
	f.Add([]byte{0, 97, 115, 109, 1, 0, 0, 0, 1, 255})
	k := KernelConfig{ID: 1, Export: "wago_gpu.kernel.twice", CPUExport: "wago_gpu.cpu.twice", Bindings: []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessWrite}}}
	f.Fuzz(func(t *testing.T, source []byte) {
		shader, e := CompileBufferWGSL(source, k)
		if e == nil && len(shader) > 512<<10 {
			t.Fatal("shader limit")
		}
		if e != nil {
			if _, ok := e.(*CompileError); !ok {
				t.Fatalf("untyped compiler error: %T", e)
			}
		}
	})
}
