package wagogpu

import (
	"os"
	"testing"

	"github.com/jtenner/wago-gpu/internal/fixtures"
	wago "github.com/wago-org/wago"
)

type compileCase struct {
	name   string
	source []byte
	config Config
}

func compilationCases(b *testing.B) []compileCase {
	b.Helper()
	slots := []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessWrite}}
	config := Config{Disabled: true}
	for i, name := range []string{"twice", "square", "copy", "polynomial"} {
		config.Kernels = append(config.Kernels, KernelConfig{ID: uint32(i + 1), Export: "wago_gpu.kernel." + name,
			CPUExport: "wago_gpu.cpu." + name, Bindings: slots, RelaxedFloat: true})
	}
	mandelbrot, err := os.ReadFile("examples/mandelbrot/gpu.wasm")
	if err != nil {
		b.Fatal(err)
	}
	mandelConfig := Config{Disabled: true, Kernels: []KernelConfig{{ID: 1,
		Export: "wago_gpu.kernel.step", CPUExport: "wago_gpu.cpu.step", RelaxedFloat: true,
		Bindings: []BindingConfig{{0, TypeF32, AccessRead}, {1, TypeF32, AccessRead},
			{2, TypeF32, AccessReadWrite}, {3, TypeF32, AccessReadWrite}}}}}
	large := append([]byte(nil), fixtures.BufferWork...)
	large = append(large, 0)
	for n := uint32((2 << 20) + 1); ; n >>= 7 {
		v := byte(n & 127)
		if n < 128 {
			large = append(large, v)
			break
		}
		large = append(large, v|128)
	}
	large = append(large, make([]byte, (2<<20)+1)...)
	return []compileCase{{"FourKernels", fixtures.BufferWork, config},
		{"Polynomial", fixtures.BufferWork, Config{Disabled: true, Kernels: config.Kernels[3:]}},
		{"Mandelbrot", mandelbrot, mandelConfig}, {"LargeCustom", large, config}}
}

// GPU/device setup is excluded. PrepareSource measures Wago's preparation plus
// our source hook; it uses fresh preparation identities and bounded eviction.
func BenchmarkCompilePath(b *testing.B) {
	for _, tc := range compilationCases(b) {
		b.Run(tc.name, func(b *testing.B) {
			b.Run("Decode", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := decodeBufferModule(tc.source); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("StandaloneKernel", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := CompileBufferWGSL(tc.source, tc.config.Kernels[0]); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("PrepareSource", func(b *testing.B) {
				rt, _ := setup(b, tc.config, nil)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					prepared, err := rt.PrepareCompile(tc.source)
					if err != nil {
						b.Fatal(err)
					}
					if err = prepared.Close(); err != nil {
						b.Fatal(err)
					}
				}
			})
			for _, plugin := range []bool{false, true} {
				name := "NativeOnly"
				if plugin {
					name = "NativeWithPlugin"
				}
				b.Run(name, func(b *testing.B) {
					var rt *wago.Runtime
					if plugin {
						rt, _ = setup(b, tc.config, nil)
					} else {
						rt = wago.NewRuntime()
						b.Cleanup(func() {
							if e := rt.Close(); e != nil {
								b.Error(e)
							}
						})
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						module, err := rt.Compile(tc.source)
						if err != nil {
							b.Fatal(err)
						}
						if err = module.Close(); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
	b.Run("LegacyHeavy", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := CompileWGSL(fixtures.Heavy, "kernel"); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkShaderBuild(b *testing.B) {
	if os.Getenv("WAGO_GPU_TEST") != "1" {
		b.Skip("requires real GPU with -tags webgpu")
	}
	for _, tc := range compilationCases(b)[:3] {
		b.Run(tc.name, func(b *testing.B) {
			cfg := tc.config
			cfg.Disabled = false
			_, p := setup(b, cfg, nil)
			d := p.bufferDevice()
			if d == nil {
				b.Fatalf("hardware required: %s", p.Snapshot().Reason)
			}
			shader, err := CompileBufferWGSL(tc.source, cfg.Kernels[0])
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				pipeline, err := d.BuildBuffer(shader)
				if err != nil {
					b.Fatal(err)
				}
				pipeline.Close()
			}
			b.StopTimer()
			b.ReportMetric(float64(len(shader)), "WGSL-B")
		})
	}
}
