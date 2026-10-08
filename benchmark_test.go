package wagogpu

import (
	"fmt"
	"os"
	"testing"

	"wago-gpu/internal/fixtures"
)

// BenchmarkCPU always uses Wago's native loop and needs no GPU.
func BenchmarkCPU(b *testing.B) { benchmarkExecution(b, false, 1) }

// BenchmarkGPU uses real hardware or skips. Set WAGO_GPU_TEST=1 with -tags
// webgpu to make hardware absence a test failure (see hardware benchmark).
func BenchmarkGPU(b *testing.B)      { benchmarkExecution(b, true, 1) }
func BenchmarkBatchCPU(b *testing.B) { benchmarkExecution(b, false, 8) }
func BenchmarkBatchGPU(b *testing.B) { benchmarkExecution(b, true, 8) }
func benchmarkExecution(b *testing.B, gpu bool, passes uint32) {
	for _, kernel := range []struct {
		name   string
		source []byte
	}{{"twice", fixtures.Twice}, {"square", fixtures.Square}, {"heavy", fixtures.Heavy}} {
		if kernel.name == "square" && passes > 4 {
			continue
		}
		for _, n := range []uint32{1024, 16384, 1_000_000, 10_000_000} {
			b.Run(fmt.Sprintf("%s/%d", kernel.name, n), func(b *testing.B) {
				cfg := config()
				cfg.Disabled = !gpu
				rt, p := setup(b, cfg, nil)
				_, i := instance(b, rt, kernel.source)
				if e := fixtures.Prepare(i, n); e != nil {
					b.Fatal(e)
				}
				cpuExport, gpuExport := "cpu", "gpu"
				args := []uint64{0, uint64(n) * 4, uint64(n)}
				if passes > 1 {
					cpuExport = "cpu_batch"
					gpuExport = "gpu_batch"
					args = append(args, uint64(passes))
				}
				invoke(b, i, cpuExport, args...)
				export := cpuExport
				output := uint64(n) * 4
				if gpu {
					export = gpuExport
					output = uint64(n) * 8
					args[1] = output
					if got := invoke(b, i, export, args...); got != Success {
						if os.Getenv("WAGO_GPU_TEST") == "1" {
							b.Fatalf("hardware GPU required: %s", p.Snapshot().Reason)
						}
						b.Skipf("GPU unavailable: %s", p.Snapshot().Reason)
					}
					if _, e := fixtures.VerifyKernel(i, n, kernel.name, passes); e != nil {
						b.Fatal(e)
					}
				}
				b.ReportAllocs()
				b.SetBytes(int64(n) * 4)
				b.ResetTimer()
				for j := 0; j < b.N; j++ {
					v, e := i.Invoke(export, args...)
					if e != nil {
						b.Fatal(e)
					}
					if gpu && (len(v) != 1 || int32(v[0]) != Success) {
						b.Fatal("GPU failed")
					}
				}
				b.StopTimer()
				if gpu {
					if _, e := fixtures.VerifyKernel(i, n, kernel.name, passes); e != nil {
						b.Fatal(e)
					}
				}
			})
		}
	}
}
