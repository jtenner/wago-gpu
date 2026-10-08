package wagogpu

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/jtenner/wago-gpu/internal/fixtures"
	wago "github.com/wago-org/wago"
)

// These authored reference shaders measure the backend. They do not extend the
// restricted Wasm compiler. CPU references execute actual Wasm in native Wago.
const commonBindings = `
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
`

func matrixShader(n uint32, tiled bool) string {
	if !tiled {
		return commonBindings + fmt.Sprintf(`
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 let i = id.x; if (i >= params.count) { return; }
 let row = i / %[1]du; let col = i %% %[1]du;
 var sum = 0.0;
 for (var k = 0u; k < %[1]du; k++) { sum += a[row * %[1]du + k] * b[k * %[1]du + col]; }
 c[i] = sum;
}`, n)
	}
	// All tile invocations reach both barriers. The host admits only complete
	// 16x16 tiles; the uniform guard excludes whole workgroups, never lanes.
	return commonBindings + fmt.Sprintf(`
var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 if (group.x >= params.count / 256u) { return; }
 let r = lane / 16u; let s = lane %% 16u;
 let row = (group.x / %[2]du) * 16u + r;
 let col = (group.x %% %[2]du) * 16u + s;
 var sum = 0.0;
 for (var block = 0u; block < %[2]du; block++) {
  tileA[lane] = a[row * %[1]du + block * 16u + s];
  tileB[lane] = b[(block * 16u + r) * %[1]du + col];
  workgroupBarrier();
  for (var k = 0u; k < 16u; k++) { sum += tileA[r * 16u + k] * tileB[k * 16u + s]; }
  workgroupBarrier();
 }
 c[row * %[1]du + col] = sum;
}`, n, n/16)
}

const saxpyShader = commonBindings + `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 if (id.x < params.count) { c[id.x] = 2.0 * a[id.x] + b[id.x]; }
}`

const dotShader = commonBindings + `
var<workgroup> sums: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 var value = 0.0;
 if (id.x < params.count) { value = a[id.x] * b[id.x]; }
 sums[lane] = value;
 workgroupBarrier();
 for (var stride = 128u; stride > 0u; stride /= 2u) {
  if (lane < stride) { sums[lane] += sums[lane + stride]; }
  workgroupBarrier();
 }
 if (lane == 0u) { c[group.x] = sums[0]; }
}`

func commonInputs(count uint32) ([]byte, []byte) {
	a, b := make([]byte, int(count)*4), make([]byte, int(count)*4)
	for i := uint32(0); i < count; i++ {
		binary.LittleEndian.PutUint32(a[i*4:], math.Float32bits(float32(int32(i%31)-15)/32))
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(int32((i*7+5)%29)-14)/32))
	}
	return a, b
}

func commonCPU(t testing.TB, count, regions uint32, a, b []byte) *wago.Instance {
	t.Helper()
	rt := wago.NewRuntime()
	t.Cleanup(func() {
		if e := rt.CloseContext(context.Background()); e != nil {
			t.Error(e)
		}
	})
	_, in := instance(t, rt, fixtures.Common)
	pages := (uint64(count)*16 + 65535) / 65536
	if pages > 1 {
		v, e := in.Invoke("reserve", pages-1)
		if e != nil || len(v) != 1 || int32(v[0]) < 0 {
			t.Fatal("reserve", v, e)
		}
	}
	if !in.Write(0, a) || !in.Write(count*4, b) {
		t.Fatal("input bounds")
	}
	return in
}
func commonInvoke(t testing.TB, in *wago.Instance, name string, n uint32) []uint64 {
	t.Helper()
	v, e := in.Invoke(name, uint64(n))
	if e != nil {
		t.Fatal(e)
	}
	return v
}
func commonResult(t testing.TB, in *wago.Instance, count uint32) []byte {
	t.Helper()
	v, ok := in.Read(count*8, count*4)
	if !ok {
		t.Fatal("output bounds")
	}
	return v
}
func closeFloat(a, b float32) bool {
	return !math.IsNaN(float64(a)) && !math.IsNaN(float64(b)) && !math.IsInf(float64(a), 0) && !math.IsInf(float64(b), 0) &&
		math.Abs(float64(a)-float64(b)) <= 8e-6*math.Max(1, math.Abs(float64(a)))
}
func compareCommon(t testing.TB, got, want []byte) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("result length")
	}
	for i := 0; i < len(got); i += 4 {
		a, b := math.Float32frombits(binary.LittleEndian.Uint32(got[i:])), math.Float32frombits(binary.LittleEndian.Uint32(want[i:]))
		if !closeFloat(a, b) {
			t.Fatalf("element %d: got %g want %g", i/4, a, b)
		}
	}
}
func sumPartials(data []byte) float32 {
	// The final CPU reduction is part of the end-to-end dot product, not of the
	// device-resident partial reduction. A bounded sum uses no scratch allocation.
	var sum float32
	for i := 0; i < len(data); i += 4 {
		sum += math.Float32frombits(binary.LittleEndian.Uint32(data[i:]))
	}
	return sum
}

func TestCommonCPUReferences(t *testing.T) {
	for _, n := range []uint32{1, 3, 16, 31} {
		t.Run(fmt.Sprintf("matmul/%d", n), func(t *testing.T) {
			count := n * n
			a, b := commonInputs(count)
			in := commonCPU(t, count, 4, a, b)
			commonInvoke(t, in, "matmul", n)
			got := commonResult(t, in, count)
			want := make([]byte, len(got))
			for r := uint32(0); r < n; r++ {
				for col := uint32(0); col < n; col++ {
					var sum float64
					for k := uint32(0); k < n; k++ {
						sum += float64(math.Float32frombits(binary.LittleEndian.Uint32(a[(r*n+k)*4:]))) * float64(math.Float32frombits(binary.LittleEndian.Uint32(b[(k*n+col)*4:])))
					}
					binary.LittleEndian.PutUint32(want[(r*n+col)*4:], math.Float32bits(float32(sum)))
				}
			}
			compareCommon(t, got, want)
			commonInvoke(t, in, "transpose", n)
			commonInvoke(t, in, "matmul_t", n)
			compareCommon(t, commonResult(t, in, count), want)
		})
	}
	for _, n := range []uint32{1, 255, 256, 257, 1024} {
		a, b := commonInputs(n)
		in := commonCPU(t, n, 3, a, b)
		commonInvoke(t, in, "saxpy", n)
		want := make([]byte, len(a))
		for i := uint32(0); i < n; i++ {
			value := 2*math.Float32frombits(binary.LittleEndian.Uint32(a[i*4:])) + math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
			binary.LittleEndian.PutUint32(want[i*4:], math.Float32bits(value))
		}
		compareCommon(t, commonResult(t, in, n), want)
	}

	for _, n := range []uint32{1, 255, 256, 257, 1024} {
		a, b := commonInputs(n)
		in := commonCPU(t, n, 2, a, b)
		var want float64
		for i := uint32(0); i < n; i++ {
			want += float64(math.Float32frombits(binary.LittleEndian.Uint32(a[i*4:]))) * float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:])))
		}
		got := math.Float32frombits(uint32(commonInvoke(t, in, "dot", n)[0]))
		if !closeFloat(got, float32(want)) {
			t.Fatal(n, got, want)
		}
	}
}

type referenceGPU struct {
	device         bufferDevice
	pipeline       bufferPipeline
	uniform        deviceBuffer
	resources      [3]deviceBuffer
	staging        deviceBuffer
	count          uint32
	parameterDirty bool
	output         []byte
}

func newReferenceGPU(t testing.TB, shader string, count, outputCount uint32, a, b []byte) *referenceGPU {
	t.Helper()
	backend, e := openBackend()
	if e != nil {
		t.Fatal("real hardware required:", e)
	}
	t.Log(backend.Info())
	t.Cleanup(backend.Close)
	d, ok := backend.(bufferDevice)
	if !ok {
		t.Fatal("buffer backend unavailable")
	}
	g := &referenceGPU{device: d, count: count, parameterDirty: true, output: make([]byte, int(outputCount)*4)}
	g.pipeline, e = d.BuildBuffer(shader)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(g.pipeline.Close)
	g.uniform, e = d.AllocateUniform(16)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(g.uniform.Close)
	for i, size := range []uint64{uint64(len(a)), uint64(len(b)), uint64(len(g.output))} {
		g.resources[i], e = d.AllocateBuffer(size)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(g.resources[i].Close)
	}
	g.staging, e = d.AllocateReadback(uint64(len(g.output)))
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(g.staging.Close)
	if _, e = g.upload(a, b); e != nil {
		t.Fatal(e)
	}
	return g
}
func (g *referenceGPU) upload(a, b []byte) (time.Duration, error) {
	start := time.Now()
	if e := g.device.UploadBuffer(context.Background(), g.resources[0], a); e != nil {
		return 0, e
	}
	if e := g.device.UploadBuffer(context.Background(), g.resources[1], b); e != nil {
		return 0, e
	}
	return time.Since(start), nil
}
func (g *referenceGPU) execute() (time.Duration, error) {
	start := time.Now()
	op := BufferOperation{}
	e := g.device.ExecuteBuffers(context.Background(), g.pipeline, g.uniform, g.resources[:], nil, g.count, g.parameterDirty, &op)
	if e == nil {
		g.parameterDirty = false
	}
	return time.Since(start), e
}
func (g *referenceGPU) download() (time.Duration, error) {
	start := time.Now()
	e := g.device.ReadBuffer(context.Background(), g.resources[2], g.staging, uint64(len(g.output)), func(data []byte) { copy(g.output, data) })
	return time.Since(start), e
}

func TestCommonReferenceHardware(t *testing.T) {
	if os.Getenv("WAGO_GPU_TEST") != "1" {
		t.Skip("set WAGO_GPU_TEST=1 for real GPU tests")
	}
	for _, n := range []uint32{1, 17, 32, 64} {
		for _, tiled := range []bool{false, true} {
			if tiled && n%16 != 0 {
				continue
			}
			t.Run(fmt.Sprintf("matmul/%d/tiled=%v", n, tiled), func(t *testing.T) {
				count := n * n
				a, b := commonInputs(count)
				in := commonCPU(t, count, 4, a, b)
				commonInvoke(t, in, "matmul", n)
				want := commonResult(t, in, count)
				g := newReferenceGPU(t, matrixShader(n, tiled), count, count, a, b)
				for pass := 0; pass < 3; pass++ {
					// Change input while reusing all resources and the cached bind group.
					if pass > 0 {
						binary.LittleEndian.PutUint32(a[0:], math.Float32bits(float32(pass)))
						in.Write(0, a)
						commonInvoke(t, in, "matmul", n)
						want = commonResult(t, in, count)
						if _, e := g.upload(a, b); e != nil {
							t.Fatal(e)
						}
					}
					if _, e := g.execute(); e != nil {
						t.Fatal(e)
					}
					if _, e := g.download(); e != nil {
						t.Fatal(e)
					}
					compareCommon(t, g.output, want)
				}
			})
		}
	}
	for _, n := range []uint32{1, 255, 256, 257, 16384} {
		t.Run(fmt.Sprintf("saxpy/%d", n), func(t *testing.T) {
			a, b := commonInputs(n)
			in := commonCPU(t, n, 3, a, b)
			commonInvoke(t, in, "saxpy", n)
			want := commonResult(t, in, n)
			g := newReferenceGPU(t, saxpyShader, n, n, a, b)
			if _, e := g.execute(); e != nil {
				t.Fatal(e)
			}
			if _, e := g.download(); e != nil {
				t.Fatal(e)
			}
			compareCommon(t, g.output, want)
		})
	}

	for _, n := range []uint32{1, 255, 256, 257, 16384} {
		t.Run(fmt.Sprintf("dot/%d", n), func(t *testing.T) {
			a, b := commonInputs(n)
			in := commonCPU(t, n, 2, a, b)
			want := math.Float32frombits(uint32(commonInvoke(t, in, "dot", n)[0]))
			g := newReferenceGPU(t, dotShader, n, (n+255)/256, a, b)
			if _, e := g.execute(); e != nil {
				t.Fatal(e)
			}
			if _, e := g.download(); e != nil {
				t.Fatal(e)
			}
			if got := sumPartials(g.output); !closeFloat(got, want) {
				t.Fatal(got, want)
			}
		})
	}
}

func BenchmarkCommon(b *testing.B) {
	for _, n := range []uint32{64, 128, 256, 512} {
		for _, method := range []string{"cpu-naive", "cpu-transposed", "cpu-transpose-and-matmul", "gpu-naive-resident", "gpu-naive-end-to-end", "gpu-tiled-resident", "gpu-tiled-end-to-end"} {
			b.Run(fmt.Sprintf("matmul/%d/%s", n, method), func(b *testing.B) { benchmarkCommonCase(b, n, "matmul", method) })
		}
	}
	for _, n := range []uint32{1024, 16384, 1_000_000, 10_000_000} {
		for _, method := range []string{"cpu", "gpu-resident-partials", "gpu-end-to-end"} {
			b.Run(fmt.Sprintf("dot/%d/%s", n, method), func(b *testing.B) { benchmarkCommonCase(b, n, "dot", method) })
		}
	}
	for _, n := range []uint32{1024, 16384, 1_000_000, 10_000_000} {
		for _, method := range []string{"cpu", "gpu-resident", "gpu-end-to-end"} {
			b.Run(fmt.Sprintf("saxpy/%d/%s", n, method), func(b *testing.B) { benchmarkCommonCase(b, n, "saxpy", method) })
		}
	}
}
func benchmarkCommonCase(b *testing.B, n uint32, workload, method string) {
	dot := workload == "dot"
	matrix := workload == "matmul"
	gpu := method[0:3] == "gpu"
	if gpu && os.Getenv("WAGO_GPU_TEST") != "1" {
		b.Skip("set WAGO_GPU_TEST=1 to require real hardware")
	}
	count := n
	if matrix {
		count = n * n
	}
	a, inputB := commonInputs(count)
	regions := uint32(4)
	if dot {
		regions = 2
	} else if !matrix {
		regions = 3
	}
	in := commonCPU(b, count, regions, a, inputB)
	var want []byte
	var dotWant float32
	if dot {
		dotWant = math.Float32frombits(uint32(commonInvoke(b, in, "dot", n)[0]))
	} else {
		commonInvoke(b, in, workload, n)
		want = commonResult(b, in, count)
		if matrix {
			commonInvoke(b, in, "transpose", n)
		}
	}
	var g *referenceGPU
	var first time.Duration
	if gpu {
		shader := dotShader
		outputCount := (count + 255) / 256
		if matrix {
			shader = matrixShader(n, method == "gpu-tiled-resident" || method == "gpu-tiled-end-to-end")
			outputCount = count
		}
		if workload == "saxpy" {
			shader = saxpyShader
			outputCount = count
		}
		start := time.Now()
		g = newReferenceGPU(b, shader, count, outputCount, a, inputB)
		if _, e := g.execute(); e != nil {
			b.Fatal(e)
		}
		if _, e := g.download(); e != nil {
			b.Fatal(e)
		}
		first = time.Since(start)
		if dot {
			if !closeFloat(sumPartials(g.output), dotWant) {
				b.Fatal("dot mismatch")
			}
		} else {
			compareCommon(b, g.output, want)
		}
	}
	resident := method == "gpu-naive-resident" || method == "gpu-tiled-resident" || method == "gpu-resident-partials" || method == "gpu-resident"
	var upload, compute, download, finish time.Duration
	b.ReportAllocs()
	b.ResetTimer()
	for trial := 0; trial < b.N; trial++ {
		if !gpu {
			name := workload
			switch method {
			case "cpu-transposed":
				name = "matmul_t"
			case "cpu-transpose-and-matmul":
				if _, e := in.Invoke("transpose", uint64(n)); e != nil {
					b.Fatal(e)
				}
				name = "matmul_t"
			}
			if _, e := in.Invoke(name, uint64(n)); e != nil {
				b.Fatal(e)
			}
		} else {
			var d time.Duration
			var e error
			if !resident {
				d, e = g.upload(a, inputB)
				upload += d
				if e != nil {
					b.Fatal(e)
				}
			}
			d, e = g.execute()
			compute += d
			if e != nil {
				b.Fatal(e)
			}
			if !resident {
				d, e = g.download()
				download += d
				if e != nil {
					b.Fatal(e)
				}
				var result float32
				if dot {
					start := time.Now()
					result = sumPartials(g.output)
					finish += time.Since(start)
				}
				b.StopTimer()
				if dot {
					if !closeFloat(result, dotWant) {
						b.Fatal(result, dotWant)
					}
				} else {
					compareCommon(b, g.output, want)
				}
				b.StartTimer()
			}
		}
	}
	b.StopTimer()
	if gpu && resident {
		if _, e := g.download(); e != nil {
			b.Fatal(e)
		}
		if dot {
			if !closeFloat(sumPartials(g.output), dotWant) {
				b.Fatal("resident mismatch")
			}
		} else {
			compareCommon(b, g.output, want)
		}
	}
	if !gpu && !dot {
		compareCommon(b, commonResult(b, in, count), want)
	}
	if !gpu && dot {
		if !closeFloat(math.Float32frombits(uint32(commonInvoke(b, in, "dot", n)[0])), dotWant) {
			b.Fatal("CPU dot mismatch")
		}
	}
	if gpu {
		b.ReportMetric(float64(first.Nanoseconds())/1e6, "first-ms")
		b.ReportMetric(float64(upload.Nanoseconds())/float64(b.N)/1000, "upload-us/op")
		b.ReportMetric(float64(compute.Nanoseconds())/float64(b.N)/1000, "compute-us/op")
		b.ReportMetric(float64(download.Nanoseconds())/float64(b.N)/1000, "download-us/op")
		b.ReportMetric(float64(finish.Nanoseconds())/float64(b.N)/1000, "final-sum-us/op")
		b.ReportMetric(float64(16+len(a)+len(inputB)+len(g.output)*2), "device-buffer-bytes")
	}
	flops := float64(2) * float64(n) * float64(n) * float64(n)
	if !matrix {
		flops = float64(2) * float64(n)
	}
	b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}
