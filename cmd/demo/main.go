// Command demo checks the Wasm kernels and measures complete guest calls.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"time"

	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/internal/fixtures"
	wago "github.com/wago-org/wago"
)

type measurement struct {
	MeanNS, MedianNS, MinNS, MaxNS int64
	BytesPerCall, AllocsPerCall    float64
	HeapAllocAfter, HeapSysAfter   uint64
	RSSAfterKiB                    uint64
}
type result struct {
	Kernel                                                                             string
	Elements                                                                           uint32
	Passes                                                                             uint32
	MinElements                                                                        uint32
	MaxAbsoluteError                                                                   float64
	SeparateGPU                                                                        *measurement `json:",omitempty"`
	Device, Mode, Reason                                                               string
	Verified                                                                           bool
	DeviceInitNS, NativeCompileNS, PluginCompileNS, WGSLTranslateNS, PipelineCompileNS int64
	CompileGoBytes, CompileGoAllocs                                                    uint64
	FirstCPU, FirstGPU, CPU, GPU                                                       measurement
	// These host stage times require -profile and include API and queue waits.
	Stages                     gpu.Timing
	GuestBytes, GPUBufferBytes uint64
}
type report struct {
	Time, Go, OS, Arch, Wago, WebGPU string
	GOMAXPROCS, Repetitions          int
	ProfileStages                    bool
	Cases                            []result
}

func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func run() error {
	bench := flag.Bool("bench", false, "measure all four required sizes")
	n := flag.Uint("n", 16384, "array element count")
	reps := flag.Int("reps", 20, "repeated calls per case")
	disabled := flag.Bool("cpu-only", false, "force the guest CPU fallback")
	require := flag.Bool("require-gpu", false, "fail if any call uses CPU fallback")
	profile := flag.Bool("profile", false, "add queue waits for separate stage times")
	asJSON := flag.Bool("json", false, "write JSON; all duration fields are nanoseconds")
	kernelFlag := flag.String("kernel", "all", "all, twice, square, or heavy")
	passes := flag.Uint("passes", 1, "kernel passes per batch, 1..64 (square demo: at most 4)")
	minimum := flag.Uint("min-elements", 0, "use CPU below this element count; zero always attempts GPU")
	separate := flag.Bool("separate", false, "also measure one upload/download per pass; requires a GPU")
	sizesFlag := flag.String("sizes", "", "comma-separated sizes for a crossover scan")
	flag.Parse()
	if *passes < 1 || *passes > uint(gpu.MaxBatchPasses) || *minimum > uint(gpu.DefaultMaxElements) || *reps < 1 || *n < 1 || *n > uint(gpu.DefaultMaxElements) {
		return fmt.Errorf("invalid repetitions or array size")
	}
	sizes := []uint32{uint32(*n)}
	if *bench {
		sizes = []uint32{1024, 16384, 1_000_000, 10_000_000}
	}
	if *sizesFlag != "" {
		sizes = nil
		for _, s := range strings.Split(*sizesFlag, ",") {
			v, e := strconv.ParseUint(s, 10, 32)
			if e != nil || v == 0 || v > uint64(gpu.DefaultMaxElements) {
				return fmt.Errorf("invalid size %q", s)
			}
			sizes = append(sizes, uint32(v))
		}
	}
	r := report{Time: time.Now().UTC().Format(time.RFC3339), Go: runtime.Version(), OS: runtime.GOOS, Arch: runtime.GOARCH, GOMAXPROCS: runtime.GOMAXPROCS(0), Repetitions: *reps, ProfileStages: *profile}
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, d := range info.Deps {
			switch d.Path {
			case "github.com/wago-org/wago":
				r.Wago = d.Version
			case "github.com/oliverbestmann/webgpu":
				r.WebGPU = d.Version
			}
		}
	}
	foundKernel := false
	for _, kernel := range []struct {
		name   string
		source []byte
	}{{"twice", fixtures.Twice}, {"square", fixtures.Square}, {"heavy", fixtures.Heavy}} {
		if *kernelFlag != "all" && *kernelFlag != kernel.name {
			continue
		}
		foundKernel = true
		if kernel.name == "square" && *passes > 4 {
			return fmt.Errorf("square demo inputs overflow after more than four passes; select twice or heavy")
		}
		for _, n := range sizes {
			v, e := runCase(kernel.name, kernel.source, n, *reps, *disabled, *require, *profile, uint32(*passes), uint32(*minimum), *separate)
			if e != nil {
				return fmt.Errorf("%s/%d: %w", kernel.name, n, e)
			}
			r.Cases = append(r.Cases, v)
			if !*asJSON {
				fmt.Printf("%s n=%d passes=%d: %s, verified; CPU %.3f ms, GPU path %.3f ms, first GPU path %.3f ms\n", v.Kernel, n, v.Passes, v.Mode, float64(v.CPU.MedianNS)/1e6, float64(v.GPU.MedianNS)/1e6, float64(v.FirstGPU.MedianNS)/1e6)
				if v.Device != "" {
					fmt.Println("  device:", v.Device)
				} else {
					fmt.Println("  fallback:", v.Reason)
				}
				if v.SeparateGPU != nil {
					fmt.Printf("  separate transfers: %.3f ms\n", float64(v.SeparateGPU.MedianNS)/1e6)
				}
				if *profile {
					fmt.Printf("  upload=%s compute=%s download=%s commit=%s\n", v.Stages.Upload, v.Stages.Compute, v.Stages.Download, v.Stages.Commit)
					fmt.Printf("  device compute=%s timestamps=%v upload=%d B download=%d B\n", v.Stages.DeviceCompute, v.Stages.TimestampsValid, v.Stages.UploadBytes, v.Stages.DownloadBytes)
				}
			}
		}
	}
	if !foundKernel {
		return fmt.Errorf("unknown kernel %q", *kernelFlag)
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	return nil
}

func runCase(name string, source []byte, n uint32, reps int, disabled, require, profile bool, passes, minimum uint32, separate bool) (v result, err error) {
	v.Kernel = name
	v.Elements = n
	v.Passes = passes
	v.MinElements = minimum
	v.GuestBytes = ((uint64(n)*12 + 65535) / 65536) * 65536
	start := time.Now()
	native, e := wago.Compile(source)
	v.NativeCompileNS = time.Since(start).Nanoseconds()
	if e != nil {
		return v, e
	}
	if e = native.Close(); e != nil {
		return v, e
	}
	p, e := gpu.New(gpu.Config{KernelExport: "kernel", RelaxedFloat: true, Disabled: disabled, ProfileStages: profile, MinElements: minimum})
	if e != nil {
		return v, e
	}
	set, e := p.PluginSet()
	if e != nil {
		return v, e
	}
	rt := wago.NewRuntime()
	defer func() {
		if e := rt.CloseContext(context.Background()); err == nil {
			err = e
		}
	}()
	if e = rt.LoadPlugins(context.Background(), set); e != nil {
		return v, e
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start = time.Now()
	module, e := rt.Compile(source)
	v.PluginCompileNS = time.Since(start).Nanoseconds()
	runtime.ReadMemStats(&after)
	v.CompileGoBytes = after.TotalAlloc - before.TotalAlloc
	v.CompileGoAllocs = after.Mallocs - before.Mallocs
	if e != nil {
		return v, e
	}
	defer module.Close()
	inst, e := rt.Instantiate(context.Background(), module)
	if e != nil {
		return v, e
	}
	defer inst.Close()
	if e = fixtures.Prepare(inst, n); e != nil {
		return v, e
	}
	s := p.Snapshot()
	v.Device = s.Device
	v.DeviceInitNS = s.DeviceInit.Nanoseconds()
	v.WGSLTranslateNS = s.Build.Translate.Nanoseconds()
	v.PipelineCompileNS = s.Build.Pipeline.Nanoseconds()
	cpu := func() error {
		if passes == 1 {
			_, e := inst.Invoke("cpu", 0, uint64(n)*4, uint64(n))
			return e
		}
		_, e := inst.Invoke("cpu_batch", 0, uint64(n)*4, uint64(n), uint64(passes))
		return e
	}
	mode := ""
	var stageTotal gpu.Timing
	collectStages := false
	accelerated := func() error {
		var out []uint64
		var e error
		if passes == 1 {
			out, e = inst.Invoke("run", 0, uint64(n)*8, uint64(n))
		} else {
			out, e = inst.Invoke("run_batch", 0, uint64(n)*8, uint64(n), uint64(passes))
		}
		if e != nil {
			return e
		}
		status := wago.AsI32(out[0])
		current := "GPU"
		if status == gpu.Fallback {
			current = "CPU fallback"
		} else if status != gpu.Success {
			return fmt.Errorf("GPU import status %d", status)
		}
		if require && status != gpu.Success {
			return fmt.Errorf("real GPU required: %s", p.Snapshot().Reason)
		}
		if mode != "" && mode != current {
			return fmt.Errorf("execution mode changed during measurement")
		}
		mode = current
		if collectStages {
			s := p.Snapshot().Last
			stageTotal.Upload += s.Upload
			stageTotal.Compute += s.Compute
			stageTotal.Download += s.Download
			stageTotal.Commit += s.Commit
			stageTotal.Total += s.Total
			stageTotal.DeviceCompute += s.DeviceCompute
			stageTotal.TimestampsValid = stageTotal.TimestampsValid && s.TimestampsValid
			stageTotal.UploadBytes += s.UploadBytes
			stageTotal.DownloadBytes += s.DownloadBytes
		}
		return nil
	}
	if v.FirstCPU, e = measure(1, cpu); e != nil {
		return v, e
	}
	if v.FirstGPU, e = measure(1, accelerated); e != nil {
		return v, e
	}
	if v.MaxAbsoluteError, e = fixtures.VerifyKernel(inst, n, name, passes); e != nil {
		return v, e
	}
	// Repeated calls reuse the same input, result arrays, pipeline, and buffers.
	if v.CPU, e = measure(reps, cpu); e != nil {
		return v, e
	}
	collectStages = profile
	stageTotal.TimestampsValid = true
	if v.GPU, e = measure(reps, accelerated); e != nil {
		return v, e
	}
	if v.MaxAbsoluteError, e = fixtures.VerifyKernel(inst, n, name, passes); e != nil {
		return v, e
	}
	s = p.Snapshot()
	v.Mode = mode
	v.Reason = s.Reason
	v.Verified = true
	v.GPUBufferBytes = s.BufferBytes
	v.Stages = gpu.Timing{Upload: stageTotal.Upload / time.Duration(reps), Compute: stageTotal.Compute / time.Duration(reps), Download: stageTotal.Download / time.Duration(reps), Commit: stageTotal.Commit / time.Duration(reps), Total: stageTotal.Total / time.Duration(reps), DeviceCompute: stageTotal.DeviceCompute / time.Duration(reps), TimestampsValid: profile && stageTotal.TimestampsValid, Passes: passes, UploadBytes: s.Last.UploadBytes, DownloadBytes: s.Last.DownloadBytes}
	if separate {
		if mode != "GPU" {
			return v, fmt.Errorf("separate-transfer control requires successful GPU execution")
		}
		measurement, e := measure(reps, func() error {
			out, e := inst.Invoke("gpu_separate", 0, uint64(n)*8, uint64(n), uint64(passes))
			if e != nil {
				return e
			}
			if wago.AsI32(out[0]) != gpu.Success {
				return fmt.Errorf("separate pass failed: %s", p.Snapshot().Reason)
			}
			return nil
		})
		if e != nil {
			return v, e
		}
		v.SeparateGPU = &measurement
		if _, e = fixtures.VerifyKernel(inst, n, name, passes); e != nil {
			return v, e
		}
	}
	return v, nil
}

func measure(reps int, fn func() error) (measurement, error) {
	times := make([]int64, reps)
	// Setup, GC, verification, and memory-stat reads are outside timed calls.
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	var sum int64
	for j := range times {
		start := time.Now()
		if e := fn(); e != nil {
			return measurement{}, e
		}
		times[j] = time.Since(start).Nanoseconds()
		sum += times[j]
	}
	runtime.ReadMemStats(&after)
	slices.Sort(times)
	median := times[reps/2]
	if reps%2 == 0 {
		median = (times[reps/2-1] + median) / 2
	}
	return measurement{MeanNS: sum / int64(reps), MedianNS: median, MinNS: times[0], MaxNS: times[reps-1], BytesPerCall: float64(after.TotalAlloc-before.TotalAlloc) / float64(reps), AllocsPerCall: float64(after.Mallocs-before.Mallocs) / float64(reps), HeapAllocAfter: after.HeapAlloc, HeapSysAfter: after.HeapSys, RSSAfterKiB: rss()}, nil
}
func rss() uint64 {
	b, e := os.ReadFile("/proc/self/status")
	if e != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			fields := strings.Fields(line)
			v, _ := strconv.ParseUint(fields[1], 10, 64)
			return v
		}
	}
	return 0
}
