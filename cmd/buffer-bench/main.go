// Command buffer-bench measures native Wago and the buffer plugin. It verifies
// every GPU result and emits JSON; absent hardware is an error, never a result.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	gpu "github.com/jtenner/wago-gpu"
	"github.com/jtenner/wago-gpu/internal/fixtures"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type measurement struct {
	TotalNS                                       int64
	UploadNS, DeviceCopyNS, ComputeNS, DownloadNS int64
	Allocations, AllocatedBytes                   uint64
	HeapAlloc, HeapSys                            uint64
	Dispatch, Copy                                gpu.BufferOperation
}
type result struct {
	DirectCPU, BufferCPU                                                              measurement
	Resident, SmallPrefix                                                             measurement
	PeakTrackedBytes                                                                  uint64
	DirectCPURepeatedNS                                                               []int64
	Kernel                                                                            string
	Count                                                                             uint32
	Device, OS, Arch, Go                                                              string
	ProfileStages                                                                     bool
	DeviceInitNS, CompileTotalNS, TranslationNS, PipelineNS, DirectCPUNS, BufferCPUNS int64
	First                                                                             measurement
	Repeated                                                                          []measurement
	RetainedBytes                                                                     uint64
}

func measure(f func() error) (measurement, error) {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	e := f()
	elapsed := time.Since(start)
	runtime.ReadMemStats(&after)
	return measurement{TotalNS: int64(elapsed), Allocations: after.Mallocs - before.Mallocs, AllocatedBytes: after.TotalAlloc - before.TotalAlloc, HeapAlloc: after.HeapAlloc, HeapSys: after.HeapSys}, e
}
func run(kernel string, n uint32, repeats int) (out result, resultErr error) {
	k := gpu.KernelConfig{ID: 1, Export: "wago_gpu.kernel." + kernel, CPUExport: "wago_gpu.cpu." + kernel, RelaxedFloat: true, Bindings: []gpu.BindingConfig{{Slot: 0, Type: gpu.TypeF32, Access: gpu.AccessRead}, {Slot: 1, Type: gpu.TypeF32, Access: gpu.AccessWrite}}}
	host, e := gpu.NewHost(context.Background(), gpu.Config{Kernels: []gpu.KernelConfig{k}, ProfileStages: true})
	if e != nil {
		return out, e
	}
	defer func() { resultErr = errors.Join(resultErr, host.Close(context.Background())) }()
	start := time.Now()
	module, e := host.Compile(fixtures.BufferWork)
	compileTime := time.Since(start)
	if e != nil {
		return out, e
	}
	defer func() { resultErr = errors.Join(resultErr, module.Close()) }()
	instance, e := host.Instantiate(context.Background(), module)
	if e != nil {
		return out, e
	}
	defer func() { resultErr = errors.Join(resultErr, instance.Close()) }()
	if e = fixtures.Prepare(instance, n); e != nil {
		return out, e
	}
	if _, e = instance.Invoke("setup", uint64(n)); e != nil {
		return out, e
	}
	invoke := func(name string, args ...uint64) error {
		v, e := instance.Invoke(name, args...)
		if e != nil {
			return e
		}
		if len(v) > 0 && v[0] != 0 {
			return fmt.Errorf("%s returned status %d: %+v", name, v[0], host.BufferSnapshot())
		}
		return nil
	}
	cpu, e := measure(func() error { return invoke("direct."+kernel, uint64(n)) })
	if e != nil {
		return out, e
	}
	if e = invoke("upload"); e != nil {
		return out, e
	}
	bufferCPU, e := measure(func() error { return invoke(k.CPUExport, uint64(n)) })
	if e != nil {
		return out, e
	}
	sample := func() (measurement, error) {
		var dispatch, copyOp gpu.BufferOperation
		m, e := measure(func() error {
			if e := invoke("upload"); e != nil {
				return e
			}
			if e := invoke("dispatch", 1, uint64(n)); e != nil {
				return e
			}
			dispatch = host.BufferSnapshot().Last
			if dispatch.Outcome != "GPU" || !dispatch.HardwareSubmitted {
				return fmt.Errorf("GPU execution was not confirmed")
			}
			if e := invoke("download"); e != nil {
				return e
			}
			copyOp = host.BufferSnapshot().Last
			return nil
		})
		if e != nil {
			return m, e
		}
		if e = fixtures.Verify(instance, n); e != nil {
			return m, e
		}
		m.Dispatch = dispatch
		m.Copy = copyOp
		m.UploadNS = int64(dispatch.Upload)
		m.DeviceCopyNS = int64(dispatch.DeviceCopy)
		m.ComputeNS = int64(dispatch.Compute)
		m.DownloadNS = int64(copyOp.Download)
		return m, nil
	}
	first, e := sample()
	if e != nil {
		return out, e
	}
	out = result{Kernel: kernel, Count: n, Device: host.Snapshot().Device, OS: runtime.GOOS, Arch: runtime.GOARCH, Go: runtime.Version(), ProfileStages: true, DeviceInitNS: int64(host.Snapshot().DeviceInit), CompileTotalNS: int64(compileTime), DirectCPUNS: cpu.TotalNS, BufferCPUNS: bufferCPU.TotalNS, First: first}
	for j := 0; j < repeats; j++ {
		m, e := measure(func() error { return invoke("direct."+kernel, uint64(n)) })
		if e != nil {
			return out, e
		}
		out.DirectCPURepeatedNS = append(out.DirectCPURepeatedNS, m.TotalNS)
	}
	for _, d := range host.BufferSnapshot().Kernels {
		out.TranslationNS += int64(d.Translate)
		out.PipelineNS += int64(d.Pipeline)
	}
	for j := 0; j < repeats; j++ {
		m, e := sample()
		if e != nil {
			return out, e
		}
		out.Repeated = append(out.Repeated, m)
	}
	resident := func(count uint32) (measurement, error) {
		m, e := measure(func() error { return invoke("dispatch", 1, uint64(count)) })
		m.Dispatch = host.BufferSnapshot().Last
		m.UploadNS = int64(m.Dispatch.Upload)
		m.DeviceCopyNS = int64(m.Dispatch.DeviceCopy)
		m.ComputeNS = int64(m.Dispatch.Compute)
		if e == nil && (m.Dispatch.Outcome != "GPU" || !m.Dispatch.HardwareSubmitted) {
			e = fmt.Errorf("resident GPU execution was not confirmed")
		}
		// Check outside the timed dispatch. A prefix dispatch must preserve its tail.
		if e == nil {
			e = invoke("download")
		}
		if e == nil {
			e = fixtures.Verify(instance, n)
		}
		return m, e
	}
	out.Resident, e = resident(n)
	if e != nil {
		return out, e
	}
	out.SmallPrefix, e = resident(1)
	if e != nil {
		return out, e
	}
	out.DirectCPU = cpu
	out.BufferCPU = bufferCPU
	out.PeakTrackedBytes = host.BufferSnapshot().PeakRuntimeBufferBytes
	out.RetainedBytes = host.BufferSnapshot().RuntimeBufferBytes
	return out, nil
}
func main() {
	sizes := flag.String("sizes", "1024,16384,1000000,10000000", "element counts")
	repeats := flag.Int("repeats", 3, "warm runs after first run")
	flag.Parse()
	if *repeats < 1 {
		fmt.Fprintln(os.Stderr, "repeats must be positive")
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	for _, kernel := range []string{"twice", "square"} {
		for _, s := range strings.Split(*sizes, ",") {
			n, e := strconv.ParseUint(s, 10, 32)
			if e != nil || n == 0 || n > 10_000_000 {
				fmt.Fprintln(os.Stderr, "invalid size", s)
				os.Exit(1)
			}
			r, e := run(kernel, uint32(n), *repeats)
			if e != nil {
				fmt.Fprintln(os.Stderr, e)
				os.Exit(1)
			}
			if e = enc.Encode(r); e != nil {
				panic(e)
			}
		}
	}
}
