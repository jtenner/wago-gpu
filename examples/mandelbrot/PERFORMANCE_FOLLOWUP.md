# CPU input reuse and GPU copy checks

Measured on 2026-10-08. The baseline is commit `2bff5d6`. These changes keep
Wago, the public plugin ABI, the four-buffer Mandelbrot layout, and the 256 MiB
instance limit unchanged.

## Changes kept

- The private Mandelbrot render loop uses its current guest arrays for CPU
  fallback. Coordinates stay constant. Each completed GPU pass copies both
  orbit outputs into those arrays; each CPU pass updates them. This removes
  four input copies per CPU pass. The exported CPU runner still refreshes its
  inputs for an independent call. This is a property of this guest program,
  not a general permission to use stale plugin data.
- The compiler records reads before the first store for each slot. Accepted
  bodies have no branches and use the original invocation index. A full-range
  output with no read of old data can use dirty private scratch without a seed
  copy or an upload of its old contents. Prefix writes and in-place increments
  still seed the output. Stores followed by reads use the new value.
- The count uniform remains valid after successful completion. A repeated
  count needs no new upload. Count changes, allocation, or eviction require
  another upload. The cache belongs to one instance. Pending payload storage
  is charged only when a write occurs.
- Stage profiling omits the extra seed submission when there are no seeds.
  Its device-copy duration is zero in that case.

Outputs remain private until the existing validation, completion, error, and
commit checks pass. Lost contents still require a full guest set operation.
These changes do not weaken cancellation or cleanup rules.

The generated CPU loop already loads slice metadata and checks bounds before
its pixel loop. No additional bounds-check change was needed. No loop whose
work grows quadratically with pixel count was found.

## Method and hardware

NVIDIA RTX 4060 Laptop GPU, 8 GiB, Vulkan driver 550.163.01; AMD Ryzen 7 8845HS;
Debian 13, Linux 6.12.111; Go 1.27.1; TinyGo 0.42.0.
`GOMAXPROCS=1`, `ProfileStages=false`, no race detector during timing.
Each Mandelbrot value is the median of three samples, each averaging three
renders at **32 iterations**. The example default remains 64 iterations.

`ReuseModule` retains the module, GPU device, and pipeline. Each command creates
a new WASI instance and buffers. Timing includes setup, rendering, output to
`io.Discard`, result checks, and instance cleanup. Full image comparisons run
outside timing. Baseline and changed runs use the same settings. The final
changed run also reports guest-copy bytes and count-parameter writes.

GPU clocks were not fixed, and temperature was not controlled. Runs were
sequential, with no other test process started by this task during timing.
An additional changed-code run is retained to show run-to-run variation.
There are no GPU device timestamps. These local timings do not establish a
fixed speedup or isolate the cost of each change.

```sh
GOMAXPROCS=1 WAGO_GPU_TEST=1 go test -tags webgpu ./examples/mandelbrot \
  -run '^$' \
  -bench '^BenchmarkMandelbrotReuseModule/(512x512|1920x1080|2560x1440)/(cpu|fallback|gpu)$' \
  -benchtime=3x -count=3 -benchmem
```

## Mandelbrot results

| Image | CPU fallback before ms | CPU fallback after ms | GPU before ms | GPU after ms | Direct CPU after ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512×512 | 27.444 | 22.917 | 36.283 | 30.279 | 5.671 |
| 1920×1080 | 236.856 | 188.217 | 224.742 | 203.488 | 43.505 |
| 2560×1440 | 489.652 | 402.079 | 405.065 | 377.683 | 77.535 |

Direct CPU execution remains faster. It stops updating escaped pixels. Both
buffer paths update every pixel on each pass.

| Per command | Before | After |
| --- | ---: | ---: |
| CPU fallback Go allocations | 1,923 | 1,155 |
| GPU Go allocations | 2,480 | 2,449 |
| GPU count-parameter writes across 32 passes | 32 | 1 |
| 1440p CPU fallback input copies, MiB | 1,800 | 0 |
| 1440p GPU peak tracked storage, MiB | 154.70 | 154.70 |
| 1440p CPU fallback peak tracked storage, MiB | 56.25 | 56.25 |

The old input-copy total follows from four F32 arrays copied on each of 32
passes. The new counter and tests require zero copies for CPU-only rendering.
Output sets remain: the 1440p CPU command transfers about 928.1 MiB into plugin
CPU storage, including initial coordinates. The GPU still reads back 900 MiB,
copies 900 MiB into guest memory, and seeds 900 MiB on the device per image.
Mandelbrot reads old orbit values before it writes, so its seeds cannot be
omitted with the retained layout.

No new scratch allocation is needed. Go byte counts include fresh CPU mirrors
per instance; native driver allocations are outside Go counters and tracked
plugin buffer bytes.

## Separate-output layout tested and removed

A six-buffer Mandelbrot variant alternated separate orbit inputs and outputs.
It passed image checks and removed all output seeds. Its 1440p GPU median was
401.894 ms against 405.065 ms for the baseline. Peak tracked GPU-path storage
rose from 154.70 to 210.90 MiB; Go allocations rose from 2,480 to 2,630.
Its full-HD GPU median was 222.460 ms against 224.742 ms.

These small timing changes did not justify the added memory and allocation
cost. The variant was removed. The general compiler proof remains useful for
kernels that already have separate outputs, including the original math
kernels. Its raw measurements are retained as a rejected experiment.

## Original array kernels

The array command used stage profiling and three repeats after its first GPU
run. The table reports median repeated fresh-input end-to-end GPU time and
median repeated direct CPU time. Every result, resident dispatch, and one-element
prefix was checked. All full covering math outputs reported zero seed copies;
repeated counts reported zero parameter uploads. Prefixes still preserved tails.
The peak includes first-run, repeated, resident, and prefix checks.

| Kernel | Elements | Repeated GPU ms | Direct CPU ms | Peak tracked MiB |
| --- | ---: | ---: | ---: | ---: |
| `2*x` | 1,024 | 0.208 | 0.001 | 0.03 |
| `2*x` | 16,384 | 0.153 | 0.011 | 0.44 |
| `2*x` | 1,000,000 | 1.413 | 0.551 | 26.70 |
| `2*x` | 10,000,000 | 82.743 | 5.722 | 228.88 |
| `x*x+1` | 1,024 | 0.207 | 0.002 | 0.03 |
| `x*x+1` | 16,384 | 0.133 | 0.014 | 0.44 |
| `x*x+1` | 1,000,000 | 1.381 | 0.761 | 26.70 |
| `x*x+1` | 10,000,000 | 83.554 | 7.970 | 228.88 |

Direct CPU remains faster at every required size. The earlier array report used
one repeated GPU sample; it is not a controlled baseline for these medians.

```sh
GOMAXPROCS=1 go run -tags webgpu ./cmd/buffer-bench -repeats=3
```

## Checks and evidence

CPU fallback images still match direct CPU exactly. At 32 iterations the GPU
differs at zero pixels for 512×512, five for full HD, and thirteen for 1440p.
These counts match the baseline. Float arithmetic remains relaxed.

The 64-iteration full-HD and 1440p runs also completed all GPU passes without
fallback. Their CPU fallback images match exactly. GPU images differ at 198
and 366 pixels, respectively, matching the earlier counts.

The CPU-only suite, full real-GPU race suite, cgo pointer checks, both builds of
`go vet`, all 11 specification checks, and compiler fuzzing passed. New fake and
hardware tests cover dirty scratch, prefix preservation, reads before writes,
saved old values, reads after writes, repeated count reuse, changed counts,
uniform eviction, and cleanup. The mixed test checks a GPU pass followed by a
failed private dispatch and eleven CPU fallback passes, with no stale input use.

- [Baseline samples](../../results/perf-followup-before.txt),
  [final samples](../../results/perf-followup-after.txt), and
  [additional changed-code samples](../../results/perf-followup-cache-only.txt).
- [Removed separate-output variant](../../results/perf-followup-pingpong.txt).
- [Median summary](../../results/perf-followup-summary.json) and
  [array measurements](../../results/perf-followup-arrays.jsonl).
- [CPU-only suite](../../results/perf-followup-cpu-tests.txt),
  [real-GPU race suite](../../results/perf-followup-hardware-race.txt),
  [focused hardware tests](../../results/perf-followup-hardware-focused.txt),
  [cgo checks](../../results/perf-followup-cgocheck.txt), and
  [compiler fuzzing](../../results/perf-followup-fuzz.txt).
- [64-iteration image hashes and comparisons](../../results/perf-followup-images.json).

## Next work

1. Move escape checks and iteration to the GPU to reduce the two readbacks per
   pass. This requires an explicit compiler-subset extension and new tests.
2. Measure bulk CPU runners for the general array examples. Scalar buffer
   imports remain useful for the contract but are costly inside large CPU loops.
3. Measure batched uploads and readbacks before changing queue submission.
   Completion, error checks, failure atomicity, and resource charging must stay
   intact. None of these three changes is implemented here.
