# wago-gpu

This is an external Go plugin for Wago. It compiles one selected Wasm function into WGSL and runs it across an array. Wago remains an unmodified dependency. No GPU code is installed in Wago.

The [full plugin specification](BUFFER_API_PROPOSAL.md) defines the existing interface and the proposed named-kernel and typed-buffer extension. It marks proposed features separately from implemented behavior.

The [boundary review](BUFFER_API_PROPOSAL.md#235-confirmed-callback-defect-and-backend-acceptance-gate) found callback-lifetime and error-reporting gaps in the pinned GPU binding. The historical hardware results remain valid observations, but the binding needs repair or replacement before the proposed buffer GPU phase can be accepted. The draft ABI is not frozen.

The experiment works on a real NVIDIA RTX 4060 Laptop GPU. The original examples match Wago's native CPU results bit for bit. A third kernel has 32 multiply/add steps and uses an explicit float tolerance. Batch calls retain intermediate results on the GPU. GPU execution is slower for small arrays. See [IMPROVEMENTS.md](IMPROVEMENTS.md) for current results and limits. [REPORT.md](REPORT.md) records the first experiment.

## Build and run

Run these commands from this directory. The tested system uses Go 1.27.1 on Linux/amd64. The module declares Go 1.22 as its minimum version.

```sh
# CPU tests and demonstration. No GPU or C compiler is needed.
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go run ./cmd/demo -cpu-only

# GPU build. This requires CGO, a C compiler, and a working GPU driver.
go build -tags webgpu -o demo ./cmd/demo
./demo -require-gpu

# Require real GPU tests. Device absence is a failure, not a skipped test.
WAGO_GPU_TEST=1 go test -tags webgpu -v ./...

# Check concurrent calls with Go's race detector.
go test -race ./...
WAGO_GPU_TEST=1 go test -race -tags webgpu ./...
```

The GPU build uses `github.com/cogentcore/webgpu/wgpu` v0.23.0. That dependency includes native `wgpu` libraries and a CGO binding. This project contains no C or C++ source. The backend uses the native default adapter selection with a high-performance preference. It rejects software adapters. Only Linux/amd64 with NVIDIA Vulkan was tested.

Without `-tags webgpu`, the GPU import returns fallback status. A build with `CGO_ENABLED=0` also uses fallback. With GPU support built, device or shader errors also select fallback. Use `-require-gpu` when a successful hardware run is required.

The checked-in Wasm files were built from the adjacent WAT files. Tests do not need WABT. To rebuild the files with `wat2wasm`:

```sh
go generate ./internal/fixtures
```

## Use the plugin

For a complete module that calls the GPU during startup, see
[examples/start/module.wat](examples/start/module.wat) and its
[Go host](examples/start/main.go). The Wasm start function fills four input
values, calls the GPU import, and uses its CPU fallback if needed. The results
are ready when `Runtime.Instantiate` returns. Go does not call a processing
export after that.

```sh
go run -tags webgpu ./examples/start -require-gpu
go run ./examples/start -cpu-only
```

Both paths produce `[2 4 6 8]` from `[1 2 3 4]`. The checked-in Wasm file makes
the example runnable without WABT. Rebuild it with `go generate ./examples/start`.

The host selects a function export. Each module can have one selected kernel. The examples use separate modules for `2*x`, `x*x+1`, and the heavier kernel.

```go
p, err := wagogpu.New(wagogpu.Config{
    KernelExport: "kernel",
    RelaxedFloat: true, // Explicit consent; read the float limits below.
})
if err != nil { return err }
set, err := p.PluginSet()
if err != nil { return err }

rt := wago.NewRuntime()
defer rt.CloseContext(context.Background())
if err := rt.LoadPlugins(context.Background(), set); err != nil { return err }
mod, err := rt.Compile(wasmBytes)
if err != nil { return err }
defer mod.Close()
inst, err := rt.Instantiate(context.Background(), mod)
if err != nil { return err }
defer inst.Close()
```

Use imports `wagogpu "wago-gpu"` and `wago "github.com/wago-org/wago"` within this local module. The plugin ID and repository use reserved `example.com` values. They identify an unpublished experiment. Set real module and repository names before distribution.

The guest imports:

```wat
(import "wago_gpu" "run"
  (func $gpu (param i32 i32 i32) (result i32)))
```

The arguments are input byte offset, output byte offset, and `f32` element count. Memory index 0 is used. Values use Wasm's little-endian byte order.

| Status | Meaning | Guest action |
| --- | --- | --- |
| `0` | GPU result committed | Continue |
| `1` | GPU path unavailable | Execute the original CPU loop |
| `2` | Invalid guest memory range | Report an error; do not run an unchecked CPU loop |
| `3` | Invalid batch pass count | Report an error; do not run the CPU loop |

The example guest's `run` export implements this branch. Its `cpu` export runs the same scalar Wasm function in Wago's native loop. Its `gpu` export exposes the raw import status for tests. See [twice.wat](internal/fixtures/twice.wat) and [square.wat](internal/fixtures/square.wat).

For several applications of the same kernel, import:

```wat
(import "wago_gpu" "run_batch"
  (func $gpu_batch (param i32 i32 i32 i32) (result i32)))
```

The first three arguments are the same. The fourth is the number of passes, from 1 to 64. Each pass uses the prior pass's output. The plugin uploads once, dispatches separate compute passes, and downloads once. It commits only the final result. The guest's `run_batch` export falls back to `cpu_batch` for the entire operation. No GPU handle or borrowed guest-memory pointer survives the call.

Use `MinElements` to select CPU fallback for arrays below a measured size. Zero, the default, always attempts the GPU. The limit applies to the complete call, including all batch passes. Choose it for your device, kernel, and pass count; there is no universal threshold. For example:

```sh
# Compare batched execution with a full transfer on every pass.
GOMAXPROCS=1 ./demo -kernel heavy -passes 8 -bench -reps 15 -require-gpu -separate

# Select CPU for small arrays. Do not combine this with -require-gpu.
./demo -kernel heavy -n 1024 -min-elements 16384
```

`gpu_separate` is a benchmark control. It uses one guest call but one host import and transfer pair per pass. It aborts on failure and can leave earlier pass results in memory. Applications must use `run_batch` for transactional fallback. The square demonstration limits batches to four passes to keep its fixed input range finite. The plugin itself permits up to 64 passes; the host must accept the kernel's numerical behavior.

Both ranges are checked through `WithGuestStorage`, `MemoryInfo`, and `MemoryRange`. They remain inside the callback until the GPU call finishes. Input is uploaded before output is committed. The plugin commits only after a successful GPU result read. A failed call leaves guest output unchanged. Exact in-place operation is supported. Partial overlap selects CPU fallback because a forward CPU loop can have different results.

Make one plugin per runtime. GPU calls from different instances are safe, but are serialized. A single cache and reusable buffers serve those calls. Close the instance and module to release their references. A module can close before its existing instances. Its pipeline remains valid until the last instance closes. Use `Runtime.CloseContext` to wait for device cleanup; `Runtime.Close` starts asynchronous shutdown.

`Snapshot()` reports device information, build times, resource counts, the last call's times, pass count, transfer byte counts, and the last rejection reason. `GPUFailed` means an execution error or timeout disabled GPU calls for this runtime. Make a new runtime and plugin to retry the device. `ProfileStages` enables separate stage timings. Leave it off during normal use.

## Compiler scope and float limits

`CompileWGSL(wasmBytes, exportName)` reads the actual function body. It uses a small stack and emits one WGSL `let` per instruction. It preserves operand order. Constants use their original 32-bit representation. The compiler does not contain example shaders.

The selected function must have type `(f32) -> f32`. Supported instructions are `local.get 0`, `f32.const`, `f32.add`, `f32.sub`, and `f32.mul`, followed by `end`. Other instructions, extra locals, imported kernels, and wrong signatures are rejected for GPU execution. Other guest functions stay on the CPU.

This is a bounded subset decoder. Wago validates the complete Wasm module before a GPU pipeline is made. Limits are 4 MiB of module source, 16,384 section entries, 16 KiB of kernel code, and 4,096 instructions. The module must use MVP section order and only function imports. Shared memory, imported memory, Memory64, and multiple memories cannot use this GPU path. Array size is limited to 10 million elements. `MaxElements` can lower this limit. GPU device limits can also cause fallback.

`RelaxedFloat` defaults to false. With that default, calls use the original CPU function. Set it to true only when WGSL floating-point behavior is acceptable. WGSL permits behavior that differs from Wasm, including fused operations, reassociation, subnormal flushing, and differences for signed zero and non-finite values. Separate `let` statements do not enforce strict Wasm rounding. See the [WGSL floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation).

The original single-pass examples use bounded binary-fraction inputs and check every output bit. The heavy kernel repeats `x = x * 0.9990234375 + 0.0009765625` 32 times. Its Wasm body contains 64 arithmetic instructions. It uses only the compiler's existing instruction subset.

For the heavy kernel and multi-pass square, verification requires finite results and an absolute error no greater than `8e-6 * passes * max(1, abs(CPU))`. This is an acceptance tolerance for the fixed demonstration inputs, not a general numerical error guarantee. The maximum observed absolute error is saved in JSON. Neither these tests nor the plugin promise strict Wasm float behavior for all `f32` values.

## Wago API investigation

The dependency is pinned to `7aa401f29a33`, the current Wago commit when this work started. The design follows these examples:

- [08-custom-plugin](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/examples/08-custom-plugin/main.go): providers, authority grants, and host imports.
- [11-source-transform](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/examples/11-source-transform/main.go): source inspection and compile observation.
- [21-guest-storage](https://github.com/wago-org/wago/blob/7aa401f29a33eb9050559aab5488cdb307f5a078/examples/21-guest-storage/main.go): checked callback-scoped memory views.

| Public API | Use |
| --- | --- |
| `ModuleSourceTransformer` | Inspect source and return it unchanged |
| `CompilationIdentity`, `ModuleCompileObserver` | Connect generated WGSL to successful compilation |
| `ModuleSourceDigest` | Reject stale WGSL if a later plugin changes the source |
| `InstanceInstantiateInterceptor.After` | Connect the exact instance before its start function |
| `HostCallers`, `HostImports` | Resolve the caller and expose `run` and `run_batch` |
| `GuestStorageHostModule` | Check and borrow both memory ranges |
| Module and instance close observers | Release pipeline references |
| `PluginLifecycle.Stop` | Release all remaining GPU resources |

No Wago API change was required. One optional gap remains: `PreparedCompile.Close` emits no abandonment event. The plugin bounds pending translations to 16 entries and evicts old entries. An evicted preparation uses CPU fallback if completed later. A generic compile-abandonment event carrying `CompilationIdentity` would permit immediate cleanup.

## Benchmarks

```sh
# Total call time; includes the guest import, transfers, and result commit.
GOMAXPROCS=1 go run -tags webgpu ./cmd/demo \
  -bench -reps 50 -require-gpu -json > results/current-benchmark.json

# Separate stage measurements. Extra queue waits change total time.
GOMAXPROCS=1 go run -tags webgpu ./cmd/demo \
  -bench -reps 50 -require-gpu -profile -json > results/current-stages.json

# Go benchmark output and allocation counts.
GOMAXPROCS=1 WAGO_GPU_TEST=1 go test -tags webgpu -run '^$' \
  -bench 'Benchmark(Batch)?(CPU|GPU)$' -benchtime=50x -benchmem

# Compare nearby sizes. These results depend on the device and system load.
GOMAXPROCS=1 go run -tags webgpu ./cmd/demo \
  -sizes 65536,131072,262144,393216,524288,786432,1000000 \
  -reps 100 -require-gpu -json
```

Use `-kernel` to select a fixture and `-passes` to set the batch size. All JSON durations are nanoseconds. Timings use the host clock. Compute-stage times include command encoding, submission, and a queue wait. They are not device timestamps. Transfer timings include API and synchronization costs. Setup, Go GC, result checks, and memory-stat reads are outside timed calls. Each case uses a fresh runtime and pipeline. Driver caches are not cleared.

The normal call combines compute and result-copy commands. The profile run separates upload, compute, and download with waits. Do not add its stage means to the normal run's median total.

Each call uploads current guest input and downloads final output. A batch uses the same transfer byte count as one pass. Buffers and pipelines are reused. No unchanged-input assumption is made. Buffer capacity grows geometrically, up to 40,000,000 bytes per buffer. Two buffers are retained per cached pipeline. A smaller later call can retain that capacity until module cleanup. Go allocations stay independent of array size and grow linearly with pass count. GPU-driver allocations and Wasm's native memory mappings are outside Go heap statistics.

The binding has two avoided paths: optional instance extras are not fully initialized, and its device-loss callback fails Go's pointer checks. This plugin uses default descriptors, returned errors, and result-read completion status. `RunTimeout` defaults to five seconds. The plugin checks the invocation context and this deadline between nonblocking device polls. On failure, it cancels result mapping, leaves guest memory unchanged, and stops further GPU use. An already canceled call is rejected before submission without setting `GPUFailed`.

This does not interrupt GPU commands already submitted. A native driver entry point or resource cleanup can still block. There is no hard driver-hang timeout and no direct device-loss notification. Device timestamps remain unavailable: `DeviceCompute` is zero and `TimestampsValid` is false. Do not read zero as a measured GPU execution time. The report records why the newer bindings were rejected.
