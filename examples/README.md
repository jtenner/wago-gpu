# Examples

Start with one of these commands from the repository root. Checked-in Wasm
files let you run the examples without installing a guest compiler.

| Example | Command | Result |
| --- | --- | --- |
| Two buffer kernels | `go run ./examples/buffers -first-cpu -second-cpu` | `[3 5 7 9]` |
| TinyGo guest | `go run ./examples/tinygo -cpu` | `[2 4 6 8]` |
| F16 storage | `go run ./examples/storage -kind f16 -cpu` | `2` |
| Memory64 transfer | `go run ./examples/storage -kind memory64 -cpu` | `7` |
| GC-array transfer | `go run ./examples/storage -kind gc -cpu` | `7` |
| WASI Mandelbrot | `go run ./examples/mandelbrot > mandelbrot-cpu.pgm` | A grayscale image |

The small hosts declare a kernel, run a guest, and check its result.
[runwasm](internal/runwasm/run.go) contains their common compilation, optional
WASI imports, and cleanup. Read that file to see the complete host sequence.
It uses source compilation and keeps WASI imports in their own namespace.

Build the patched native library with `./native/build.sh` to run GPU examples.
Then add `-tags webgpu` and `-require-gpu` to a GPU-capable command. That flag
fails if any required computation uses fallback.

The [Mandelbrot examples](mandelbrot/README.md) show WASI arguments, stdout,
CPU execution, and explicit GPU iteration with CPU fallback. The earlier
[start example](start/main.go) still demonstrates a real Wasm start function.

## Benchmarks

[Measured results for all examples](BENCHMARKS.md) include CPU, GPU, mixed
execution, full setup, repeated commands, and Go allocations. The report also
includes 64-iteration WASI Mandelbrot images up to 2560×1440 and both CLI demos.
All GPU measurements used real hardware. The report has raw data and commands
for a new run. Memory64 and GC transfer examples have no GPU kernel.
