# Follow-up report: wago-gpu

The batch, failure-handling, heavier-kernel, and CPU-selection changes work. Real GPU tests passed on the NVIDIA RTX 4060 Laptop GPU. Wago and its public APIs were not changed.

Device timestamps remain incomplete. They are not reported as measured zero time. Hard interruption of a stuck GPU driver and direct device-loss notification also remain unavailable.

## Changes

- Added `wago_gpu.run_batch(input, output, count, passes)`, for 1–64 passes of the selected kernel. It performs one upload and one download. Separate compute passes preserve the dependency between dispatches. Only the final result is committed to checked guest memory.
- Added a complete Wasm CPU batch loop. A failed GPU batch leaves guest output unchanged, so the guest can repeat the entire batch on the CPU. Invalid pass counts return status 3. Invalid memory ranges still return status 2.
- Added `MinElements`, with zero as the default. The host can use a measured CPU/GPU threshold for its kernel and pass count. No machine-specific threshold is built in.
- Added context-aware result polling and a five-second default `RunTimeout`. Execution errors and timeouts disable further GPU calls in that runtime. A cancellation detected before execution does not mark the device failed. Calls do not start a background polling goroutine.
- Reused pipelines and buffers. Buffer capacity now grows geometrically, bounded at 40,000,000 bytes per buffer. A batch needs the same two buffers as one pass. Input is uploaded again on every call.
- Added `heavy.wat`: 32 steps of `x = x * 0.9990234375 + 0.0009765625`. Its 64 arithmetic instructions are translated from the actual Wasm body. The shader is not hardcoded. This remains a small arithmetic example; it does not represent general Wasm programs or a known count of native GPU instructions after driver optimization.
- Added batch, transfer-control, threshold, and heavier-kernel options to the demonstration and Go benchmarks.

## Test system and method

The system is the same as the initial report: Debian 13.7, Linux 6.12.111+deb13-amd64, AMD Ryzen 7 8845HS, NVIDIA RTX 4060 Laptop GPU with 8,188 MiB, driver 550.163.01, Vulkan, Go 1.27.1 on linux/amd64. The current power-limit query returned N/A. The initial environment record is [environment.txt](results/environment.txt).

The Wago dependency remains `7aa401f29a33`. The retained GPU binding is `github.com/cogentcore/webgpu` v0.23.0. Workgroup size is 256. Benchmarks use `GOMAXPROCS=1`, default GC settings, and no race instrumentation. CPU affinity, clocks, and other system activity were not controlled. CPU calls run before GPU calls. The separate-transfer control runs after the batch. This order can affect clocks and caches.

Each case uses a new runtime, GPU device, and pipeline. The first call is recorded separately. Driver caches are not cleared. Input setup, explicit GC, result checks, and memory-stat reads are outside timed calls. Repeated GPU calls retain buffers and pipelines, but upload current input. CPU timing is Wago's native Wasm loop, not Go arithmetic. All reported totals include the guest call and, for GPU work, upload, dispatch, download, and result commit.

## One pass

Times are milliseconds. Repeated times are medians of 30 calls. Device creation and compilation are separate from the first-call columns.

| Kernel | Elements | First CPU | CPU median | First GPU | GPU median | CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.1130 | 0.0020 | 5.0171 | 0.0511 | 0.04x |
| twice | 16,384 | 0.0285 | 0.0213 | 5.4968 | 0.0695 | 0.31x |
| twice | 1,000,000 | 2.4033 | 1.3050 | 7.6458 | 0.9855 | 1.32x |
| twice | 10,000,000 | 25.4017 | 14.9880 | 49.2853 | 11.1657 | 1.34x |
| square | 1,024 | 0.0143 | 0.0020 | 4.7069 | 0.0437 | 0.05x |
| square | 16,384 | 0.0302 | 0.0211 | 4.9649 | 0.0589 | 0.36x |
| square | 1,000,000 | 2.4148 | 1.3025 | 7.3872 | 0.9288 | 1.40x |
| square | 10,000,000 | 26.1006 | 15.0646 | 51.7220 | 10.8100 | 1.39x |
| heavy | 1,024 | 0.0308 | 0.0216 | 5.2082 | 0.0449 | 0.48x |
| heavy | 16,384 | 0.3244 | 0.3198 | 5.1771 | 0.0668 | 4.79x |
| heavy | 1,000,000 | 21.0548 | 18.9720 | 7.5912 | 0.9480 | 20.01x |
| heavy | 10,000,000 | 207.2677 | 189.1137 | 49.8092 | 10.8745 | 17.39x |

A ratio above 1 means the GPU path was faster. The small original workloads remain slower on the GPU. The heavy kernel has enough arithmetic to use the GPU at much smaller sizes. These are local measurements, not general speed guarantees. Raw data: [improvements-single.json](results/improvements-single.json).

## Eight-pass batches

Each row compares the same eight kernel applications. CPU uses one Wasm batch loop. GPU batch uses one import and one transfer pair. Separate GPU uses one guest call with eight imports and eight transfer pairs. The latter is a benchmark control and has no whole-batch rollback on failure.

Times are milliseconds. Twice uses 30 repetitions; heavy uses 15. The batch speed ratio compares separate GPU transfers with a GPU batch.

| Kernel | Elements | CPU median | GPU separate median | GPU batch median | Separate / batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 1,024 | 0.0115 | 0.3912 | 0.1398 | 2.80x |
| twice | 16,384 | 0.1656 | 0.5185 | 0.1463 | 3.54x |
| twice | 1,000,000 | 10.2456 | 8.2216 | 1.2166 | 6.76x |
| twice | 10,000,000 | 101.7165 | 96.1851 | 20.0372 | 4.80x |
| heavy | 1,024 | 0.1587 | 0.4694 | 0.1285 | 3.65x |
| heavy | 16,384 | 2.5264 | 0.5097 | 0.1378 | 3.70x |
| heavy | 1,000,000 | 149.4649 | 8.2754 | 1.2312 | 6.72x |
| heavy | 10,000,000 | 1502.6082 | 95.9394 | 13.4352 | 7.14x |

At one million elements, the heavy batch transfers 4 MB up and 4 MB down. Eight separate calls transfer 32 MB in each direction. At ten million elements the counts are 40 MB per direction versus 320 MB. These are application buffer byte counts, not PCIe counters.

The single-call and batch results vary between runs. For example, the twice batch at ten million elements had a 20.04 ms median above, while the separate five-call Go benchmark averaged 13.73 ms. The heavy batch at 1,024 elements was faster than CPU in the main run but slower in the size scan. Both results are retained. Do not choose thresholds from a marginal gain in one run.

Raw data: [twice batches](results/improvements-batch-twice.json), [heavy batches](results/improvements-batch-heavy.json), [Go benchmarks](results/improvements-go-benchmarks.txt).

## Transfer and compute-stage measurements

These are mean host wall times from separate profile runs. One-pass cases use 15 calls. The eight-pass heavy cases use 10. Extra queue waits separate the stages. Compute includes encoding, submission, and waiting. It is not a device timestamp. Download includes result-read completion. Plugin total includes guest-memory checks and commit, but excludes the outer Wago invocation overhead.

| Kernel | Passes | Elements | Upload | Compute stage | Download | Commit | Plugin total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| twice | 1 | 1,024 | 0.0238 | 0.0301 | 0.0212 | 0.0003 | 0.0766 |
| twice | 1 | 16,384 | 0.0275 | 0.0261 | 0.0248 | 0.0029 | 0.0823 |
| twice | 1 | 1,000,000 | 0.4831 | 0.0441 | 0.3406 | 0.1626 | 1.0335 |
| twice | 1 | 10,000,000 | 5.6734 | 0.4154 | 3.0231 | 1.6889 | 10.8119 |
| square | 1 | 1,024 | 0.0221 | 0.0258 | 0.0199 | 0.0003 | 0.0692 |
| square | 1 | 16,384 | 0.0273 | 0.0255 | 0.0247 | 0.0023 | 0.0807 |
| square | 1 | 1,000,000 | 0.4966 | 0.0488 | 0.3422 | 0.1739 | 1.0642 |
| square | 1 | 10,000,000 | 5.8822 | 0.4429 | 3.0234 | 1.7838 | 11.1446 |
| heavy | 1 | 1,024 | 0.0238 | 0.0294 | 0.0220 | 0.0004 | 0.0767 |
| heavy | 1 | 16,384 | 0.0318 | 0.0290 | 0.0262 | 0.0027 | 0.0909 |
| heavy | 1 | 1,000,000 | 0.4736 | 0.0440 | 0.3336 | 0.1647 | 1.0178 |
| heavy | 1 | 10,000,000 | 5.8733 | 0.4376 | 3.0050 | 1.7130 | 11.0402 |
| heavy | 8 | 1,024 | 0.0252 | 0.1044 | 0.0220 | 0.0004 | 0.1531 |
| heavy | 8 | 16,384 | 0.0417 | 0.1299 | 0.0300 | 0.0029 | 0.2066 |
| heavy | 8 | 1,000,000 | 0.5163 | 0.2438 | 0.3559 | 0.1632 | 1.2819 |
| heavy | 8 | 10,000,000 | 11.3489 | 18.9086 | 7.1833 | 1.5168 | 38.9664 |

The ten-million-element heavy batch profile contains a large outlier. Its full-call median was 13.3319 ms, mean 38.9941 ms, and maximum 133.4570 ms. It is retained in the stage means above. The cause was not established. Do not add profile means to, or substitute them for, normal-run medians.

Raw data: [one-pass stages](results/improvements-stages.json), [batch stages](results/improvements-batch-stages.json).

## First-run costs and memory

The tables above separate first calls from repeated work. The following are compilation and device-creation observations at ten million elements, in milliseconds. WGSL translation and pipeline creation are parts of plugin compilation, not additional costs.

| Kernel | Native Wasm compile | Wasm to WGSL | GPU pipeline | Plugin compile | Device creation |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 0.2785 | 0.0105 | 0.3446 | 0.6277 | 122.5308 |
| square | 0.2686 | 0.0103 | 0.3544 | 0.6123 | 98.8310 |
| heavy | 0.2761 | 0.0318 | 0.5977 | 0.9211 | 105.6089 |

Repeated native CPU calls used zero Go allocations. The normal GPU path used about 1.6 KiB and 52 allocations per call. Eight-pass GPU batches used about 2.4 KiB and 94 allocations. The deadline and cancellation work adds allocations relative to the first implementation. Allocation size stays independent of element count; dispatch allocation count grows with pass count. There is no array-sized Go allocation in the plugin execution path.

The following are process snapshots after repeated calls at ten million elements. MiB uses 1,048,576 bytes. GPU process RSS includes native Wasm memory, buffers, driver state, and staging. CPU-only RSS is from a separate process. RSS is not a peak measurement.

| Kernel | Guest MiB | GPU buffers MiB | CPU-only RSS MiB | GPU RSS MiB | GPU process Go heap MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| twice | 114.50 | 76.29 | 127.53 | 373.82 | 0.21 |
| square | 114.50 | 76.29 | 127.55 | 378.32 | 0.21 |
| heavy | 114.50 | 76.29 | 129.43 | 385.75 | 0.22 |

Batch buffers have the same logical size as one-pass buffers. A smaller call can retain prior capacity until the last module/instance reference closes. Per-call allocations, compilation allocations, heap size, RSS, first-call times, and native memory estimates are saved in the JSON files. CPU-only data: [improvements-cpu-only.json](results/improvements-cpu-only.json).

## CPU/GPU selection

The first sampled warm GPU gain in a 50-call scan was:

| Kernel | Passes | First sampled gain | Conservative example `MinElements` |
| --- | ---: | ---: | ---: |
| twice | 1 | 131,072 | 262,144 |
| square | 1 | 131,072 | 262,144 |
| heavy | 1 | 4,096 | 4,096 |
| twice | 8 | 16,384 | 32,768 |
| heavy | 8 | 2,048 | 2,048 |

These values are warm-call samples on this machine. They exclude startup and do not identify a universal smallest profitable workload. The first original-kernel gains were only 4–8%, so a larger limit gives some margin. For one-off calls, device creation and first-run costs can change the choice. The host selects its limit explicitly.

A demonstration with `-kernel heavy -sizes 1024,16384 -min-elements 4096` used CPU fallback at 1,024 and GPU at 16,384. Both results passed verification. See [selection.json](results/improvements-selection.json) and the [single](results/improvements-crossover-single.json), [twice batch](results/improvements-crossover-batch-twice.json), and [heavy batch](results/improvements-crossover-batch-heavy.json) scans.

## Correctness and failure tests

CPU-only tests passed without GPU support. GPU tests ran on the physical NVIDIA device. The following also passed:

- CPU and GPU race-detector runs, including separate concurrent instances sharing a serialized GPU queue.
- Real instruction translation for all three kernels, unsupported instructions, invalid memory ranges, and invalid batch pass counts.
- Repeated batches, changed inputs, shrinking and growing buffers, exact in-place execution, shared pipelines, and instance/module/runtime cleanup.
- CPU fallback for missing devices, shader failures, execution failures, low element counts, disabled GPU use, and strict float mode.
- Timeout and cancellation polling tests. A simulated waiting backend verifies the failure latch and full CPU fallback.
- A hardware test destroys only its own WebGPU readback buffer. The resulting validation failure leaves guest output unchanged, disables later GPU calls, and uses the CPU batch fallback.
- A five-second compiler fuzz run completed 63,729 executions with two workers and found no failure.
- CPU and GPU demonstrations, the `-require-gpu` failure path in a CPU-only build, and all single/batch Go benchmarks.

The original one-pass examples and repeated doubling require exact output bits. The heavy kernel and repeated square require finite results within `8e-6 * passes * max(1, abs(CPU))`. This is a stated acceptance tolerance for these demonstration inputs. It is not a guarantee for arbitrary float values. Tests also reject deliberately wrong, infinite, and NaN results.

The largest heavy-kernel absolute errors were about 0.00001335 for one pass and 0.00003433 for eight passes. A 64-pass hardware test had a maximum near 0.00002956. Repeated square produced an absolute difference of 128 on some large results; these passed the stated relative tolerance. Strict Wasm arithmetic is not promised. Physical device removal or a genuinely hung driver was not tested.

Test logs: [CPU](results/improvements-cpu-tests.txt), [GPU](results/improvements-hardware-tests.txt), [CPU race](results/improvements-race-tests.txt), [GPU race](results/improvements-hardware-race-tests.txt), [fuzz](results/improvements-fuzz.txt), [CPU demo](results/improvements-demo-cpu.txt), [GPU demo](results/improvements-demo-gpu.txt).

## Limits and rejected backend changes

The retained binding does not offer a usable direct device-loss callback in this project: the optional callback path fails Go pointer checks. Returned errors and readback status detect failures. A failed execution disables GPU reuse. Result polling checks a context deadline, but this cannot interrupt a native call that blocks inside the driver. Cleanup can also block. This is not a hard driver-hang timeout.

Device timestamp work was investigated but not completed safely:

- `go-webgpu/webgpu` v0.5.5 aborted during adapter creation with `invalid callback`. It was not retained.
- `gogpu/wgpu` v0.34.5 ran the simple kernels and an initial batch on the device. Source review then found that its Vulkan buffer-transition conversion omitted host map access flags. The required host-read memory dependency was not assured. Its timestamp feature reporting also omitted support, and its query creation used host query reset without enabling that device feature. It was not retained.
- The existing binding was restored. New dependency binaries and module changes were removed. GPU timestamp fields stay explicitly unavailable. Host stage timings are measured.

The Vulkan host-read requirement is described in the [Vulkan memory specification](https://docs.vulkan.org/spec/latest/chapters/memory.html) and [synchronization specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html). A later backend must expose safe timestamp queries, the timestamp scale, required feature negotiation, and proper host memory synchronization before it can replace the tested backend.

The batch API currently repeats one selected function. It does not retain GPU arrays between imports or chain different kernels. GPU work is serialized across instances. The compiler remains limited to the original five arithmetic/local instruction kinds. CPU fallback remains the answer for other Wasm functions.

## Wago API changes

None were required. The same public source-transform, compile-observer, host-import, caller-resolution, and checked guest-storage APIs support the new batch call. No Wago source or runtime code was modified. The existing optional compile-abandonment event gap remains as described in the initial report. The timestamp and driver-notification gaps are in GPU dependencies, not Wago.

## Next three improvements

1. Use or repair a GPU binding that passes synchronization tests and exposes device timestamps and safe device-loss notification. Add a separate-process hang test before claiming a hard timeout.
2. Extend one-call batches to a bounded sequence of different selected kernels. This can avoid transfers in more useful workloads without retaining guest pointers across calls.
3. Use broader nonlinear workloads and interleaved repeated benchmark rounds. Record timing distributions, then select thresholds with a margin for startup, pass count, and system variation. Reduce deadline and binding allocations where measurements show a useful gain.
