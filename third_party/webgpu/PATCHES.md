# Local WebGPU binding changes

Base: `github.com/oliverbestmann/webgpu` v1.36.0, MIT license.
Only its binding, Linux headers, JSX support, and module metadata are copied here.
Unused native platform dependencies are removed. Identical Linux headers are
stored once for amd64 and arm64.
Native binaries are not checked in. The Linux library is built by
[`native/build.sh`](../../native/build.sh).

Changes in this copy:

- Pin each callback's Go userdata byte until native ownership ends. The handle
  table keeps the value and its `runtime.Pinner` alive.
- Keep the device-loss callback until the final explicit device release. The
  plugin releases every child before releasing the device.
- Wait for each validation and allocation error-scope callback, including a
  successful no-error result. Use C-owned callback storage and C atomic
  completion markers. Do not recycle error storage before completion.
- Add `Device.Check` and `Queue.TrySubmit`. Check raw command encoding as well
  as submission. Remove `nocallback` hints from checked native calls.
- Disable automatic resource finalizers. The plugin uses explicit ownership
  and `Release`; wrappers cannot be collected during a native operation.
- Return nil for a failed mapped-range pointer so that checked callers can
  receive its error.

The native library needs a separate patch. See
[`wgpu-native-errors.patch`](../../native/wgpu-native-errors.patch). It routes
submission, polling, and mapped-range errors through the device error sink.
The upstream paths aborted the process. Device-drop polling errors are logged
while cleanup proceeds.

The pinned native implementation reports validation errors, allocation errors,
and device loss. It does not accept an Internal error-scope filter. Other
reported operation errors go through validation. The plugin waits for both
successful queue completion and completed error reporting before output commit.

The supported GPU build is Linux with Vulkan. Only Linux/amd64 on the GPU named
in the buffer report has been tested. CPU fallback builds need no native library.
A native driver call can still hang or terminate the process; a Go context is
not a hard native-driver timeout.

Do not regenerate the upstream binding files without reapplying these changes.
This dependency copy is part of the experiment, not a published upstream fix.
