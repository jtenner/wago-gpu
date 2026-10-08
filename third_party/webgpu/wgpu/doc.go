// Package wgpu provides WebGPU bindings for Go.
//
// # Memory ownership in the wago-gpu fork
//
// This fork requires explicit Release calls. Garbage collection does not release
// native resources. Release children before their device, and keep callback
// contexts alive until native code can no longer call them. The wago-gpu backend
// owns this order. Do not use Share: its no-op Release would leak in this fork.
package wgpu
