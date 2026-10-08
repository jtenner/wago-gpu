//go:build !js

package wgpu

/*
#include "gen_wgpu_wrappers.h"
static inline void checked_finish(WGPUDevice dev, char *err) {
 WGPUPopErrorScopeCallbackInfo cb = {.callback=webgpu_error_callback,.userdata1=err};
 checked_pop(dev, cb);
}
*/
import "C"
import "errors"

// Check runs fn under validation and allocation error scopes.
// Pinned wgpu-native routes non-device-loss errors to these two categories.
// It returns only after all scope callbacks completed. Callers must serialize
// device access. The native driver entry points can block.
func (g *Device) Check(fn func() error) (err error) {
	cb := acquireErrorCallback()
	defer cb.Done()
	C.checked_push(g.ref)
	defer func() { C.checked_finish(g.ref, cb.ToPointer()); err = errors.Join(err, cb.ToError()) }()
	return fn()
}
func (q *Queue) TrySubmit(commands ...*CommandBuffer) error {
	refs, slice := callocSlice[C.WGPUCommandBuffer](len(commands))
	defer free(refs)
	for i, c := range commands {
		slice[i] = c.ref
	}
	cb := acquireErrorCallback()
	defer cb.Done()
	C.go_wgpuQueueSubmit(q.device, cb.ToPointer(), q.ref, C.size_t(len(commands)), refs)
	return cb.ToError()
}
