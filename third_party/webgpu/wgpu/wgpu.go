//go:build !js

package wgpu

import (
	"errors"

	_ "github.com/oliverbestmann/webgpu/libs-linux"
)

/*

// Linux headers are identical for the supported amd64 and arm64 builds.
#cgo linux CFLAGS: -I${SRCDIR}/lib/linux

#include <stdio.h>
#include <wgpu.h>

#ifdef __ANDROID__
#include <android/log.h>
void logCallback_cgo(WGPULogLevel level, char const *msg) {
	switch (level) {
	case WGPULogLevel_Error:
		__android_log_write(ANDROID_LOG_ERROR, "GoLogWGPU", msg);
		break;
	case WGPULogLevel_Warn:
		__android_log_write(ANDROID_LOG_WARN, "GoLogWGPU", msg);
		break;
	default:
		__android_log_write(ANDROID_LOG_INFO, "GoLogWGPU", msg);
		break;
	}
}
#else
void logCallback_cgo(WGPULogLevel level, char const *msg) {
	char const *level_str;
	switch (level) {
	case WGPULogLevel_Error:
		level_str = "Error";
		break;
	case WGPULogLevel_Warn:
		level_str = "Warn";
		break;
	case WGPULogLevel_Info:
		level_str = "Info";
		break;
	case WGPULogLevel_Debug:
		level_str = "Debug";
		break;
	case WGPULogLevel_Trace:
		level_str = "Trace";
		break;
	default:
		level_str = "Unknown Level";
	}
	fprintf(stderr, "[wgpu] [%s] %s\n", level_str, msg);
}
#endif


*/
import "C"

func init() {
	C.wgpuSetLogCallback(C.WGPULogCallback(C.logCallback_cgo), nil)
}

func SetLogLevel(level LogLevel) {
	C.wgpuSetLogLevel(C.WGPULogLevel(level))
}

func GetVersion() Version {
	return Version(C.wgpuGetVersion())
}

func (g *Device) addRef() C.WGPUDevice {
	if g.ref == nil {
		panic(errors.New("device already released"))
	}

	C.wgpuDeviceAddRef(g.ref)

	return g.ref
}

type releaser interface{ release() }

func releaseOnGC[T releaser](value T) T {
	// Local fork: ownership is explicit. Native calls must not race a finalizer.
	return value
}

func releaseNow(value releaser) {
	value.release()
}

// cBool converts the given Go bool to a C.WGPUBool.
func cBool(b bool) C.WGPUBool {
	if b {
		return 1
	}
	return 0
}

// goBool converts the given C.WGPUBool to a Go bool.
func goBool(b C.WGPUBool) bool {
	return b != 0
}
