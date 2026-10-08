//go:build !webgpu || !cgo

package wagogpu

import "fmt"

func openBackend() (backend, error) {
	return nil, fmt.Errorf("WebGPU backend not built; use go run -tags webgpu")
}
