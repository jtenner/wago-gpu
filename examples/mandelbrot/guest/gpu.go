//go:build tinygo && gpu

package main

import (
	"unsafe"
)

//go:wasmimport wago_gpu_v1 createBufferPacked
func createBuffer(kind, count uint32) uint64

//go:wasmimport wago_gpu_v1 freeBuffer
func freeBuffer(handle uint32) uint32

//go:wasmimport wago_gpu_v1 bindBuffer
func bindBuffer(slot, handle uint32) uint32

//go:wasmimport wago_gpu_v1 getBuffer
func getBuffer(slot uint32) uint32

//go:wasmimport wago_gpu_v1 readBufferF32
func readBuffer(handle, index uint32) float32

//go:wasmimport wago_gpu_v1 writeBufferF32
func writeBuffer(handle, index uint32, value float32)

//go:wasmimport wago_gpu_v1 setBuffer32
func setBuffer(handle, offset, memory, ptr, count uint32) uint32

//go:wasmimport wago_gpu_v1 copyBuffer32
func copyBuffer(handle, offset, memory, ptr, count uint32) uint32

//go:wasmimport wago_gpu_v1 dispatch
func dispatch(kernel, count uint32) uint32

// One pass of z = z*z+c. Only supported Wasm arithmetic occurs in this body.
// Slots 0/1 contain c; slots 2/3 contain the real/imaginary parts of z.
//
//export wago_gpu.kernel.step
func step(index uint32) {
	x := readBuffer(getBuffer(2), index)
	y := readBuffer(getBuffer(3), index)
	cr := readBuffer(getBuffer(0), index)
	ci := readBuffer(getBuffer(1), index)
	writeBuffer(getBuffer(2), index, x*x-y*y+cr)
	writeBuffer(getBuffer(3), index, 2*x*y+ci)
}

// CPU scratch is allocated once per command and shared with the escape check.
var cr, ci, x, y []float32

//export wago_gpu.cpu.step
func cpuStep(count uint32) {
	if count == 0 {
		return
	}
	// Read current plugin contents once, then compute in native Wasm memory.
	// This runner has the same math as step without per-element host calls.
	handles := [4]uint32{getBuffer(0), getBuffer(1), getBuffer(2), getBuffer(3)}
	check(copyBuffer(handles[0], 0, 0, address(cr), count))
	check(copyBuffer(handles[1], 0, 0, address(ci), count))
	check(copyBuffer(handles[2], 0, 0, address(x), count))
	check(copyBuffer(handles[3], 0, 0, address(y), count))
	for i := uint32(0); i < count; i++ {
		nextX, nextY := x[i]*x[i]-y[i]*y[i]+cr[i], 2*x[i]*y[i]+ci[i]
		x[i], y[i] = nextX, nextY
	}
	check(setBuffer(handles[2], 0, 0, address(x), count))
	check(setBuffer(handles[3], 0, 0, address(y), count))
}

var gpuPasses, cpuPasses uint32

//export gpu_passes
func usedGPU() uint32 { return gpuPasses }

//export cpu_passes
func usedCPU() uint32 { return cpuPasses }

func check(status uint32) {
	if status != 0 {
		panic("buffer operation failed")
	}
}
func address(data []float32) uint32 { return uint32(uintptr(unsafe.Pointer(&data[0]))) }

func render(width, height, iterations uint32) []byte {
	count := width * height
	cr, ci = make([]float32, count), make([]float32, count)
	x, y = make([]float32, count), make([]float32, count)
	escaped := make([]uint32, count)
	pixels := make([]byte, count)
	for row := uint32(0); row < height; row++ {
		for col := uint32(0); col < width; col++ {
			cr[row*width+col], ci[row*width+col] = point(col, row, width, height)
		}
	}
	var buffers [4]uint32
	for slot := range buffers {
		packed := createBuffer(8, count) // F32 type, zero-filled contents.
		check(uint32(packed >> 32))
		buffers[slot] = uint32(packed)
		defer freeBuffer(buffers[slot])
		check(bindBuffer(uint32(slot), buffers[slot]))
	}
	check(setBuffer(buffers[0], 0, 0, address(cr), count))
	check(setBuffer(buffers[1], 0, 0, address(ci), count))
	for pass := uint32(1); pass <= iterations; pass++ {
		switch dispatch(1, count) {
		case 0:
			gpuPasses++
			// The shader subset has no comparison or conditional branch.
			// Reuse the orbit scratch for the guest's first-escape check.
			check(copyBuffer(buffers[2], 0, 0, address(x), count))
			check(copyBuffer(buffers[3], 0, 0, address(y), count))
		case 1:
			cpuStep(count)
			cpuPasses++
		default:
			panic("dispatch failed")
		}
		for i := uint32(0); i < count; i++ {
			if escaped[i] == 0 && x[i]*x[i]+y[i]*y[i] > 4 {
				escaped[i] = pass
			}
		}
	}
	for i, n := range escaped {
		pixels[i] = shade(n, iterations)
	}
	text(2, "Mandelbrot buffers: GPU passes=")
	digits(2, gpuPasses)
	text(2, ", CPU passes=")
	digits(2, cpuPasses)
	text(2, "\n")
	return pixels
}
