//go:build tinygo

package main

import "unsafe"

//go:wasmimport wago_gpu_v1 createBufferPacked
func createBufferPacked(kind, count uint32) uint64

//go:wasmimport wago_gpu_v1 freeBuffer
func freeBuffer(handle uint32) uint32

//go:wasmimport wago_gpu_v1 bindBuffer
func bindBuffer(slot, handle uint32) uint32

//go:wasmimport wago_gpu_v1 getBuffer
func getBuffer(slot uint32) uint32

//go:wasmimport wago_gpu_v1 readBufferF32
func readBufferF32(handle, index uint32) float32

//go:wasmimport wago_gpu_v1 writeBufferF32
func writeBufferF32(handle, index uint32, value float32)

//go:wasmimport wago_gpu_v1 setBuffer32
func setBuffer32(handle, offset, memory, ptr, count uint32) uint32

//go:wasmimport wago_gpu_v1 copyBuffer32
func copyBuffer32(handle, offset, memory, ptr, count uint32) uint32

//go:wasmimport wago_gpu_v1 dispatch
func dispatch(kernel, count uint32) uint32

//export wago_gpu.kernel.double
func double(index uint32) { writeBufferF32(getBuffer(1), index, readBufferF32(getBuffer(0), index)*2) }

//export wago_gpu.cpu.double
func cpuDouble(count uint32) {
	for i := uint32(0); i < count; i++ {
		double(i)
	}
}

var input = [4]float32{1, 2, 3, 4}
var output [4]float32

//export run
func run() uint32 {
	a := createBufferPacked(8, 4)
	if uint32(a>>32) != 0 {
		return uint32(a >> 32)
	}
	b := createBufferPacked(8, 4)
	if uint32(b>>32) != 0 {
		freeBuffer(uint32(a))
		return uint32(b >> 32)
	}
	defer freeBuffer(uint32(a))
	defer freeBuffer(uint32(b))
	if s := bindBuffer(0, uint32(a)); s != 0 {
		return s
	}
	if s := bindBuffer(1, uint32(b)); s != 0 {
		return s
	}
	if s := setBuffer32(uint32(a), 0, 0, uint32(uintptr(unsafe.Pointer(&input[0]))), 4); s != 0 {
		return s
	}
	status := dispatch(1, 4)
	if status == 1 {
		cpuDouble(4)
	} else if status != 0 {
		return status
	}
	if s := copyBuffer32(uint32(b), 0, 0, uint32(uintptr(unsafe.Pointer(&output[0]))), 4); s != 0 {
		return s
	}
	for i, v := range output {
		if v != float32((i+1)*2) {
			return 99
		}
	}
	return status
}
func main() {}
