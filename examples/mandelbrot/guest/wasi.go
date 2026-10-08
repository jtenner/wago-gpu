//go:build tinygo

package main

import "unsafe"

// This small WASI I/O layer needs no scheduler. The math files do not need to
// know about guest pointers, argument storage, or fd_write iovecs.
//
//go:wasmimport wasi_snapshot_preview1 args_sizes_get
func argsSizes(argc, bytes uint32) uint32

//go:wasmimport wasi_snapshot_preview1 args_get
func argsGet(argv, buffer uint32) uint32

//go:wasmimport wasi_snapshot_preview1 fd_write
func fdWrite(fd, iovecs, count, written uint32) uint32

func pointer(p unsafe.Pointer) uint32 { return uint32(uintptr(p)) }

func arguments() []string {
	var count, size uint32
	if argsSizes(pointer(unsafe.Pointer(&count)), pointer(unsafe.Pointer(&size))) != 0 {
		panic("cannot read arguments")
	}
	if count == 0 || count > 4 || size == 0 || size > 4096 {
		panic("usage: mandelbrot [width height iterations]")
	}
	argv := make([]uint32, count)
	data := make([]byte, size)
	base := pointer(unsafe.Pointer(&data[0]))
	if argsGet(pointer(unsafe.Pointer(&argv[0])), base) != 0 {
		panic("cannot read arguments")
	}
	result := make([]string, count)
	for i, p := range argv {
		if p < base || p-base >= size {
			panic("invalid argument address")
		}
		start := p - base
		end := start
		for end < size && data[end] != 0 {
			end++
		}
		if end == size {
			panic("unterminated argument")
		}
		result[i] = string(data[start:end])
	}
	return result
}

func write(fd uint32, data []byte) {
	for len(data) != 0 {
		vector := [2]uint32{pointer(unsafe.Pointer(&data[0])), uint32(len(data))}
		var written uint32
		if fdWrite(fd, pointer(unsafe.Pointer(&vector[0])), 1, pointer(unsafe.Pointer(&written))) != 0 || written == 0 || written > uint32(len(data)) {
			panic("cannot write output")
		}
		data = data[written:]
	}
}
func text(fd uint32, value string) { write(fd, []byte(value)) }
func digits(fd, value uint32) {
	var data [10]byte
	start := len(data)
	for {
		start--
		data[start] = '0' + byte(value%10)
		value /= 10
		if value == 0 {
			break
		}
	}
	write(fd, data[start:])
}
