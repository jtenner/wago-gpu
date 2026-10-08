// Package fixtures contains real Wasm guests and common demonstration checks.
package fixtures

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"

	wago "github.com/wago-org/wago"
)

//go:generate wat2wasm heavy.wat -o heavy.wasm
//go:generate wat2wasm twice.wat -o twice.wasm
//go:generate wat2wasm square.wat -o square.wasm
//go:embed twice.wasm
var Twice []byte

//go:embed square.wasm
var Square []byte

//go:embed heavy.wasm
var Heavy []byte

func Prepare(in *wago.Instance, n uint32) error {
	pages := (uint64(n)*12 + 65535) / 65536
	if pages > 1 {
		v, e := in.Invoke("reserve", pages-1)
		if e != nil {
			return e
		}
		if wago.AsI32(v[0]) < 0 {
			return fmt.Errorf("memory.grow failed")
		}
	}
	return Fill(in, n)
}

func Fill(in *wago.Instance, n uint32) error {
	// Keep host setup memory bounded, including the ten-million-element case.
	buf := make([]byte, 64<<10)
	for first := uint32(0); first < n; {
		count := min(uint32(len(buf)/4), n-first)
		for j := uint32(0); j < count; j++ {
			v := float32(int32((first+j)%2049)-1024) / 32
			binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(v))
		}
		if !in.Write(first*4, buf[:count*4]) {
			return fmt.Errorf("input write failed")
		}
		first += count
	}
	return nil
}

// Verify compares two separate guest result arrays, without a large host copy.
// The supplied test inputs give exact results for both demonstration functions.
func Verify(in *wago.Instance, n uint32) error {
	for first := uint32(0); first < n*4; {
		length := min(uint32(64<<10), n*4-first)
		cpu, ok := in.Read(n*4+first, length)
		if !ok {
			return fmt.Errorf("CPU read failed")
		}
		gpu, ok := in.Read(n*8+first, length)
		if !ok {
			return fmt.Errorf("GPU read failed")
		}
		if !bytes.Equal(cpu, gpu) {
			return fmt.Errorf("result mismatch in byte range [%d,%d)", first, first+length)
		}
		first += length
	}
	return nil
}

// VerifyKernel reports the largest absolute error. Multi-pass square and the heavier demonstration
// accepts finite f32 roundoff: 8e-6 * passes * max(1, abs(CPU)). This tolerance
// applies only to these bounded demonstration inputs, not arbitrary Wasm f32.
func VerifyKernel(in *wago.Instance, n uint32, kernel string, passes uint32) (float64, error) {
	if kernel == "twice" || (kernel == "square" && passes == 1) {
		return 0, Verify(in, n)
	}
	var largest float64
	for first := uint32(0); first < n*4; {
		length := min(uint32(64<<10), n*4-first)
		cpu, ok := in.Read(n*4+first, length)
		if !ok {
			return 0, fmt.Errorf("CPU read failed")
		}
		gpu, ok := in.Read(n*8+first, length)
		if !ok {
			return 0, fmt.Errorf("GPU read failed")
		}
		for j := uint32(0); j < length; j += 4 {
			a := float64(math.Float32frombits(binary.LittleEndian.Uint32(cpu[j:])))
			b := float64(math.Float32frombits(binary.LittleEndian.Uint32(gpu[j:])))
			delta := math.Abs(a - b)
			largest = max(largest, delta)
			if math.IsNaN(delta) || math.IsInf(a, 0) || math.IsInf(b, 0) || delta > 8e-6*float64(passes)*max(1, math.Abs(a)) {
				return largest, fmt.Errorf("element %d: CPU=%g GPU=%g", (first+j)/4, a, b)
			}
		}
		first += length
	}
	return largest, nil
}
