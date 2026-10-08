//go:build webgpu && cgo && linux

package main

import (
	"bytes"
	"os"
	"testing"
)

func TestHardwareMandelbrot(t *testing.T) {
	if os.Getenv("WAGO_GPU_TEST") != "1" {
		t.Skip("set WAGO_GPU_TEST=1 to require hardware")
	}
	for _, size := range [][3]uint32{{19, 17, 12}, {32, 24, 16}} {
		want := image(t, "cpu", true, false, size[0], size[1], size[2])
		got := image(t, "buffers", false, true, size[0], size[1], size[2])
		if !bytes.Equal(got, want) {
			t.Fatal("hardware image differs from direct CPU for the tested grid", size)
		}
	}
}
