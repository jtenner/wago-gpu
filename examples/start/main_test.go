package main

import (
	"os"
	"testing"
)

func TestCPUFallbackDuringStart(t *testing.T) {
	if err := run(true, false); err != nil {
		t.Fatal(err)
	}
}

func TestHardwareDuringStart(t *testing.T) {
	if os.Getenv("WAGO_GPU_TEST") != "1" {
		t.Skip("set WAGO_GPU_TEST=1 with -tags webgpu to require a hardware GPU")
	}
	if err := run(false, true); err != nil {
		t.Fatal(err)
	}
}
