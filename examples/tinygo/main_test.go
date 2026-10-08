//go:build !tinygo

package main

import "testing"

func TestCompleteCPU(t *testing.T) {
	if e := run(true, false); e != nil {
		t.Fatal(e)
	}
}
