package main

import "testing"

func TestCPU(t *testing.T) {
	for _, kind := range []string{"f16", "memory64", "gc"} {
		t.Run(kind, func(t *testing.T) {
			if e := run(kind, true, false); e != nil {
				t.Fatal(e)
			}
		})
	}
}
