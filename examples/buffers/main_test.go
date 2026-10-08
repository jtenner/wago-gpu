package main

import "testing"

func TestCPUStart(t *testing.T) {
	if err := run(true, true, false); err != nil {
		t.Fatal(err)
	}
}
