package wagogpu

import (
	"context"
	"runtime"
)

// A deadline is checked between nonblocking native polls. This cannot interrupt
// a driver that blocks inside a native call. No background polling task survives
// a timeout or resource close.
func waitGPU(ctx context.Context, poll func() (bool, error)) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		ready, err := poll()
		if err != nil {
			return err
		}
		if ready {
			return ctx.Err()
		}
		runtime.Gosched()
	}
}
