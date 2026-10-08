package wagogpu

import (
	"context"
	"reflect"
	"testing"
)

func TestControlledHostRejectsForeignModule(t *testing.T) {
	a, e := NewHost(context.Background(), bufferConfig())
	if e != nil {
		t.Fatal(e)
	}
	defer a.Close(context.Background())
	b, e := NewHost(context.Background(), bufferConfig())
	if e != nil {
		t.Fatal(e)
	}
	defer b.Close(context.Background())
	module, e := a.Compile(bufferFixture(t))
	if e != nil {
		t.Fatal(e)
	}
	defer module.Close()
	if _, e = b.Instantiate(context.Background(), module); e == nil {
		t.Fatal("foreign artifact accepted")
	}
	prep, e := a.Prepare(bufferFixture(t))
	if e != nil {
		t.Fatal(e)
	}
	defer prep.Close()
	if _, ok := reflect.TypeOf(prep).MethodByName("Adopt"); ok {
		t.Fatal("adoption exposed")
	}
	compiled, e := prep.Compile()
	if e != nil {
		t.Fatal(e)
	}
	defer compiled.Close()
	i, e := a.Instantiate(context.Background(), compiled)
	if e != nil {
		t.Fatal(e)
	}
	defer i.Close()
	if v, _ := i.ReadUint32Le(32); v != 1 {
		t.Fatal(v)
	}
}
