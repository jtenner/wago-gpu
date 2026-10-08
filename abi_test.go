package wagogpu

import (
	"encoding/json"
	"os"
	"testing"
)

func TestCanonicalABI(t *testing.T) {
	var doc struct {
		Imports []struct {
			Name            string
			Params, Results []string
		}
		Elements []struct {
			ID   uint32
			Name string `json:"go_constant"`
			Size uint64 `json:"stored_bytes"`
		} `json:"element_types"`
	}
	b, e := os.ReadFile("spec/abi_v1.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &doc); e != nil {
		t.Fatal(e)
	}
	if len(doc.Imports) != len(bufferImports) {
		t.Fatal("ABI import count")
	}
	names := map[byte]string{0x7f: "i32", 0x7e: "i64", 0x7d: "f32", 0x7c: "f64", 0x6e: "anyref"}
	for idx, entry := range doc.Imports {
		generated := bufferImports[idx]
		if entry.Name != generated.name || len(entry.Params) != len(generated.params) || len(entry.Results) != len(generated.results) {
			t.Fatal("ABI descriptor mismatch", entry.Name)
		}
		for j, p := range entry.Params {
			if names[wasmType(generated.params[j])] != p {
				t.Fatal("parameter mismatch", entry.Name)
			}
		}
		for j, r := range entry.Results {
			if names[wasmType(generated.results[j])] != r {
				t.Fatal("result mismatch", entry.Name)
			}
		}
	}
	for _, e := range doc.Elements {
		if ElementType(e.ID).spec().size != e.Size {
			t.Fatal("storage size", e.Name)
		}
	}
}
