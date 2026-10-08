#!/usr/bin/env python3
"""Check the draft specification. This does not test plugin execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def run(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(' '.join(args) + '\n' + result.stderr)
    return result.stdout


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tinygo', action='store_true', help='rebuild the compile-only guest probe')
    args = parser.parse_args()
    doc = (ROOT / 'BUFFER_API_PROPOSAL.md').read_text()
    abi = json.loads((ROOT / 'spec/abi_v1.json').read_text())
    probe = (ROOT / 'spec/guest-tinygo.go.txt').read_text()
    checks = []
    imports = {i['name']: i for i in abi['imports']}
    check(len(imports) == len(abi['imports']) == abi['import_count'] == 34, 'import count/uniqueness')
    check(abi['module'] == 'wago_gpu_v1' and abi['status'] == 'draft-unfrozen', 'ABI identity')
    check([t['id'] for t in abi['element_types']] == list(range(1, 12)), 'type IDs')
    check([t['value'] for t in abi['statuses']] == list(range(13)), 'status values')
    constants = [t['go_constant'] for t in abi['element_types'] + abi['statuses']]
    check(len(set(constants)) == 24, 'Go constant names')
    for t in abi['element_types']:
        row = rf"^\| {t['id']} \| `{t['suffix']}` \| {t['stored_bytes']} \| `{t['wasm_scalar']}` \|"
        check(re.search(row, doc, re.M), 'type table: ' + t['suffix'])
        check('`' + t['go_constant'] + '`' in doc, 'type constant: ' + t['go_constant'])
        for prefix in ('readBuffer', 'writeBuffer'):
            entry = imports[prefix + t['suffix']]
            expected = (['i32', 'i32'], [t['wasm_scalar']]) if prefix == 'readBuffer' else (['i32', 'i32', t['wasm_scalar']], [])
            check((entry['params'], entry['results']) == expected, 'typed access signature: ' + entry['name'])
    for status in abi['statuses']:
        check(re.search(rf"^\| {status['value']} \| `{status['name']}` \|", doc, re.M), 'status table: ' + status['name'])
    management = re.findall(r'^\| `(createBuffer|createBufferPacked|freeBuffer|bindBuffer|getBuffer|dispatch)` \| `\(([^)]*)\) -> ([^`]+)`', doc, re.M)
    check(len(management) == 6, 'management signature table')
    for name, params, results in management:
        want = (re.findall(r': (i32|i64)', params), re.findall(r': (i32|i64)', results))
        check((imports[name]['params'], imports[name]['results']) == want, 'management signature: ' + name)
    transfers = re.findall(r'(setBuffer(?:32|64|GC)|copyBuffer(?:32|64|GC))\((.*?)\) -> status: i32', doc, re.S)
    check(len(transfers) == 6, 'bulk transfer signature declarations')
    for name, params in transfers:
        check((imports[name]['params'], imports[name]['results']) == (re.findall(r': (i32|i64|anyref)', params), ['i32']), 'bulk signature: ' + name)
    checks.append('All 34 import signatures and the type/status tables agree with the document; Go constant names are unique.')

    for name in ('BUFFER_API_PROPOSAL.md', 'README.md'):
        path = ROOT / name
        text = path.read_text()
        check(text.endswith('\n') and text.count('```') % 2 == 0, name + ': fences/newline')
        check(all(line == line.rstrip() for line in text.splitlines()), name + ': trailing whitespace')
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if link.startswith(('https://', 'http://')):
                continue
            target, _, anchor = link.partition('#')
            linked = path.parent / target if target else path
            check(linked.exists(), 'missing local link: ' + link)
            if anchor:
                headings = re.findall(r'^#{1,6}\s+(.+)$', linked.read_text(), re.M)
                slugs = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}
                check(anchor in slugs, 'missing anchor: ' + link)
    check([int(n) for n in re.findall(r'^## (\d+)\.', doc, re.M)] == list(range(1, 32)), 'sections 1-31')
    checks.append('Local links, heading anchor, code fences, whitespace, and sections 1-31 pass.')

    with tempfile.TemporaryDirectory(prefix='wago-gpu-spec-') as temporary:
        temp = Path(temporary)
        declarations = []
        for item in abi['imports']:
            params = '(param ' + ' '.join(item['params']) + ')' if item['params'] else ''
            results = '(result ' + ' '.join(item['results']) + ')' if item['results'] else ''
            declarations.append(f'(import "{abi["module"]}" "{item["name"]}" (func {params} {results}))')
        (temp / 'abi.wat').write_text('(module\n' + '\n'.join(declarations) + '\n)\n')
        run(['wasm-tools', 'parse', str(temp / 'abi.wat'), '-o', str(temp / 'abi.wasm')])
        run(['wasm-tools', 'validate', str(temp / 'abi.wasm')])
        checks.append('All 34 proposed signatures form a valid import-only Wasm module, including GC reference and multi-result types.')

        blocks = re.findall(r'```wat\n(.*?)\n```', doc, re.S)
        check(len(blocks) == 1, 'one complete WAT example')
        matched = re.findall(r'\(import "wago_gpu_v1" "(\w+)"\s+\(func \$\w+ \(param ([^)]+)\)(?: \(result ([^)]+)\))?\)\)', blocks[0])
        check(len(matched) == 9, 'WAT example import count')
        for name, params, results in matched:
            check((imports[name]['params'], imports[name]['results']) == (params.split(), results.split()), 'WAT signature: ' + name)
        (temp / 'example.wat').write_text(blocks[0] + '\n')
        run(['wat2wasm', str(temp / 'example.wat'), '-o', str(temp / 'example.wasm')])
        run(['wasm-validate', str(temp / 'example.wasm')])
        checks.append('Complete proposed WAT example passes syntax/type validation; all nine imports match the canonical ABI.')

        for block in re.findall(r'```go\n(.*?)\n```', doc, re.S):
            if block.startswith('package '):
                source = block + '\n'
            elif block.startswith('p, err :='):
                source = 'package doccheck\nfunc example() error {\n' + block + '\nreturn nil\n}\n'
            else:
                source = 'package doccheck\n' + block + '\n'
            run(['gofmt'], input=source)
        checks.append('All Go snippets parse. This does not type-check unimplemented plugin APIs.')

        formula = abi['packed_result']['go_encoding']
        check(formula == '(uint64(uint32(status)) << 32) | uint64(uint32(handle))' and formula in doc, 'packed formula')
        check(abi['packed_result']['handle_bits'] == [0, 31] and abi['packed_result']['status_bits'] == [32, 63], 'packed layout')
        source = '''package main
func pack(handle, status int32) uint64 { return FORMULA }
func main() {
    cases := []struct{ h, s uint32 }{{0, 6}, {1, 0}, {0x80000001, 0}, {0xffffffff, 0}, {0, 12}}
    for _, c := range cases {
        got := pack(int32(c.h), int32(c.s))
        if uint32(got) != c.h || uint32(got >> 32) != c.s { panic("packed result mismatch") }
    }
}
'''.replace('FORMULA', formula)
        (temp / 'packed.go').write_text(source)
        run(['go', 'run', str(temp / 'packed.go')], cwd=ROOT)
        checks.append('Canonical packed-result Go formula round-trips sign-bit handles and failure statuses.')

        mapping = {'uint32': 'i32', 'uint64': 'i64', 'float32': 'f32', 'float64': 'f64'}
        found = re.findall(r'//go:wasmimport wago_gpu_v1 (\w+)\nfunc \w+\(([^)]*)\)\s*(\w*)', probe)
        check(len(found) == 4, 'TinyGo source import count')
        for name, params, result in found:
            types = []
            pending = 0
            for group in params.split(','):
                words = group.split()
                pending += 1
                if len(words) == 2:
                    types.extend([mapping[words[1]]] * pending)
                    pending = 0
            check(pending == 0, 'incomplete Go parameter group')
            results = [mapping[result]] if result else []
            check((types, results) == (imports[name]['params'], imports[name]['results']), 'TinyGo import: ' + name)
        checks.append('Four TinyGo source import declarations match the canonical ABI.')
        if args.tinygo:
            (temp / 'guest.go').write_text(probe)
            run(['tinygo', 'build', '-target=wasm-unknown', '-scheduler=none', '-panic=trap', '-gc=leaking', '-opt=2', '-no-debug', '-o', str(temp / 'guest.wasm'), str(temp / 'guest.go')], cwd=ROOT)
            run(['wasm-validate', str(temp / 'guest.wasm')])
            wat = run(['wasm2wat', str(temp / 'guest.wasm')])
            body = wat.split('(func $wago_gpu.kernel.double ', 1)[1].split('\n  (func ', 1)[0]
            check('i32.load' not in body and 'f32.add' in body and 'call $main.readBufferF32' in body, 'TinyGo kernel shape')
            check('(param i32)' in body and '(local f32)' in body, 'TinyGo kernel signature/locals')
            checks.append('TinyGo probe rebuilt and validated; inspected selected body has the expected compatible form. No plugin execution ran.')

    report = json.loads((ROOT / 'results/improvements-single.json').read_text())
    rows = re.findall(r'^\| `(2\*x|x\*x\+1)` \| ([\d,]+) \| ([\d.]+) \| ([\d.]+) \|$', doc, re.M)
    check(len(rows) == 8, 'historical benchmark row count')
    for name, size, cpu, gpu in rows:
        case = next(c for c in report['Cases'] if c['Kernel'] == {'2*x': 'twice', 'x*x+1': 'square'}[name] and c['Elements'] == int(size.replace(',', '')))
        check(f"{case['CPU']['MedianNS']/1e6:.4f}" == cpu and f"{case['GPU']['MedianNS']/1e6:.4f}" == gpu, 'historical figures')
    checks.append('Eight historical benchmark rows match the saved JSON. No benchmark was rerun.')
    files = ['BUFFER_API_PROPOSAL.md', 'spec/abi_v1.json', 'spec/guest-tinygo.go.txt', 'spec/check_spec.py']
    evidence = {
        'scope': 'Draft specification validation; no proposed plugin runtime or hardware acceptance',
        'result': 'passed',
        'sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
        'tools': {'go': run(['go', 'version']).strip(), 'wabt': run(['wat2wasm', '--version']).strip(), 'wasm_tools': run(['wasm-tools', '--version']).strip()},
        'checks': checks,
        'not_run': ['Production v1 host registration and generated guest-binding comparison: not implemented.', 'Proposed buffer CPU/fake-backend and real Wago interaction tests.', 'Native callback repair/error-protocol tests.', 'Full guest management execution.', 'Real GPU execution and new benchmarks.'],
    }
    if args.tinygo:
        evidence['tools']['tinygo'] = run(['tinygo', 'version']).strip()
    else:
        evidence['not_run'].append('TinyGo recompilation: pass --tinygo to include it.')
    (ROOT / 'spec/validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f"PASS: {len(checks)} specification checks; report: spec/validation.json")


if __name__ == '__main__':
    main()
