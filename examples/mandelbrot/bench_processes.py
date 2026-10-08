#!/usr/bin/env python3
"""Measure nine fresh default commands on Linux; emit JSON on stdout."""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('executable', help='Mandelbrot host built with -tags webgpu')
args = parser.parse_args()
env = dict(os.environ, GOMAXPROCS='1')
results = []
reference = None
for mode, options in [('cpu', []), ('fallback', ['-program', 'buffers', '-cpu']),
                      ('gpu', ['-program', 'buffers', '-require-gpu'])]:
    for trial in range(3):
        command = [args.executable] + options
        # Files prevent a full pipe from blocking the child before wait4.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            start = time.perf_counter_ns()
            process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
            _, status, usage = os.wait4(process.pid, 0)
            elapsed = time.perf_counter_ns() - start
            process.returncode = os.waitstatus_to_exitcode(status)
            out.seek(0)
            err.seek(0)
            image, stderr = out.read(), err.read().decode()
        if process.returncode != 0:
            raise RuntimeError(f'{command} exited {process.returncode}: {stderr}')
        if reference is None:
            reference = image
        if image != reference:
            raise RuntimeError('Default process image differs from CPU')
        results.append(dict(Mode=mode, Trial=trial+1, TotalNS=elapsed,
                            UserSeconds=usage.ru_utime, SystemSeconds=usage.ru_stime,
                            MaxRSSKiB=usage.ru_maxrss, Command=command, GOMAXPROCS=1,
                            Width=96, Height=64, Iterations=32, StdoutBytes=len(image),
                            SHA256=hashlib.sha256(image).hexdigest(), Stderr=stderr))
print(json.dumps(results, indent=2))
