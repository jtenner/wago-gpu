#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
build_dir="$project_root/native/build"
revision=d2e3330ade4ae1bb238d76b485926f067e7ee64c
if [ ! -d "$build_dir/.git" ]; then
  git clone https://github.com/gfx-rs/wgpu-native.git "$build_dir"
  git -C "$build_dir" checkout "$revision"
  git -C "$build_dir" submodule update --init ffi/webgpu-headers
fi
if [ "$(git -C "$build_dir" rev-parse HEAD)" != "$revision" ]; then
  echo 'Unexpected native source revision' >&2
  exit 1
fi
if git -C "$build_dir" apply --check "$project_root/native/wgpu-native-errors.patch"; then
  git -C "$build_dir" apply "$project_root/native/wgpu-native-errors.patch"
else
  git -C "$build_dir" apply --reverse --check "$project_root/native/wgpu-native-errors.patch"
fi
mkdir -p "$project_root/.cache/tmp"
TMPDIR="$project_root/.cache/tmp" CC="${CC:-/usr/bin/cc}" CXX="${CXX:-/usr/bin/c++}" \
  cargo build --manifest-path "$build_dir/Cargo.toml" --release --locked \
  --no-default-features --features vulkan,wgsl,spirv,glsl
case "$(uname -m)" in
 x86_64) architecture=amd64 ;;
 aarch64) architecture=arm64 ;;
 *) echo 'This build supports Linux amd64 and arm64 only' >&2; exit 1 ;;
esac
install -D "$build_dir/target/release/libwgpu_native.a" \
  "$project_root/third_party/webgpu/libs-linux/$architecture/libwgpu_native.a"
