#!/bin/bash
# 编译 stable-diffusion.cpp 的 gfx906 HIP 后端
# 用法: bash build-sdcpp.sh [源码目录]
set -euo pipefail

SRC=${1:-stable-diffusion.cpp-master}
ROCM=${ROCM:-/opt/rocm}
BUILD="$SRC/build-hip"

echo "=== 0/3 检查 ROCm 编译器"
[ -x "$ROCM/llvm/bin/clang++" ] || { echo "缺 $ROCM/llvm/bin/clang++，先跑 setup-rocm101.sh"; exit 1; }

echo "=== 1/3 修正 ggml submodule（关键）"
# sd.cpp 的 ggml 必须是仓库锁定的 commit，master 会缺 i8_convrot / sage_attn 符号
PIN=$(curl -s "https://api.github.com/repos/leejet/stable-diffusion.cpp/contents/ggml?ref=master" \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])")
echo "锁定 commit: $PIN"
if [ ! -f "$SRC/ggml/include/ggml.h" ] || ! grep -q "ggml_quantize_i8_convrot" "$SRC/ggml/include/ggml.h"; then
  echo "ggml 缺失或不匹配，下载 $PIN"
  curl -L -o /tmp/ggml-pin.tar.gz \
    "https://github.com/leejet/ggml/archive/${PIN}.tar.gz"
  rm -rf "$SRC/ggml"; mkdir -p "$SRC/ggml"
  tar xzf /tmp/ggml-pin.tar.gz --strip-components=1 -C "$SRC/ggml"
  rm -f /tmp/ggml-pin.tar.gz
fi

echo "=== 2/3 配置"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_HIP_COMPILER="$ROCM/llvm/bin/clang++" \
  -DSD_HIPBLAS=ON \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DSD_WEBP=OFF -DSD_WEBM=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-multiple-definition

echo "=== 3/3 编译（i5-4590 四核约 25 分钟）"
cmake --build "$BUILD" -j"$(nproc)"

echo "=== 产物"
ls -lh "$BUILD/bin/"
echo "运行前: export LD_LIBRARY_PATH=$ROCM/lib"