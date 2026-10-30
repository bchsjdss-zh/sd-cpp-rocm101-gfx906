#!/bin/bash
# 在 Ubuntu 24.04 上安装 gfx906 可用的 ROCm 10.1（社区补丁版）
# 用法: sudo bash setup-rocm101.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then echo "需要 root"; exit 1; fi

echo "=== 1/4 基础依赖"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg

echo "=== 2/4 导入 gfx906 源签名"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL --retry 3 https://s3.arkprojects.space/apt-gfx906/ubuntu/gpg \
  -o /etc/apt/keyrings/apt-gfx906.asc
chmod a+r /etc/apt/keyrings/apt-gfx906.asc

echo "=== 3/4 写入 apt 源"
cat > /etc/apt/sources.list.d/gfx906.sources <<EOF
Types: deb
URIs: https://s3.arkprojects.space/apt-gfx906/ubuntu
Suites: noble
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/apt-gfx906.asc
EOF

apt-get update

echo "=== 4/4 安装 ROCm 10.1 完整 SDK（约 910MB）"
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  amdrocm-core-sdk10.1-gfx906

echo "=== 验证"
/opt/rocm/bin/rocminfo | grep -E "gfx906|Marketing" || echo "警告: 未识别到 gfx906"
ls /opt/rocm/lib/rocblas/library/*gfx906* | head -2

echo "完成。运行时需要: export LD_LIBRARY_PATH=/opt/rocm/lib"