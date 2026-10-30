#!/bin/bash
# 出图示例（Anima + Q8_0 DiT）
# 用法: bash run-anima.sh "prompt" [seed] [W] [H]
set -euo pipefail

SD=${SD:-./build-hip/bin/sd-cli}
ROCM=${ROCM:-/opt/rocm}
# 按实际路径改
DIT=${DIT:-/data/models/image/anima/diffusion/anima_q8.gguf}
TE=${TE:-/data/models/image/anima/text_encoder/Qwen3-0.6B-Base.Q4_K_M.gguf}
VAE=${VAE:-/data/models/image/anima/vae/qwen_image_vae.safetensors}

PROMPT=${1:?"需要 prompt"}
SEED=${2:-871204}
W=${3:-896}
H=${4:-1152}

export LD_LIBRARY_PATH=$ROCM/lib:$LD_LIBRARY_PATH

S=$(date +%s)
"$SD" \
  --diffusion-model "$DIT" \
  --vae "$VAE" \
  --llm "$TE" \
  -p "$PROMPT" \
  -n "text, watermark, cartoon, deformed, extra limbs, bad anatomy" \
  --steps 8 --cfg-scale 1.0 \
  -W "$W" -H "$H" \
  --seed "$SEED" \
  -o "out_$(date +%H%M%S).png"
E=$(date +%s)
echo "=== 总耗时 $((E-S)) 秒"

# 量化对照（把 bf16 转 Q8）
# "$SD" -M convert --diffusion-model anima_bf16.safetensors --type q8_0 -o anima_q8.gguf