# sd.cpp on gfx906 with ROCm 10.1

**Radeon Pro VII (gfx906) 上跑 stable-diffusion.cpp HIP 后端 · 2026-10 战报**

官方 ROCm 从 6.x 起不再为 gfx906 编 GPU kernel，官方 sd.cpp 二进制也只编 gfx908+。
本仓库记录**唯一跑通的组合**：社区补丁版 ROCm 10.1 + 自编 sd.cpp master，含实测数据与全部踩坑。

---

## 一、结论先行

| 项目 | 结果 |
|---|---|
| ROCm 版本 | **10.1（社区补丁版，TheRock 构建）** |
| sd.cpp 版本 | master @ `2026-10-29`（`e239e4b0`） |
| HIP / clang | HIP 7.16.26385 / AMD clang 24.0.0git |
| 卡 | Radeon Pro VII 16GB HBM2（gfx906:sramecc+:xnack-） |
| 编译产物 | `sd-cli` 104MB / `sd-server` 105MB |
| **Anima 896×1152 · 8 步 · bf16** | **采样 20.9s / 端到端 24.9s** |
| **Anima 896×1152 · 8 步 · Q8_0** | **采样 17.7s / 端到端 21.6s** |
| 同一张卡 ComfyUI 同参数 | 采样 53s / 端到端 57.1s |

**sd.cpp 比 ComfyUI 快 2.3 倍**（同模型、同卡、同参数），Q8 量化再快 15%。

---

## 二、为什么必须用社区补丁 ROCm

实测四份官方兼容矩阵，**gfx906 全都不在里面**：

| ROCm 官方版本 | 兼容矩阵含 gfx906 | 说明 |
|---|---|---|
| 5.7.1 | ✅ | 最后一版开箱即用 |
| 6.3.3 | ❌ | 最老只到 gfx908 |
| 6.4.0 | ❌ | 同上 |
| 7.0.2 | ❌ | 同上 |
| 10.1.0 | ❌ | 同上 |

官方包不再为 gfx906 生成 GPU kernel，装上去 `rocminfo` 可能认卡但 rocBLAS 没有对应 `.hsaco`，跑起来直接失败。

**解法**：`mixa3607/ML-gfx906` 项目用 TheRock 体系重编了 rocBLAS/RCCL，把 gfx906 kernel 补回去，版本对齐到 10.1：

```bash
# apt 源（Ubuntu 24.04 / noble）
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://s3.arkprojects.space/apt-gfx906/ubuntu/gpg \
  -o /etc/apt/keyrings/apt-gfx906.asc
# sources.list.d/gfx906.sources
Types: deb
URIs: https://s3.arkprojects.space/apt-gfx906/ubuntu
Suites: noble
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/apt-gfx906.asc

apt-get install -y amdrocm-core-sdk10.1-gfx906   # 完整 SDK，含 HIP 编译器
```

验证 kernel 真的在：

```bash
ls /opt/rocm/lib/rocblas/library/*gfx906*
# Kernels.so-000-gfx906.hsaco
# TensileLibrary_Type_4xi8I_HPA_..._gfx906.hsaco
```

> ⚠️ 已知限制：gfx906 有 rocBLAS device library，但**没有 hipBLASLt device library**。
> 影响 INT8（`torch._int_mm`）；sd.cpp 走 hipBLAS/rocBLAS，不受影响。

---

## 三、编译

### 3.1 关键坑：ggml submodule 必须用锁定 commit

sd.cpp 的 `ggml/` 是 submodule（`github.com/leejet/ggml`）。**直接下 master 会编译失败**：

```
error: 'ggml_quantize_i8_convrot' was not declared in this scope
error: 'GGML_SAGE_ATTN_AUTO' was not declared in this scope
```

master 已经不含这些 sd.cpp 需要的符号。必须用 sd.cpp 仓库锁定的那个 commit：

```bash
# 查询锁定 commit
curl -s "https://api.github.com/repos/leejet/stable-diffusion.cpp/contents/ggml?ref=master" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])"
# → 7b2b435da087614be772ee6f5ea80e68eca81de6

curl -L -o ggml-pin.tar.gz \
  "https://github.com/leejet/ggml/archive/7b2b435da087614be772ee6f5ea80e68eca81de6.tar.gz"
rm -rf stable-diffusion.cpp-master/ggml
mkdir -p stable-diffusion.cpp-master/ggml
tar xzf ggml-pin.tar.gz --strip-components=1 -C stable-diffusion.cpp-master/ggml
```

### 3.2 编译命令

```bash
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSD_HIPBLAS=ON \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DSD_WEBP=OFF -DSD_WEBM=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-multiple-definition

cmake --build build-hip -j4
```

产物：`build-hip/bin/sd-cli`、`build-hip/bin/sd-server`

四条历史上必须的修复（ROCm 10.1 下实测只有第 3 条仍需注意）：
1. CPU 侧用 gcc/g++，HIP 侧用 ROCm 裸 clang++（不用 hipcc wrapper）
2. `-DSD_USE_UPSTREAM_GGML` 若启用需同时注入编译宏
3. **ggml submodule 版本必须匹配**（见 3.1）—— 这是本次唯一的失败点
4. `-Wl,--allow-multiple-definition` 吞 bfloat16 重复符号

---

## 四、运行

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib

./sd-cli --list-devices
# ggml_cuda_init: found 1 ROCm devices (Total VRAM: 16368 MiB):
#   Device 0: AMD Radeon Pro VII, gfx906:sramecc+:xnack-, VRAM: 16368 MiB

# Anima 出图（注意 TE 用 --llm，不是 --clip_l）
./sd-cli \
  --diffusion-model anima_dit.safetensors \
  --vae qwen_image_vae.safetensors \
  --llm Qwen3-0.6B-Base.Q4_K_M.gguf \
  -p "your prompt" -n "text, watermark, cartoon" \
  --steps 8 --cfg-scale 1.0 -W 896 -H 1152 \
  --seed 871204 -o out.png
```

**参数名易错点**：
- `--clip_l`（下划线）不是 `--clip-l`
- Anima 的 Qwen3 TE 走 `--llm`；用 `--clip_l` 会报
  `Conditioner model tensor 'text_encoders.llm.model.embed_tokens.weight' not in model metadata`

---

## 五、量化实测（bf16 vs Q8_0）

sd.cpp 自带量化：`-M convert --type q8_0`，4.18G bf16 → 2.1G。

| seed | 格式 | 体积 | 8 步采样 | 端到端 | 总耗时 |
|---|---|---|---|---|---|
| 871204 | bf16 | 4.18G | 20.94s | 24.86s | 27s |
| 871204 | Q8_0 | 2.10G | 17.68s | 21.63s | 23s |
| 12345 | bf16 | 4.18G | 20.90s | 24.91s | 27s |
| 12345 | Q8_0 | 2.10G | 17.67s | 21.62s | 23s |

**Q8 稳定快 15.5%，体积减半。** 两组数据几乎完全重合。

原因：gfx906 的瓶颈在 HBM 带宽而非算力。bf16 权重 4.18G 每步要完整读一遍，Q8 只读一半，
带宽节省超过 kernel 内反量化的开销。

> 注意：这与「显存装不下时才该量化」的常见经验相反——在 gfx906 上带宽容量的收益更早兑现。

---

## 六、与 ComfyUI 的对比

| 指标 | ComfyUI + ROCm 6.3 | sd.cpp + ROCm 10.1 |
|---|---|---|
| 每步 | 6.7 s/it | 2.22 s/it |
| 8 步采样 | 53s | 20.9s |
| 端到端 | 57.1s | 24.9s |
| 骨架控制 | LLLite / Fun-CN ✅ | Z-Image Fun-CN ✅ / Anima LLLite ❌ |

sd.cpp 胜在速度与内存（mmap，权重不全量驻留）；
ComfyUI 胜在生态（Anima LLLite、LoRA 链、workflow）。

**分工建议**：
- 纯文生图、批量 → sd.cpp
- 骨架控制 + LoRA → ComfyUI（Anima LLLite）或 sd.cpp（Z-Image + Fun-CN）

---

## 七、环境固化

为避免容器重建后重装 6.2G ROCm，推荐把 `/opt/rocm` 打包持久化：

```bash
# 打包（容器内 → 挂载目录）
tar -czf /work/rocm-10.1-gfx906.tar.gz -C /opt rocm      # 6.2G → 1.7G

# 恢复（新容器）
tar -xzf rocm-10.1-gfx906.tar.gz -C /opt
```

同时把 deb 归档（`/data/downloads` 或机械盘），需要时可 `dpkg -i` 离线重装。

---

## 八、实测环境

- Ubuntu 26.04 宿主 / Ubuntu 24.04 容器（`--network=host`，挂载 `/dev/kfd` `/dev/dri`）
- i5-4590 4C4T / 16GB DDR3 / Radeon Pro VII 16GB HBM2
- 代理：本地 mihomo `127.0.0.1:7890`（社区 apt 源在国内需代理，直连被 Cloudflare 403）

---

## License

MIT（本文档与脚本）。上游：stable-diffusion.cpp (MIT) · ggml (MIT) · ROCm (AMD) · ML-gfx906 (社区)。