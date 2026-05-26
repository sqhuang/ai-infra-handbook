# 关键数字速查

> AI infra 工程师面试与日常估算高频用到的"硬件 / 软件 / 业务"数字，按 2025–2026 年生态整理。每条尽量给出"数量级 + 典型范围 + 出处或场景"，避免精确到末位的伪精度。所有数字仅供估算，正式决策需以厂商最新数据为准。

## 计算精度

| 精度 | 比特 | 表示形式 | 主要用途 | 备注 |
| --- | ---: | --- | --- | --- |
| FP32 | 32 | E8M23 | 累加 / master weight | 训练 master weight、梯度累加 |
| TF32 | 19 (实) | E8M10 | Ampere 默认 GEMM | TF32 没有 19-bit 存储格式，只是计算路径 |
| BF16 | 16 | E8M7 | 训练默认 | 与 FP32 同 exponent，dynamic range 一致 |
| FP16 | 16 | E5M10 | 旧训练默认 | dynamic range 小，需 loss scaling |
| FP8 (E4M3) | 8 | E4M3 | 训练前向 / 推理 | Hopper+ 原生，weight/activation 主用 |
| FP8 (E5M2) | 8 | E5M2 | 梯度 | range 更大但精度低，gradient 路径 |
| MXFP4 / NVFP4 | 4 (+block scale) | E2M1 | Blackwell 训练 / 推理 | 32 元素一组 shared scale |
| INT8 | 8 | 整数 | 推理量化 | per-channel scale 典型 |
| INT4 | 4 | 整数 | 推理量化 | per-group/channel scale，激进量化 |

精度选择经验：训练 BF16 (master FP32) → 加 FP8 → B200 上加 FP4。推理 FP16/BF16 → FP8 → INT8/INT4 → 端侧 INT4/Q4_K_M。

## GPU 算力与带宽（旗舰代次）

| 卡 | TF32 / BF16 TFLOPs | FP8 TFLOPs | FP4 TFLOPs | HBM 容量 | HBM 带宽 | NVLink 双向 | TDP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V100 | ~125 | – | – | 32GB HBM2 | 0.9 TB/s | 300 GB/s | 300W |
| A100 80G | ~312 | – | – | 80GB HBM2e | 2.0 TB/s | 600 GB/s | 400W |
| H100 SXM | ~990 | ~1980 | – | 80GB HBM3 | 3.35 TB/s | 900 GB/s | 700W |
| H200 SXM | ~990 | ~1980 | – | 141GB HBM3e | 4.8 TB/s | 900 GB/s | 700W |
| B200 SXM | ~2250 | ~4500 | ~9000 | 192GB HBM3e | 8 TB/s | 1.8 TB/s | 1000W |
| GB200 (2× B200) | ~4500 | ~9000 | ~18000 | 384GB HBM3e | 16 TB/s | 1.8 TB/s | 2700W (含 Grace) |
| MI300X | ~1300 | ~2600 | – | 192GB HBM3 | 5.3 TB/s | 896 GB/s (Infinity) | 750W |
| TPU v5p | ~459 (bf16) | ~918 (int8) | – | 96GB HBM | 2.8 TB/s | – | – |

口诀：H100 比 A100 算力翻 3x、HBM 带宽涨 67%。B200 比 H100 算力翻 ~2x、HBM 翻 2.4x。FP4 在 B200 上又翻 ~2x。

## 内存层级带宽 / 延迟

| 层级 | 典型带宽 | 典型延迟 | 备注 |
| --- | ---: | ---: | --- |
| Register / Shared mem | 数 TB/s | 1-10 cycle | 单 SM 内 |
| L1 / L2 cache | TB/s 量级 | 10-100 cycle | 跨 SM 共享 L2 |
| HBM3e | 4-8 TB/s | ~500ns | 单卡显存 |
| NVLink 5 (B200) | 1.8 TB/s 双向 | ~1µs | 同 NVL 域 |
| PCIe 5 x16 | 128 GB/s 双向 | 几 µs | CPU↔GPU |
| InfiniBand NDR 400G | 50 GB/s | ~2µs RDMA | 跨机 |
| InfiniBand XDR 800G | 100 GB/s | ~2µs RDMA | 2025+ 主流 |
| 25/100 Gbps ethernet | 3-12 GB/s | µs-ms | 普通 LAN |
| NVMe SSD | 5-12 GB/s | 50-100 µs | 节点本地盘 |
| Lustre / GPFS | 10-100 GB/s 聚合 | ms | 并行文件系统 |
| S3 | 1-10 GB/s 单连接 | 10-100ms | 对象存储 |

记忆窍门：每跨一级介质带宽降 10-50x、延迟涨 10-100x。

## 模型规模与显存估算

### 权重显存（仅参数）
- BF16：每参数 2 字节
- FP8：每参数 1 字节
- FP4 (含 scale)：每参数 ~0.6 字节

例：70B 模型 BF16 ≈ 140GB；FP8 ≈ 70GB；FP4 ≈ 42GB。

### 训练总显存（粗略）
单卡训练（无任何并行）：
- 模型权重：1×
- 梯度：1×
- 优化器状态（Adam）：4× (m, v, FP32 master)
- 激活：1-2× （依赖 sequence / batch）

合计 7-8× 模型大小。70B BF16 单卡训练理论 ~560GB——单卡无解，必须 FSDP / TP。

### FSDP / ZeRO-3 + 激活重计算
- 权重切片：1× / world_size
- 梯度切片：1× / world_size
- 优化器切片：4× / world_size
- 激活：保留 attention 边界，~0.3-0.5× 模型大小

256 张 H100 训 70B：每张卡 ~7GB 状态 + ~30GB 激活 = ~37GB（FP8 状态时更省）。

### KV cache（推理）
每 token KV size：2 (K+V) × n_layer × n_kv_head × head_dim × dtype_bytes。
- Llama-3-70B FP16：~140 KB / token
- Llama-3-70B FP8：~70 KB / token
- DeepSeek-V3 (MLA) FP16：~10 KB / token（MLA 压缩）

例：70B FP16 / 16k context / batch 32 ≈ 70GB KV。

## 通信原语量级

| 操作 | 数据量 | 在 NVL8 内耗时 | 跨机 IB 400G |
| --- | --- | --- | --- |
| all-reduce 1MB | 1MB | ~100µs | ~500µs |
| all-reduce 1GB | 1GB | ~5ms | ~30ms |
| all-reduce 10GB | 10GB | ~50ms | ~300ms |
| all-to-all (MoE EP=8) per layer | depends | 50-200µs | 1-5ms |
| broadcast 70B weight | 140GB | ~80ms | ~3s |

口诀：NVLink 内通信比 IB 快 5-10x；NVL72 内通信带宽相当于 IB 36x。

## Kernel 启动 / 调度量级

| 事件 | 典型量级 |
| --- | --- |
| Python overhead per op | 1-10µs |
| CUDA kernel launch | 5-10µs |
| CUDA Graph 复用 | <1µs |
| NCCL collective launch | 几 µs |
| FA-2 attention call | 几十 µs ~ ms（depends N） |
| 一次 H100 forward (Llama 70B, 1 token) | ~25-35ms |
| 一次 H100 prefill (Llama 70B, 1k token) | ~150ms |

CUDA Graph 在 decode 阶段（高频小 kernel）能省 30-50% Python overhead。

## 推理吞吐与延迟（参考）

| 配置 | TTFT (1k prompt) | TPOT (decode) | tokens/s/replica |
| --- | --- | --- | --- |
| 7B Llama, 1× H100 FP8 | <100ms | 5-10ms | 2000-4000 |
| 13B Llama, 1× H100 FP8 | <150ms | 8-15ms | 1500-3000 |
| 70B Llama, TP=4 H100 FP8 | <300ms | 25-40ms | 1500-2500 |
| 70B Llama, TP=8 H200 FP8 | <250ms | 20-35ms | 2500-4000 |
| 70B Llama, TP=8 B200 FP4 | <150ms | 15-25ms | 5000-8000 (估) |
| 671B MoE (DeepSeek-V3), 256 GPU | ~500ms | ~25ms | 大集群高吞吐 |

说明：以上是 vLLM/SGLang 类引擎在中等 batch 下的常见量级，具体随 batch、prompt 长度、KV cache 量化等差异显著。

## 训练吞吐与时间（参考）

| 模型 | 集群 | tokens/s/GPU | MFU |
| --- | --- | --- | --- |
| Llama 7B BF16 | 64× A100 | ~10k | ~35% |
| Llama 70B BF16 | 256× A100 | ~1.5k | ~45% |
| Llama 70B FP8 | 256× H100 | ~3k | ~50% |
| Llama 70B FP8 | 256× H200 | ~3.5k | ~52% |
| Llama 405B FP8 | 1024× H100 | ~1k | ~45% |
| MoE 671B FP8 | 2048× H800 | ~2k | – (HFU) |

预训量估算：训 70B 一万亿 token (~1T tokens) 在 256× H100 FP8 ~14 天；同样规模在 256× B200 FP4 缩到 ~5 天（估）。

## 业务单位经济学

| 指标 | 典型范围 (2025-2026) | 备注 |
| --- | --- | --- |
| 70B 推理 token 成本 | $0.5-2 / 1M token | 中等并发 |
| 7B 推理 token 成本 | $0.05-0.2 / 1M token | 中等并发 |
| H100 现货月租 | $1.5-2.5 / 小时 | 长期合约更低 |
| B200 现货月租 | $3-5 / 小时 | 早期紧俏 |
| 训 70B 一次端到端预算 | $1-5M | 数据 + 算力 |
| 训 405B 一次端到端预算 | $30-100M | 含 SFT/RL |

## 故障与运维

| 事件 | 典型频率 / 影响 |
| --- | --- |
| ECC corrected error | 每 GPU 每月数次（正常） |
| ECC uncorrected (XID 48/63) | 每千 GPU 每月数次（不正常需更换） |
| GPU hang / fall off bus | 每千 GPU 每月数次 |
| NCCL timeout | 每大集群每天数次（多为单卡问题） |
| 节点掉网 | 每千节点每周数次 |
| 千卡训练 24 小时不挂的概率 | ~70-90% (调优后) |
| 万卡训练 24 小时不挂的概率 | <30% (必须自动恢复) |

口诀：千卡及以上规模训练必须有 ckpt + 自动恢复 + 节点排查自动化。

## 长上下文参考

| 上下文长度 | 单 sample KV (70B FP16) | 推荐并行 |
| --- | --- | --- |
| 8k | 1.1 GB | 单卡 / TP=2 |
| 32k | 4.4 GB | TP=4-8 |
| 128k | 17.5 GB | TP=8 + KV 量化 |
| 512k | 70 GB | TP=8 + CP=2 + KV-FP8 |
| 1M | 140 GB | TP=8 + CP=4 + KV-FP8 |

训练长上下文：1M context 1 step time 约是 32k 的 30-50x（O(N²)）。

## 常用命令 / 环境变量量级

```bash
# NCCL 调试
NCCL_DEBUG=INFO              # 看 ring / tree 选择
NCCL_TOPO_DUMP_FILE=topo.xml # dump 探测拓扑
NCCL_SOCKET_IFNAME=ib0       # 强制网卡
NCCL_IB_HCA=mlx5_0,mlx5_1    # 指定 HCA

# 容量监控
nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv
dcgmi diag -r 3              # health check
nvidia-smi nvlink --status   # NVLink 状态

# vLLM 启动核心参数
vllm serve <model> \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8
```

记住这几条命令在生产里就够用 90%。
