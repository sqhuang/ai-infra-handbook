#!/usr/bin/env python3
"""Driver: run enrich_volumes.process_volume for V03/V05/V06/V07/V08
with hand-crafted 综合题 templates per volume.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_volumes import process_volume  # noqa: E402


# =====================================================================
# Volume 03: CUDA/Triton 与自定义算子
# =====================================================================
V03_SYNTHETIC = [
    {
        "title": "比较题：CUDA C++ vs Triton vs CUTLASS 三种 kernel 实现路径",
        "lede": "三条路径各自有典型适用区——把它们错配是工程效率塌方的根本原因。",
        "s1": "CUDA C++ 控制力最强、Triton 开发效率最高、CUTLASS 是 GEMM 类高性能模板库的最强基础设施。新写自定义 kernel 默认用 Triton；要榨干最后 5% 性能 / 涉及 TMA + WGMMA 高级特性时降到 CUDA C++ + CUTLASS。",
        "s2": "Triton 抽象了 block / shared memory / vectorize，把多数 90 分 kernel 用几十行 Python 写出来；CUDA C++ 暴露所有硬件原语，但代码量多 3-5x；CUTLASS 用 C++ 模板封装 Tensor Core / TMA 调度，FA-3、Marlin、TRT-LLM 大量基于它构建。",
        "s3": "Triton：`@triton.jit` 装饰、`tl.load/store`、`tl.dot`，autotune 自动调 BLOCK/num_warps。CUDA C++：手写 cooperative group + shared memory + async copy。CUTLASS：`GemmUniversal<...>` 模板按 epilogue / tile / cluster 组合。三者可混用：上层 Triton 拼调度，关键 GEMM 走 CUTLASS extern。",
        "s4": "Triton 编译产物质量取决于 autotune 配置质量；CUDA C++ 性能上限高但维护贵；CUTLASS 学习曲线陡但一旦用上单 kernel 性能接近 cuBLAS。生产里 90% kernel 走 Triton 5% 走 CUTLASS 5% 走 CUDA C++ 是一个合理分布。",
        "s5": "Triton kernel 慢通常是 autotune configs 不全或 BLOCK size 选差；CUTLASS 写错 epilogue 容易触发 fallback；CUDA C++ 性能写对一遍困难、要反复 profile。三者最大的共同陷阱是「测了 microbenchmark 快但端到端没快」——profile 一定要在真实 batch 下做。",
        "s6": "新 kernel 默认 Triton 起步；性能差距 > 10% 再考虑 CUTLASS。把 cuBLAS/cuDNN 作为 baseline，自研 kernel 至少要打平 cuBLAS 才算交付。`triton.testing.do_bench` 是最常用的单 kernel 测时工具。",
        "s7": "- 默认 Triton，性能差距 >10% 才换 CUTLASS / CUDA C++\n- 三者可混用：Triton 调度 + CUTLASS GEMM extern\n- 90/5/5 是合理生产分布\n- 必须以 cuBLAS / cuDNN 为 baseline 验证",
        "s8": "- [ ] 你能不能给出 Triton vs CUDA C++ vs CUTLASS 各自的「杀手锏」场景？\n- [ ] 你能不能解释为什么不直接全用 CUDA C++？\n- [ ] 你能不能描述一个三者混用的真实 kernel 案例（如 FA-3）？\n- [ ] 你能不能说出 microbench 与端到端 perf 脱钩的典型原因？",
    },
    {
        "title": "场景题：自定义 attention kernel microbench 提升 5%，但 vLLM 端到端反而变慢",
        "lede": "kernel 单测快不等于端到端快——调度、KV 布局、launch 开销都可能把收益吞掉。",
        "s1": "三层定位：1) Kernel 本身在生产 batch / shape 下是否仍快；2) 集成路径是否多了 Python 边界 / 内存拷贝 / launch 开销；3) 是否破坏了 vLLM 的 CUDA Graph / PagedAttention 假设。端到端慢通常根因是后两层。",
        "s2": "vLLM 用 CUDA Graph 把 decode 阶段的 launch 序列固化。如果自定义 kernel 不支持被 CUDA Graph capture（带 host-device sync / 动态 shape），就会触发 fallback 到 eager 调用，每步增加几百微秒。同时 PagedAttention 要求 KV layout 是 page-block；自定义 kernel 用 contiguous KV 时需要额外重排，开销巨大。",
        "s3": "定位工具：`nsys profile` 看 launch 次数、`vllm bench` 跑端到端对比、`torch.profiler` 看每 op 时间。关键检查项：是否在 forward 里有 `.item()` / `.cpu()` / `print()`、是否使用了 `torch.compile` 的可 capture API、KV layout 是否对齐 vLLM 的 page block。",
        "s4": "kernel 「局部最优」与「全局最优」经常不一致。生产对集成友好性的要求大于绝对 perf：5% kernel 收益但破坏 graph capture 通常净负。",
        "s5": "为什么 microbench 没暴露？因为 microbench 通常 batch=1 / 单 kernel 测，不触发 graph capture、不走 KV layout。生产场景要用「压测脚本 + 完整 pipeline」做端到端 perf。",
        "s6": "新 kernel 上线前必做三件事：1) graph capture 兼容性测试；2) 真实 batch & shape 分布下的端到端 bench；3) 与 baseline 在多 (TTFT, TPOT, throughput) 维度对比。",
        "s7": "- 三层定位：kernel 本身、集成路径、CUDA Graph 兼容性\n- 常见根因：launch 增多、KV layout 重排、graph capture 失败\n- microbench 不能反映生产 perf，必须端到端压测\n- 5% kernel 收益破坏 graph capture 通常净负",
        "s8": "- [ ] 你能不能用 `nsys` 验证 CUDA Graph 是否真的捕获你的 kernel？\n- [ ] 你能不能识别一个 kernel 里隐藏的 host-device sync？\n- [ ] 你能不能描述 PagedAttention 对 KV layout 的约束？\n- [ ] 你能不能给出「kernel microbench → 端到端」的标准化验收流程？",
    },
    {
        "title": "估算题：H100 上 GEMM (M=N=K=8192) FP16 的理论 SOL 与实际 cuBLAS 差距",
        "lede": "估算「理论上限」与「实际值」的差距，是判断「还有多少优化空间」的根本依据。",
        "s1": "H100 FP16 Tensor Core 峰值 ~990 TFLOPs。GEMM FLOPs = 2 × M × N × K = 1.1 TFLOP。理论 SOL = 1.1 / 990 ≈ 1.1ms。实际 cuBLAS 在该 size 上通常跑到 1.4–1.6ms，效率 70–80%。差距来自 launch 开销、tile boundary、L2 cache miss、Tensor Core 利用率。",
        "s2": "Tensor Core 要求 tile 对齐到 16/32/64；M=N=K=8192 是完美对齐，所以基本能跑满 Tensor Core。剩余 20–30% 损失来自：(a) 数据从 HBM 到 SM 的搬运 vs 计算并行度；(b) split-K 决策、stream-K 调度差异；(c) cuBLAS heuristic 选 kernel 不一定最优。CUTLASS / FA-3 风格 kernel 在某些 shape 上能超过 cuBLAS 10–15%。",
        "s3": "估算的 building blocks：FLOPs = 2MNK；峰值算力（FP16/FP8/FP4 各不同）；带宽 SOL = (A+B+C 字节) / HBM 带宽，HBM 带宽 H100 = 3.35 TB/s，8192² × 2 bytes × 3 = 384MB，理论搬运 ~114µs。FLOPs SOL > 带宽 SOL 时是 compute-bound（这道题就是）；反之 memory-bound（小 batch GEMV、attention decode）。",
        "s4": "搞清是 compute-bound 还是 memory-bound 是优化方向的分水岭。compute-bound 优化 Tensor Core 利用率、tile shape、autotune；memory-bound 优化 tiling 减少重读、用 shared memory / TMA prefetch。两类问题的解法不能错配。",
        "s5": "为什么不是 100% SOL？因为算力峰值只在所有 SM 满载、所有 Tensor Core 同时跑 FMA、没有任何 stall 时才达到——实际不可能。70–80% 已是顶级 kernel。",
        "s6": "做估算时先写 FLOPs / 带宽 两条 SOL 线，看哪条更紧。然后从 cuBLAS 测一个 baseline，差距大小决定还有多少空间。差距 < 10% 不值得自研 kernel。",
        "s7": "- SOL = 2MNK / 峰值算力，8192³ FP16 在 H100 ≈ 1.1ms\n- 实际 cuBLAS ~1.4–1.6ms，效率 70–80%\n- 差距来源：launch、tile 边界、L2 miss、heuristic\n- compute-bound 还是 memory-bound 决定优化方向",
        "s8": "- [ ] 你能不能为任意 shape 算出 FLOPs SOL 与带宽 SOL？\n- [ ] 你能不能判断给定 GEMM 是 compute-bound 还是 memory-bound？\n- [ ] 你能不能解释 cuBLAS 离 100% SOL 的差距来源？\n- [ ] 你能不能说出什么时候自研 kernel 才划算？",
    },
    {
        "title": "设计题：生产推理引擎的自定义算子注册 + dispatch 框架（FP8/FP4 + tree mask + PagedAttention）",
        "lede": "现代推理引擎要在 Hopper 和 Blackwell 上自动选最优 kernel，还要支持 tree mask、PagedAttention，是一个典型的「按 capability + shape + dtype dispatch」的设计题。",
        "s1": "三层架构：1) 算子 schema 层（每个算子定义 input/output shape & dtype）；2) kernel 注册层（per-(arch, dtype, layout) 注册多份实现）；3) dispatch 层（按 runtime info 选最优 kernel + fallback chain）。FP8/FP4 / tree mask / PagedAttention 都是单独的 capability flag。",
        "s2": "Dispatch 的「key」通常是 (arch, dtype_in, dtype_out, layout, mask_type)。注册表是 hash map。runtime 收到 input 后构造 key，查表选 kernel；找不到完美匹配走 fallback（如 FP4 找不到 fall back 到 FP8，FP8 fall back 到 BF16）。CUTLASS 的 GemmUniversal 模板系统就是这种思想的工业级实现。",
        "s3": "注册 API: `@register_kernel(arch=「hopper」, dtype=「fp8」, layout=「paged」, mask=「tree」)`。每个 kernel 实现一个标准签名。运行时调度器把 PyTorch tensor 转 capability key、查表、调用。同步要处理 stream / event。",
        "s4": "framework 取舍：写死所有 kernel 直观但难维护；用动态分发灵活但 dispatch 开销几 µs（要 cache）。生产里通常用 cache + sticky routing：首次解析后缓存 (key, kernel_ptr) 元组。",
        "s5": "常见坑：1) fallback chain 没设计好，FP4 不支持时直接报错而非降级；2) tree mask 和 PagedAttention 的复合 kernel 没注册，运行时找不到；3) arch 探测错误（应使用 `torch.cuda.get_device_capability()` 而非硬编码）。",
        "s6": "从最小集起步：(Hopper FP8 + paged + causal) → (Blackwell FP4 + paged + causal) → 加 tree mask → 加 sliding window。每加一个 capability 都要补 fallback 路径。CI 跑 dispatch 矩阵：每 (arch, dtype, layout, mask) 组合都要有 kernel 命中。",
        "s7": "- 三层：schema、注册、dispatch + fallback chain\n- key = (arch, dtype, layout, mask)；查表后 cache\n- 必须有 fallback：FP4 → FP8 → BF16\n- CI 跑 dispatch 矩阵确保所有组合命中",
        "s8": "- [ ] 你能不能列出推理 kernel 的 dispatch key 完整字段？\n- [ ] 你能不能设计一个 fallback chain 处理 capability 缺失？\n- [ ] 你能不能描述 CUTLASS GemmUniversal 模板与这个设计的对应关系？\n- [ ] 你能不能为新增 capability（如 NVFP4 + sliding window）写出最小添加步骤？",
    },
]


# =====================================================================
# Volume 05: 显存、通信与 Profiling
# =====================================================================
V05_SYNTHETIC = [
    {
        "title": "比较题：activation checkpointing 全层 vs selective vs attention 边界三种策略",
        "lede": "重计算策略选错就是「训练时间多 25% 但只省了 10% 显存」——分场景选很关键。",
        "s1": "三种策略：全层重计算最省显存但慢约 25%；selective（每 N 层重计算一次）权衡灵活；attention 边界重计算（只保留 attention 输入）平衡最优，是 70B 训练 default。70B BF16 训练里 attention 边界 + FSDP2 + TP 是当前主流组合。",
        "s2": "重计算的本质：反向需要 activation，正向算完释放，反向时再正向一次。全层重计算要再做一遍完整 forward，cost 25% 左右；selective 跳过部分层减少 overhead；attention 边界利用了 FA-2/3 内重计算的特性，只在 attention boundary 重新算。",
        "s3": "API：PyTorch `torch.utils.checkpoint.checkpoint`、FSDP2 `apply_activation_checkpointing`、Megatron `activations-checkpoint-method=selective`。每种策略要决定 (granularity, frequency, reentrant) 三个参数。",
        "s4": "全层适合显存极紧、能忍 25% 慢；selective 适合中等显存；attention 边界适合现代 Transformer + FA-3。混合并行（FSDP+TP+CP）时，重计算要和 CP 协同——CP 已经摊薄显存，可以放宽重计算。",
        "s5": "常见坑：1) selective 频率没调好（每层 vs 每 4 层）；2) 与 dropout / random op 配合不当导致前后向不一致；3) 重计算开启后 MFU 看起来下降，要看 HFU 才公平。",
        "s6": "推荐组合：70B BF16 训练 → attention 边界 + FSDP2 + activation offload。先 profile 看 activation 占显存比例，再决定哪种策略。",
        "s7": "- 三种策略：全层（最省最慢）、selective（折中）、attention 边界（现代 default）\n- 全层 ~25% overhead；attention 边界 < 10%\n- 与 FSDP/CP/TP 协同，CP 摊薄显存后可放宽重计算\n- 比较时看 HFU 而非 MFU 才公平",
        "s8": "- [ ] 你能不能解释三种策略的显存 vs 时间权衡？\n- [ ] 你能不能描述 attention 边界重计算与 FA-3 的协同？\n- [ ] 你能不能说出 MFU / HFU 在重计算开启时的差异？\n- [ ] 你能不能为 70B 训练给出推荐配置？",
    },
    {
        "title": "场景题：NCCL all-reduce p99 从 30ms 涨到 200ms，怎么三层定位？",
        "lede": "p99 突涨是分布式训练最常见的事故——按 NIC / 拓扑 / 应用 三层排查，能在 30 分钟内定位 80% 的情况。",
        "s1": "三层定位：NIC 层（硬件错误 / 限速 / 重传）→ 拓扑层（NCCL 路径选错 / 网卡绑定漂移 / NUMA cross）→ 应用层（gradient bucket 大小 / 通信叠 compute 失败）。先用 DCGM + nccl-tests 排除硬件，再用 NCCL_DEBUG=INFO 看拓扑，最后看应用 profile。",
        "s2": "NCCL 通信延迟 = α + β × bytes。α 由 ring/tree 选型与节点跳数决定，β 由带宽决定。p99 跳到 200ms 通常是 α 端：某些 ring step 卡住 / 重传 / wait。具体根因可能是 NIC 错误 retry、IB switch 拥塞、某个节点 NCCL 版本漂、PCIe 链路降速。",
        "s3": "工具栈：`nvidia-smi`（看 GPU/PCIe 状态）、`ibstat`（IB 链路）、`dcgmi diag -r 3`（GPU 健康）、`NCCL_DEBUG=INFO`（拓扑选择 + warn）、`nccl-tests`（单独 all-reduce bench）、`nsys`（端到端 timeline）。",
        "s4": "排查顺序：先快速确认是单节点问题还是集群问题（看 timeline 上是不是固定 rank 慢）；再看 NCCL_DEBUG 输出有无 retry / fallback；最后看应用层 bucket / overlap 配置。错误顺序会浪费大量时间。",
        "s5": "常见根因 top 5：单节点 NIC 错误 / IB switch 端口异常 / PCIe 降速 / NUMA 绑定漂移 / NCCL 版本不一致。前 3 个最常见。",
        "s6": "p99 监控必须有，p50 看不出突发。设阈值告警：all-reduce p99 > 2x 7d baseline 就告警。常备 nccl-tests 一键 bench 脚本，5 分钟内出诊断报告。",
        "s7": "- 三层：NIC（硬件错误）、拓扑（路径选错）、应用（bucket/overlap）\n- 先 nccl-tests 排硬件，再 NCCL_DEBUG 排拓扑，最后看应用\n- 单节点慢 vs 集群慢 是分支关键\n- 必须监控 p99 而非 p50",
        "s8": "- [ ] 你能不能列出 NCCL p99 突涨的 top 5 根因？\n- [ ] 你能不能用 nccl-tests 在 5 分钟内出诊断报告？\n- [ ] 你能不能解释 α + β × bytes 模型对故障定位的指导意义？\n- [ ] 你能不能区分单节点问题与集群问题？",
    },
    {
        "title": "估算题：H100 NVL8 单机 70B BF16 FSDP2 训练每 step 的通信量 / HBM 带宽 / launch 次数",
        "lede": "把训练 step 拆成可估算的几个量，是判断「瓶颈在哪」的核心能力。",
        "s1": "70B BF16 模型权重 140GB，FSDP2 每 step all-gather + reduce-scatter 全部参数：通信量 ~280GB / step / rank。NVL8 内 NVLink 900 GB/s 双向，理论通信时间 ~310ms / step（不计算 compute overlap）。HBM 带宽 H100 3.35 TB/s，每 step 访问总量 ~500GB（activation + 权重），带宽端 SOL ~150ms。launch 数千级（FSDP + autograd + optimizer）。",
        "s2": "FSDP2 通信结构：forward 前 all-gather 参数（140GB），forward 后丢弃；backward 前再 all-gather，backward 后 reduce-scatter（140GB）梯度。所以总通信 280GB/rank/step（粗算）。compute 与通信 overlap 能把暴露通信压到 30–50%。",
        "s3": "估算分解：（a）通信量 = 模型参数 × 2（fwd + bwd）；（b）通信时间 = 量 / 带宽 × overlap 因子；（c）compute 时间 = FLOPs / 算力；（d）launch 次数 = 层数 × 每层 op 数 × （fwd + bwd + optimizer）。70B 32 层 × ~40 op × 3 ≈ 4k launch。",
        "s4": "通信 vs compute 谁瓶颈决定优化方向。NVL8 内 70B BF16 是 compute-bound（compute > 通信暴露），扩到 8 机 NVL8（DP=8）后 cross-node 通信暴露变大，开始 communication-bound。这就是为什么 8x NVL8 训练性能比单 NVL8 不等比扩展。",
        "s5": "为什么 overlap 因子能到 50%？因为 PyTorch 用 stream / async collective 把通信和下一层 compute 叠流水。但要写对 prefetch 配置。bucket size 太小会让 overlap 失效。",
        "s6": "做估算时先列三条线：FLOPs / 带宽 / 通信。瓶颈线决定优化重点。70B BF16 FSDP2 优化重点通常是减少 cross-node 通信（升级 NVL72 或 减 DP 维）。",
        "s7": "- 通信 ~280GB / step / rank（FSDP2 all-gather + reduce-scatter）\n- HBM 带宽 SOL ~150ms / step；NVL8 通信 SOL ~310ms（含 overlap 后实际更低）\n- launch ~4k / step（32 层 × 40 op × 3）\n- 单 NVL8 compute-bound，跨域 communication-bound",
        "s8": "- [ ] 你能不能为任意模型 / 并行配置做出三条 SOL 线？\n- [ ] 你能不能解释 FSDP2 通信结构与 ZeRO-3 的对应？\n- [ ] 你能不能描述 overlap 因子如何由 bucket / prefetch 影响？\n- [ ] 你能不能区分单机 compute-bound 与跨机 communication-bound 的 perf 表现？",
    },
    {
        "title": "设计题：万卡集群显存与通信可观测性栈（OOM / 掉速 5 分钟定位）",
        "lede": "万卡训练 5 分钟内定位 OOM / 通信掉速，靠的不是经验是 instrumentation——这是一个标准的「可观测性平台」设计题。",
        "s1": "四层栈：1) 节点级（DCGM + 自研 GPU agent，每 5s 上报 power/util/mem/Xid）；2) 通信级（NCCL_DEBUG 日志 + nccl-tests 周期 benchmark）；3) 应用级（torch.profiler 抽样 + 自定义 trace）；4) 平台级（dashboard + alert + 自动诊断）。核心数据流：节点 → kafka → ClickHouse / Prometheus → 看板 + 自动 root cause analysis。",
        "s2": "可观测性 = metric + trace + log。metric 适合做 dashboard 与 alert，trace 适合做事故复盘，log 适合做事后取证。三者要打通（trace ID 串联 metric 和 log）。万卡场景下 sampling 必不可少：每节点 1% 抽样 + 关键事件全量。",
        "s3": "关键指标：HBM usage p50/p99、PCIe link state、NCCL p50/p99 by op、torch op time by layer、Xid event rate、ECC event rate。所有指标 tag 必须有 (cluster, rack, node, gpu, job_id, rank)。",
        "s4": "万卡数据量挑战：1 万张卡 × 每 GPU 30 指标 × 每 5s = 60k metric/s。ClickHouse 能撑这个量级，Prometheus 远端存储要 sharding。alert 要分 severity：oncall page（节点级硬件错）/ email（性能下降）/ ticket（容量预测）。",
        "s5": "常见坑：日志量太大 oncall 无效（必须分级 + AI 聚合）；trace 没串联（找不到关联事件）；alert 太敏感 oncall 疲劳；alert 太迟事故已经升级。",
        "s6": "落地路线：Q1 节点级 metric + dashboard 铺通；Q2 通信级 + 自动 NCCL bench；Q3 应用级 trace 抽样；Q4 自动 RCA + 智能 alert 聚合。一个 critical 设计点：所有数据必须可被 self-service 查询（SQL / dashboard），不然 oncall 永远要找平台团队取数。",
        "s7": "- 四层栈：节点（DCGM）、通信（NCCL）、应用（profiler）、平台（dashboard + RCA）\n- 万卡场景必须 sampling + trace ID 串联\n- alert 分 severity：page / email / ticket\n- self-service 查询是 oncall 效率根本",
        "s8": "- [ ] 你能不能列出万卡集群必须的最小指标集（< 30 个）？\n- [ ] 你能不能设计 metric / trace / log 的统一 tag 与串联方式？\n- [ ] 你能不能为 OOM 故障设计自动 RCA 流程？\n- [ ] 你能不能解释 1 万卡每 5s 指标的数据量与存储选型？",
    },
]


# =====================================================================
# Volume 06: 推理量化与服务化
# =====================================================================
V06_SYNTHETIC = [
    {
        "title": "比较题：vLLM v1 vs SGLang 0.4+ vs TensorRT-LLM 四类业务下的选型",
        "lede": "不是「谁更快」而是「什么业务下哪个的工程综合分最高」——按场景拉表才能选对。",
        "s1": "普通 chat：vLLM v1（生态最广、模型最全）；agent / 多轮 / RAG：SGLang（RadixAttention 命中率高）；长上下文 / 极致 perf：TRT-LLM（NVIDIA stack 上限最高）；多模型 / 多硬件：vLLM v1（可移植性最好）。选型时间 → 上线时间 < 1 周用 vLLM，对 perf 要求极致允许投入 4-8 周接 TRT-LLM。",
        "s2": "vLLM v1：开源生态 default、PagedAttention + chunked prefill + spec decode 都已集成、模型支持广；SGLang：RadixAttention 让 agent / multi-turn cache hit 显著高、Python 控制流友好；TRT-LLM：闭源 NVIDIA stack、kernel 优化最深、但 engine build 慢、迭代慢、模型支持需要 NVIDIA 适配。",
        "s3": "选型 4 维：(模型支持, 业务模式, perf 上限, 运维复杂度)。chat → vLLM 全部满足；agent → SGLang 在业务模式维胜；长上下文 → TRT-LLM 在 perf 维胜，但运维复杂。三者都支持 FP8/FP4 quantization、PagedAttention、speculative decoding。",
        "s4": "迁移成本与运维。vLLM 升级与新模型适配快；SGLang 迭代速度同样快；TRT-LLM 升级慢（要重新 build engine 与适配）。生产里常见混合：主 fleet vLLM、高价值 / 长上下文 fleet TRT-LLM、agent 子 fleet SGLang。",
        "s5": "为什么不能「全 TRT-LLM」？支持模型少 + 迭代慢 + 运维重。为什么不能「全 vLLM」？某些极致 perf 场景上限不够。为什么不能「全 SGLang」？模型生态略窄、企业级支持弱。",
        "s6": "起步默认 vLLM。流量起来后按业务分 fleet：agent → SGLang；长上下文 → TRT-LLM；其他 → vLLM。每个 fleet 都要有独立 SLO 与 dashboard。",
        "s7": "- chat → vLLM；agent → SGLang；长上下文 → TRT-LLM；多模型 → vLLM\n- 上线 < 1 周用 vLLM，极致 perf 投 4-8 周接 TRT-LLM\n- 生产常混合：vLLM 主、TRT-LLM 高价值、SGLang agent\n- 选型 4 维：模型支持 / 业务模式 / perf 上限 / 运维",
        "s8": "- [ ] 你能不能给出 4 类业务的具体引擎选型？\n- [ ] 你能不能解释 RadixAttention 在 agent 场景的命中率优势？\n- [ ] 你能不能列出 TRT-LLM 的运维成本来源？\n- [ ] 你能不能描述混合 fleet 的拆分逻辑？",
    },
    {
        "title": "场景题：70B 推理服务 TTFT p99 从 300ms 涨到 1.5s 但 TPOT 不变",
        "lede": "TTFT 涨而 TPOT 不变是典型「prefill 阶段问题」——chunked prefill / 队列 / cache miss 三选一。",
        "s1": "三层定位：1) 队列堆积（chunked prefill 配置错或 max_num_seqs 不够）；2) prefix cache miss（流量模式变化 / 系统 prompt 更新）；3) prefill 算力被 decode 抢（chunked prefill 没开 / batch size 不当）。TPOT 不变说明 decode 阶段健康。",
        "s2": "vLLM v1 默认是 prefill + decode 共存 batch，scheduler 决定每 step 配多少 prefill chunk vs decode。如果 prompt 集中变长（如新业务上线），prefill workload 飙升、scheduler 倾向 prefill、新请求排队（队列效应让 p99 爆炸）。",
        "s3": "排查工具：vLLM metrics endpoint（scheduler_queue_depth / num_running / num_waiting）、prefix_cache_hit_rate、chunked_prefill_count、ttft histogram。日志侧看 prompt 长度分布。",
        "s4": "修复路径：1) 增大 max_num_batched_tokens 让 prefill chunk 更大；2) 开 prefix_caching；3) 限流（长 prompt 单独队列）；4) 横向扩 replica。先 metrics 看清根因再修，不要一上来就扩容。",
        "s5": "常见错排：1) 看到 TTFT 涨就以为 decode 慢；2) cache hit rate 没监控；3) 没区分 long-prompt vs short-prompt 队列。",
        "s6": "TTFT 监控按 prompt 长度分桶（<1k, 1-4k, 4-16k, >16k）。每桶 SLO 不同。流量模式变化要在 dashboard 上直接看出来。",
        "s7": "- TTFT 涨 + TPOT 不变 = prefill 阶段问题\n- 三层：队列 / cache miss / prefill 被 decode 抢\n- 工具：vLLM metrics + prefix_cache_hit_rate + prompt 长度直方图\n- 修复：max_num_batched_tokens、prefix_caching、限流、扩 replica",
        "s8": "- [ ] 你能不能用 vLLM metrics 5 分钟内判断 TTFT 涨的根因？\n- [ ] 你能不能解释 chunked prefill 与 scheduler 的协同？\n- [ ] 你能不能描述 prefix cache hit rate 与流量模式的关系？\n- [ ] 你能不能为 TTFT 设计按 prompt 长度分桶的 SLO？",
    },
    {
        "title": "估算题：70B FP8 服务在 4× H200 上的 tokens/$ 估算",
        "lede": "把硬件单价、token 吞吐、SLO 等价量代到一起，是判断推理业务能不能赚钱的核心估算。",
        "s1": "70B FP8 TP=4 H200：tokens/s ≈ 2000–3000（中等 batch、平均 prompt 2k、output 300）。4× H200 月租 ~$3000 × 4 × 720h ≈ $8.6k/月，等于每秒 ~$0.0033。假设 2500 tokens/s，1 美元能跑 2500 / 0.0033 ≈ 75 万 tokens，对外报价 $5-10/M tokens 利润率合理。",
        "s2": "tokens/$ = (tokens/s / 单卡时价) × utilization × cache hit gain。变量：(a) 模型大小 (70B 是 7B 的 ~10x cost)；(b) GPU 代次 (B200 比 H200 单 token 便宜 30%)；(c) batch 利用率 (峰谷流量决定 fleet 大小)；(d) cache hit rate (agent 场景能省 30-80%)。",
        "s3": "估算 5 步：1) 列模型大小与并行 (70B TP=4)；2) 查吞吐基准 (~2500 tokens/s @ batch=32)；3) 查 GPU 单价 ($2-3/h H200)；4) 估 utilization (75% 是健康值)；5) 算 tokens/$ 与 $/1M tokens。",
        "s4": "影响最大的变量：cache hit rate（agent 多轮）+ batch 利用率（流量稳定性）+ model size（每翻倍 cost ~10x）。优化方向也是这三个。",
        "s5": "为什么不能算到精确？因为 utilization 与 cache hit rate 都波动 ±30%。估算只能给区间，5x 差距以内算合理。",
        "s6": "做估算时给上下限（保守 + 激进），差距通常 5-10x。商业模型据此定价。Profile + 灰度 + cost dashboard 是把估算变现实的三件套。",
        "s7": "- 公式：(tokens/s / 单卡时价) × utilization × cache hit\n- 70B FP8 TP=4 H200 ~75 万 tokens/$\n- 对外报价 $5-10/1M tokens 利润合理\n- 三大变量：cache hit、batch 利用率、模型大小",
        "s8": "- [ ] 你能不能 5 分钟做出 70B 服务的 tokens/$ 估算？\n- [ ] 你能不能列出至少 5 个影响 tokens/$ 的变量？\n- [ ] 你能不能解释 cache hit rate 在 agent 业务下的杠杆作用？\n- [ ] 你能不能为新模型上线给出报价建议？",
    },
    {
        "title": "设计题：支持多模型 / 多版本 / 多硬件后端的 LLM serving 网关",
        "lede": "现代 LLM 平台不只是「跑模型」，是要 A/B 灰度、prefix sticky、KV migration、多租户隔离——这是一个标准的服务网关 + ML 平台融合设计题。",
        "s1": "五层架构：1) gateway（鉴权、限流、路由）；2) router（A/B、灰度、sticky）；3) engine pool（vLLM / SGLang / TRT-LLM 多 backend）；4) KV cache layer（LMCache 跨节点 cache）；5) 控制面（model registry / config / deploy）。多租户隔离贯穿全栈：quota、cache namespace、metrics tag。",
        "s2": "Sticky routing 让 agent / multi-turn 命中同实例 prefix cache。A/B 灰度 by request tag。多版本：每模型 N 个 version，灰度按比例分流。KV migration 让 prefill / decode 分池。",
        "s3": "核心数据流：request → gateway 鉴权 + tag → router 选 (model, version, replica) → engine 推理 → response 回写。每个 hop 都打 trace ID + cost 字段。控制面用 CRD（KServe-style）管理 deploy。",
        "s4": "复杂度 vs 灵活度。完全自研网关周期长（6-12 月）；用 Istio + Envoy + 自研 plugin 折中；用现成 KServe + 改造最快但定制能力弱。生产里通常 Envoy + 自研 plugin。",
        "s5": "常见坑：1) sticky routing 实现错让 cache hit 失效；2) 多租户没 isolation 导致 noisy neighbor；3) deploy CRD 跟 engine 状态不同步；4) KV migration 没考虑 reshard。",
        "s6": "起步用 vLLM + 简单 LB；流量上来加 sticky + A/B；agent 业务上线加 RadixAttention / LMCache；多模型多硬件后加 model registry + 多 backend dispatch。",
        "s7": "- 五层：gateway / router / engine pool / KV layer / 控制面\n- sticky routing + A/B 灰度是 agent / 多模型核心\n- 多租户隔离贯穿：quota / cache namespace / metrics tag\n- Envoy + 自研 plugin 是常见折中",
        "s8": "- [ ] 你能不能画出请求从 gateway 到 response 的完整数据流？\n- [ ] 你能不能解释 sticky routing 与 prefix cache 的依赖关系？\n- [ ] 你能不能设计 A/B 灰度的路由逻辑？\n- [ ] 你能不能描述多租户 quota 的实现层级？",
    },
]


# =====================================================================
# Volume 07: 训练与推理平台
# =====================================================================
V07_SYNTHETIC = [
    {
        "title": "比较题：SLURM vs Kubernetes (Volcano/Kueue) vs Ray 三套调度的取舍",
        "lede": "HPC 圈 SLURM、云原生 K8s、Python 生态 Ray——三套调度系统对训练 / RL rollout / 在线推理偏好完全不同。",
        "s1": "SLURM：HPC 老大，gang scheduling、prolog/epilog 成熟，训练首选；K8s + Volcano/Kueue：云原生默认，资源 CRD 与 ML pipeline 集成好，推理 + 中型训练首选；Ray：Python 优先，分布式编程模型，RL rollout / 多 agent 首选。生产常常混用：训练 SLURM、推理 K8s、RL Ray。",
        "s2": "三者抽象层不同：SLURM 把 job 当一等公民（sbatch + node list），K8s 把 pod 当一等公民（CRD + scheduler），Ray 把 actor / task 当一等公民（remote function）。训练负载与 SLURM 模型契合，推理 / 微服务与 K8s 契合，编程式分布式与 Ray 契合。",
        "s3": "核心机制：SLURM 用 `srun`/`sbatch` + prolog；K8s + Volcano 用 PodGroup + Gang；K8s + Kueue 用 Workload 排队；Ray 用 Actor 分布式 + AutoScaler。所有都支持 GPU 资源声明 (nvidia.com/gpu)。",
        "s4": "训练用 K8s 也行但 gang scheduling 没 SLURM 成熟；推理用 SLURM 也能跑但 autoscaling / multi-version 差；RL 用 SLURM 不灵活（rollout/train 解耦难表达）。错配会让运维 cost 翻倍。",
        "s5": "为什么不能「全 K8s 统一」？SLURM 在大规模训练（千卡以上）的 gang scheduling 与节点亲和上仍更成熟。为什么不能「全 SLURM」？K8s 的 autoscaling、灰度、HTTP routing 是 SLURM 没有的。",
        "s6": "中小团队首选 K8s + Kueue 覆盖训 + 推 + RL；大团队按负载分：训练 SLURM + 推理 K8s + RL Ray。混用要解决 cluster federation / 共享存储 / 共享镜像问题。",
        "s7": "- 训练 → SLURM；推理 → K8s + Volcano/Kueue；RL → Ray\n- 抽象不同：job / pod / actor 三种模型\n- gang scheduling SLURM 最成熟、K8s 凑合、Ray 不需要\n- 混用是常态，要解决 federation + 共享栈",
        "s8": "- [ ] 你能不能用一句话说出三套调度的核心抽象差异？\n- [ ] 你能不能解释 gang scheduling 为什么在大训练上至关重要？\n- [ ] 你能不能描述 Ray 在 RL rollout 上的优势？\n- [ ] 你能不能为虚拟团队设计混合调度架构？",
    },
    {
        "title": "场景题：1k 卡 H100 集群 JCT p99 比同行慢 2×，怎么从 4 个角度定位？",
        "lede": "JCT 慢通常不是单点问题——调度 / 排队 / 容错 / 资源碎片每一层都可能贡献，必须按层挖。",
        "s1": "四角度：1) 调度（gang scheduling 配置 / 优先级权重）；2) 排队（队列深度、长任务饥饿）；3) 容错（节点故障率 + 恢复时间）；4) 资源碎片（分配策略让大 job 拿不到连续 node）。先看 metrics dashboard 定位主因，再深挖。",
        "s2": "JCT = 排队时间 + 启动时间 + 运行时间 + 恢复时间。p99 慢通常是排队 + 恢复主导。运行时间慢说明硬件 / 通信问题。需要把 4 段时间分别统计才能看出哪段是 bottleneck。",
        "s3": "工具栈：SLURM `sacct`（每 job 各段时间）、自研 metrics（每 job 排队 / 启动 / 恢复）、cluster dashboard（队列深度 / 节点状态 / 碎片化指数）。每个 job 必须打 trace ID 串联各阶段。",
        "s4": "调度算法影响碎片化。Best-fit 减少碎片但要求 job 准确声明资源；Backfill 让小 job 填空但大 job 优先级要够。生产里 backfill + reservation + preemption 三者结合。",
        "s5": "常见根因：1) 节点故障率高 + checkpoint 间隔过长（恢复慢）；2) backfill 策略错让大 job 永远排不上；3) 没有 fair-share 让一个团队占满 cluster；4) GPU 配额声明错让 job 拿不到完整 NVL8。",
        "s6": "优化路线：1) 装 trace 把 JCT 拆 4 段；2) 看哪段是 bottleneck；3) 针对性修（短 job 加 backfill、长 job 加 reservation、故障节点加自动隔离）；4) 季度跑 fair-share 审计。",
        "s7": "- 4 角度：调度 / 排队 / 容错 / 资源碎片\n- JCT 拆 4 段：排队 + 启动 + 运行 + 恢复\n- p99 慢通常是排队 + 恢复主导\n- 优化前先 trace 拆段，盲修是浪费",
        "s8": "- [ ] 你能不能把 JCT 拆成 4 个可独立统计的阶段？\n- [ ] 你能不能解释 backfill 与 reservation 的协同？\n- [ ] 你能不能列出节点故障率与 checkpoint 频率的关系？\n- [ ] 你能不能为 fair-share 设计季度审计流程？",
    },
    {
        "title": "估算题：5k 卡集群 MTBF=30 天 + 30min ckpt 间隔下的 24h 训练成功率",
        "lede": "节点故障率 × 时间 × ckpt 频率 → 端到端成功率，是平台 SRE 必须熟练的费米估算。",
        "s1": "5k 卡 / 单卡 MTBF=30 天 → 集群故障率 ~5000 / (30 × 24) ≈ 7 次/小时。一次故障让训练中断 + 恢复时间 = ckpt 间隔 / 2 + 恢复时间 ≈ 15min + 5min = 20min。24h 内总恢复时间 ≈ 7 × 24 × 20min ≈ 2800min ≈ 47h——比 24h 还多，说明 5k 卡裸跑根本无法持续 24h。需要 elastic training（剔除坏节点继续）才能维持。",
        "s2": "MTBF 不是单点故障概念——集群 MTBF 与卡数线性反比。5k 卡集群 MTBF ≈ 30d / 5000 = ~10 min。即每 10 分钟有节点故障。30min ckpt 间隔意味着每次故障平均丢 15min 进度。",
        "s3": "elastic training 关键：节点故障 → 检测（DCGM event）→ 通知（SLURM scancel + requeue）→ ckpt reload + 剔除坏节点 → 继续训练（DP world size -1）。整个流程要 < 5min 才能维持 80%+ 有效训练时间。",
        "s4": "提高有效训练时间的杠杆：1) 缩短 ckpt 间隔（每 10min 而非 30min）；2) 加速 ckpt save/load（async + 分布式）；3) elastic training（不重启）；4) 预先隔离坏节点（每节点 boot 时 DCGM diag）。",
        "s5": "为什么不能简单「卡数翻倍」？因为故障率也翻倍，恢复时间是 amortize cost。10k 卡集群必须有更短 ckpt + elastic + 自动诊断。",
        "s6": "万卡训练标配：async ckpt（每 10min）+ elastic training + 节点池预热 + 自动诊断 + 故障节点自动隔离。这一套缺一不可。",
        "s7": "- 5k 卡集群 MTBF ≈ 10min，每 10min 有节点故障\n- 30min ckpt 间隔意味着每次故障丢 15min\n- 裸跑 24h 不可行，必须 elastic\n- elastic + 短 ckpt + 自动诊断 三件套",
        "s8": "- [ ] 你能不能为任意卡数估算集群 MTBF？\n- [ ] 你能不能描述 elastic training 的完整流程？\n- [ ] 你能不能解释 async ckpt 在大集群的关键作用？\n- [ ] 你能不能为万卡训练给出运维 SLO 设计？",
    },
    {
        "title": "设计题：训练 + 在线推理 + RL rollout 三类负载的统一 GPU 平台",
        "lede": "三类负载对 SLO、抢占性、资源粒度要求完全不同——统一平台是优雅设计 vs 工程现实的对话。",
        "s1": "三层抽象：1) 资源池（按 priority class 划：critical / standard / preemptible）；2) 调度层（按负载选 SLURM/K8s/Ray sub-scheduler）；3) 业务层（训练 job / 推理 deployment / RL pipeline 三种 CRD）。资源在三池间可借用但有 SLO 边界。",
        "s2": "三类负载的根本差异：训练对 latency 不敏感但要 gang + 长任务；推理对 latency 敏感且要 autoscale；RL rollout 既要 latency 又要弹性。无法用单一调度器搞定全部，必须分层。",
        "s3": "核心机制：(a) priority preemption（推理高优、训练中、批处理低）；(b) gang scheduling for training；(c) HPA for inference；(d) actor-based for RL；(e) 统一 monitoring + cost。CRD 例：TrainingJob / InferenceService / RLPipeline。",
        "s4": "统一 vs 隔离取舍。完全统一难度极高（不同负载冲突）；完全隔离浪费资源。生产里通常是「资源池统一 + 调度子系统隔离」。",
        "s5": "常见坑：1) preemption 实现错让推理被训练抢；2) gang scheduling 与 HPA 冲突；3) 三类负载的 cost attribution 不统一让财务对不上；4) 节点重启清理脚本错让 RL state 丢失。",
        "s6": "落地路线：Q1 资源池统一；Q2 训练 SLURM + 推理 K8s 集成；Q3 RL Ray 接入；Q4 统一 cost + monitoring。最后一步通常是把「借用规则」治理化（什么时候推理能借训练池）。",
        "s7": "- 三层：资源池 / 调度子系统 / 业务 CRD\n- 三负载差异：训练 gang+长 / 推理 latency+autoscale / RL 弹性+latency\n- 资源池统一 + 调度子系统隔离 是务实折中\n- preemption + gang + HPA + actor 四种机制都要支持",
        "s8": "- [ ] 你能不能列出三类负载的 SLO 与资源粒度差异？\n- [ ] 你能不能描述资源池借用的安全边界？\n- [ ] 你能不能设计统一的 cost attribution 模型？\n- [ ] 你能不能为新 RL pipeline 上线给出最小适配清单？",
    },
]


# =====================================================================
# Volume 08: 系统基础
# =====================================================================
V08_SYNTHETIC = [
    {
        "title": "比较题：cgroup v2 vs Docker vs Kubernetes namespace 三层隔离对 GPU 负载的影响",
        "lede": "三层都叫「隔离」但作用层级不同——错放就是「以为隔离了但 GPU 资源仍互相干扰」。",
        "s1": "cgroup v2 是 Linux 内核级（CPU/内存/IO/PID）；Docker 是容器化层（namespace + cgroup + image）；K8s namespace 是命名空间逻辑分组（不真正隔离资源）。GPU 隔离不靠 namespace，靠 nvidia-container-toolkit + MIG / MPS。这三层各自负责的隔离维度完全不同。",
        "s2": "cgroup v2 提供 CPU / 内存 / IO 的硬隔离；Docker 加上 filesystem / network / process 的命名空间隔离；K8s namespace 只是 API 层逻辑分组（不限制资源）。GPU 隔离是另一套：MIG 硬件分区（最强）、MPS 进程共享（中等）、Time-sharing（最弱）。",
        "s3": "Kubernetes 上 GPU pod 实际跑的是 nvidia-container-runtime + cgroup（限 CPU/内存）+ MIG/MPS（限 GPU 算力 / 显存）。K8s ResourceQuota 限制 namespace 内 GPU 数但不隔离 GPU 之间的干扰。",
        "s4": "训练 job 通常占整张卡（不需 MIG），推理 / agent 可用 MIG（一卡多实例）。cgroup CPU 配额要给 dataloader 留够；忘了限制会导致 noisy neighbor 拖 IO。",
        "s5": "常见坑：1) 只用 K8s namespace 以为做了隔离，实际 noisy neighbor；2) MIG 配置后忘改 driver 模式重启；3) MPS 进程共享在大 model 下显存抢占严重；4) cgroup IO 限制忘配让 HDFS dataloader 互相干扰。",
        "s6": "推理多租户：MIG + ResourceQuota + cgroup IO 三件套。训练单租户：整卡 + cgroup CPU + 节点亲和。绝不要用 K8s namespace 当资源隔离手段。",
        "s7": "- cgroup v2 = 内核级资源；Docker = 容器命名空间；K8s namespace = API 逻辑分组\n- GPU 隔离另一套：MIG（硬）/ MPS（中）/ Time（弱）\n- K8s namespace 不限资源，只做 quota 与 API 隔离\n- 推理多租户：MIG + Quota + cgroup IO 三件套",
        "s8": "- [ ] 你能不能讲清三层隔离各自的作用范围？\n- [ ] 你能不能描述 MIG / MPS / Time-sharing 的 GPU 隔离粒度？\n- [ ] 你能不能列出「以为隔离了实际没」的常见配置错误？\n- [ ] 你能不能为多租户推理给出完整隔离方案？",
    },
    {
        "title": "场景题：GPU util 正常但 step time 慢 30%，IO wait 不高 CPU 也正常——怎么排查？",
        "lede": "经典「指标都正常但 step time 慢」——多半在内核 / NUMA / 页表 / IRQ 层面，不是常规 monitor 能看出来。",
        "s1": "五条排查路径：1) NUMA cross node（cudaMalloc 落到远端内存）；2) page cache pressure（系统在 reclaim）；3) IRQ 不均（中断都落在 CPU 0）；4) THP（Transparent Huge Pages）影响 NCCL；5) syscall overhead（频繁 mmap/munmap）。每一条都需要专门工具检测。",
        "s2": "看上去「指标都正常」是因为常规 monitor 颗粒太粗。GPU util 是 SM 是否在跑，与 step time 直接关系不大。CPU util 是 user/sys，不算 NUMA 远端访问开销。要用 `numactl --hardware`、`perf stat`、`/proc/interrupts`、`turbostat` 才能看到这些深层问题。",
        "s3": "工具：`numastat`（NUMA hit/miss）、`perf top`（热点函数）、`pidstat -d`（IO 细粒度）、`/proc/zoneinfo`（page reclaim）、`/proc/interrupts`（IRQ 分布）、`turbostat`（频率 / C-state）。配 `dmesg` 看是否有 kernel warn。",
        "s4": "修复手段：`numactl --cpunodebind=N --membind=N`（绑定 NUMA）、`echo 0 > /sys/.../enabled` 关 THP、`irqbalance` 或手动绑 IRQ、`echo 3 > /proc/sys/vm/drop_caches` 清 page cache（短期）。每一项修复后 step time 通常立刻有变化。",
        "s5": "常见错排：1) 看 GPU util 高就以为 GPU 没问题；2) 忽略 NUMA topology；3) 不知道 THP 默认开会让 NCCL 慢；4) 用 top 而非 `pidstat -t -p` 看具体线程。",
        "s6": "训练节点 baseline：NUMA-aware 启动、THP disable、IRQ 绑定到 CPU 0 之外、HugePages 预留、ulimit 调大。把这些写进 prolog script。",
        "s7": "- 五条路径：NUMA / page cache / IRQ / THP / syscall\n- GPU util 不反映 system-level 问题\n- 工具：numastat / perf top / /proc/interrupts / turbostat\n- 训练节点 baseline 必备：NUMA bind + THP off + IRQ + HugePages",
        "s8": "- [ ] 你能不能用 numastat 5 分钟内确认 NUMA cross 问题？\n- [ ] 你能不能解释 THP 为什么影响 NCCL？\n- [ ] 你能不能描述 IRQ 不均对 NCCL 的影响？\n- [ ] 你能不能写一个完整的训练节点 baseline 脚本？",
    },
    {
        "title": "估算题：dataloader 在 H100 服务器（2 socket / 8 卡）上的最大吞吐与瓶颈",
        "lede": "dataloader 瓶颈出现的位置随 sample 大小和并发数变化——估算能在硬件确定前就预判瓶颈层。",
        "s1": "200KB/sample × num_workers=32 × pin_memory=True，单 worker 吞吐 ~5000 sample/s（NVMe 读 + CPU decode + pin memory），32 worker 总吞吐 ~50k sample/s（受 CPU 核数与 NVMe 带宽限制）。瓶颈：CPU decode（如果是图像）、NVMe 带宽（如果 sample 大）、PCIe（pin memory → GPU 拷贝）。8 卡需求 ~80k sample/s（batch=2k / step=20ms × 8 卡），dataloader 不太够。",
        "s2": "dataloader 链路：NVMe → page cache → CPU process → decode → tensor → pin memory → GPU。每一段都可能是瓶颈。NVMe 5-12 GB/s（200KB × 50k = 10GB/s，刚好打满）、CPU decode（jpeg 单核 ~1k/s，32 核 ~32k）、PCIe (16GB/s 双向，瓶颈出现晚)。",
        "s3": "估算分解：所需 sample/s = batch × world_size / step_time；可达 sample/s = min(NVMe / sample_size, num_workers × per_worker_throughput, PCIe / sample_size)。两者差距决定 dataloader 是否 bottleneck。",
        "s4": "扩 num_workers 不一定有用：超过 CPU 核数后切换开销盖过收益。增大 NVMe 数量（多盘 RAID0）能解 IO 瓶颈。decode 慢可改 webdataset / FFCV 风格的预 decode 格式。",
        "s5": "常见坑：1) `num_workers` 比 CPU 核还多；2) `pin_memory=True` 但不 prefetch；3) sample shuffle 没设缓冲区导致内存抖动；4) 用 jpeg 而非预 decode 让 CPU decode 是瓶颈。",
        "s6": "排查 5 步：1) profile dataloader iter time vs GPU step time；2) 测 NVMe 带宽；3) 测 CPU decode 单 worker 吞吐；4) 测 PCIe 拷贝；5) 找出最慢一段。pythorch `torch.profiler` 有 dataloader 段计时。",
        "s7": "- 链路：NVMe → CPU → tensor → pin → GPU，每段可能瓶颈\n- 200KB × 32 worker 大概 50k sample/s\n- 8 卡需求可能 80k+，需要 NVMe 多盘 / 预 decode\n- num_workers > CPU 核数有反效果",
        "s8": "- [ ] 你能不能列出 dataloader 全链路的所有可能瓶颈？\n- [ ] 你能不能为给定 sample 大小估算 NVMe / CPU / PCIe 各自上限？\n- [ ] 你能不能解释 pin_memory + prefetch 的协同？\n- [ ] 你能不能用 torch.profiler 定位 dataloader 瓶颈？",
    },
    {
        "title": "设计题：训练节点完整系统 baseline（kernel param + cgroup + ulimit + IRQ + NUMA + HugePages）",
        "lede": "训练节点上线前一份「调好的 baseline」决定后续运维的轻重——把 6 个层面写进 prolog 是 SRE 的基本功。",
        "s1": "六层 baseline：1) kernel 参数（vm.swappiness=10、vm.nr_hugepages=256、net.core.somaxconn=4096）；2) cgroup v2（CPU/IO 给 dataloader 留余量）；3) ulimit（nofile=1M、stack=unlimited）；4) IRQ 亲和（IB / NIC IRQ 绑 NUMA local CPU）；5) NUMA bind（GPU 与 CPU socket 对齐）；6) HugePages（预留 256GB）。每一项都有具体配置文件。",
        "s2": "原理：kernel 默认参数面向通用 workload，训练负载需要：少 swap（vm.swappiness 低）、大量并发 socket（somaxconn 高）、大页减少 TLB miss（HugePages）、NUMA local 内存（GPU 数据流）。每项都能让训练吞吐变化 5-30%。",
        "s3": "配置位置：`/etc/sysctl.d/`（kernel param）、`/etc/cgroup/`（cgroup）、`/etc/security/limits.d/`（ulimit）、`/proc/irq/<N>/smp_affinity`（IRQ）、`numactl --hardware`（NUMA topology）、`/etc/default/grub` + `hugepagesz=1G`（HugePages）。所有改完要重启或 reload。",
        "s4": "训练节点 baseline 一次写好，作为 image 入到所有训练节点。任何手工改都要走 PR + review。这是规模化运维的根本。",
        "s5": "常见坑：1) HugePages 预留过多影响普通进程；2) NUMA bind 写错让 GPU 数据 cross；3) IRQ 绑定漂移没监控；4) ulimit 改了但 systemd 没继承；5) cgroup IO 限制太严让 ckpt save 慢。",
        "s6": "落地：写一份 `prolog.sh` 作为 SLURM prolog，每次 job 启动前 reload 一遍 baseline。K8s 用 init container 同样思路。所有改动版本化（git）+ CI 测过。",
        "s7": "- 六层：kernel param + cgroup + ulimit + IRQ + NUMA + HugePages\n- 每项都影响 5-30% 训练吞吐\n- 写进 image + prolog，所有手工改走 PR\n- 必须有 CI 验证 baseline 是否生效",
        "s8": "- [ ] 你能不能列出训练节点必调的 kernel param？\n- [ ] 你能不能描述 NUMA bind 与 GPU topology 的对齐方式？\n- [ ] 你能不能解释 HugePages 预留多少合理？\n- [ ] 你能不能写一份完整的 prolog.sh？",
    },
]


# =====================================================================
# Run
# =====================================================================
def main():
    plan = [
        ("03-cuda-triton-custom-ops",         "CUDA / Triton / 自定义算子", 49, V03_SYNTHETIC),
        ("05-memory-communication-profiling", "显存 / 通信 / Profiling",    69, V05_SYNTHETIC),
        ("06-inference-quantization-serving", "推理量化 / 服务化",          59, V06_SYNTHETIC),
        ("07-training-inference-platform",    "训练 / 推理平台",            84, V07_SYNTHETIC),
        ("08-systems-foundations",            "Linux / 系统基础",           39, V08_SYNTHETIC),
    ]
    for stem, topic_hint, expected, synth in plan:
        process_volume(stem, topic_hint, expected, synth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
