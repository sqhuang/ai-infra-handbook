# 术语表

> 收录 11 卷语料中高频出现的英文缩写、专有名词与中文术语对应关系。每条 1–2 句白话定义 + 第一次出现的位置（用 (Vol N, QM) 形式）作为深读入口。

## 训练 / 框架

### Autograd
PyTorch 的自动求导引擎；前向期间记录每个操作对应的反向算子，反向时按拓扑序回传梯度。详见 (Vol 01, Q1)。

### `torch.compile` / PT2 / TorchDynamo / AOTAutograd / Inductor
PyTorch 2.x 编译栈一组件：`torch.compile` 是入口，TorchDynamo 在前端字节码层捕获 FX 图，AOTAutograd 联合捕获前向 / 反向，Inductor 把图降到 Triton / C++ 内核。详见 (Vol 01, Q9)。

### `torch.export`
把模型显式导出成"无 Python 依赖"的 ExportedProgram，便于送入 AOT 推理后端（TensorRT、ExecuTorch 等）。详见 (Vol 01, Q14)。

### DDP（DistributedDataParallel）
PyTorch 最经典的数据并行实现，每张卡保留完整副本，反向时通过 all-reduce 同步梯度。详见 (Vol 04, Q1)。

### FSDP / FSDP2
Fully Sharded Data Parallel；把参数 / 梯度 / 优化器状态切片到各 rank，用通信换显存。FSDP2 是基于 DTensor 的重写版，组合性更好。详见 (Vol 04, Q3)。

### ZeRO / ZeRO-Offload / ZeRO-Infinity
DeepSpeed 的分级内存优化方案：Stage 1/2/3 分别分片优化器状态、梯度、参数；Offload / Infinity 进一步把状态下沉到 CPU 内存乃至 NVMe。详见 (Vol 04, Q5)。

### DTensor
PyTorch 原生的分布式张量抽象，用 placement（Replicate/Shard/Partial）描述切分语义，是 FSDP2、TP 等的统一底座。详见 (Vol 04, Q48)。

### TorchTitan
PyTorch 官方维护的大模型训练参考实现，覆盖 FSDP2 + TP + PP + CP + FP8/FP4 组合并行的开源样板。详见 (Vol 11, Q14)。

### NeMo-Aligner / verl / OpenRLHF / AReaL
四个常见的 RL 后训练框架：NeMo-Aligner 是 NVIDIA 全栈，verl 是字节开源（rollout-train 解耦），OpenRLHF 偏轻量，AReaL 强调异步 + 高吞吐。详见 (Vol 11, Q1)。

### Transformer Engine
NVIDIA 维护的 FP8 / FP4 训练加速库，包装了 FP8 GEMM、microscaling、loss scaling 等数值技巧。详见 (Vol 11, Q14)。

## 并行

### TP / PP / DP / CP / EP / SP（张量 / 流水线 / 数据 / 上下文 / 专家 / 序列并行）
六种主要并行维度：TP 切单个矩阵乘、PP 按层切流水、DP 复制模型 + 切数据、CP 沿序列维切长上下文、EP 切 MoE expert、SP 在 LayerNorm 等处沿序列维省激活。实际系统通常是它们的组合。详见 (Vol 04, Q10)。

### Context Parallelism (CP)
沿 sequence 维度把单条样本切到多卡，让长上下文训练成为可能；attention 必须 ring 传递 KV（Ring Attention）。详见 (Vol 11, Q8)。

### Ring Attention / Star Attention
两种长上下文 attention 实现：Ring 在 CP 卡间环形传递 KV chunk，去中心化；Star 把 KV 全 gather 到一张中心卡，通信量小但 memory 集中。详见 (Vol 11, Q8)。

### 1F1B / Zero-bubble
两类流水线并行调度：1F1B 在稳态阶段交替执行 1 次前向 / 1 次反向控制 in-flight 数量；Zero-bubble 把反向拆成 dW/dX 两段精细排程，把空闲气泡压到接近 0。详见 (Vol 04, Q15)。

### Megatron-LM
NVIDIA 的大模型训练框架，业界 TP/PP/SP 实现的事实参考。详见 (Vol 04, Q40)。

### async TP
"通信和矩阵乘叠流水"的张量并行变种，把 all-gather/reduce-scatter 拆细与 GEMM 重叠，减少 TP 暴露通信。详见 (Vol 04, Q12)。

## 通信

### NCCL
NVIDIA Collective Communications Library，GPU 集合通信事实标准；all-reduce/all-gather 等都走它。详见 (Vol 04, Q2)。

### all-reduce / reduce-scatter / all-gather / all-to-all
四种核心集合通信原语：求和广播、分片求和、分片汇总、全交换。FSDP、TP、EP 各自重度依赖其中一两种。详见 (Vol 05, Q11)。

### NVLink / NVSwitch / NVL8 / NVL72
NVIDIA 卡间高速互连：NVLink 是链路、NVSwitch 是交换芯片，单机内带宽数百 GB/s；NVL8 是单机 8 卡全互连，NVL72 是 GB200 一柜 72 卡 NVLink 域。详见 (Vol 05, Q20)。

### NVLink 5
Blackwell 时代第五代 NVLink，单 link 100 GB/s 单向，每卡 18 link，总双向带宽 1.8 TB/s。详见 (Vol 11, Q12)。

### IB（InfiniBand）/ RoCE / NDR
两类高带宽 RDMA 网络与代次：IB 是专用网络，RoCE 是跑在以太网上的 RDMA；NDR 是 IB 400G/800G 代次。多机训练的常见选择。详见 (Vol 05, Q23)。

### NVLS / SHARP
在 NVSwitch / IB 交换机上做 in-network reduction 的能力，可以把 all-reduce 的部分计算下沉到网络。详见 (Vol 05, Q24)。

### DeepEP
DeepSeek 开源的 MoE 专家并行通信库，针对 all-to-all 路由做了大量低延迟优化。详见 (Vol 11, Q6)。

### GPUDirect RDMA
让 NIC 直接 DMA 到 GPU 显存而不绕 CPU 内存的能力，是跨机训练通信效率的基石。详见 (Vol 05, Q22)。

### UCX
开源高性能通信中间件，封装 IBVerbs / TCP / shared memory，许多自定义 RL 同步通道、DeepEP 后端用它。详见 (Vol 11, Q4)。

## 显存与性能

### KV cache
解码阶段缓存的历史 K/V 张量，避免每步重算前缀；推理显存主要被它吃掉。详见 (Vol 06, Q4)。

### KV cache 量化（FP8 / INT8 / INT4 KV）
把 KV cache 用低精度存储，显存压一半到 1/4；要求 attention kernel 原生支持低精度读入（FA-3 FP8 KV）。详见 (Vol 11, Q9)。

### 激活重计算（activation checkpointing）
反向时按需重新前向算激活而非全量保存，是训练显存最重要的杠杆之一。详见 (Vol 05, Q3)。

### MFU / HFU
Model FLOPs Utilization / Hardware FLOPs Utilization；衡量训练算子有效利用峰值算力的比例，HFU 把重计算等"重复 FLOPs"也算进来。详见 (Vol 05, Q40)。

### mixed precision / FP8 / FP4 / MXFP4 / NVFP4
混合精度训练：权重保存 FP32、计算与梯度走 BF16/FP16；FP8 已在 H100/H200 训练常态化，FP4 / MXFP4 / NVFP4 是 B200 时代的微缩放新格式。详见 (Vol 11, Q14)。

### microscaling
每个小 block（典型 32 元素）一组 scale 的量化策略，让 outlier 不污染整 tensor；FP4 训练能跑起来的根本原因。详见 (Vol 11, Q14)。

### gradient bucket
DDP / FSDP 把多个梯度合并成一个 bucket 再发起 all-reduce，减少小通信启动开销。详见 (Vol 04, Q2)。

### FlashAttention / FA-1 / FA-2 / FA-3
内存高效的 attention 实现，按 tile 分块 + softmax 重写把 O(N²) 中间矩阵从 HBM 拿掉；三代分别提出 tiling、并行优化、Hopper TMA/WGMMA 重写。详见 (Vol 03, Q30)。

### PagedAttention / RadixAttention
两种 KV cache 管理算法：PagedAttention（vLLM）按 page 管理把利用率推到接近 100%；RadixAttention（SGLang）用 radix tree 复用共享前缀的 KV。详见 (Vol 06, Q4)。

### SDPA / FlexAttention
PyTorch 的两个 attention API：`scaled_dot_product_attention` 自动按硬件选 FlashAttention / mem-efficient / math 后端；FlexAttention 让用户用 DSL 描述自定义 mask / 偏置同时仍走 fused 内核。详见 (Vol 01, Q16)。

### MLA（Multi-head Latent Attention）
DeepSeek 提出的 attention 变体，把 KV 投影到一个低维 latent 空间存储，显著压缩 KV cache（比 GQA 还小几倍）。详见 (Vol 06, Q4)。

### GQA / MQA
Grouped-Query Attention / Multi-Query Attention，让多个 query head 共享同一组 KV head，是 Llama 系列的 KV 压缩主流方案。详见 (Vol 06, Q4)。

### Tensor Memory（B200）
Blackwell 引入的异步张量搬运/暂存单元，是 Hopper TMA 的进化，支持更精细的 producer/consumer pipeline。详见 (Vol 11, Q12)。

### NTK / YaRN
两种 RoPE 长上下文外推方法：NTK（neural tangent kernel scaling）按维度调节 base，YaRN（Yet another RoPE extensioN）综合 NTK + 注意力温度，主流长上下文模型都用 YaRN。详见 (Vol 11, Q15)。

## 推理与服务化

### vLLM / SGLang / TensorRT-LLM
三大主力 LLM 推理引擎：vLLM 因 PagedAttention + continuous batching 成为开源默认起点；SGLang 强调结构化 output 与 RadixAttention；TensorRT-LLM 深度绑定 NVIDIA 硬件，吞吐 / 时延上限通常更高但可移植性差。详见 (Vol 06, Q3)。

### vLLM v0 / v1
vLLM 的两代调度器：v0 是 PagedAttention + continuous batching 的早期形态；v1 重写执行栈，集成 cuda graph、TP、speculative decoding 等。详见 (Vol 11, Q18)。

### continuous batching
推理调度策略：每步动态加入 / 移出请求，而不是等整个 batch 一起结束，是高吞吐推理的基线。详见 (Vol 06, Q5)。

### chunked prefill
把长 prompt 的 prefill 拆成多个 chunk，与 decode 阶段交错调度，平滑 GPU 占用。详见 (Vol 06, Q6)。

### speculative decoding / Medusa / EAGLE / Lookahead / Jacobi
"小模型起草、大模型验证"的解码加速套路；Medusa 用多 head 并行预测，EAGLE 用轻量 draft head 利用 hidden state，Lookahead/Jacobi 是 fixed-point 迭代。详见 (Vol 11, Q11)。

### prefix caching
跨请求复用相同 prompt 前缀对应的 KV cache，对系统提示词、agent 多轮对话尤其有效。详见 (Vol 06, Q13)。

### TTFT / TPOT / ITL / p99 / SLO
推理服务的核心 SLO 词汇：TTFT 是首 token 时延、TPOT/ITL 是后续每 token 时延，通常按 p95/p99 分位数定 SLO 而非平均值。详见 (Vol 06, Q20)。

### disaggregated prefill/decode / Mooncake
把 prefill 与 decode 拆到不同 GPU 池，分别按各自的瓶颈（compute vs memory bw）扩容；Mooncake、DeepSeek inference 是公开代表。详见 (Vol 11, Q10)。

### LMCache
跨节点 KV cache 服务，让多个推理实例共享 prefix / persistent KV，对 agent / multi-tenant 场景命中率显著提升。详见 (Vol 11, Q19)。

### AWQ / GPTQ / SmoothQuant
三种主流权重量化算法：AWQ 看激活分布选保护通道，GPTQ 用二阶信息校准，SmoothQuant 用激活迁移让 W8A8 稳定。详见 (Vol 06, Q30)。

### GGUF / Marlin / Machete
三个常被混提的工程件：GGUF 是 llama.cpp 系生态的统一权重格式（端侧 / 本地推理常见）；Marlin 是 W4A16 GEMM 的高性能 kernel；Machete 是 vLLM 的 W4A8 GEMM 后继。详见 (Vol 06, Q33)。

### tree decoding / tree mask
投机解码里多个 candidate sequence 形成 tree，主模型一次 forward 用 tree mask 验证整棵 tree。详见 (Vol 11, Q11)。

### flash-decoding
FA 在 decode 阶段的特化（KV 维度 split + 二次 reduce），让单序列 decode 也能用满 SM。详见 (Vol 03, Q30)。

## 后训练 / RL

### RLHF / RLAIF / RLVR
三种 reward 来源：RLHF 用人类偏好训练的 reward model；RLAIF 用更强 LLM 当 judge；RLVR 用规则可验证奖励（代码跑通 / 数学答对）。详见 (Vol 11, Q3)。

### PPO / GRPO / DPO / KTO / Reinforce++
五种主流后训练算法：PPO 经典 RL 需 critic；GRPO 组内归一化无 critic（DeepSeek 路线）；DPO/KTO 是 off-policy 偏好优化，不需要 rollout；Reinforce++ 是无 critic 的 advantage 变体。详见 (Vol 11, Q21)。

### rollout pool / train pool
RL 解耦架构里的两个 GPU 池：rollout pool 跑 vLLM/SGLang 做高吞吐采样，train pool 跑 FSDP/Megatron 做梯度更新。详见 (Vol 11, Q1)。

### staleness
rollout 用的 actor 权重与 train 当前权重的步数差；解耦 RL 里 "off-policy 漂移程度" 的核心指标。详见 (Vol 11, Q4)。

### capacity factor
MoE 里给每个 expert 设的 token 数上限（典型 1.25），超过部分被 drop 或 re-route。详见 (Vol 11, Q7)。

### load balancing loss / aux-loss-free
两种 MoE 防止 router collapse 的手段：load balancing loss 把"分散度"加进 loss；aux-loss-free 用 expert-wise bias 自动调节（DeepSeek-V3）。详见 (Vol 11, Q7)。

### Constitutional AI
Anthropic 提出的 RLAIF 范式，用一组成文原则 + LLM judge 替代人类标注。详见 (Vol 11, Q3)。

### reward hacking
模型学到"骗 reward model 给高分而非真的好"的退化模式；RLVR 主要为解决这一问题，但 verifier 本身仍可能被 hack。详见 (Vol 11, Q3)。

## 编译器与 IR

### XLA / HLO
Google 的加速线性代数编译器及其中间表示 HLO；JAX/TF 的默认后端。详见 (Vol 02, Q1)。

### MLIR / IREE
LLVM 系新一代多层 IR 框架与基于它的端到端编译运行时；TritonIR、StableHLO、IREE 都基于 MLIR。详见 (Vol 02, Q5)。

### TVM
开源深度学习编译器，强调 schedule + autotune 的编译范式。详见 (Vol 02, Q8)。

### ONNX Runtime / TensorRT
两个传统的图级推理后端：ORT 是微软的跨框架推理 runtime，支持多 EP；TensorRT 是 NVIDIA 的图级推理优化器与 runtime，GPU 推理的传统重型武器。详见 (Vol 02, Q12)。

### AOT vs JIT
Ahead-of-Time 与 Just-in-Time 编译；前者部署期固定形状，后者运行时按 shape 触发。详见 (Vol 02, Q4)。

### Triton DSL
OpenAI 开源的 Python-like GPU kernel DSL，吃掉 CUDA 的大量样板，是 TorchInductor、FlashAttention 的内核语言。详见 (Vol 03, Q3)。

### CUTLASS
NVIDIA 的高性能 GEMM 模板库，FA-3、Marlin、TensorRT-LLM 的底层 building block。详见 (Vol 03, Q10)。

## 硬件与平台

### H100 / H200 / B200 / GB200
NVIDIA Hopper（H100/H200，HBM 容量不同）与 Blackwell（B200，FP4 + 更大 HBM）系列旗舰训练 / 推理 GPU；GB200 是双 B200 + Grace CPU 的 superchip。详见 (Vol 11, Q12)。

### HBM3 / HBM3e / HBM4
GPU 上的高带宽显存代际，决定单卡内存带宽上限。详见 (Vol 05, Q43)。

### TMA / Tensor Core
两类核心计算 / 搬运单元：Tensor Core 是矩阵乘累加专用单元（FP16/BF16/FP8 算力主力），TMA 是 Hopper 引入的异步张量搬运单元，FlashAttention-3、CUTLASS 大量利用。详见 (Vol 03, Q5)。

### NUMA
Non-Uniform Memory Access；多 socket 服务器里 CPU 访问不同内存节点延迟不一致，会影响数据加载与 NCCL 拓扑。详见 (Vol 05, Q33)。

### Xid / ECC / DCGM
节点健康监测的核心信号：Xid 是 NVIDIA 驱动报错码、ECC 是显存纠错事件、DCGM 是集群级 GPU 健康与遥测的事实标准。详见 (Vol 07, Q40)。

### MIG / MPS
两种 GPU 切分手段：MIG 把一张 GPU 切成多个硬件分区，MPS 是 CUDA 进程级的软共享。详见 (Vol 07, Q34)。

### NCCL_TOPO_FILE
NCCL 拓扑配置文件，自动探测错误时可手动指定，避免选错通信路径。详见 (Vol 05, Q26)。

### Lustre / GPFS / S3 / NVMe
四类常见存储栈：Lustre/GPFS 是高性能并行文件系统，S3 是对象存储，NVMe 是节点本地高速盘；训练数据与 checkpoint 通常分层放置。详见 (Vol 05, Q50)。

### MI300X / TPU v5p / TPU v6 / 昇腾 910B/910C / 寒武纪
非 NVIDIA 主流加速器：AMD MI300X（HBM 192GB 早于 B200）、Google TPU v5p/v6（pod 互联强）、华为昇腾 910B/910C（国产替代主力）、寒武纪（推理为主）。详见 (Vol 09, Q8)。

## 调度 / 编排

### SLURM
HPC 圈的事实调度器；`sbatch`、`gres`、prolog/epilog、preempt/requeue 等都是它的组件。详见 (Vol 07, Q5)。

### Volcano / Kueue
Kubernetes 上的批调度器：Volcano 偏 gang scheduling，Kueue 是 Kubernetes 官方的队列层。详见 (Vol 07, Q15)。

### Ray / Ray Serve / KServe
Ray 是分布式 Python 计算框架（RL rollout 主流）；Ray Serve 是 Ray 生态的服务框架；KServe 是 Kubernetes 上的标准化推理 CRD。详见 (Vol 07, Q22)。

### Capacity Block / spot / RI
三种 GPU 容量产品：Capacity Block 是按整段时间预订一组 GPU 的 burst 容量；spot/preemptible 是可被随时回收的低价容量；RI 是长周期预留实例，单价低但锁定容量。详见 (Vol 09, Q5)。

### tokens/$
单位成本能产出多少 token；推理业务的核心单位经济学指标。详见 (Vol 09, Q15)。

### JCT（Job Completion Time）
作业从入队到结束的端到端时间；训练平台的核心 SLO 维度之一。详见 (Vol 07, Q8)。

### gang scheduling
对一个分布式作业的所有 worker "同时调度成功才下发"的策略，避免 partial 死锁；Volcano / SLURM 都原生支持。详见 (Vol 07, Q15)。

### preempt / requeue
两种作业回收机制：preempt 是抢占（运行中作业被打断），requeue 是失败重排队；spot 资源主要靠这两条机制。详见 (Vol 09, Q5)。

## 数据与评测

### eval harness / lm-eval-harness
开源 LLM 评测框架，覆盖 MMLU / GSM8K / HumanEval 等百余 benchmark；社区报数的事实标准。详见 (Vol 09, Q20)。

### NeedleInHaystack
长上下文评测，往长 prompt 里塞一个特殊事实，看模型能不能在 1M context 里"找出针"。详见 (Vol 11, Q15)。

### Constitutional / Anthropic HH
两类 RLHF 数据集：Constitutional 是基于成文原则的 LLM-judged；HH（helpful/harmless）是人类标注的对比偏好。详见 (Vol 11, Q3)。

### perplexity / loss curve
两个最基础的训练监控量：perplexity 是 exp(NLL)，loss curve 是 train/val loss 随 step 的曲线。是 FP4/FP8 等量化训练的对照基准。详见 (Vol 11, Q14)。

## 安全与隔离

### gvisor / firejail / seccomp
三类沙箱机制：gvisor 用户态 kernel 重新实现 syscall 过滤；firejail Linux namespace + seccomp；seccomp 内核级 syscall 白名单。详见 (Vol 11, Q3)。

### prompt injection / jailbreak
两类 LLM 安全攻击：prompt injection 让 LLM 忽略系统 prompt 执行恶意指令；jailbreak 让 LLM 突破安全限制。详见 (Vol 09, Q25)。

### tenancy / multi-tenancy
单租户 vs 多租户隔离；多租户推理要做 prompt 隔离 + cache 隔离 + quota 隔离。详见 (Vol 09, Q22)。

## 端侧 / 本地推理

### llama.cpp / GGUF
端侧 / 本地推理事实工具：llama.cpp 是 C++ 实现的轻量推理引擎，GGUF 是它统一的量化权重格式。详见 (Vol 06, Q45)。

### MLX / Core ML / ExecuTorch
三个端侧推理框架：MLX 是 Apple Silicon 的训练 + 推理；Core ML 是 Apple 设备的部署 SDK；ExecuTorch 是 PyTorch 官方端侧部署路径。详见 (Vol 06, Q45)。

### W4A16 / W4A8
两类 4-bit 权重量化：W4A16 权重 4-bit 激活 FP16，Marlin 是代表 kernel；W4A8 权重 4-bit 激活 INT8，端侧更激进。详见 (Vol 06, Q33)。
