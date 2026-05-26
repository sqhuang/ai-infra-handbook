# 后训练与新一代推理范式卷

## 主题边界
本卷聚焦 2025–2026 年快速成熟的几条工程主线：RL 后训练（PPO/GRPO/DPO/RLAIF）的 infra 形态、MoE 与专家并行推理、长上下文训练 / 推理（Context Parallelism、Ring Attention、KV 量化）、解耦推理（disaggregated prefill/decode）、投机解码（Medusa、EAGLE、Lookahead、Jacobi），以及 Blackwell 时代 FP4 / NVL72 / Tensor Memory 的实战经验。这些主题在卷 04 / 05 / 06 已有零散涉及，本卷把它们集中重写一遍，便于体系化复习与跨子题对照。

## Q1. RLHF / RL 后训练的 infra 形态有哪些？rollout 与 train 共置还是解耦怎么选？

> 🔴 专家 · 后训练的 rollout 既要"像推理一样高吞吐"，又要"像训练一样跟权重一致"——把 rollout 跟 train 揉在一台机器还是拆成两套池，是决定整条流水线 cost / SLO 形态的根本选择。

### 1. 核心结论
现代 RL 后训练（PPO、GRPO、Reinforce++ 等）由三段循环组成：actor 用当前策略 rollout 生成轨迹、critic / reward model 打分、optimizer 用得到的 advantage 更新 actor。infra 形态的主要分歧在于「rollout 引擎与 train 引擎是不是同一组 GPU」。共置（co-located，OpenRLHF 早期、TRL、vanilla DeepSpeed-Chat）实现简单、权重无需迁移、但 rollout 阶段大量算力闲置；解耦（disaggregated，Anthropic / OpenAI / xAI / DeepSeek 内部、verl、NeMo-Aligner、AReaL）把 rollout pool 用 vLLM / SGLang 撑起来，train pool 跑 FSDP / Megatron，权重通过显存对显存的高速通道或共享 checkpoint 周期同步，是当前大规模做法。

选择维度可以收敛成三条：rollout/train 算力比、权重同步频率与冷启动开销、调度复杂度。模型越大、rollout 越占主导（特别是 long-CoT / 多步 tool use）就越倾向解耦；模型小、rollout 短、迭代节奏快则共置更省心。

### 2. 底层原理
RL 后训练对 infra 的真正挑战不是算法，而是"在两种执行形态间高频切换"。Rollout 阶段是 decode 主导的推理负载：KV cache、continuous batching、Speculative decoding、PagedAttention 都希望 batch 大、context 长、调度激进；而 train 阶段是大批 forward + backward + optimizer，更看重 throughput、AllReduce 带宽与 activation checkpoint。两者对显存布局、kernel 选择、并行策略的偏好截然不同。

共置方案常见做法是把模型加载两份"模式"：训练态用 FSDP 的 sharded 参数 + 优化器状态，rollout 时把参数 unshard 成全副本送进 vLLM；rollout 结束再返回训练态。这种切换的代价是显存峰值（要同时容纳两套元数据）以及 unshard / shard 的通信。

解耦方案中，actor 权重在 train pool 训完一步后，需要尽快"刷"到 rollout pool。常用的几种通道：NCCL p2p 直接拷贝、NVSwitch 内 broadcast、写共享存储再让 rollout pool reload、走 RDMA write。生产里通常是「step-aligned 异步同步」：train 出新 weight A，rollout 仍用旧 weight B 跑完已起飞的轨迹，新轨迹用 A，避免阻塞。

### 3. 关键机制 / 流程 / 数据结构
第一，rollout 引擎的选型。vLLM、SGLang、TRT-LLM 都能跑 rollout，但要 expose 「在线热替换权重」能力。vLLM 的 `LLM.llm_engine.model_executor.driver_worker.model_runner.model.load_weights` 是常见入口；SGLang 提供 `update_weights_from_*` API；TRT-LLM 通常通过 engine rebuild。

第二，advantage 估计。PPO 用 GAE 需要 critic 网络（多一份模型显存），GRPO 用组内归一化省掉 critic（DeepSeek 提出，国内开源社区主流），Reinforce++ / RLOO / DPO（off-policy） 各有变体。GRPO 让 rollout / train 的 GPU 比例可以更激进倾向 rollout。

第三，奖励来源。RLHF 的 reward model 是显式打分模型；RLAIF（Anthropic Constitutional AI 风格）用一个更强模型当 judge；RLVR（DeepSeek-R1）用规则可验证奖励（代码跑通、数学答对）。后两类的 infra 形态显著影响 rollout cluster 大小：RLVR 几乎不需要 reward model 推理。

第四，On-policy 与 off-policy 边界。每 N 步同步一次权重，期间 rollout 在用"略旧"的策略；超出某个 KL 阈值前都能视为近似 on-policy。这是 disaggregated infra 能成立的工程前提。

### 4. 工程权衡 / 性能影响
共置最大问题是"利用率塌方"。一个 70B 模型 rollout 一条 8k token 的 long-CoT 轨迹可能花十几秒，期间所有训练算力闲置；如果用 256 张卡共置，rollout 阶段平均利用率可能只有 30–40%。解耦能让 rollout pool 跑 vLLM 的高吞吐 batched decoding，单卡 tokens/s 是共置方案的 3–10 倍。

解耦的代价是「同步与一致性」。如果同步频率太低，rollout 策略与 train 策略偏离过大，advantage 估计带偏 bias；如果太高，每次都要全模型 broadcast，几十 GB 数据在 cluster 内来回跑，反而成新瓶颈。NCCL p2p（同 NVSwitch domain）能把同步压到 1–3 秒，RDMA 跨机一般 5–15 秒。

调度上，解耦让 rollout 与 train 形成「生产者-消费者」管道，rollout queue 是天然的 buffer。生产里要监控的两个量：buffer 长度（太短说明 rollout 不够，train 饥饿）、staleness（buffer 里轨迹用的权重版本与当前权重相差多少）。

### 5. 常见追问 / 易错点
第一，为什么不能直接拿 SFT 的训练代码改一改跑 RL？因为 SFT 不需要 rollout，所以也不需要推理引擎；RL 需要"模型作为函数"的高吞吐采样能力，PyTorch eager 单 batch 跑 decode 慢到不可用，必须接 vLLM / SGLang 一类的引擎。

第二，rollout 用 FP8 / INT8 量化推理可不可以？可以，但要校准量化误差对 reward 估计的影响。RLVR 场景因为只看「答案对错」，对量化更鲁棒；RLHF 用 reward model 的连续打分则更敏感。

第三，critic 是不是必须？PPO 算法上需要 value head，但工程上：1) GRPO 完全不要 critic；2) PPO 的 critic 可以共享 actor 主干只加 head；3) RLOO / Reinforce++ 也不需要 critic。"是否要 critic" 直接决定要不要多准备一份模型权重和优化器状态，是 infra 容量规划的关键开关。

第四，DPO 的位置。DPO / IPO / KTO 是 off-policy、不需要 rollout 的偏好优化，infra 形态就是普通 SFT 训练，不需要本卷讨论的两段式。但效果上 DPO 通常做不到 PPO/GRPO 的上限，所以"省 infra"是要付效果代价的。

### 6. 实践建议
团队规模小、模型 13B 以下、迭代快，共置足够。大模型、long-CoT、想做 R1 类 RLVR 的，直接上解耦。两种之间没有"逐步升级"的过渡形态——硬要做就是把 train 与 rollout 引擎从同一份代码彻底拆成两套服务。

权重同步通道优先 NCCL p2p > RDMA write > 共享存储。同步频率从「每步」开始放宽到「每 N 步」直到 KL(actor‖rollout) 超阈值再回收。监控指标的最小集：rollout tokens/s、train tokens/s、buffer 长度、staleness、KL 漂移。

面试回答时强调「不是算法选 PPO vs GRPO 这层选择，而是 rollout/train 算力比和权重同步通道这两条决定 infra 形态」。能讲清「同步频率与一致性边界」就是高分。

### 7. 30 秒速答
- RL 后训练 infra 二选一：rollout 与 train 共置还是解耦成两个池。
- 解耦后 rollout 用 vLLM 高吞吐采样，train 用 FSDP/Megatron，权重周期同步。
- 易踩坑：同步频率太高变瓶颈，太低 staleness 让 advantage 估计带偏。
- 加分关键词：on-policy 边界、KL 漂移、step-aligned 异步同步、weight bridge。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 rollout 与 train 共置 vs 解耦的核心差异？
- [ ] 你能不能解释为什么共置方案 GPU 利用率会塌方到 30–40%？
- [ ] 你能不能举一个解耦方案下权重同步的具体通道（NCCL p2p / RDMA / 共享 ckpt）？
- [ ] 你能不能说出 buffer staleness 过大时的 RL 训练失败模式？

## Q2. GRPO 与 PPO 在工程实现上有什么差异？怎么影响 infra 设计？

> 🔴 专家 · GRPO 在算法层省掉了 critic、在工程层省掉了一份 70B 模型——这一个差异决定了你需要的 GPU 是 256 还是 384。

### 1. 核心结论
GRPO（Group Relative Policy Optimization）是 DeepSeek 在 DeepSeekMath / R1 中推开的 RL 算法，核心思想是「同一个 prompt 多次采样得到一组回答，用组内奖励的均值方差归一化代替 value function」。这相当于把 PPO 里 GAE + critic 那一支换成「组内 z-score」，因此 GRPO 完全不需要 critic 网络。在 infra 层面这意味着：少一份与 actor 同等大小的模型权重 + 优化器状态、少一支训练前向反向、少一套学习率调度，对显存与卡数的影响非常显著。

### 2. 底层原理
PPO 的 advantage 是 GAE：A_t = δ_t + γλ A_{t+1}，其中 δ_t = r_t + γ V(s_{t+1}) − V(s_t)。这里的 V 来自 critic 模型，跟 actor 一样大、一样训练。GRPO 用一个 prompt 采 G 条（典型 G=4–16）回答，组内奖励减去均值除以标准差作为 advantage：A_{g,i} = (r_{g,i} − mean_g) / std_g。这种归一化对长序列稀疏奖励（只在最后一个 token 给奖励）特别友好，因为不需要 step-level value 估计。

KL 约束部分两者都保留，对 reference policy（通常是 SFT 后的模型）做 KL 惩罚或加到 reward 里。GRPO 的 KL 写在 loss 里更稳，PPO 习惯把 KL 加到 reward 上。

### 3. 关键机制 / 流程 / 数据结构
第一，rollout 形态。PPO 每个 prompt 通常采 1 条；GRPO 每个 prompt 采 G 条。这意味着 rollout pool 的吞吐需求是 PPO 的 G 倍——但每组内的 prefix 是同一个 prompt，prefix caching 能把这 G 倍的代价压下来。

第二，显存与卡分配。70B 模型，PPO 需要 actor + critic + reference + reward 四份模型（reference / reward 可以 freeze、可以 offload），GRPO 砍掉 critic 后是三份。如果 reference 用同一份 actor + 控制 KL，能压到两份。

第三，loss 形态。GRPO loss 形态接近 importance-weighted policy gradient + KL，所有量都 token-level 累加。实现时要注意 padding token 的 mask 与 token-level 归一化基数，不然 long 回答会被惩罚或被放大。

### 4. 工程权衡 / 性能影响
GRPO 省 critic 的代价是：1) 组内归一化在 G 较小（4 以下）时方差大，advantage 噪声高；2) 长尾分布的 reward 容易让 std 接近 0，需要 clipping；3) 因为没有 value baseline，对稀疏奖励的样本效率比 PPO 略低，但用 G 倍样本通常能补回来。

infra 实践里 GRPO 让你可以把"省下来的卡"全部投入 rollout pool，rollout/train 比例 3:1 甚至 5:1 在 R1 类训练里很常见。这对 RLVR 是非常合适的——你需要让模型采集大量 long-CoT 样本，让规则奖励有足够基数。

PPO 在 reward 噪声小、可以训 critic 的常规 RLHF 场景仍然更稳。DeepSeek 自己也强调 GRPO 主要在 RLVR / 数学代码场景上验证，通用对话 PPO + reward model 仍然是参考方案。

### 5. 常见追问 / 易错点
第一，G 应该取多少？经验上 G=8–16 是 sweet spot。低于 4 噪声过大，高于 32 边际收益递减、rollout 显著放慢。

第二，组内 std≈0 怎么办？clipping 到一个 floor（如 1e-4），或者直接跳过这一组。生产里要监控 "degenerate group rate" 这个指标。

第三，KL 是 token-level 还是 sequence-level？建议 token-level，跟 loss 一致。Sequence-level KL 在长回答上易爆。

第四，重复采样浪费算力吗？不浪费，因为 prefix（prompt）只跑一次 prefill，G 条回答共享 KV cache。这是 GRPO 与 prefix caching 天然契合的关键。

### 6. 实践建议
做 RLVR / 数学 / 代码：先 GRPO，原生匹配。做开放对话 RLHF：仍以 PPO + reward model 为基线，GRPO 作为「砍 critic 省卡」的备选。

实现时关键的工程开关：组内 normalization 的 clip floor、KL coefficient（GRPO 常用 0.001–0.01）、token-level loss mask、prefix caching 是否开启。这四个开关每一个错都会让训练崩或 reward 不涨。

面试问到 GRPO 时强调「砍 critic 是算法+infra 的联合优化点」——只讲算法是讲一半，能把"省下来的卡可以投到哪里、对 rollout/train 比例的影响"讲清楚才有体感。

### 7. 30 秒速答
- GRPO 用组内 z-score 替代 critic 的 V(s)，一刀砍掉一份与 actor 同等大小的模型。
- 一个 prompt 采 G=8–16 条，prefix prefill 共享 KV，rollout 吞吐压力被 prefix caching 吸收。
- 易踩坑：组内 std≈0（全过或全不过）让 advantage 爆炸，必须 clip floor 或丢弃 degenerate group。
- 加分关键词：aux-loss-free、token-level loss mask、degenerate group rate、3:1 rollout/train 配比。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 GRPO 与 PPO 在 advantage 估计上的差异？
- [ ] 你能不能解释为什么砍 critic 能直接释放 1/4 集群算力？
- [ ] 你能不能举一个 G 取值（4 / 8 / 16 / 32）对噪声与吞吐的具体影响？
- [ ] 你能不能说出 GRPO 在通用对话 RLHF 上不如 PPO 的原因？

## Q3. RLVR（可验证奖励）相比 RLHF 在 infra 上简化了什么？还需要哪些新组件？

> 🔴 专家 · 你不再需要训一个 reward model、不再需要怕 reward hacking 漂出哲学层面，但你需要一个能跑代码 / 跑数学验证器的沙箱集群。

### 1. 核心结论
RLVR（Reinforcement Learning with Verifiable Rewards，DeepSeek-R1 / Kimi K1.5 / o1 类系统的训练范式）用「程序化的、可验证的奖励信号」替代 RLHF 里那个昂贵的 reward model。常见 reward 来源：单元测试通过率（代码）、数学答案是否等于参考（数学）、形式验证器、API 调用是否成功。infra 上的影响是「reward model pool 消失，verifier service 出现」——后者通常是 CPU-bound 的隔离沙箱集群，跟 GPU 训练完全解耦。

### 2. 底层原理
RLHF 把 human preference 蒸馏成一个奖励模型（reward model，RM），训练 actor 时拿 RM 当 oracle。问题是 RM 也是模型，会被 actor reward hacking：模型学到"如何骗 RM 给高分"而不是"真的答对"。RLVR 把 reward 钉死在客观正确性上：能跑通的代码、能算对的数学、能解析的 JSON——actor 没法骗一个 Python 解释器。

代价是「适用面窄」。开放对话、创意写作没有客观可验证的 reward，RLVR 不适用。RLVR 主要在数学、代码、形式推理、工具调用等可验证领域大放异彩。

### 3. 关键机制 / 流程 / 数据结构
第一，verifier 沙箱。代码 verifier 是个隔离的 Python / 多语言执行环境（常用 firejail、gvisor、Docker、custom seccomp）；数学 verifier 是 SymPy / Mathematica / LaTeX 比对；形式证明用 Lean / Coq。每条样本可能要并发跑数十个测试用例，吞吐压力大。

第二，奖励聚合。pass@k 通常聚合成 0/1 或 0–1 区间。GRPO 配 RLVR 时，组内 reward 多为 0/1，std 会塌陷（要么全过要么全不过），所以 R1 论文里有 "filter out groups with std=0" 的处理。

第三，与 long-CoT 的耦合。RLVR 几乎只对长 CoT 推理有用——短答案的数学 / 代码用 SFT 就够了。这导致 rollout 极长（数千甚至数万 token），rollout pool 必须是高吞吐推理引擎，且要支持 long context。

### 4. 工程权衡 / 性能影响
省掉 RM 是真省。RM 通常跟 actor 同级别（7B / 30B / 70B），算力占整个集群 1/3 到 1/4。RLVR 把这部分算力释放出来，加到 rollout 或 train 池都行。

新引入的 verifier 是 CPU-bound（代码 / 数学执行）或 GPU 但小（一些用更小模型当 LLM-as-judge 的混合方案）。verifier cluster 的容量规划：rollout 吞吐 × 平均测试用例数 × 单用例耗时。代码 verifier 单核可能 100ms-10s（取决于算法题难度），需要的 vCPU 数会让人意外。

安全是 RLVR 不能跳过的工程点。verifier 执行的是模型生成的代码 / 命令——隔离不到位等于让模型有 RCE 能力。生产里通常用 firejail + 网络断开 + 只读 fs + 时间限制 + 内存限制五件套。

### 5. 常见追问 / 易错点
第一，reward hacking 真的没了吗？不是消失，是变形。模型仍可能学到「找输出格式漏洞」「猜测 verifier 实现细节」「构造让 verifier 超时但跑到检查前的程序」。verifier 的健壮性是新的 reward hacking 战场。

第二，没有客观 reward 的领域能否混合？可以，称为 hybrid reward：RLVR + RM 加权。但要小心两路 reward 的尺度统一与 reward shaping。

第三，verifier 怎么并发？rollout pool 出来的样本进入 verifier queue，verifier 进程池（pidpool）并发处理，结果回写。生产里 verifier 用 Ray Actor / Celery / Kubernetes Job 都见过；规模上来后用专门的 sandbox cluster（比如 Anthropic 公开的 sandboxing infra）。

第四，verifier 失败怎么处理？timeout / 测试用例本身有 bug / 测例不全。常见策略：timeout 给中性 reward（不 punish 也不奖励）；用例 bug 在数据准备阶段拦掉；测例不全用 hidden test 评估 generalization。

### 6. 实践建议
做数学 / 代码 RLVR：先把 verifier 沙箱建起来，再写 RL 训练代码。Verifier 是最容易被低估的工程量。

容量规划：rollout pool : verifier pool : train pool ≈ 3 : 1 : 1 是常见比例，但代码题密集场景 verifier 可能要顶到 1.5。

监控指标：verifier throughput、average verification latency、timeout rate、reward distribution（特别关注 std=0 的组占比）、reward hacking 探测（模型输出结构异常率）。

面试讲 RLVR 突出「reward model 消失 + verifier 出现」这一对动力学变化，再讲 verifier 是新的 attack surface（既是被攻击面，也是模型学习去探测的对象）。

### 7. 30 秒速答
- RLVR 用程序化可验证奖励（单测、数学等式、形式证明）替代昂贵的 reward model。
- reward model pool 消失，CPU-bound 的 verifier 沙箱集群出现，两者算力形态完全不同。
- 易踩坑：verifier 执行模型生成代码，隔离不到位等于给模型 RCE 能力；std=0 的组要过滤。
- 加分关键词：firejail/gvisor 五件套、pass@k 聚合、verifier timeout、reward hacking 变形。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 RLVR 相对 RLHF 在 reward 信号上的本质差异？
- [ ] 你能不能解释为什么 RLVR 几乎只在 long-CoT 推理任务上有效？
- [ ] 你能不能举一个 verifier 沙箱的具体隔离方案（firejail / gvisor / seccomp）？
- [ ] 你能不能说出 RLVR 下 reward hacking 的新形态（如骗过 verifier 实现细节）？

## Q4. PPO/GRPO 的权重同步通道有哪几种？怎么权衡？

> 🔴 专家 · 你训完一步要把 70B 权重刷到 rollout pool，是走 NCCL p2p 一秒搞定，还是走 NFS 慢慢拷？这是 RL 后训练吞吐的核心瓶颈之一。

### 1. 核心结论
解耦 RL 训练里 actor 权重要持续从 train pool 同步到 rollout pool。常见通道按速度排序：NVSwitch / NVLink 内 broadcast > NCCL p2p 跨节点 > RDMA write 跨机 > 共享内存映射 > 共享文件系统 reload。生产里通常组合用：同 NVSwitch domain 内走 NVLink、跨机走 RDMA / NCCL、冷启动 / 故障恢复走文件。

### 2. 底层原理
train pool 一般用 FSDP / Megatron 把权重 shard 到多张卡；rollout pool 用 vLLM / SGLang 通常是 TP（张量并行）或 PP，权重也是 sharded。两者 shard 形态不同——FSDP 是 ZeRO-3 风格的 parameter-dim shard，TP 是 row/column-wise。同步前要做 reshard：先把 FSDP 的 shard gather 成全参，再按 vLLM 的 TP 切分发出去。这个 reshard 步骤的计算和通信都非平凡。

NCCL p2p：用 NCCL 直接 send/recv 到目标 rank，绕过 reduction。要求两端在同一个 NCCL Communicator 里。

RDMA write：用 GPUDirect RDMA 让 train rank 直接写到 rollout rank 的显存。需要驱动支持、注册 memory region、用 UCX / GLEX / IBVerbs 自己写一层。

文件系统：train 写 checkpoint，rollout reload。简单可靠但慢——70B safetensors 写读各几十秒，几乎是 RL step 时间的 100%。

### 3. 关键机制 / 流程 / 数据结构
第一，热替换 API 需要在推理引擎里 expose 出来。vLLM 的 v0 用 `load_weights`，v1 在 `Worker.update_weights`；SGLang 用 `update_weights_from_distributed`。需要确认是「就地替换」还是「new engine swap」——就地替换避免重新分配 KV cache page table。

第二，同 NVSwitch domain 与跨域。Hopper NVL8 单机 8 卡互连 900 GB/s 双向，一份 70B 权重 ~140GB，理论几百毫秒就能 broadcast 完。跨机走 IB 400G 大约 50 GB/s，~3 秒。NVL72 域内 72 卡互连，一个 broadcast 能覆盖一整柜。

第三，异步流水线。同步与 rollout 不要串行。常见做法：train 步 N 结束后，把权重 enqueue 到同步 worker；rollout 当前 batch 跑完用旧权重，下一批 prompt 用新权重。这种 "asynchronous off-policy" 通常 1–2 步 staleness 是 acceptable 的。

第四，partial update。LoRA / 部分层微调时同步的 delta 很小，可以高频同步；全参数 RL 同步代价就是上面分析。

### 4. 工程权衡 / 性能影响
同步代价占比是关键指标。对 70B 全参数 RL，同步时间 / step 时间 ≥ 30% 说明同步通道选错或频率过高。理想范围是 ≤10%。

带宽换 staleness 是关键权衡。带宽充裕（NVL72 / 同机）就可以每步同步，做近 on-policy；跨机带宽紧张就放宽到每 N 步同步，引入 staleness，必要时 importance sampling 修正。

通信库选型上，NCCL 最成熟但要求严格的 collective 边界；UCX / GLOO 更灵活但稳定性差一些；自己写 ibverbs 性能最优但是工程坑多。生产里 90% 用 NCCL。

### 5. 常见追问 / 易错点
第一，FSDP 和 vLLM 的 shard 形态不一样怎么对接？要写一个 "weight bridge"：FSDP 侧 unshard 成 full state_dict（或按 vLLM 的 TP plan 直接 shard），通过 NCCL 发到 vLLM 的 Worker。注意 dtype 转换（训练 BF16，推理 FP16 / FP8 都常见）。

第二，KV cache 在权重更新后还有效吗？通常无效——权重变了之前的 prefill 结果不再正确。生产里在权重更新时 flush KV cache。但 RLVR 的多 G 采样里，每组的 prefix prefill 用同一份新权重就可以缓存。

第三，optimizer state 要同步吗？不需要。rollout pool 只跑推理，只要 actor 权重。

第四，rollout 中途权重更新会让正在生成的轨迹出错吗？会。要么等当前 batch 跑完再换，要么允许"半段旧+半段新"（一般不推荐）。

### 6. 实践建议
单机 / 同 NVL domain：每 1–2 步同步一次，走 NVLink。跨机训练：每 4–8 步同步一次，走 RDMA。大集群 + 容错：再加一个共享 ckpt 作为冷启动 backstop。

实测同步耗时：先 dry run 几次记录 P50/P99；定义 SLO（如同步时间 < step 时间 5%），超阈值就告警。

监控四个量：每次同步耗时、staleness（rollout 用的 weight 与最新 weight 步数差）、reshard 时间、KV cache invalidation 比例。

面试讲到这里时强调"权重同步是 RL 后训练独有的工程问题，SFT/pretrain 没有"。能讲清楚 NCCL p2p 与 RDMA 的差异、reshard 的存在、staleness 的可接受性，就完整覆盖了高分线。

### 7. 30 秒速答
- 权重通道四级：NVLink broadcast > NCCL p2p > RDMA write > 共享 ckpt reload。
- FSDP shard 和 vLLM TP shard 形态不同，每次同步要走 weight bridge 做 reshard。
- 易踩坑：同步耗时 / step 时间 > 30% 说明通道选错；权重换了 KV cache 必须 flush。
- 加分关键词：step-aligned 异步、reshard、热替换 API、optimizer state 不同步、partial update。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 RL 后训练为何需要"权重从 train pool 刷到 rollout pool"？
- [ ] 你能不能解释 FSDP shard 与 vLLM TP shard 之间的 reshard 过程？
- [ ] 你能不能举一个具体的同步频率选择（每 1 步 / 每 N 步 / 按 KL 阈值）？
- [ ] 你能不能说出权重热替换时 KV cache 失效的常见处理？

## Q5. RL 后训练的 long-CoT rollout 怎么调度？为什么这么吃显存？

> 🔴 专家 · R1 类训练里一条样本可能跑 16k token、用掉单卡 40 GB KV cache——你需要的不是更快的 decoder，而是一个能动态 evict、抢占、迁移 KV 的调度器。

### 1. 核心结论
Long-CoT（思维链）rollout 是 R1 / o1 类 RL 后训练独有的负载：单条样本生成 4k–32k token，KV cache 占用线性增长，显存压力是常规 chat 推理的几十倍。调度策略上需要：1) PagedAttention 把 KV 切 block 管理避免外碎片；2) 抢占机制让短样本优先完成、长样本可被换出；3) prefix caching 让组内多条样本共享 prompt KV；4) chunked prefill + decode 重叠，避免长 prompt 阻塞调度。

### 2. 底层原理
解码阶段每 token 都要读全部历史 KV，KV 显存随序列长度线性涨。70B 模型 KV per token ~140KB（FP16），16k token = ~2.3GB / sample；一个 batch 32 条就是 70+ GB——已经超过单卡显存。这就是 long-CoT rollout 不能简单"加大 batch"的根因。

PagedAttention 把每个序列的 KV 切成 16 或 32 token 的 block，按需分配；不同序列在同一显存池里共存，碎片化降到接近 0。RadixAttention（SGLang）进一步用 radix tree 维护 prefix 共享。

抢占（preemption）：当显存吃满时，调度器选择"换出"一些低优先级序列（把其 KV 拷到 CPU 或临时不调度），优先让接近完成的 / 高优先级的 sample 继续。vLLM 1.x 有 swap 机制；SGLang 有 recompute（不存 KV，下次重新 prefill）。

### 3. 关键机制 / 流程 / 数据结构
第一，prefix caching 与 GRPO 协同。一个 prompt 采 G 条（G=8–16）。组内 G 条共享 prompt 的 KV，prefix caching 直接把 prompt prefill cost 摊掉。GRPO 的吞吐能跑起来很大程度依赖这个。

第二，chunked prefill。长 prompt（4k+ token）的 prefill 会阻塞 decode batch。chunked prefill 把 prefill 切成 512/1024 token 的 chunk，与正在 decode 的 batch 交错调度，让 GPU 利用率不塌方。

第三，speculative decoding 与 long-CoT。Medusa / EAGLE 这类投机解码能让单序列 decode 速度提 2–3 倍，对 long-CoT 收益巨大。但它要消耗额外显存放 draft model 或额外 head，需要在 KV 与 draft 间做容量权衡。

第四，dynamic batch resizing。当某条样本因为 long-CoT 跑很久，调度器要支持新样本动态加入 batch、完成的样本动态退出。这是 continuous batching 的基本能力，但在 RL 场景下"序列长度跨度极大"会让动态性需求更强。

### 4. 工程权衡 / 性能影响
显存利用率优先级最高。生产里 KV cache 通常占 GPU 显存的 60–80%（剩 20–40% 给权重和激活）。这要求 KV block size 调优（block 越小利用率越高，但调度开销越大；典型 16）。

吞吐 vs 时延。RL rollout 不太关心单序列时延，更关心 tokens/s 总吞吐。所以可以激进地放大 batch size、用 chunked prefill、用 speculative decoding。这跟在线推理服务（要保 P99 TTFT）的优化方向截然不同。

显存峰值要给 RL 同步留 headroom。如果 KV 占满 80%，权重热替换时 spike 显存可能 OOM。预留 5–10% buffer 是 RL rollout pool 与在线推理 pool 的关键差别。

### 5. 常见追问 / 易错点
第一，max_model_len 设置多少？设到模型支持上限即可（如 32k）。设太小会 truncate 推理，设太大会让 vLLM 预留过多 KV cache header。

第二，prefix caching 在 RL 里要不要持久化？跨 batch 持久化收益不大（prompt 不重复），但组内共享必须开。

第三，long-CoT 中途模型权重换了怎么办？正在生成的 sample 用旧权重跑完，新 sample 用新权重。中途换权重等于 reset，要避免。

第四，CPU offload KV 划算吗？只在 NVLink 内多卡 KV 转移时划算（带宽 200 GB/s+）；跨 PCIe 走 CPU offload 大概率慢于直接 recompute。

### 6. 实践建议
rollout pool 配置：tensor_parallel_size 跟随模型大小（70B 通常 TP=4-8），enable_chunked_prefill=True，enable_prefix_caching=True，max_num_batched_tokens 调到 8192–16384。

监控的最小指标集：tokens/s（rollout 吞吐）、avg sequence length、avg KV usage、preemption rate、prefix cache hit rate、scheduler queue depth。

面试讲 long-CoT rollout 时主线是「KV cache 主宰显存、调度器主宰吞吐」。能讲清 PagedAttention + prefix caching + chunked prefill 三件套的协同就到位。

### 7. 30 秒速答
- Long-CoT 单条样本 4k–32k token，KV 显存线性涨，常规推理 batch 模式直接 OOM。
- 三件套：PagedAttention 切 block + 组内 prefix caching + chunked prefill 交错 decode。
- 易踩坑：KV 占满 80% 时权重热替换 spike OOM，必须留 5–10% headroom。
- 加分关键词：抢占 / swap / recompute、动态 batch resize、RadixAttention、speculative decoding 协同。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 long-CoT rollout 与常规 chat 推理在显存上的差异？
- [ ] 你能不能解释 PagedAttention 为什么是 long-CoT 调度的前提？
- [ ] 你能不能举一个 GRPO 组内 prefix caching 共享 prompt KV 的具体场景？
- [ ] 你能不能说出 chunked prefill 不开时长 prompt 阻塞 decode 的表现？

## Q6. MoE 推理的专家并行（EP）是什么？跟 TP 怎么组合？

> 🔴 专家 · 你想跑 DeepSeek-V3 671B 推理，把所有专家都复制到每张卡显存都不够——必须切专家，每张卡只放一部分专家，token 通过 all-to-all 找到自己的专家。

### 1. 核心结论
专家并行（Expert Parallelism, EP）把 MoE 模型里的众多 expert FFN 切分到不同 GPU：每张卡只持有一部分 expert，token 在 router 决策后通过 all-to-all 路由到目标卡，目标卡执行对应 expert 的 FFN 后再 all-to-all 路由回原卡继续后续层。EP 通常与 TP 组合：attention 部分走 TP（同卡共享），FFN 走 EP（卡间切 expert）。这种组合让单层 FFN 算力可以横跨更多卡，让 DeepSeek-V3 671B / Mixtral-8x22B 这类巨型 MoE 能在合理卡数上推理。

### 2. 底层原理
MoE 模型把 FFN 层换成"router + N 个 expert（每个本身是个 FFN）"。给定 token，router 输出 top-k expert 索引（典型 k=2），token 的 FFN 输出是这 k 个 expert 输出的加权和。

如果用 TP 实现 MoE，所有 expert 都复制在每张卡上，每张卡都跑完整路由——显存爆炸。EP 让每张卡只负责一部分 expert：例如 256 个 expert 切到 64 张卡，每张卡 4 个 expert。Token 在每层都要 "去找自己的 expert"，所以每层 FFN 前后各一次 all-to-all。

Attention 仍然是 TP 友好的（按 head 切），所以 attention 走 TP、FFN 走 EP 是常见组合。DeepSeek-V3 在 256 GPU 上跑 TP=1 + EP=64 + DP=4 风格的混合并行。

### 3. 关键机制 / 流程 / 数据结构
第一，router 与 dispatch。router 是个小的 linear，输出 N（专家数）维的 score。top-k 选择给出每个 token 要去的 k 个 expert。Dispatch 把 token embedding 按目标 expert 重新打包，发起 all-to-all。

第二，all-to-all 与 DeepEP。所有卡同时发送 / 接收，是通信模式最复杂的之一。DeepSeek 开源的 DeepEP 是针对 MoE all-to-all 做了极致优化的通信库，把延迟压到几十微秒级，对 EP 推理吞吐至关重要。

第三，token 路由不均衡。Router 决策导致某些 expert "热"（被大量 token 选中）、某些 "冷"。热 expert 所在卡的 FFN 算力打满，冷卡闲置。这是 EP 推理的核心痛点之一。解决方法：1) router 训练时加 load balancing loss；2) expert 副本（一些热 expert 在多张卡有副本）；3) 调度器层做 token re-routing。

第四，capacity factor。给每个 expert 预定一个最大 token 数（capacity），超过的 token 被丢弃或路由到次优 expert。capacity factor=1.25 是常见值。

### 4. 工程权衡 / 性能影响
EP 的核心 cost 是 all-to-all 通信。每层两次 all-to-all，模型几十层下来通信量很可观。NVLink 域内 all-to-all 通常很快（DeepEP 在 NVL8 内可以做到 50µs 量级），跨域 RoCE/IB all-to-all 慢一个量级。所以 EP 的"域"通常对齐到 NVL8 / NVL72。

EP 大小不能任意增大。expert 数 N 是模型固定的（DeepSeek-V3 是 256），EP 大小 E 必须能整除 N。E 越大每张卡 expert 越少（显存更省），但每层 all-to-all 通信量更大。

与 TP 组合时，TP 内的卡需要复制同样的 expert；TP=8 + EP=32 意味着 256 卡集群，每 8 卡一组持有相同的 32 个 expert。这个 layout 让通信和复制都最优。

### 5. 常见追问 / 易错点
第一，为什么不用 TP 处理 expert 切分？因为 expert 是离散的、token 是稀疏路由的——TP 切 expert 内部矩阵无法解决"每个 token 只去 2 个 expert"这个 sparsity。EP 切的是 expert 维度，正好匹配 sparsity 模式。

第二，dropped tokens 怎么处理？超过 capacity 的 token 可以跳过 FFN（直接接 residual）或路由到 top-2 中可用的那个。生产里 capacity factor 调到 1.25–1.5 让 drop rate <1%。

第三，EP 跟 PP 怎么配合？PP 切层、EP 切 FFN 内 expert，两者正交，可以叠加。但通信复杂度会迅速升高。

第四，EP 推理时 expert offload 可不可以？可以——冷 expert 放 CPU 内存，按需 swap 进 GPU。但增加延迟，主要用于"想跑超大 MoE 但卡不够"的场景。

### 6. 实践建议
配置上 attention 走 TP（4 或 8），FFN 走 EP（对齐到 expert 数）。监控 expert utilization 分布、all-to-all 延迟、drop rate。

DeepEP 是当前 EP 通信的开源最优解，强烈推荐用。

面试问到 MoE EP 时强调"它解决的是 expert 维度的 sparsity，所以 router + all-to-all 是核心，不是简单的 TP 拓展"。能讲清 expert 不均衡问题与 DeepEP 的存在就到位。

### 7. 30 秒速答
- EP 把 expert 切到不同 GPU，token 经 router 后通过 all-to-all 路由到目标卡再回来。
- attention 走 TP、FFN 走 EP 是标准组合；EP 域必须对齐 NVLink 拓扑（NVL8 / NVL72）。
- 易踩坑：hot expert 拖慢全卡（all-to-all 等最慢的卡）；capacity factor 太小 drop 高、太大浪费。
- 加分关键词：DeepEP 50µs 级 all-to-all、capacity factor 1.25、expert 副本、top-k=2 sparsity。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 EP 解决的是 expert 维度的什么问题？
- [ ] 你能不能解释为什么 EP 不能用 TP 切 expert 内部矩阵替代？
- [ ] 你能不能举一个具体的 EP + TP 组合（DeepSeek-V3 TP=1 + EP=64 + DP=4）？
- [ ] 你能不能说出 EP 域跨 NVLink 与跨 IB 的延迟数量级差异？

## Q7. MoE 训练里 expert load balancing 是怎么做的？为什么 router collapse 是个 infra 问题？

> 🔴 专家 · 你训了一周发现 256 个 expert 里只有 5 个被用得多，剩下 251 个权重几乎没更新——这不是模型问题，是 infra 在调度时没把"router 退化"当成一等指标。

### 1. 核心结论
MoE 训练有个天然的退化倾向：router 学到把绝大多数 token 都送到少数几个 expert，其他 expert 缺训练信号变成"死 expert"。这叫 router collapse。从 infra 视角看不仅是模型质量问题——它会让卡间负载严重不均，热 expert 所在卡持续 OOM 风险，冷 expert 所在卡持续闲置。解决一般靠两路：损失函数侧加 load balancing loss（Switch Transformer 提出）/ aux-free balancing（DeepSeek-V3 的 bias 调节）；调度侧加 capacity 限制与 expert dropping。

### 2. 底层原理
Load balancing loss 形式：L_balance = α · N · Σ_i f_i · P_i，其中 f_i 是 expert i 被选为 top-k 的频率，P_i 是 router 对 expert i 的平均概率。这个 loss 在 router 给 expert i 高概率且 expert i 被高频选中时变大，反向梯度推动 router 把概率分散开。

DeepSeek-V3 改用 aux-loss-free：每个 expert 维护一个 bias b_i，每个 batch 后根据负载情况调节 bias（被多选的减 bias、被少选的加 bias）。优点是不污染主 loss、不需要调 α。

Capacity factor 是硬截断：每个 expert 一个 batch 内最多处理 C 个 token，C = capacity_factor × (batch_tokens × top_k / N)。超过的 token 被丢弃或重路由。这从工程上给热 expert 设了上限。

### 3. 关键机制 / 流程 / 数据结构
第一，监控 expert utilization。每个 expert 收到的 token 数除以平均预期数，理想值 1.0。<0.5 或 >2.0 都说明不均衡。生产里这是 dashboard 上 MoE 训练的核心指标。

第二，hot expert 的硬件影响。某些 expert 所在卡 FFN 计算量是别的卡 5x，会拖慢 step time（all-to-all 必须等最慢的卡完成）。expert utilization 不均衡直接反映在 step time variance 上。

第三，capacity overflow 处理。Token 超 capacity 时：a) drop（最简单）；b) re-route 到次优 expert；c) "shadow expert"（DeepSeek 在某些层加冗余 expert 吸收溢出）。

第四，跨 DP 副本的全局视角。EP 切 expert 后，一个 expert 在不同 DP rank 上是同一个副本。load balancing loss 应当在全局 token 上算，不是 per-DP-rank 算，否则 collapse 检测不到。

### 4. 工程权衡 / 性能影响
α 调参敏感：太小不起作用、collapse 发生；太大主任务 loss 被压制、模型质量差。Switch Transformer 推荐 α=0.01。aux-loss-free 把这部分 hyper-param 移到 bias 调节速率，更易调。

Capacity factor 与吞吐取舍：cf=1.0 严格但 drop 高；cf=1.25 是 sweet spot；cf=2.0 浪费容量。

Router 训练阶段不稳定。前几千步 router 学得慢，token 路由几乎随机，load balance 自然好；几万步后 router 学到结构，开始倾向少数 expert——这时候 collapse 风险最高，要密集监控。

### 5. 常见追问 / 易错点
第一，为什么不直接强制均匀路由？强制均匀（如 random routing）破坏了 router 的语义价值——MoE 的核心假设就是 token 应该去最相关的 expert。Balancing 是"软"约束，要让 router 在尽量分散与尽量准确之间取平衡。

第二，aux-loss-free 怎么工作？给每个 expert i 一个 bias b_i，路由打分变成 score_i + b_i。被多选的 expert 减 b_i（让它下次少被选），被少选的加。bias 调整步长是 hyper-param 但比 α 直观。

第三，dropped token 对训练有损失吗？有但小。FFN 被跳过的 token 直接走 residual，相当于该层 "no-op"。Drop rate <1% 通常不影响收敛。

第四，推理时 collapse 怎么办？推理时 router 是 freeze 的，collapse 已经形成。只能从训练侧治本；推理侧只能用 capacity / 重路由减损。

### 6. 实践建议
新 MoE 训练默认开 aux-loss-free（DeepSeek 路线）。capacity_factor=1.25 是稳健起点。

监控五个量：per-expert token count、step time variance（卡间）、router entropy、drop rate、aux loss / bias 调整量。任一异常都是 collapse 早期信号。

面试讲 MoE balancing 时强调"它是模型 + infra 的联合问题"——分别讲 load balancing loss 和 capacity 限制就到位，能引用 DeepSeek-V3 的 aux-loss-free 是加分项。

### 7. 30 秒速答
- Router collapse：256 expert 退化为只用 5 个，剩下变死 expert + 卡间负载严重不均。
- 解法两路：损失侧 aux loss（Switch Transformer）/ aux-free bias 调节（DeepSeek-V3）；调度侧 capacity 截断。
- 易踩坑：load balance loss 要在全局 token 上算，per-DP-rank 算检测不到 collapse。
- 加分关键词：aux-loss-free bias、capacity factor 1.25、step time variance、shadow expert、router entropy。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 router collapse 是什么、为什么是 infra 问题？
- [ ] 你能不能解释 aux-loss-free 的 bias 调节机制相对 α 系数的优势？
- [ ] 你能不能举一个 hot expert 拖慢 step time 的具体表现？
- [ ] 你能不能说出 capacity overflow 的三种处理方式（drop / reroute / shadow expert）？

## Q8. Context Parallelism (CP) 是什么？跟 Ring Attention、Sequence Parallelism 怎么区分？

> 🔴 专家 · 你要训 1M context 的模型，单卡放不下 1M 的 attention——把 sequence 切到不同卡，每张卡只负责一段，相互发 KV 拼接，这就是 CP。

### 1. 核心结论
Context Parallelism (CP) 是沿 sequence 维度把单条样本切到多张卡的并行方式，让每张卡只处理一部分 token 的 forward / backward。它的核心挑战是 attention：每个 query token 要看到全部 key/value，所以 CP 必须让 KV 在卡间流动。Ring Attention 是 CP 的标志性实现：KV 在卡间形成 ring 传递，每次计算一段 attention 后把 KV 转给下一张卡，避免一次性 gather 全部 KV。Sequence Parallelism (SP) 是 Megatron-LM 提出的窄义概念，只在 LayerNorm/Dropout 等非 attention 层沿序列维切，本质是减少 activation 显存。CP 是更广义的 sequence 切分，包括 attention 计算。

### 2. 底层原理
Attention 是 O(N²) 的 query × key：query 沿 N 切到 G 张卡，每张卡 N/G 个 query；但 attention 输出依赖全部 N 个 key/value，所以每张卡需要在某个时点看到完整的 KV。

Ring Attention 的关键 trick：把全局 KV 也切到 G 张卡，每张卡持有 N/G 个 token 的 KV。计算时 ring 传递：第一轮每张卡用本地 KV 算本地 query 的部分 attention；然后所有卡的 KV 顺时针传一格，再算下一段；走完 G 步，每个 query 都"看到"了所有 KV。每步只发 1/G 的 KV，通信摊到 G 步上。

数学上每步算的是 partial softmax，要用 FlashAttention 类的"online softmax"累加机制，确保最终结果与一次性 attention 等价。

### 3. 关键机制 / 流程 / 数据结构
第一，Ring Attention 通信模式。每步 send_recv 1/G KV，是 NCCL p2p 通信。延迟敏感，CP 域一般要在 NVLink 内。

第二，CP 与 FA-2/3 的整合。FlashAttention 内核要支持"接收外部 KV chunk"——FA-3 已有 Ring Attention friendly 实现。

第三，与其他并行的组合。CP 与 DP、TP、PP、EP 都正交：DP 跨数据维、TP 切 head、PP 切层、EP 切 expert、CP 切 sequence。生产里 70B + 1M context 训练可能用 DP=2 × TP=8 × CP=16 = 256 卡。

第四，causal mask 处理。Causal attention 让前面的 query 不需要看后面的 KV，可以在 ring 步数上 skip 一些通信。CP 实现需要把 mask 拆分到每个 ring step。

### 4. 工程权衡 / 性能影响
CP 解锁的能力：1M、10M context 训练。但 attention 通信开销随 CP 大小线性涨，CP=32 时 attention 时间可能 doubled。所以 CP 通常用在 attention bound 的极长序列上。

CP 内存收益与 sequence 长度成正比：32k context CP=4 让每张卡只放 8k 的 KV，激活和 KV 都按比例缩小。

FFN 部分 CP 几乎免费：FFN 是 token-wise 的，切到不同卡就是 ZeRO-2 style 切 token。所以 CP 的开销基本都来自 attention。

### 5. 常见追问 / 易错点
第一，CP 与 TP 都"切单层"，为什么不冲突？TP 切矩阵乘的输出 dim（如 head 维），CP 切 sequence 维，两者是不同维度，正交。

第二，CP 与 PP 谁先切？PP 在最外层切层，CP 在层内切 sequence。所以 PP 优先，CP 是 PP 内的进一步切分。

第三，Ring Attention 与 Star Attention 区别？Star 是把 KV 全 gather 到一张"reducer"卡再算（中心化），通信量更小但单卡内存压力更大；Ring 是去中心化的 ring 通信，通信量较大但内存分布更均匀。生产里 Ring 是主流。

第四，CP 推理可不可以？可以，但推理一般不需要——推理 KV 总长度 = prompt + 已生成，远小于训练时的 sequence 长度。极长 context 推理（>1M）会用 CP，但常见还是单卡 / TP 内 KV cache 足够。

### 6. 实践建议
单卡 sequence 长度限制是 CP 的触发条件。一般 32k 内 SP+激活重计算就够；64k–256k 上 CP=2-4；1M+ 用 CP=8-16。

CP 域优先放在 NVLink 内（NVL8 或 NVL72）。跨机 CP 通信代价过高。

监控：attention compute time、ring send/recv time、CP scaling efficiency（实际加速 / 理论加速）。

面试讲 CP 时主线是「沿 sequence 切，attention 必须 ring 传 KV」。能区分 CP / SP / Ring / Star 是高分。

### 7. 30 秒速答
- CP 沿 sequence 切，每张卡持有 N/G 个 token 的 query 和 KV，attention 必须看到全 KV。
- Ring Attention：KV 在卡间环形传递 G 步，每步算 partial softmax，online softmax 累加保证等价。
- 易踩坑：CP 域必须放在 NVLink 内（NVL8/NVL72），跨机 ring 通信慢一个量级直接拖垮 attention。
- 加分关键词：FA-3 ring 友好、causal mask 跳通信、Star vs Ring、与 TP/EP/PP 正交。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 CP 与 SP（Megatron 窄义）的区别？
- [ ] 你能不能解释 Ring Attention 的 online softmax 为何与一次性 attention 数学等价？
- [ ] 你能不能举一个 1M context 训练的具体并行配比（DP×TP×CP）？
- [ ] 你能不能说出 CP 与 Star Attention 在通信 / 内存上的取舍？

## Q9. KV cache 量化（INT8 / FP8 / INT4）是怎么做的？工程坑在哪？

> 🔴 专家 · 你把 KV cache 从 FP16 压到 FP8 显存省一半——但前提是你能让 attention kernel 直接读 FP8 KV，否则反复转换就把省下来的带宽吃回来。

### 1. 核心结论
KV cache 量化是把推理时缓存的 K/V 张量用 INT8 / FP8 / INT4 存储，显存压一半到 1/4。技术核心：1) per-head 或 per-channel 量化（per-token 太细粒度，per-tensor 误差过大）；2) attention kernel 必须原生支持低精度 KV 读入（FA-3 + FP8 KV / SGEMM-on-FP8）；3) 通常 K、V 分别量化（K 更敏感所以 K 用更高位、V 可以更激进）。

### 2. 底层原理
KV cache 在 long context 推理里占显存大头。FP16 KV 每 token ~140KB（70B 模型）；FP8 减半到 70KB；INT4 进一步到 17.5KB。对 16k context 单 sample，差距是 2.2GB vs 0.28GB——单卡上 batch size 可以提 8 倍。

量化方式：1) per-tensor（最粗，accuracy 损失最大）；2) per-token（每 token 一组 scale，太细粒度运行时开销大）；3) per-head per-channel（典型选择，每个 head 的每个 channel 一组 scale）；4) per-block（block-wise，平衡 accuracy 与开销）。

K 量化误差通过 softmax 放大，V 量化误差是线性传播——所以 K 更敏感，常见 K=FP8 V=INT4 的非对称组合。

### 3. 关键机制 / 流程 / 数据结构
第一，attention kernel 集成。FlashAttention-3 在 Hopper 上原生支持 FP8 KV；vLLM 内置 INT8 / FP8 KV cache（`kv_cache_dtype="fp8"`）；TensorRT-LLM 在 Hopper 后支持 FP8 KV。

第二，calibration。FP8 用 per-tensor scale 时需要 calibration 样本算 scale；INT8 / INT4 用 per-channel scale 通常 online 算（每次 store 时统计）。

第三，online vs offline 量化。online 是每次 store KV 时实时算 scale 与量化；offline 是预先 calibrate 后固定 scale（仅 FP8 静态量化）。生产里 INT8/INT4 用 online，FP8 可以静态也可以 dynamic。

第四，dequant 位置。理想情况下 attention kernel 直接读 quantized KV，内部 dequantize 到寄存器；如果 kernel 不支持，要在主存里 dequantize 回 FP16 再算——这就把显存收益吃回来了。所以"是否有原生 FP8/INT8 attention kernel"是决定能不能用 KV 量化的根本。

### 4. 工程权衡 / 性能影响
显存收益直接。70B / 16k context / batch 32 在 FP16 KV 下要 70+ GB，FP8 减到 35GB，INT4 减到 17.5GB。这直接决定单卡能跑多大 batch。

精度损失。FP8 几乎无损（perplexity 涨幅 <1%）；INT8 perplexity 涨 1–3%；INT4 涨 3–10%。生产里 FP8 / INT8 是 long-context 推理默认；INT4 要谨慎评估。

吞吐影响有正有负：显存省下来可以加大 batch，吞吐涨；但 dequant 开销可能让 attention 慢一些。FA-3 FP8 KV 在 Hopper 上是双赢，整体吞吐能涨 30–50%。

### 5. 常见追问 / 易错点
第一，KV 量化和权重量化能叠加吗？能。W8A8 + KV-FP8 是常见组合，但 calibration 要分开做。

第二，scale 怎么算最稳？per-head per-channel 用 max abs / 127（INT8）或 max abs / 448（FP8 E4M3）。outlier 通过 clip-to-percentile（99.9%）处理。

第三，量化后的 cache 跨 batch reuse 怎么办？scale 必须随 cache 一起持久化。prefix caching 共享 KV 时 scale 也要共享。

第四，FA-3 FP8 KV 在 Ampere 上能用吗？不能，必须 Hopper 及以后（FP8 tensor core）。Ampere 上能用 INT8 / INT4 KV 但需要专门 kernel。

### 6. 实践建议
Hopper / Blackwell 推理：FP8 KV 默认开。Ampere：INT8 KV，INT4 仅在显存极度紧张时。

实测路径：先 perplexity 评估（wiki-text / C4），再 long-context 任务评估（NeedleInHaystack），最后业务任务评估。三个都通过才上线。

监控：KV memory usage、attention latency（quant vs no-quant 对比）、accuracy drift（线上 A/B 测）。

面试问 KV 量化时强调「不只是节省显存，还要 attention kernel 原生支持，否则得不偿失」。能讲 K/V 非对称量化和 FA-3 FP8 是加分项。

### 7. 30 秒速答
- KV cache 量化把 FP16 KV 压到 FP8/INT8/INT4，显存减半到 1/4，batch 上限直接翻倍。
- per-head per-channel 是最常用粒度；K 比 V 敏感（softmax 放大误差），常 K=FP8、V=INT4 非对称。
- 易踩坑：attention kernel 不原生支持低精度 KV 就要 dequant 回 FP16，省下的带宽吃回来。
- 加分关键词：FA-3 FP8 KV、E4M3 scale=max_abs/448、prefix cache 共享 scale、Hopper 限定。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 KV 量化在显存与吞吐上的双重收益？
- [ ] 你能不能解释为什么 K 比 V 对量化更敏感？
- [ ] 你能不能举一个 FA-3 FP8 KV 在 Hopper 上的吞吐涨幅区间？
- [ ] 你能不能说出 KV 量化与权重量化叠加时的 calibration 注意点？

## Q10. Disaggregated prefill/decode 是什么？为什么 Mooncake / DeepSeek 在生产里用它？

> 🔴 专家 · prefill 是 compute-bound（每秒万 token），decode 是 memory-bound（每秒数十 token）——把它们放同一张卡就像让短跑选手和长跑选手轮流上场，资源永远只有一半在用。

### 1. 核心结论
Disaggregated prefill/decode 把 LLM 推理的两个阶段拆到不同 GPU 池：prefill pool（高算力卡，处理长 prompt 一次性 forward）和 decode pool（高带宽卡或更多卡分担 KV，处理 token-by-token 生成）。Prefill 完成后 KV cache 通过 RDMA / NVLink 迁移到 decode pool，decode 继续生成。这种架构让 prefill 卡按算力扩、decode 卡按 memory bandwidth 扩，单元经济学比共置方案好 30–100%。Mooncake（月之暗面）、DeepSeek inference 是公开的代表系统。

### 2. 底层原理
Prefill 阶段：长 prompt 一次性 forward，输出每个 token 的 logits（只用最后一个）+ 完整 KV cache。计算量 ~O(L × N²)（attention）+ O(L × N × d²)（FFN），L 是 prompt 长度。Compute-bound——算力越强吞吐越高。

Decode 阶段：每步一个 token 的 forward，attention 要读全部历史 KV（O(L)），FFN 是 O(d²)。Memory-bandwidth-bound——HBM 带宽越高 tokens/s 越高。

把两者放同一张卡时，prefill 突发吃算力但只占很短时间；decode 持续吃带宽。GPU 算力大部分时间在等带宽，HBM 带宽大部分时间用不满。Disaggregated 把这俩 decouple：prefill pool 全用 compute-rich GPU（H100 / B200），decode pool 用同款但配合超大 batch 把带宽吃满。

### 3. 关键机制 / 流程 / 数据结构
第一，KV cache 迁移。prefill 完得到完整 KV（典型几 GB），通过 RDMA 或 NVLink 拷到 decode pool 的目标 GPU。要做到迁移延迟 < 100ms 才不显著影响 TTFT（首 token 时延）。

第二，调度策略。prefill 请求进 prefill queue，decode 请求进 decode queue。一个 request 的生命周期是 "prefill queue → prefill GPU → KV migration → decode queue → decode GPU → tokens"。调度器要平衡两个 queue 的长度。

第三，KV layout 一致性。如果 prefill 用 TP=4，decode 用 TP=2，KV 需要 reshard。生产里通常让两个 pool 的 TP/PP 一致避免 reshard。

第四，热卡冷卡。prefill 用更新更贵的卡（B200），decode 用稍旧但显存大的卡（H100/H200）；或者反过来——具体看哪类卡 supply 紧张。

### 4. 工程权衡 / 性能影响
单元经济学胜出。共置 prefill+decode 时一张卡的算力利用率 ~50%、HBM 带宽利用率 ~50%；解耦后两个 pool 都能跑到 80%+，整体单 token 成本下降 30–100%。

延迟权衡。KV 迁移加了一段 latency，但因为 decode pool 不再被 prefill 抢占，TPOT 更稳；prefill pool 也不再被 decode 阻塞，TTFT 实际反而更好。

调度复杂度上升。两个 queue 长度的平衡是新的难题。Mooncake 用了"prefill 排队预估 + decode 反压" 的双向信号。

跨域通信压力。KV 迁移每秒级几 GB 量级，要求 prefill pool 与 decode pool 同 NVL 域或 IB 直连。跨地域不可行。

### 5. 常见追问 / 易错点
第一，KV 迁移会不会成新瓶颈？理论上会，但 RDMA 带宽 400 Gbps = 50 GB/s，一个 8k context 70B KV ~3GB，60ms 就完了，比 prefill 本身快得多。

第二，prefill 与 decode 共用 weight 副本吗？不共用——每个 pool 持有自己的 weight 副本。这是 disaggregated 的代价之一（多份 weight 显存）。

第三，prefix caching 怎么处理？在 prefill pool 端做 prefix caching（system prompt KV 复用），命中后直接送 KV migration。

第四，能不能 chunked prefill 替代 disaggregated？chunked prefill 是把 prefill 切片与 decode 交错，**仍在同卡**；disaggregated 是真正物理隔离。前者是穷人方案，后者是富人方案。

### 6. 实践建议
小规模（<100 GPU）共置 + chunked prefill 足够；大规模生产（>500 GPU）解耦能省下显著成本。

监控指标：prefill queue depth、decode queue depth、KV migration latency、prefill GPU util、decode HBM bandwidth util、TTFT、TPOT、E2E latency。

面试讲 disaggregation 时主线是「两阶段不同瓶颈，解耦让两边各自饱和」。能提 Mooncake / DeepSeek inference 公开方案是加分。

### 7. 30 秒速答
- Prefill 是 compute-bound、decode 是 memory-bandwidth-bound，瓶颈完全不同。
- 解耦成两个 GPU 池，KV cache 通过 RDMA / NVLink 从 prefill 迁到 decode，两边各自打满。
- 易踩坑：两 pool 的 TP/PP 形态不一致就要 reshard KV；KV 迁移 latency 不能超过 100ms。
- 加分关键词：Mooncake、DeepSeek inference、chunked prefill 是穷人方案、双 queue 反压调度。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 prefill / decode 两阶段的瓶颈差异？
- [ ] 你能不能解释 KV migration 在 RDMA 400 Gbps 下的可行性测算？
- [ ] 你能不能举一个共置方案下算力与带宽利用率都只 ~50% 的具体原因？
- [ ] 你能不能说出 chunked prefill 与 disaggregated 在物理隔离上的本质差异？

## Q11. 投机解码（speculative decoding）有哪几类方案？Medusa、EAGLE、Lookahead 各自工程取舍？

> 🔴 专家 · 你想让 decode 加速 2-3x 但又不想牺牲质量——找一个又快又准的"草稿者"是核心，模型本身当 verifier 一致性自然保证。

### 1. 核心结论
投机解码（speculative decoding）核心思想：用一个小且快的 draft model 一次生成 N 个 candidate token，主模型一次 forward 验证这 N 个，接受到第一个不匹配为止。一次 forward 通常落 1-N 个 token（期望 2-4），decode 吞吐相应翻倍。主流方案：1) classic draft-verifier（小模型当 draft，最经典）；2) Medusa（在主模型上加多个 head 并行预测后续 N 个 token）；3) EAGLE / EAGLE-2/3（学一个轻量 draft head 用主模型 hidden 状态作为输入）；4) Lookahead / Jacobi decoding（多步迭代 fixed-point）。工程取舍主要在「draft 训练成本 vs 接受率 vs 实现复杂度」。

### 2. 底层原理
Draft 提议 token x_1...x_N；主模型一次 forward 计算 P(x_1|context), P(x_2|context, x_1)... P(x_N|...) 同时验证。验证规则（rejection sampling）保证最终分布与原始 decode 一致。

Medusa：主模型最后一层加 K 个并行 head，每个 head 预测后续 t+1, t+2... t+K 位置的 token。优点是无需独立 draft model、训练简单（只训 head）；缺点是后位预测准度低、接受率衰减快。

EAGLE：训一个轻量 draft 模块，输入是主模型的 hidden state（不只是 token id），输出后续 token。EAGLE-2 引入动态 tree 验证，EAGLE-3 引入更激进的 hidden 跨层信息。接受率显著高于 Medusa。

Lookahead / Jacobi：不需要训新参数，用迭代 fixed-point 求解一段 token——理论优雅但实际加速一般 1.5x。

### 3. 关键机制 / 流程 / 数据结构
第一，draft tree。多个 candidate sequence 形成 tree（如 Medusa-2 / EAGLE-2 的 tree decoding）。主模型一次 forward 验证整棵 tree，选择被接受最长的路径。tree 越宽期望接受 token 越多但 forward 算力越大。

第二，verification kernel。tree 验证需要变形 attention mask（不同 candidate path 看不同前缀）。FA-2/3 有 tree mask 支持。

第三，acceptance length。每 forward 接受的平均 token 数。理想 4-6，差的 1.5-2。EAGLE-2 在常见 benchmark 上能到 4-5。

第四，draft model 选型。classic 用 1B 模型 draft 70B，速度比 1:60；Medusa 用主模型 head，速度比 1:1（同 forward）但接受率更低；EAGLE 用 draft head ~100M，速度比 1:700。

### 4. 工程权衡 / 性能影响
吞吐 vs 显存。draft model / Medusa head / EAGLE head 都额外占显存。EAGLE 最省（100M 量级），Medusa 中等，classic 1B draft 最贵。

实现复杂度。Medusa 最简单（加 head 训练），classic draft 中等（要管两个 model），EAGLE 复杂（hidden state 接入要小心 layer alignment）。

加速效果在 batch size 上的依赖。投机解码主要降单序列 latency；batch size 大时主模型 forward 已经 amortize，投机收益变小。所以投机更适合在线推理（小 batch、低延迟），不太适合 batch rollout（大 batch、高吞吐）。

### 5. 常见追问 / 易错点
第一，投机解码会不会改变输出分布？严格的 rejection sampling 不会——数学上等价于直接 decode。但近似变体（accept-all、temperature mixing）会改变分布。

第二，为什么 batch 大时收益小？大 batch 下主模型 forward 已经把算力打满，加 spec decode 的 forward 时间不变但 batch 多——投机的"省 forward 次数"价值降低。

第三，EAGLE 训练数据怎么准备？用 SFT 数据 + 主模型 forward 取 hidden state 作为 EAGLE 的训练 input/target。生产里 LMSYS / chatbot 数据 100k 量级够用。

第四，多个用户的 spec decode 能复用 draft 吗？draft model 本身是共享的（多 batch 复用同一份 weight）；draft tree 是 per-request 的。

### 6. 实践建议
在线 chat 服务：EAGLE-2/3 是当前最佳选择。简单实现优先 Medusa。

吞吐主导（RL rollout）：投机解码不是必须。大 batch + chunked prefill + PagedAttention 收益更明显。

监控：acceptance length、draft forward time、verify forward time、effective tokens/s。监控 acceptance length 是判断 spec decode 是否健康的最关键指标。

面试讲投机解码时主线是「draft 提议 + verifier 一次性验证 + rejection 保证一致」。能区分 Medusa / EAGLE / classic 三类方案的取舍就到位。

### 7. 30 秒速答
- Draft 一次预测 N 个 token，主模型一次 forward 验证，rejection sampling 保证分布等价。
- Medusa 加多头并行预测，EAGLE 用 hidden state 训轻量 draft head，classic 小模型独立 draft。
- 易踩坑：投机收益随 batch 变大而消失，吞吐主导场景（rollout）不要开。
- 加分关键词：acceptance length 4–5、tree decoding、FA tree mask、EAGLE-2/3、Jacobi/Lookahead。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清投机解码为什么不改变输出分布？
- [ ] 你能不能解释为什么大 batch 下投机收益变小？
- [ ] 你能不能举一个 Medusa / EAGLE / classic 三方案的具体取舍（显存 vs 接受率 vs 训练成本）？
- [ ] 你能不能说出 acceptance length 是 spec decode 健康度的核心指标？

## Q12. Blackwell B200 与 Hopper H100/H200 在训练 / 推理 infra 上的差异是什么？

> 🔴 专家 · B200 不是简单的"H100 + 更多 HBM"——它把 FP4 做成一等公民、把 NVLink 域从 8 扩到 72、加了 Tensor Memory，整个 software stack 都要重写。

### 1. 核心结论
Blackwell (B200/GB200) 相对 Hopper (H100/H200) 的核心升级：1) FP4（MXFP4 / NVFP4）tensor core 原生支持，FP4 算力是 H100 FP8 的 ~5x；2) NVL72 单柜 72 GPU 全互连，NVLink 域从 8 卡扩到 72 卡（NVL8 → NVL72）；3) HBM3e 更大容量（192GB vs H100 80GB）；4) Tensor Memory（异步张量存储单元，FA-3 风格 kernel 进一步演化）；5) 第五代 NVLink 1.8 TB/s 双向。Software stack 上需要：CUDA 12.4+、cuDNN 9+、NCCL 2.20+、新版 Triton / FlashAttention 适配 FP4 与 Tensor Memory。

### 2. 底层原理
FP4 格式有两种：NVFP4（NVIDIA 自家，E2M1，4-bit shared exponent per block）；MXFP4（OCP 标准，类似但 block 定义不同）。两者数学上接近，硬件上 B200 都原生支持。FP4 训练效果依赖 microscaling（每个小 block 一个 scale，避免 outlier 主导）。

NVL72 把 72 张 B200 用 NVSwitch 全互连成一个域，域内任意两卡 1.8 TB/s 双向带宽。这让"模型放进一个 NVL72 域"成为新的部署单元——单域内 TP / EP / CP 都能高吞吐运行。

Tensor Memory 是异步张量搬运单元的演化，比 H100 的 TMA 更细粒度，FA-3 的 producer/consumer 模型在 B200 上能进一步演化为 multi-producer multi-consumer。

### 3. 关键机制 / 流程 / 数据结构
第一，FP4 训练实践。权重存 BF16，矩阵乘走 FP4 + microscale。常见路径：W BF16 → FP4 quantize（per-block scale）→ FP4 GEMM → BF16 accumulate。NVIDIA Transformer Engine 提供了这条 path 的封装。

第二，FP4 推理。直接量化权重到 FP4 是一条路径；Marlin / CUTLASS 在 B200 上有 W4A4 kernel。AWQ / GPTQ 等校准方法在 FP4 上也适用。

第三，NVL72 部署。一个 NVL72 域内：70B 模型 TP=8 + DP=9，或 256-expert MoE EP=72 + TP=1，都能跑得很 happy。跨 NVL72 域走 IB 400G/800G。

第四，软件兼容性。CUDA 12.4 起支持 B200，cuDNN 9 才有 FP4 kernel，NCCL 2.20 才认 NVL72 拓扑。早期版本会跑但跑得慢。

### 4. 工程权衡 / 性能影响
FP4 训练比 FP8 再快 ~2x（算力翻倍），但 perplexity 涨幅 0.5-1.5%（视训练长度 / 数据量）。MoE 模型对 FP4 更宽容；dense 模型对 FP4 更敏感。

NVL72 域大带来的实质变化：以前模型并行被限制在 NVL8 内（8 卡 TP），跨域走 IB 慢；现在 NVL72 内 72 卡都是 NVLink，可以 TP=72 / EP=72，让单层内并行度大幅扩展。

显存容量翻倍。B200 192GB 让长上下文 KV cache 不再是瓶颈——单卡能放 32k context × batch 64 的 KV，common case 不需要 CP。

供应紧张。2025-2026 年 B200 / GB200 供应受限，capacity blocking 抢得很厉害。这反过来让 H200 + NVL8 + FP8 仍是大量 production fleet 的现实选择。

### 5. 常见追问 / 易错点
第一，FP4 训练真的稳定吗？短期看 DeepSeek、xAI 都公开了 FP4 训练初步结果，loss 曲线与 FP8 接近。长期 SFT / 后训练对量化噪声更敏感的部分（embedding、router）仍保持 BF16。

第二，NVL72 内为什么不直接 TP=72？因为 TP 切 head，head 数得整除。Llama-3 70B 有 64 head，不能 TP=72；常用 TP=8 + EP=8 + DP=...。

第三，B200 推理还要不要 FP8？要。FP8 在 inference 上仍然是 sweet spot——比 FP4 准确、比 BF16 快 ~2x。FP4 inference 适合 batch rollout、long-context decode 等吞吐优先场景。

第四，能否买不到 B200 就买 H200？H200 是 H100 + 更大 HBM（141GB），不带 FP4 也不带 NVL72。能做的事 H100 都能做，只是 KV cache 更宽松。预算紧张的项目仍是合理选择。

### 6. 实践建议
新集群优先 B200 + NVL72；存量集群继续 H100/H200 + FP8。FP4 训练先在小模型 / 中等规模实验，大规模 production 还是 FP8 为主，FP4 优先用于 inference + 后训练。

软件栈对齐：CUDA 12.4+、cuDNN 9+、NCCL 2.20+、Transformer Engine 1.10+、vLLM v0.6+ / SGLang 0.3+。

面试讲 B200 时强调三点：FP4 一等公民、NVL72 域、Tensor Memory。再讲清楚"软件适配"的滞后窗口（B200 出来到 ecosystem 完备约 6-9 个月）就到位。

### 7. 30 秒速答
- B200 三大升级：FP4 一等公民（~5x H100 FP8）、NVL72 域、HBM3e 192GB + Tensor Memory。
- NVL8 → NVL72 让单域内 TP/EP/CP 都能高吞吐，跨域走 IB 800G。
- 易踩坑：CUDA 12.4 / cuDNN 9 / NCCL 2.20 / TE 1.10 不齐就跑得慢，软件滞后 6–9 个月。
- 加分关键词：MXFP4 vs NVFP4、microscale block、第五代 NVLink 1.8 TB/s、FP4 推理 vs FP8 sweet spot。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 B200 与 H100/H200 的三大代际差异？
- [ ] 你能不能解释为什么 NVL72 让"模型放进单域"成为新部署单元？
- [ ] 你能不能举一个 B200 上 FP4 推理 vs FP8 推理的场景取舍？
- [ ] 你能不能说出 H200 相对 H100 仅是显存升级、不带 FP4 / NVL72？

## Q13. NVL72 vs NVL8 对模型并行策略的影响是什么？

> 🔴 专家 · 你以前的 TP=8 + 跨机 PP 现在可以变成 NVL72 内 TP=8 + EP=9 + DP=1——并行拓扑被硬件重画了一遍。

### 1. 核心结论
NVL72 单域 72 卡全 NVLink 互连，对模型并行的影响：1) 大模型可以"塞进单域"——70B 用 NVL72 内 8-way TP + 9-way DP，单域吞吐顶到极致；2) MoE 模型 EP 可以扩到 72（甚至 144 通过 2 个 NVL72），all-to-all 通信全走 NVLink；3) PP 需求下降——以前因 NVL8 卡数不够要靠 PP 切层，现在 72 卡同域 PP 不再必须；4) 跨 NVL72 训练时，NVL72 内做"快通信"（TP/EP/CP），NVL72 间做"慢通信"（DP/PP）。

### 2. 底层原理
NVLink 5th gen 在 B200 上单 link 100 GB/s 单向，每张 B200 有 18 个 link → 1.8 TB/s 双向。NVL72 用 NVSwitch 4 实现 72 卡全互连，任意两卡 1.8 TB/s。

这相当于把以前 IB 400G（50 GB/s）的跨机通信换成 NVLink 1.8 TB/s——通信速度 30x。结果就是以前必须分层处理的并行策略可以"压平"。

NVL8 时代典型布局：每机 8 卡 NVL8 做 TP=8，机间 IB 做 DP/PP。NVL72 时代变成：72 卡 NVL72 做 TP+EP+DP 任意组合，跨 NVL72 做 DP/PP。

### 3. 关键机制 / 流程 / 数据结构
第一，并行拓扑映射。在 NVL72 内：把高频通信维度（TP、EP、CP）放在域内；把低频通信维度（DP、PP）放在跨域。这是 NCCL 拓扑感知所做的事情。

第二，all-to-all 在 NVL72 内的优势。MoE EP all-to-all 通信量随 EP 数线性涨。NVL72 让 EP=72 的 all-to-all 在 NVLink 上跑，延迟降到几十微秒；NVL8 时代 EP=8 跨机 IB 已经 ms 量级。

第三，NCCL 拓扑文件。NVL72 域需要 NCCL 2.20+ 才正确识别。早期版本可能把 NVL72 当成多个 NVL8 处理，丢失大量带宽。

第四，DP=1 推理服务。NVL72 单域吞吐已足够大，可以单域单副本 serve（DP=1），跨域是"多副本"扩展，而非"模型并行"扩展。

### 4. 工程权衡 / 性能影响
模型规模上限上提。70B / 200B 模型以前因显存只能跨机 TP+PP；现在 NVL72 内 8-9 张 B200 就能跑 dense 200B（每张 24 GB 容纳模型权重）。

MoE 部署优化。256-expert MoE 在 NVL72 内 EP=72，每卡 ~3.5 expert，all-to-all 在 NVLink 上跑——比 NVL8 时代快 5-10x。

PP 退场（部分）。dense 模型 200B 以下不再需要 PP，省了 1F1B / zero-bubble 等复杂调度。但 PP 在跨 NVL72 训练仍然有用。

跨 NVL72 通信仍是瓶颈。集群规模上去（数千卡）后，跨 NVL72 的 IB / NDR 400G/800G 是新的关键。NVL72 内卷得越极致，跨域成为瓶颈的概率越高。

### 5. 常见追问 / 易错点
第一，是不是 NVL72 就不用 PP 了？对于 1T 以下 dense 模型基本不用；1T+ 大概率还要 PP。MoE 普遍不需要 PP（EP 主导）。

第二，CP 在 NVL72 内更有优势吗？是。Ring Attention 通信全在 NVL72 内走 NVLink，超长 context（256k+）训练比 NVL8 时代实用很多。

第三，DP 跨 NVL72 走 IB 慢，怎么平衡？用 FSDP2 + bucket 优化、async TP、optimizer overlap 都是常见手段。

第四，能不能 NVL72 + NVL8 混合？理论上能但 NCCL 拓扑识别可能混乱，生产里一般同代硬件部署。

### 6. 实践建议
新集群按 NVL72 域规划部署。模型并行策略从"放进一个 NVL72 域"反推。

跨 NVL72 选 IB NDR 400G 起步，800G 在 B200 时代已逐步普及。

面试讲 NVL72 时主线是「同一 SKU 内的高带宽域扩张让并行拓扑被重画」。能讲清"PP 退场，EP 扩大"这两个具体后果就到位。

### 7. 30 秒速答
- NVL72 单域 72 卡全 NVLink，每对卡 1.8 TB/s，跨机 IB 通信速度被同域 30x 替代。
- 后果：dense 200B 单域装下，PP 退场；MoE EP 扩到 72，all-to-all 走 NVLink 飞快。
- 易踩坑：head 数得整除 TP，Llama-3 70B 64 head 不能 TP=72；NCCL 2.20 才正确识别 NVL72 拓扑。
- 加分关键词：拓扑感知、压平并行、跨 NVL72 走 IB 800G、FSDP2 + async TP、单域单副本。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 NVL72 相对 NVL8 在通信带宽上的差异（30x）？
- [ ] 你能不能解释为什么 PP 在 NVL72 时代 dense 模型上几乎退场？
- [ ] 你能不能举一个 256-expert MoE 在 NVL72 EP=72 的具体收益？
- [ ] 你能不能说出 NVL72 域间 IB / NDR 仍是新瓶颈的场景？

## Q14. FP4 训练需要哪些数值技巧？为什么不是简单的"把 FP8 量化代码改成 FP4"？

> 🔴 专家 · FP4 只有 16 个可表示值，outlier 一颗就让一整块 block 全军覆没——你必须用 per-block microscale + tensor-level clip + 选择性 fallback 三件套。

### 1. 核心结论
FP4 训练有效需要：1) microscaling（每 32 或 16 元素一组 scale，让 outlier 不污染整 block）；2) per-tensor stochastic rounding（避免系统性 bias）；3) loss scaling 调整（FP4 动态范围更窄，loss scale 要更激进）；4) BF16 master weight + FP4 GEMM 计算路径（DeepSeek FP4 训练公开方案）；5) sensitive layer fallback（embedding / output projection / LayerNorm 通常 BF16）。Transformer Engine、TorchTitan 等开源已开始支持。

### 2. 底层原理
FP4（E2M1）只能表示 ±0、±0.5、±1、±1.5、±2、±3、±4、±6——16 个 unique 值。这个动态范围对一个 tensor 一个 scale 来说太窄，必须用 microscale：每 32 元素（或 16/8）一组共享 exponent。

Microscale 形式：把 tensor reshape 成 N × 32，每行算 max abs，scale = max_abs / FP4_max。每行内的元素都除以这个 scale 量化到 FP4。这样 outlier 只影响所在 row，不污染整个 tensor。

MXFP4（OCP 标准）规定 32 元素 block + 1 个 E8M0 shared exponent；NVFP4（NVIDIA）类似但有自家调整。两者数学上接近。

Stochastic rounding：FP4 -> nearest 会引入 bias（小值往零偏），用随机舍入（按距离比例概率上下取）消除累积 bias。

### 3. 关键机制 / 流程 / 数据结构
第一，GEMM 路径。FP4 输入 × FP4 权重 → BF16 / FP32 累加。B200 tensor core 原生支持这种 mixed precision GEMM。

第二，主权重保留 BF16 / FP32。优化器更新走高精度，每 forward 时 cast 到 FP4 算 GEMM。这是 mixed precision 的扩展。

第三，敏感层 fallback。embedding lookup、final logits projection、LayerNorm、softmax 通常 BF16。这些层 FLOP 占比小，精度收益大。

第四，loss scaling 与 gradient handling。FP4 gradient 比 FP8 更易下溢；要用 per-channel gradient scaling 或 dynamic loss scaling 配合。

### 4. 工程权衡 / 性能影响
吞吐收益。B200 上 FP4 GEMM ~2x FP8 算力（理论上）。实际训练吞吐涨幅 30-80%，取决于 attention / FFN 占比与 sensitive layer 比例。

精度风险。微小模型（<3B）FP4 训练 perplexity 可能涨 1-2%；大模型 / MoE 更宽容，公开的 DeepSeek FP4 实验大模型损失 < 0.5%。

调参负担。loss scale、stochastic rounding 概率、sensitive layer 选择都是新 hyper-param。生产里通常用厂商提供的 default 配置作为起点。

### 5. 常见追问 / 易错点
第一，FP4 与 INT4 区别？FP4 有浮点动态范围（block 内能表示远远超 4-bit 的 range），INT4 是固定 step 的整数。FP4 更适合矩阵乘累加，INT4 更适合定点 inference。

第二，FP4 训练能拿来微调吗？目前主流 SFT / RLHF 仍用 BF16 / FP8；FP4 主要在 pretrain 大规模训练验证。微调因为对精度敏感，FP4 应用较少。

第三，block size 选 32 还是 16？OCP 标准 32 是主流；某些场景 16 精度更好但算力收益少。生产里跟硬件 default 走（B200 是 32）。

第四，需要重新 calibrate 吗？训练时不需要——scale 是 dynamic 计算的。inference 时如果用 FP4 权重，要 calibrate。

### 6. 实践建议
B200 + FP4 训练：直接用 Transformer Engine / TorchTitan 的 FP4 path，default 配置作为起点。先在小模型验证后扩到大规模。

监控：grad norm（避免 NaN）、per-layer activation distribution（看是否有持续 outlier 层）、训练 loss vs FP8 baseline（多用几个 step 对比）。

面试讲 FP4 时强调「microscale 是 FP4 能跑起来的根本原因，不是简单把 FP8 改成 FP4 就能用」。能讲 stochastic rounding 与 sensitive layer fallback 是高分。

### 7. 30 秒速答
- FP4 只有 16 个可表示值，单 outlier 让整 tensor 一个 scale 失效，必须 per-32 元素 microscale。
- 训练三件套：microscale + stochastic rounding + sensitive layer (embedding/LN) BF16 fallback。
- 易踩坑：grad 在 FP4 下易下溢，要 per-channel gradient scaling 配合 dynamic loss scaling。
- 加分关键词：MXFP4 vs NVFP4、E2M1 + shared E8M0、BF16 master weight、FP4 推理 W4A4 kernel。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 microscale 为什么是 FP4 能用的根本？
- [ ] 你能不能解释 stochastic rounding 消除累积 bias 的机制？
- [ ] 你能不能举一个 FP4 训练时 fallback 到 BF16 的具体层（embedding / final logits / LN）？
- [ ] 你能不能说出 FP4 与 INT4 在动态范围和适用场景上的差异？

## Q15. 长上下文（1M+）训练的工程瓶颈有哪些？

> 🔴 专家 · 1M context 训练时，attention 的 O(N²) 把单卡撑爆、Ring CP 通信吃满 NVLink、KV 显存放不下——这是把 attention、显存、通信三类瓶颈同时拉到极致的负载。

### 1. 核心结论
长上下文（>128k token）训练的主要瓶颈：1) attention O(N²) 计算与显存（FA-2/3 + CP 解决）；2) activation 显存爆炸（per-token activation × N tokens；激活重计算 + CP 缓解）；3) KV 在 CP 间 ring 传递通信代价（NVL 域是必要条件）；4) data loading 与 sequence packing（拼短样本到长 batch 浪费 padding）；5) loss / 梯度数值稳定性（长 sequence 梯度累积易溢出）。

### 2. 底层原理
Attention 计算 O(N²)：N=1M 时单 head 一次 attention 就要 1T FLOPs；FA-3 + flash-decoding 把内存放大降为 O(N)，但计算仍是 O(N²)。

Activation 显存：transformer 每层每 token 激活 ~10-20KB（70B 模型）；N=1M 单层激活 10-20GB——远超单卡。CP=16 摊薄到每卡 600MB-1.2GB，仍紧张但可行。激活重计算把激活打到 attention 边界，再 forward 一次。

Ring Attention 通信：每 ring step 发送 KV chunk，总量 ~N×d×L_layers。NVL72 内 1.8 TB/s 通信，1M context 一层 attention 的 ring 通信 ~10GB，10ms 量级，跑得动。

### 3. 关键机制 / 流程 / 数据结构
第一，FA-3 + CP 协同。FA-3 内 producer/consumer + ring scheduling 适配 CP 的多步 KV 传递。

第二，激活重计算粒度。整层重计算（最省显存最贵）vs FFN-only 重计算（中等）vs 选择性（最细）。长 context 训练通常全层重计算 + selective on attention。

第三，sequence packing。把多条短样本 pack 成长 batch，用 doc mask（每条样本只看自己内部）。能让长 context 训练在短数据上也有意义。

第四，position encoding 兼容。RoPE 在 long context 需要 NTK / YaRN 扩展。训练时也要扩展，否则 OOD position。

### 4. 工程权衡 / 性能影响
CP 大小与 sequence 长度匹配。128k：CP=4-8；512k：CP=8-16；1M：CP=16-32。

激活重计算开销 ~25% throughput drop，但显存收益巨大，长 context 训练几乎必开。

Step time 增长。1M context 单 step 可能是 32k context 的 30-50x（O(N²) 主导）。吞吐 tokens/s 下降但能训新能力。

### 5. 常见追问 / 易错点
第一，1M 训练数据从哪儿来？多文档拼接、book、code repo、互联网 long-form。数据多样性是大问题。

第二，长 context 训练后短 context 效果会变差吗？正确处理（mixed 短 + 长 + curriculum）不会显著退化。错误处理（全长训练）会让短上下文损失 1-3%。

第三，FlashAttention 没有 long-context 退化？FA 是 IO 优化不是 O() 优化——计算量还是 N²。Long context 主要靠 CP + FA 协同。

第四，1M 推理是不是必须用 CP？通常不用——推理 batch 小、KV 总长度通常远小于训练 sequence，可以单卡或 TP 内 KV 装下。

### 6. 实践建议
长 context 训练路线：先 32k → 128k → 256k → 1M 渐进式扩，每阶段 ~50B token 训练量。

NTK / YaRN 在每个 length 阶段调整 RoPE base / scaling。

面试讲 long-context 时主线"O(N²) + 显存 + 通信三连击"。能讲 FA-3 + CP + 激活重计算的协同是高分。

### 7. 30 秒速答
- 1M context 训练三连击：attention O(N²) 算 + 激活 O(N) 显存 + ring KV 通信。
- 路径：FA-3 + CP=16-32 + 全层激活重计算 + sequence packing + RoPE NTK/YaRN 扩展。
- 易踩坑：CP 域必须在 NVL 内，跨机 ring 通信完全打不动；全长训练让短上下文退化。
- 加分关键词：渐进 32k→128k→1M、selective 重计算、doc mask、YaRN scaling、attention 仍是 N²。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 1M context 训练的三大瓶颈？
- [ ] 你能不能解释 FA-3 解决了什么、CP 解决了什么、激活重计算解决了什么？
- [ ] 你能不能举一个 sequence packing + doc mask 的具体使用场景？
- [ ] 你能不能说出 long-context 训练后短 context 退化的常见原因？

## Q16. RAG vs long-context vs prompt caching 的 infra 权衡是什么？

> 🟡 进阶 · 你的用户要传一份 50 万字的合同问问题——是把整份合同每次塞进 prompt（long-context），还是切块检索（RAG），还是利用 prefix caching 让重复 prompt 复用 KV？三条路有不同 infra 成本。

### 1. 核心结论
三种处理超长上下文的范式：1) long-context：把全文塞 prompt，模型直接处理。优点是模型能跨段推理，缺点是 prefill cost 高、KV 显存大。2) RAG：把全文 chunk + embedding，query 时检索相关 chunk 作为 prompt 一部分。优点是 prompt 短、可扩展到无限文档量，缺点是检索召回限制 + chunk 间推理弱。3) prompt caching：同一文档对多个 query 时复用 prefill 的 KV cache。优点是首次后近零边际成本，缺点是只对重复 prompt 有效。三者不互斥，生产里常常组合用。

### 2. 底层原理
Long-context 成本 = prefill (O(N²)) + decode KV usage (O(N))。1M context 一次 prefill 几百毫秒到几秒，KV 显存几 GB。

RAG 成本 = embedding 检索 (~10ms) + 缩小后的 prefill。Top-K=20 chunk × 1k token = 20k context，prefill 几十毫秒。

Prompt caching 成本 = 第一次 prefill 全 cost + 后续 query 只算增量。LMCache / SGLang RadixAttention / vLLM prefix caching 都是这类机制。

### 3. 关键机制 / 流程 / 数据结构
第一，prompt caching 在不同引擎的实现。vLLM 的 prefix cache 是 hash-based block；SGLang 的 RadixAttention 是 radix tree，能匹配任意 prefix；LMCache 是跨节点 cache。

第二，缓存命中率。命中率取决于使用模式：agent multi-turn 通常 70%+；用户 ad-hoc query 可能 20%。

第三，RAG 检索栈。embedding model + vector DB + reranker。常见 stack：bge / e5 embedding + Milvus / Qdrant + bge-reranker。

第四，混合方案。RAG 拿回相关 chunk，long-context 把 chunk 一起塞，cached prompt 部分用 prefix cache——三者叠加常见。

### 4. 工程权衡 / 性能影响
单 query 单元成本：caching hit < RAG < long-context。但 RAG 牺牲推理质量，long-context 在跨段推理任务上更强。

延迟：caching hit ~ TPOT × output tokens（前缀几乎免费）；RAG ~ 检索 + 短 prefill；long-context ~ 长 prefill。

可扩展性：RAG 上限是 vector DB 容量（亿级文档可行）；long-context 上限是模型支持长度（当前 1M-2M）；caching 上限是 cache 容量（KV 几十 GB / 节点）。

### 5. 常见追问 / 易错点
第一，RAG 检索召回如何评估？hit rate / MRR / NDCG。生产里 hit rate @ top-K 是最实用指标。

第二，long-context 在中间段的注意力会衰减（lost in the middle）吗？早期模型会，1M context 训过的现代模型显著缓解但未消除。仍是评估重点。

第三，prefix caching 在跨用户场景下的安全性？要做好隔离——不同 tenant 的 cache 不混用，防止 prompt 泄露。

第四，RAG 的 chunk size 怎么选？典型 200-500 token。太短上下文不够，太长召回精度差。

### 6. 实践建议
文档库巨大、单 query 跨范围广：RAG 主导。文档量小、跨段推理多、用户问深问题：long-context + caching。Agent / multi-turn：caching 是必选项，常常配 RAG / long-context。

监控：cache hit rate、retrieval recall、E2E latency、tokens/$。

面试讲三者关系时强调"不是 either-or，是 stack 协同"。能讲 prefix caching 在 multi-turn 场景下的关键价值是高分。

### 7. 30 秒速答
- 三条路：long-context（全文进 prompt）、RAG（检索 chunk）、prompt caching（复用 prefix KV）。
- 单元成本：caching hit < RAG < long-context；推理质量：long-context > RAG，agent 必带 caching。
- 易踩坑：RAG chunk 200-500 token 是 sweet spot；prefix caching 跨 tenant 必须严格隔离防泄露。
- 加分关键词：RadixAttention、LMCache、lost-in-the-middle、hit rate@K、混合 stack。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清三种方案在 prompt 长度与单元成本上的差异？
- [ ] 你能不能解释为什么 multi-turn agent 场景 prefix caching 是命脉？
- [ ] 你能不能举一个 RAG + long-context + caching 混合 stack 的具体场景？
- [ ] 你能不能说出 long-context 模型 lost-in-the-middle 现象的评估方法？

## Q17. Agent infra 与 stateless LLM serving 的差异是什么？

> 🔴 专家 · 一个 agent 任务可能跑 30 步、每步调 5 个 tool、跨 10 分钟——你的推理引擎不再是"一次 forward 出一段 token"，而是要管 KV 复用、长上下文滚动、tool 调用编排、checkpoint 恢复的完整状态机。

### 1. 核心结论
Agent serving 与传统 stateless LLM serving 的关键差异：1) 多步 tool 调用：每步的 LLM 输出作为 tool input，tool output 作为下一步 LLM input；2) 长生命周期：单个 agent 任务从分钟到小时级，需要 checkpoint / 恢复；3) KV cache 极度可复用：同 agent 内每步的 prefix 大量重叠，prefix caching 是命脉；4) 跨步骤 state：tool 调用日志、思考链、scratchpad 都要持久化；5) 调度复杂：tool 调用可能慢（外部 API、代码执行），LLM 调度器要让出 GPU。

### 2. 底层原理
Agent loop：observe → think → act (tool call) → observe → ...。LLM 在每步生成"思考 + 行动指令"，被解析后调相应 tool（搜索、代码执行、API、子 agent）。

每步的 prompt = system prompt + tool 定义 + 全部历史 think-act-observe trace。所以 prompt 长度随步数线性涨。一个 30 步 agent 任务 prompt 长度可能从 1k 涨到 50k。

KV cache reuse 的角色：第 N+1 步的 prompt = 第 N 步 prompt + (新 thought + tool output)。前缀完全一样——RadixAttention / prefix caching 命中率接近 100%。这就让 agent 每步 LLM cost 只是"增量 prefill"而非"全 prefill"。

### 3. 关键机制 / 流程 / 数据结构
第一，cache key 设计。RadixAttention 用 prompt 字符串做 key；agent 长 trace 要确保串行化稳定（同样 think-act 序列产生同样 prefix）。

第二，tool 调用 latency 隔离。Tool 调用（如 API call 10s、代码执行 60s）期间 LLM 不在 forward，要让出 GPU。生产里 agent runner 是独立服务，LLM 是 RPC service，agent state 在 Redis / Postgres 持久化。

第三，checkpointing。每步结束 dump state（trace、tool outputs、KV cache reference），故障恢复时从最近 checkpoint 重启。KV cache 可以 lazy reload（重新 prefill）也可以持久化（LMCache 跨节点）。

第四，多 agent 协作。Agent-of-agents（manager-worker、debate）让 infra 复杂度再上一层。worker agent 状态独立，manager 调度 worker。

### 4. 工程权衡 / 性能影响
prefix caching 是决定 agent serving 经济性的关键。无 caching：每步 50k context full prefill，30 步 = 1.5M tokens prefill。有 caching：30 步只算每步增量（几百 token），总成本 1-2k tokens。差异 1000x。

KV cache 显存压力。长 agent trace 的 KV cache 几 GB。跨节点 KV cache（LMCache、SGLang shared cache）让多个 agent 节点能复用。

调度上 agent runner 与 LLM serving 解耦。Agent runner 处理 state machine、tool execution；LLM serving 处理 token generation。两者之间 RPC。

### 5. 常见追问 / 易错点
第一，agent KV cache 怎么管理？跨 step 同一个 agent 的 KV 要保留；不同 agent 的 KV 独立；session 结束 evict。生产里 agent ID + step ID 作为 cache namespace。

第二，tool output 怎么塞进 prompt？通常 special token / role 包装。OpenAI tool 协议、ReAct 模板各有约定。

第三，agent 任务跨 model serving 实例怎么处理？要么 sticky routing（同 agent 路由同实例，cache 命中）要么 distributed KV cache（LMCache 类）。生产里 sticky 是主流。

第四，agent 失败怎么 retry？通常从最近 successful step 重启。状态机要 idempotent。

### 6. 实践建议
Agent infra 三件套：Agent runner（state machine）+ LLM serving（with prefix caching）+ tool sandbox。三者用 RPC 解耦。

KV cache 策略：单实例 RadixAttention + sticky routing 是 80% 场景的最优选；多实例规模化需要 LMCache 类跨节点 cache。

监控：average steps per task、cache hit rate、tool latency 分布、agent failure rate、tokens/task。

面试讲 agent infra 时主线"无 cache 经济性死、有 cache 一切轻松"。能讲 RadixAttention 与 sticky routing 协同就到位。

### 7. 30 秒速答
- Agent 是有状态的多步状态机：observe → think → act → observe，prompt 随步数线性涨。
- prefix caching 是经济性命脉：每步 50k prompt 增量 vs 全 prefill，cost 差 1000x。
- 易踩坑：sticky routing 不开就把同 agent 打到不同实例，cache 全 miss；tool call 期间要让出 GPU。
- 加分关键词：RadixAttention、LMCache 跨节点、agent runner + LLM serving 解耦、checkpoint 恢复。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 agent infra 与 stateless serving 的核心差异？
- [ ] 你能不能解释 prefix caching 让 agent 每步只算增量的机制？
- [ ] 你能不能举一个 30 步 agent 任务 prompt 长度从 1k 涨到 50k 的具体场景？
- [ ] 你能不能说出 sticky routing 与 distributed KV cache 在大规模 agent 上的取舍？

## Q18. 推理引擎的"调度器演进"经历了哪几代？vLLM v0/v1、SGLang、TRT-LLM 现在各自架构什么样？

> 🔴 专家 · 从一开始的"一个 batch 一直跑到所有 sample 结束"到 continuous batching 到 chunked prefill 到 prefill-decode 解耦——推理调度器走过了五代。

### 1. 核心结论
推理引擎的调度器演进大致五代：1) static batching（batch 内同进同出，HuggingFace TGI 早期）；2) continuous batching（动态加入退出 token-level，vLLM/SGLang 起点）；3) PagedAttention（KV 分 block，碎片化降到零）；4) chunked prefill（长 prompt 切片与 decode 交错）；5) disaggregated prefill/decode（物理拆开两个池）。当前 vLLM v1 是 (1)+(2)+(3)+(4) 完备；SGLang 加 RadixAttention（前缀树 cache）；TRT-LLM 工程化最深但跑在 NVIDIA 闭源 stack 上。

### 2. 底层原理
Static batching：所有 sample 必须等最长那条 decode 完才能下一 batch。短样本被长样本拖死，GPU util 大幅波动。

Continuous batching：scheduler 每个 step 检查 batch，完成的 sample 立刻退出、queue 里新 sample 立刻加入。batch 是个滚动窗口。

PagedAttention：KV cache 按 16/32 token block 分配，每个 sequence 持有 block 列表。block 池统一管理，碎片化降到 0。

Chunked prefill：长 prompt prefill 切成 1k token chunks，与 decode batch 在同一 forward 内交错。避免长 prompt 阻塞 decode。

Disaggregated：prefill 与 decode 在物理上不同 GPU 池，已在 Q10 讲过。

### 3. 关键机制 / 流程 / 数据结构
第一，vLLM v1 架构。Scheduler + KV cache manager + Worker。v1 重写了 v0 的执行栈，把 cuda graph、tensor parallel、speculative decoding 都集成进去。Continuous batching + PagedAttention + chunked prefill 都是 default。

第二，SGLang RadixAttention。把 prefix tree 当一等公民，自动检测共享前缀复用 KV。对 agent / multi-turn 场景命中率超高。

第三，TRT-LLM。NVIDIA 闭源优化，深度绑硬件。kernel 库 + builder（编译 engine）+ Triton runtime。性能上限通常最高但可移植性差、binary 大、迭代慢。

第四，Triton inference server（与 TRT-LLM 不同）。前端调度器，可以背靠 TRT-LLM、ONNX RT、vLLM 等多个 backend。

### 4. 工程权衡 / 性能影响
vLLM 是开源 ecosystem 默认起点，社区活跃、模型支持广。SGLang 在 agent / RL rollout 场景常更优（RadixAttention 给力）。TRT-LLM 性能天花板最高，但生态绑死 NVIDIA。

调度器复杂度。从 static → continuous → chunked → disaggregated，每代都让单 token throughput 涨 30-100%，但工程实现也越来越难。

新硬件适配滞后。B200 出来后 vLLM / SGLang / TRT-LLM 都要重新写 kernel；早期 vLLM 在 B200 上 perf 远不如理论值。

### 5. 常见追问 / 易错点
第一，为什么不让 vLLM 直接调 TRT-LLM kernel？vLLM 是 PyTorch 生态、TRT-LLM 是 builder + plugin 生态，整合工作量大。社区一直有讨论但未落地。

第二，SGLang 的 RadixAttention 跟 vLLM prefix caching 区别？vLLM 是 block hash + exact prefix；SGLang 用 radix tree 能匹配任意 prefix 长度。对碎片化共享 prefix（agent）后者更优。

第三，chunked prefill 的 chunk size 选什么？典型 512-2048。太小调度开销；太大失去交错效果。

第四，能不能完全用 PyTorch 直接做生产推理？小模型可以；大模型 latency / throughput 跟专门引擎差 3-10x，生产几乎不做。

### 6. 实践建议
开源起点：vLLM。Agent 多：SGLang。性能榨干 + 接受 NVIDIA 绑定：TRT-LLM。

跟随 vLLM v1 / SGLang 0.4+ 版本，每月 release 都有关键优化。

监控四件套：tokens/s、TTFT p99、TPOT p99、GPU util。

面试讲调度器演进时按时间线讲五代，每代解决什么、引入什么。能讲 vLLM v1 与 SGLang RadixAttention 在 agent 场景的差异是高分。

### 7. 30 秒速答
- 五代演进：static batching → continuous batching → PagedAttention → chunked prefill → disaggregated。
- vLLM v1 集齐前四代，SGLang 加 RadixAttention，TRT-LLM 闭源性能上限最高但绑 NVIDIA。
- 易踩坑：B200 新硬件上 vLLM / SGLang kernel 滞后 6-9 个月才追上理论值。
- 加分关键词：cuda graph、tree mask、radix tree vs hash block、Triton inference server。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清五代调度器各自解决了什么问题？
- [ ] 你能不能解释 SGLang RadixAttention 相对 vLLM block hash 的差异？
- [ ] 你能不能举一个 chunked prefill chunk size 选择的具体取舍（512 / 2048）？
- [ ] 你能不能说出 vLLM / SGLang / TRT-LLM 在 agent / RAG / 极致性能场景的选型建议？

## Q19. KV cache offload (到 CPU / SSD / 跨节点) 什么时候值得？

> 🔴 专家 · 你显存装不下 1M context 的 KV，能不能 offload 到 CPU？答案是"看带宽差"——offload 让你能装下更多但每 token 解码慢 5-50x。

### 1. 核心结论
KV cache offload 把不活跃的 KV 从 GPU HBM 搬到 CPU 内存、SSD、甚至其他节点的 HBM。常见层级：GPU HBM > 跨 GPU NVLink (200-1800 GB/s) > CPU memory via PCIe (50-100 GB/s) > local NVMe (5-10 GB/s) > network storage (GB/s 量级)。Offload 解锁的能力：超长 context、超大 batch、KV 持久化（agent state）。但每次 attention 要从慢介质拉 KV，对解码 throughput 影响显著。

### 2. 底层原理
Attention 在 decode 阶段必须读全部 KV cache。如果 KV 在 HBM（3-8 TB/s），单 token decode ms 量级；放 CPU memory（PCIe 50 GB/s）要拉 ~100MB KV，2ms+；放 NVMe 慢一个量级。

跨 GPU NVLink offload 是 sweet spot：相邻 GPU HBM 之间走 NVLink 1.8 TB/s，带宽接近本地 HBM，能让"虚拟 KV 容量"扩到 NVL72 域内所有卡的 HBM 之和（72 × 192GB = 13.8TB）。

CPU offload 适用场景：long context training 中暂存激活；agent state 跨 session 持久化；冷 sample 暂存。

NVMe offload 仅适用极端容量需求（cross-session agent state）。

### 3. 关键机制 / 流程 / 数据结构
第一，LMCache 跨节点 KV cache。把 KV 缓存暴露成 cluster-wide 服务，多个推理节点共享。对 agent / multi-tenant 命中率涨幅显著。

第二，prefetch 与 pipelined offload。预测下一步要的 KV，提前 prefetch 到 HBM。能 hide 部分 offload latency。

第三，KV 量化 + offload 协同。先 FP8 / INT4 量化再 offload，带宽需求降 2-4x。

第四，eviction 策略。LRU / LFU / agent-aware（同 agent 同 session 优先保留）。

### 4. 工程权衡 / 性能影响
HBM 内 KV：tokens/s 接近峰值。NVLink 跨卡 offload：tokens/s 降 10-30%。PCIe CPU offload：tokens/s 降 5-10x。NVMe：降 50-100x。

经济性。CPU memory 比 HBM 便宜 10-50x，NVMe 又便宜 5-10x。容量上 CPU 内存 / NVMe 大 1-2 个数量级。

当 GPU 显存极度受限（推理服务想跑 1M context 但买不起更多卡），offload 是唯一路径。

### 5. 常见追问 / 易错点
第一，offload 后还能用 PagedAttention 吗？能——block 既可以在 HBM 也可以在 CPU memory，attention 时按需 swap。但 swap latency 是新瓶颈。

第二，offload 与 prefix caching 配合？prefix cache 持久化到 CPU memory / 跨节点，下次 query 命中 prefix 就只需要 reload。LMCache 就是这思路。

第三，跨节点 KV cache 有什么坑？序列化开销（KV 几 GB tensor 序列化耗时不小）、版本一致性（模型升级后旧 KV 失效）、网络拥塞。

第四，是不是 offload 越多越省钱？不是。在线服务 tokens/$ 取决于 latency × tokens/s，offload 让 latency 涨，可能反而单 token 成本上升。要算综合账。

### 6. 实践建议
默认不开 offload。当且仅当：1) 显存确实装不下；2) 业务对 latency 不敏感（batch processing、offline 评估）；3) 跨 session caching 价值高（agent）。

跨 NVLink offload 优先于 CPU offload。CPU offload 优先于 NVMe。

监控：offload byte/s、offload-induced latency、HBM 利用率、cache hit rate。

面试讲 KV offload 时强调"是带宽差换容量"。能讲不同层级的延迟与容量取舍是高分。

### 7. 30 秒速答
- KV offload 是带宽差换容量：HBM 8 TB/s > NVLink 1.8 TB/s > PCIe 50 GB/s > NVMe 5 GB/s。
- 跨 NVLink 是 sweet spot，等于把 NVL72 域 13.8 TB HBM 当一个大 KV 池。
- 易踩坑：PCIe CPU offload 让 tokens/s 降 5-10x，NVMe 降 50-100x，offline 场景才能用。
- 加分关键词：LMCache 跨节点、prefetch / pipelined、量化+offload 叠加、agent-aware eviction。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 KV offload 各层级的带宽数量级差异？
- [ ] 你能不能解释为什么跨 NVLink offload 是 sweet spot？
- [ ] 你能不能举一个值得开 offload 的具体场景（agent 跨 session / 1M context）？
- [ ] 你能不能说出 offload 在线服务下 tokens/$ 反而上升的情况？

## Q20. Multi-modal LLM serving 与纯文本 LLM serving 的 infra 差异是什么？

> 🔴 专家 · 用户传一张图，你要在 LLM 前面跑一段 vision encoder、再把视觉 token 跟文本 token 拼起来——视觉部分的 batching 模式跟 LLM 完全不同。

### 1. 核心结论
Multi-modal LLM（如 GPT-4V、Gemini、Claude 3、Llama 3.2 Vision）serving 与纯文本的差异：1) vision encoder 是 vision tower + projector，独立 sub-model，需要 batching；2) 视觉输入 → 视觉 token（256-2000 token / 图）拼到文本 prompt 前；3) prefill cost 显著上升（视觉 token 等价于文本 prompt 加长）；4) 多张图 / 视频时显存 / 算力压力 spike；5) 视频流式输入是新的 streaming pattern。Audio / Video LLM 在此基础上更复杂。

### 2. 底层原理
Multi-modal pipeline：image bytes → preprocess (resize/normalize) → vision encoder (ViT-L/CLIP/SigLIP) → projector (MLP) → vision tokens → 拼到 text prompt embedding → LLM forward。

Vision encoder 算力：ViT-L 224 输入约 0.3 TFLOPs；高分辨率 / 多图 / 视频帧叠加后能到几十 TFLOPs。比 LLM prefill 一段 prompt 重得多。

视觉 token 数量。Llama 3.2 Vision 一张图 1601 token；GPT-4V 多分辨率切块；Qwen-VL dynamic resolution。这些视觉 token 直接进 LLM 的 attention，占 KV cache 显存。

### 3. 关键机制 / 流程 / 数据结构
第一，vision encoder 与 LLM 是否同卡。同卡简单但负载不均；分卡（vision pool + LLM pool）能 batch 优化。生产里小模型同卡，大规模分卡。

第二，vision encoder 的 continuous batching。Vision 输入大小不一（不同分辨率 / 不同图数），batching 比 LLM 难。常用 padding 到固定 patch 数 + mask。

第三，cross-modal cache。同一张图被多个 query 引用时，vision encoder 输出可以 cache。但是同一张图与不同 text query 组合时 KV cache 不能直接复用（因为 attention 跨模态）。

第四，视频 streaming。每帧 vision encode + 滚动 prompt 更新。SGLang / vLLM 已有 video streaming 支持。

### 4. 工程权衡 / 性能影响
Vision encoder 是新的 GPU 消耗大户。生产里 vision encoder 算力可能占总算力 20-40%（取决于图大小与频率）。

视觉 token 让 prefill 时间显著上升。1601 vision token 等价于 1601 text token 的 prefill，对 TTFT 影响可观。

显存压力。Vision encoder 权重 ~1-3GB，跟 LLM 共显存时压力大。

### 5. 常见追问 / 易错点
第一，能不能直接把 vision encoder 拉到 CPU？慢。Vision encoder 是 compute-bound，CPU 跑 latency 几秒到几十秒。必须 GPU。

第二，多张图怎么 batch？每张独立编码后拼接 vision token。Batch 维度上 padding 或 packed sequence。

第三，视觉 token 怎么参与 attention？跟 text token 一起在 LLM 全 attention 内 mutual attention。不区分模态。

第四，cross-modal 任务的 prefix caching？同一张图相同 prompt 完全可缓存。但只缓存图、不缓存 prompt 实现复杂，主流方案缓存 image+prompt 整体。

### 6. 实践建议
小规模 multi-modal serving：vision encoder + LLM 同卡，简单。

大规模生产：vision pool + LLM pool 分离，vision pool 用算力卡（H100/B200），LLM pool 用带宽卡。

监控：vision encode latency、vision token count distribution、E2E TTFT、image-only encoding throughput、cache hit on images。

面试讲 multi-modal serving 时强调"vision encoder 是独立 sub-system，需要单独 batching 和容量规划"。能讲 video streaming 与 cross-modal caching 是加分。

### 7. 30 秒速答
- Multi-modal 多一段 vision tower + projector，输出几百到几千视觉 token 拼到 LLM prompt 前。
- 视觉 token 走 LLM 全 attention，prefill cost 等价 prompt 加长，显存 + 算力都涨。
- 易踩坑：vision encoder 输入大小不一，batching 比 LLM 难，需 padding patch 或 packed sequence。
- 加分关键词：vision pool + LLM pool 分离、dynamic resolution、cross-modal cache、video streaming。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 multi-modal serving 相对纯文本多了哪几个 sub-system？
- [ ] 你能不能解释 1601 vision token 对 TTFT 的影响相当于 1601 text token？
- [ ] 你能不能举一个 vision pool + LLM pool 分离的具体生产架构？
- [ ] 你能不能说出 cross-modal cache 只缓存图、不缓存 prompt 的实现难点？

## Q21. 对比题：PPO vs GRPO vs DPO 在 infra 形态、数据需求、收敛性上的取舍

> 🧭 综合 · 这三种是后训练实践里最常被拿来对比的，能不能从 infra 视角讲清差异决定了你是把它们当成同类算法还是当成三种工程范式。

### 1. 核心结论
PPO 是经典 RL，需要 actor + critic + reference + reward 四份模型 + rollout 引擎 + RL 训练框架。GRPO 砍掉 critic，省 1/4 模型 + critic 训练，rollout pool 占比扩大。DPO 完全去掉 rollout，把偏好数据直接当 SFT 训：infra 形态变回普通 SFT，但效果上限通常低于 PPO/GRPO。三者的取舍：infra 复杂度 PPO > GRPO >> DPO；效果上限 PPO ≈ GRPO > DPO（在通用对话上）；数据需求 PPO/GRPO 需要 reward signal（model 或 rule），DPO 需要 paired preference (chosen, rejected)；收敛速度 DPO 最快（一次过），PPO/GRPO 需要多轮 rollout。

### 2. 底层原理
PPO loss = clip(π/π_old · A) - β KL(π||π_ref)。Advantage 用 GAE 算，需要 critic V(s)。

GRPO loss = (π/π_old)·A - β KL(π||π_ref)。Advantage 用组内归一化算，无需 critic。

DPO loss = -log σ(β log(π(chosen)/π_ref(chosen)) - β log(π(rejected)/π_ref(rejected)))。直接最大化偏好对的对数似然差。

### 3. 关键机制 / 流程 / 数据结构
第一，infra 组件：
- PPO: rollout (vLLM) + critic train + actor train + RM serving + ref model freeze.
- GRPO: rollout (vLLM) + actor train + RM/verifier + ref model freeze.
- DPO: 普通 SFT 训练框架 + paired data.

第二，数据：
- PPO/GRPO: prompts + reward function (model or rule).
- DPO: chosen/rejected pairs (typically human-labeled or LLM-judged).

第三，调度：
- PPO: 复杂，rollout/train 协调 + critic train.
- GRPO: 中等，rollout/train 协调.
- DPO: 简单，单训练流水线.

### 4. 工程权衡 / 性能影响
PPO 训练 1 个 step 的总算力 = rollout + actor forward/backward + critic forward/backward + RM forward。GRPO 砍 critic，少 1/4。DPO 没 rollout，等于 SFT cost。

收敛速度：DPO 几千步收敛；PPO/GRPO 需要数十万到百万 rollout sample。

效果上限：在通用对话上 PPO ≈ GRPO > DPO 5-10%。在 RLVR 任务上 GRPO 显著强于 DPO（DPO 没法用 verifier reward）。

### 5. 常见追问 / 易错点
第一，能不能 DPO 起步、PPO/GRPO 调优？常见两阶段做法。先 DPO 收敛快、便宜；再 PPO/GRPO 精调上限。

第二，三者都需要 reference model 吗？是。KL 约束 / 概率比都需要 reference。reference 通常是 SFT 模型 freeze。

第三，DPO 也算 RL 吗？严格说不是——它是 off-policy 偏好优化，没有 rollout 没有 trajectory。归类一般叫"reward-free RLHF 替代"。

第四，infra 团队角度怎么选？团队规模小 / 数据是 paired preference：DPO。团队能搭 RL infra / 有 verifier 或 RM：GRPO。需要极致效果：PPO。

### 6. 实践建议
起步：DPO 跑通。验证效果上限：PPO/GRPO 跟进。数学 / 代码：GRPO + RLVR。

infra 团队优先级：先 SFT pipeline → DPO → GRPO → PPO，每步是上一步基础上的增量复杂度。

面试讲三者对比时主线"infra 复杂度递增、效果上限递增、数据形态不同"。能讲 DPO 的本质（off-policy 偏好优化、非 RL）是高分。

### 7. 30 秒速答
- PPO：actor+critic+ref+RM 四份模型 + rollout 引擎；GRPO 砍 critic；DPO 完全去 rollout 等于 SFT。
- 数据：PPO/GRPO 要 reward signal，DPO 要 paired (chosen, rejected) preference。
- 易踩坑：DPO 上限通常低 PPO/GRPO 5–10%；GRPO 在 RLVR 上显著强于 DPO（DPO 用不了 verifier）。
- 加分关键词：infra 复杂度 PPO > GRPO >> DPO、off-policy 偏好优化、DPO→GRPO→PPO 渐进路线。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清三者在 infra 组件数上的递减关系？
- [ ] 你能不能解释 DPO 为什么严格说不是 RL（没有 rollout 没有 trajectory）？
- [ ] 你能不能举一个先 DPO 起步再 PPO/GRPO 精调的两阶段路线？
- [ ] 你能不能说出三者对 reference model 的共同依赖（KL 约束）？

## Q22. 场景题：8 张 H200 + 70B Llama + 32k 上下文 + SLO TTFT<500ms TPOT<50ms，怎么部署？

> 🧭 综合 · 这是 2026 年最常见的"中型推理生产"场景，能不能讲清并行策略 + 引擎选型 + 调度配置 + 监控指标，反映了你能不能把书面知识落到一个具体决策表上。

### 1. 核心结论
推荐部署：vLLM v1，TP=8 单实例占满 8 张 H200（141GB 显存），enable_chunked_prefill=True，enable_prefix_caching=True，max_model_len=32768，FP8 quantization on weights + KV (W8A8KV8)，max_num_batched_tokens=8192，gpu_memory_utilization=0.92。预期 throughput ~1500-2500 tokens/s (depending on batch size)，TTFT p99 ~300-450ms（32k prompt），TPOT p99 ~30-45ms。

### 2. 底层原理
70B Llama FP8 权重 ~70GB，分到 8 张 TP=8 每张 ~9GB。KV cache per token FP8 ~70KB，32k context 单 sample ~2.2GB；batch 32 ≈ 70GB total，分到 8 卡每张 ~9GB。剩余每张 H200 显存 ~120GB 用于激活 + buffer。

TP=8 在 H200 的 NVL8 域内全 NVLink，attention all-reduce 微秒级，不是瓶颈。

Chunked prefill：32k prompt 切成 16 个 2k chunk，与 decode 交错。让 TTFT 不被 prefill 阻塞。

FP8 KV cache：让 batch 能开到 64+。

### 3. 关键机制 / 流程 / 数据结构
第一，引擎选型。vLLM v1：开源、模型覆盖广、性能足够。SGLang：如果有大量 multi-turn agent / RAG，RadixAttention 收益大。TRT-LLM：性能上限稍高但运维复杂。

第二，并行策略。8 张 H200 都是 NVL8，TP=8 最自然。不需要 PP（70B 单 NVL8 内能装）。不需要 EP（dense 模型）。

第三，量化配置。Weights：W8A8 (FP8) - perplexity 几乎无损。KV：FP8 - 显存减半。这两个组合在 H200 上最优。

第四，调度参数。max_num_batched_tokens=8192 平衡 prefill chunk size 与 decode batch；gpu_memory_utilization=0.92 给一些 buffer 避免 OOM；max_num_seqs=64-128 控制并发。

### 4. 工程权衡 / 性能影响
对比单 H200（70B 单卡跑得动吗？跑得动但 TPOT 高），TP=8 让 attention 算力翻 8 倍、KV 容量翻 8 倍，整体 TTFT 与 TPOT 都更稳。

对比 TP=4 + DP=2，TP=4 每实例只能用 4 张卡，整体 batch 上限低 50%；DP=2 让 2 个独立请求并发但 KV 不共享，prefix caching 跨实例不命中。生产里 TP=8 + DP=1（如果只有 8 卡）更优。

对比 FP16 / BF16，FP8 让 tokens/s 涨 30-50% 且 perplexity 几乎无损，是 H200 时代 default。

### 5. 常见追问 / 易错点
第一，max_model_len 设 32768 后所有请求都按 32k 算 KV 吗？不是。max_model_len 是上限，实际 KV 按 sequence 实际长度分配（PagedAttention）。

第二，SLO 500ms TTFT 怎么保证 p99？关键是 chunked prefill + 队列管理。如果队列深度 > 0 时 TTFT 会涨，要监控 queue depth + 限流。

第三，怎么 scale 出去更大流量？水平扩到 N 个 TP=8 实例，前面加 router（按 user / session / prefix 分发）。Prefix sticky routing 在 agent 场景命中率高。

第四，要不要开 speculative decoding？EAGLE-2 在小 batch 下 TPOT 减 30-50%。生产里如果 batch 平均 < 16 收益明显；batch 大时收益小。视流量模式决定。

### 6. 实践建议
部署 checklist：1) 跑 vLLM benchmark suite 验证基线；2) 灰度 1-5% 流量；3) 全量前定 SLO 报警；4) 跟踪 cache hit rate 决定是否加 SGLang。

容量规划：从 1 个 TP=8 实例起步，按 tokens/s 实测决定何时加第二个。

面试讲这种场景题时一定要给出"具体参数 + 数字 + 监控指标"，不能停在概念。能讲 chunked prefill + FP8 KV + prefix caching 三件套的协同就到位。

### 7. 30 秒速答
- 部署方案：vLLM v1 + TP=8 单实例 + W8A8KV8 + chunked prefill + prefix caching + max_model_len=32768。
- 预期 SLO：throughput 1500–2500 tokens/s，TTFT p99 ~300–450ms，TPOT p99 ~30–45ms。
- 易踩坑：max_num_batched_tokens 与 chunk size 不匹配会让 prefill 阻塞 decode；gpu_memory_utilization 不留 buffer 易 OOM。
- 加分关键词：TP=8 单 NVL8 全 NVLink、FP8 KV 让 batch 翻倍、sticky routing prefix 命中、EAGLE-2 小 batch 加速。

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清为什么 8 张 H200 上 TP=8 + DP=1 优于 TP=4 + DP=2？
- [ ] 你能不能解释 chunked prefill 如何保证 TTFT p99 < 500ms？
- [ ] 你能不能举一个 70B + 32k 上 FP8 权重 + FP8 KV 的显存测算（权重 ~70GB / KV 2.2GB·sample）？
- [ ] 你能不能说出 horizontal scaling 时 sticky routing 与 cache 命中率的关系？

## Q23. MLA (Multi-head Latent Attention) 数学与实现深度

> 🔴 专家 · DeepSeek-V3 用 MLA 把 KV cache 压到 GQA 的 1/4-1/8——不是简单减 head 数，是把 KV 投影到低维 latent + 推理时动态 unproject。

### 1. 核心结论
MLA (Multi-head Latent Attention) 是 DeepSeek-V2/V3 引入的 KV cache 压缩技术。核心：把 K/V 通过 low-rank projection 压到 latent 空间 c_kv（d_c 远小于 n_kv_head × head_dim），cache 只存 c_kv。Attention 时 unproject 出实际 K/V：K = W_K_uc · c_kv, V = W_V_uc · c_kv。压缩比 32-64x（vs GQA 4-8x）。代价是 inference 时多两次 unprojection matmul，但 KV cache 显存大幅减少让 batch / context length 大幅扩展。

### 2. 底层原理
MLA forward（推理时）：
```
# Compress (cache 时存这个)
c_kv = X · W_kv_compress    # [batch, seq, d_c], d_c 小
# Uncompress (attention 用)
K = c_kv · W_K_uc            # [batch, seq, n_head × head_dim]
V = c_kv · W_V_uc            # [batch, seq, n_head × head_dim]
# Attention
attn_out = attention(Q, K, V)
```

参数 d_c：DeepSeek-V3 d_c=512 (vs 传统 GQA n_kv_head × head_dim = 8 × 128 = 1024)。KV 大约减半。

实际 cache 压缩比来自：（1）latent 维度小 d_c=512；（2）单 head 不再独立存 K/V（共享 latent + per-head projection）。整体 vs MHA 压缩 ~64x；vs GQA ~8x。

### 3. 关键机制 / 流程 / 数据结构
第一，cache 形态。Cache 存 c_kv [batch, seq, d_c]。Per token 大小 d_c × 2 bytes (FP16) = 1KB。70B model 1024 token cache = 1MB / sample（vs GQA ~140KB / token / sample for similar config）。

第二，attention kernel 改造。需要新 attention kernel 支持 unproject + attention 融合。否则 unproject 输出大 tensor 浪费显存。

第三，跟 RoPE 兼容。RoPE 在 unproject 后 K 上 apply。Decoupled position encoding：position info 不在 c_kv 中，每次 attention 时加。

第四，推理 vs 训练对称。训练时 forward 也是这个 flow，gradient 自然 backprop 经过 unproject。Compatible 路径。

### 4. 工程权衡 / 性能影响
显存收益。KV cache 64x 压缩让 long context inference 显著轻松。10s context 70B 模型 KV cache 从 ~10GB 降到 ~150MB。

Compute 代价。每 attention layer 多两次 matmul（K / V unproject）。约 +5-10% compute。

Long context 优势。Context 128k 模型，MLA 让 KV cache 从 ~150GB 降到 ~2GB。可以单卡跑（GQA 几乎不可能）。

Inference engine 适配。vLLM / SGLang 等支持 MLA 需要专门 attention kernel。早期 release 后逐步加入。

### 5. 常见追问 / 易错点
第一，MLA 是 KV 压缩不是 attention 加速。Compute 反而略多（unproject）。但 KV cache 大幅减少 → batch 大幅扩展 → throughput 上去。

第二，跟 MQA / GQA 对比。MHA: 全 head 各自 K/V；MQA: 全 head 共享 1 组 K/V (压缩 32x)；GQA: 分组共享 (压缩 4-8x)；MLA: latent projection (压缩 64x + 不损 quality)。

第三，质量损失。MLA 在 DeepSeek-V2/V3 实测 quality 跟 MHA 接近，损失 <1%。比 MQA / GQA 的 quality 损失小。

第四，跟 sparse attention 关系。MLA 是 KV cache 压缩，不是 attention sparsity。可以跟 sliding window / sparse pattern 组合。

### 6. 实践建议
新模型架构选 MLA：long context 业务首选。

部署 MLA 模型：选支持 MLA 的 inference engine。vLLM 支持但配置稍复杂。

跟量化协同：MLA latent 量化效果待研究（latent 维度低，量化容忍度可能不同）。

监控：KV cache usage / attention latency / quality metric。

### 7. 30 秒速答
- MLA: K/V 通过 low-rank projection 压到 latent，cache 只存 latent
- 压缩比 64x (vs MHA), 8x (vs GQA)
- 代价: +5-10% compute (unproject)
- 长 context 推理优势大: 128k context KV 从 150GB → 2GB

### 8. 自测 checklist
- [ ] 你能不能讲清 MLA 跟 GQA 在 cache 形态上的差异？
- [ ] 你能不能算 70B MLA 模型 128k context KV 大小？
- [ ] 你能不能识别 MLA 对 attention kernel 的改造需求？
- [ ] 你能不能说出 MLA 跟 sparse attention 的关系？

## Q24. Mamba / SSM 状态空间模型与混合架构

> 🔴 专家 · Transformer attention O(N²) 让 long context 算力爆炸；Mamba 类 SSM 是 O(N) 复杂度但有 representation power 短板——2024-2025 hybrid Mamba+Transformer 架构成为长序列模型的新方向。

### 1. 核心结论
Mamba 是 2023 提出的状态空间模型（State Space Model, SSM）替代 Transformer attention：（1）**Linear complexity O(N)**：sequence 长度线性 vs Transformer O(N²)；（2）**Selective state**：可学的 state transition + 输入依赖更新；（3）**Hardware-aware kernel**：parallel scan 算法在 GPU 上高效。Mamba-2（2024）进一步改进 + tensor parallel。Hybrid 架构（如 Jamba / Falcon-Mamba / Zamba）混合 Transformer attention + Mamba 块，在 long context 上表现好。生产部署仍在 early stage，2025-2026 加速。

### 2. 底层原理
SSM 核心方程：
```
h_t = A·h_{t-1} + B·x_t   # 状态更新
y_t = C·h_t + D·x_t        # 输出
```
A / B / C / D 是参数。Mamba 让 B / C / D 输入依赖（selective SSM）让模型对 input 更敏感。

Linear vs Transformer：
- Transformer attention: O(N²) compute, O(N²) memory (KV cache)
- Mamba: O(N) compute, O(state_size × N) memory

State size: Mamba state size ~16-64 dim per channel，远小于 attention 的 head_dim × n_head。

跟 RNN 区别：RNN sequential dependency 慢；Mamba 用 parallel scan 算法（类似 cumsum）在 GPU 上并行。Selective state 让 Mamba 跟 RNN 在表达力上有差异（input-dependent transition）。

### 3. 关键机制 / 流程 / 数据结构
第一，parallel scan。Mamba forward 用 parallel scan kernel：分块算 local state + log(N) 步合并。GPU 上比朴素 sequential 快 10-100x。

第二，混合架构。Jamba: 每 8 层 = 7 Mamba + 1 Attention + MoE。Falcon-Mamba: 全 Mamba。Zamba: Mamba + Attention 交错。

第三，long context 优势。Mamba 在 1M+ context 上 compute / memory 都线性，Transformer 二次方爆炸。Jamba 支持 256k context.

第四，KV cache 替代。Mamba 没有 KV cache 概念（state 是 fixed size），跟 Transformer 工程模式不同。Inference 时维护 state vector 而非 KV cache。

### 4. 工程权衡 / 性能影响
Long context 优势。Sequence 32k+ 时 Mamba 速度明显快 Transformer。128k+ 时 Mamba 几乎是唯一选择。

短 context 劣势。Sequence < 1k 时 Transformer 略好（attention 表达力 + 优化成熟度）。

模型质量。纯 Mamba 在 NLP benchmarks 上略输纯 Transformer（同参数）。Hybrid 跟纯 Transformer 接近 + long context 优势。

工程成熟度。Mamba kernel / training infra 远不如 Transformer 成熟。生产部署需要 dedicated team。

### 5. 常见追问 / 易错点
第一，是否替代 Transformer。短期不会。Hybrid 是更现实的方向。Long context use case Mamba 优势明显，short context 仍 Transformer。

第二，KV cache 替代。Mamba state size 固定 (~16-64 dim per channel)，不像 Transformer KV cache 随 sequence length 涨。Long context 推理显存优势大。

第三，CUDA Graph 跟 Mamba。Mamba parallel scan kernel 跟 Transformer attention kernel 完全不同，CUDA Graph capture 需要专门 implementation。

第四，量化。Mamba state 量化研究还少。可能比 Transformer KV cache 量化更敏感。

### 6. 实践建议
research / experiment：Jamba / Falcon-Mamba 是好 starting point。

production 部署：long context 业务（1M+）评估 Mamba。其它场景仍 Transformer。

跟踪 trajectory：2024-2025 Mamba 生态快速发展，定期评估。

监控：Mamba state usage / parallel scan latency / quality vs Transformer baseline。

### 7. 30 秒速答
- Mamba SSM: linear O(N) complexity, selective state
- Hybrid (Jamba/Zamba): Mamba + Transformer 混合架构
- Long context (32k+) 优势明显, short context 略输 Transformer
- 工程成熟度跟 Transformer 差 1-2 个量级

### 8. 自测 checklist
- [ ] 你能不能讲清 SSM 跟 attention 的复杂度差异？
- [ ] 你能不能解释 selective state 怎么提升表达力？
- [ ] 你能不能识别 hybrid 架构的设计动机？
- [ ] 你能不能预测 Mamba 在 production 部署的 trajectory？

## Q25. 长上下文 1M+ 训练与推理实战

> 🔴 专家 · GPT-4 / Claude / Gemini 都支持 1M+ context；DeepSeek / Qwen 等开源跟进——长上下文不是简单加 context length，是训练 + 推理 + KV cache 全栈重新设计。

### 1. 核心结论
1M+ context 长上下文 LLM 实战：（1）**架构选择**：MLA / GQA 必备（KV 压缩）+ optional Mamba hybrid（更长 context）；（2）**Position encoding**：RoPE base 调大 + YaRN / NTK 扩展；（3）**训练**：Context Parallelism (CP) + Ring Attention 切 sequence 维；（4）**推理**：KV cache 量化（FP8 / INT4）+ paged + offload；（5）**Evaluation**：NeedleInHaystack / RULER 等长 context benchmark。代表模型：Claude 3.5 200k；Gemini 1.5 Pro 2M；DeepSeek-V3 128k；Qwen2.5-1M（千问 1M context）。每个加倍 context 训练 + 推理成本至少 2-4x。

### 2. 底层原理
Context length 扩展瓶颈：
- Attention O(N²) 在 1M 上 = 1T FLOPs / layer，端到端几百 TFLOPs
- KV cache 70B GQA 1M context FP16 = ~140GB（远超单卡 80GB）
- Position encoding 训练时只到 32k，1M 是 OOD

YaRN / NTK extension：让 RoPE position encoding 外推到训练之外的长度。Math 上是调 RoPE base + 频率 scaling。模型不重训仍能用更长 context 但质量略损。

Context Parallelism (CP)：训练时 sequence 切到多卡（CP=4 / 8），每卡处理一段。Attention 需要 Ring Attention 跨卡通信 K/V。

推理 KV cache 量化：FP16 → FP8 减半，INT4 减 4x。Long context 推理 KV cache 量化几乎必备。

### 3. 关键机制 / 流程 / 数据结构
第一，训练长 context。Stage 1: 32k pretrain (主体训练)。Stage 2: 128k stretch (small additional steps)。Stage 3: 1M continue-pretrain (用 long doc/code 数据)。

第二，Ring Attention。CP 卡间环形传递 K/V。每卡同时持有 part of K/V，通过环传递累积 attention output。

第三，position encoding scaling。YaRN: 调 RoPE base from 10000 to e.g. 500000。NTK-aware scaling: 按位 frequency 调。Qwen 用 YaRN，DeepSeek 用类似方案。

第四，推理 long context。FP8 KV cache 必备。Paged attention 跨多 GPU。Offload cold KV to CPU（Long context KV cache 跨 GPU+CPU 分层）。

### 4. 工程权衡 / 性能影响
训练成本。32k → 128k 训练成本 ~3-5x；128k → 1M ~10x。总成本 1M context 模型比 32k baseline 训练成本 30-50x。

推理成本。1M context prefill 一次 ~分钟级（70B 模型）。Decode 每 token ~100-200ms (KV cache 大)。

Quality。Long context 模型在 short context 上质量略下降（trade-off）。但 long context capability 是新能力。

Use case。Long context 适合 doc analysis / code review / agent。常规 chat 不需要。

### 5. 常见追问 / 易错点
第一，"支持 1M context" 不等于"1M context 上质量好"。多数模型在 32k-128k 质量好，1M context 上 NeedleInHaystack 类 benchmark 性能下降明显。

第二，YaRN extension 跟重训。YaRN 是 inference 时 trick，不需重训。但效果 limited。真正训过 1M context 才有 best quality。

第三，KV cache offload 成本。Long context KV cache 几十 GB，offload 到 CPU 让 decode latency +几十 ms / token。Trade-off。

第四，跟 RAG 关系。Long context 让 RAG 部分需求消失（直接塞文档），但 RAG 更经济 (短 context)。两者并存。

### 6. 实践建议
1M context 部署：MLA / GQA 模型 + FP8 KV + CP / TP。需要 8+ GPU fleet。

短-长 context 混合 fleet：1M context fleet 处理 long context request / 32k fleet 处理普通。

quality 评估：NeedleInHaystack + RULER + 业务 specific benchmark。Don't trust marketing claims。

监控：context length 分布 / KV cache 使用率 / long context quality drift。

### 7. 30 秒速答
- 1M context = MLA/GQA + YaRN/NTK + CP/Ring Attention + FP8 KV
- 训练 stage: 32k → 128k → 1M 渐进
- 推理 prefill 分钟级 / decode 100-200ms/token
- "支持 1M" 不等于"1M 质量好"，需 NeedleInHaystack 评估

### 8. 自测 checklist
- [ ] 你能不能讲清 1M context 训练的三个 stage？
- [ ] 你能不能算 70B 1M context KV cache 大小？
- [ ] 你能不能解释 YaRN 跟重训的 trade-off？
- [ ] 你能不能识别 long context 模型质量评估的关键？

## Q26. RLVR (Reinforcement Learning with Verifiable Rewards) 工程实战

> 🔴 专家 · DeepSeek-R1 / OpenAI o1 / Google Gemini 2.0 都用 RLVR——用规则可验证 reward 替代 reward model，infra 上 reward model pool 消失、verifier sandbox 出现。

### 1. 核心结论
RLVR 范式：用规则可验证奖励（代码跑通 / 数学答对 / 形式验证 pass）替代 reward model。代表系统：DeepSeek-R1 (math/code RLVR)、OpenAI o1 (reasoning RLVR)、Google Gemini 2.0 等。Infra 影响：（1）reward model pool 消失（不再需要训 / serve 大 reward model）；（2）verifier sandbox 出现（隔离环境跑代码 / 验证数学）；（3）rollout pool 更大（生成 long CoT 需要长 sequence）；（4）训练 fleet 配置变化：actor + verifier，比 RLHF 简化。适合场景：数学 / 代码 / 形式推理。不适合：开放对话 / 创意写作（没有客观 reward）。

### 2. 底层原理
RLHF vs RLVR：
- RLHF: actor + critic + reward model + reference model。Reward model 给 actor output 打分。
- RLVR: actor + verifier (规则) + reference model。Verifier 是程序 / API 而非 model。

Verifier 类型：
- Code execution: 跑 actor 生成的代码，看 test 是否过
- Math verification: 检查 actor 答案是否等于参考答案
- Formal verification: Lean / Coq 等定理证明
- API call: 跟外部 API 交互，验证结果

Reward = 0/1 (pass/fail) or partial credit (some tests pass)。

RL 算法：PPO / GRPO / DPO 等，跟 RLHF 类似。GRPO（DeepSeek 推出）不需要 critic，简化训练 infra。

### 3. 关键机制 / 流程 / 数据结构
第一，verifier sandbox。隔离容器（Docker / firejail / gVisor）跑生成代码，防止 actor 输出恶意代码影响系统。Sandbox 资源限制（CPU / memory / network）。

第二，rollout / verify cycle。Actor 生成 N 个 candidate → Verifier 跑 N 个 reward → 用 reward 算 advantage → 更新 actor。Loop 几千-几万次。

第三，long CoT。RLVR 训练 actor 生成长 reasoning chain（几千 token）。Rollout 时序列长，prefill / decode 慢。需要 long context inference 优化。

第四，跟 RLHF 训练 infra 区别。RLHF: 4 大 fleet（actor / reward / critic / ref）。RLVR: 2 大 fleet（actor / ref）+ verifier sandbox cluster。Verifier sandbox 是 CPU-heavy 不是 GPU。

### 4. 工程权衡 / 性能影响
训练成本。Verifier cluster 比 reward model fleet 便宜（CPU vs GPU）。Total RLVR 训练成本 ~RLHF 60-70%。

适用领域。RLVR 限于可验证 task：math / code / 形式推理 / 工具调用结果验证。Open-ended 任务（写诗 / 对话质量）仍需 reward model。

数据 cost。RLVR 不需要人类 preference 数据（替代为可验证 task）。Data acquisition 成本低。Bootstrap 容易（用现有 math / code dataset）。

模型能力。RLVR 训练让模型在 reasoning task 上飞跃（DeepSeek-R1 数学 / 代码能力大幅提升）。但对开放任务无帮助。

### 5. 常见追问 / 易错点
第一，verifier security。Actor 可能生成恶意代码尝试 escape sandbox。Strict sandbox + time limit + network isolation 必备。

第二，verifier scalability。Code execution 慢（几 ms-几秒），训练 throughput 受限。需要 verifier sandbox cluster（千 vCPU 量级）。

第三，reward shaping。0/1 reward 信号稀疏。Partial credit / step-wise reward 让训练更稳定。

第四，跟 RLHF 互补。多数生产 model 用 RLHF + RLVR 组合：RLHF 对齐通用任务，RLVR 提升 reasoning 能力。

### 6. 实践建议
RLVR 业务起步：数学 / 代码 task 用开源 framework（OpenRLHF / verl 等）+ verifier sandbox。

verifier 投入：build verifier cluster 比训练 fleet 更大。CPU heavy + sandbox security。

跟 RLHF 协同：先 SFT → RLHF 通用对齐 → RLVR specific reasoning。

监控：verifier throughput / sandbox security incident / training reward 曲线。

### 7. 30 秒速答
- RLVR: 用规则 verifier 替代 reward model
- 代表: DeepSeek-R1 / o1 / Gemini 2.0
- 适合 math / code / 形式推理；不适合开放对话
- Infra: actor + ref + verifier sandbox (CPU)，比 RLHF 简化 30-40%

### 8. 自测 checklist
- [ ] 你能不能讲清 RLVR 跟 RLHF 的 infra 区别？
- [ ] 你能不能识别 verifier sandbox 的 security 挑战？
- [ ] 你能不能说出 RLVR 适用 vs 不适用领域？
- [ ] 你能不能设计 verifier cluster 的容量规划？

## Q27. Speculative Decoding 在 2026 的演进：EAGLE-3 / Lookahead 2.0

> 🟡 进阶 · Speculative decoding 2024 主流是 EAGLE-2 / Medusa；2026 EAGLE-3 / Lookahead 2.0 等新方案让接受率再提升 + 适用场景拓展——投机解码已是工业 LLM serving 标配。

### 1. 核心结论
2026 投机解码新方案：（1）**EAGLE-3**：在 EAGLE-2 基础上加 dynamic adaptive tree（按 prompt 难度调整 tree size）；（2）**Lookahead 2.0**：n-gram + transformer 混合 draft；（3）**Suffix Decoding**：multi-tier draft tree；（4）**SpecInfer 优化**：跟 PagedAttention 紧密集成。在工业 LLM serving 上的端到端加速：1-batch 2-3x / 32-batch 1.3-1.5x。vLLM v1 / SGLang 0.4+ / TRT-LLM 都集成主流方案。生产部署：在线 chat 服务标配 EAGLE-2/3；agent 高频小请求收益更大。

### 2. 底层原理
EAGLE-2 vs EAGLE-3。EAGLE-2: 固定 tree shape (如 depth=5 width=10)。EAGLE-3: 动态 tree—简单 prompt 用窄树（节省算力），复杂 prompt 用宽树（增加 accept rate）。Tree 选择基于 prompt embedding / draft confidence。

Lookahead 2.0：早期 Lookahead 用 n-gram cache 提议 token，无 draft model。Lookahead 2.0 加 small draft model 跟 n-gram 混合，accept rate 提升 + 仍不需独立 draft model。

SpecInfer 优化：把 spec decode tree 跟 PagedAttention KV cache 紧密集成，减少 memory 拷贝 + 提升 verification efficiency。

通用 trend：accept rate 从 60-70%（EAGLE-2）→ 75-85%（EAGLE-3）。

### 3. 关键机制 / 流程 / 数据结构
第一，dynamic tree。EAGLE-3 用 lightweight predictor 决定 tree shape：input embedding → predictor → (depth, width) 配置。每 request 自适应。

第二，tree decoding kernel。Target model 一次 forward 验证整 tree（不是序列）。需要 tree mask attention kernel（FlashAttention 类支持）。

第三，accept rate measurement。Per token accept probability × tree depth = expected accept length。EAGLE-3 expected accept length 4-5 tokens / forward (vs EAGLE-2 3-4)。

第四，vLLM v1 集成。`--speculative_config "method: eagle3, ..."`。一行配置启用。

### 4. 工程权衡 / 性能影响
Throughput vs latency。Spec decode 主要降 latency (1-batch 2-3x)，对 throughput 帮助小（batch 大时 target forward 已 saturate）。

Memory cost。Draft model + tree state。EAGLE-2/3 draft ~100MB，tree state 几 MB。可接受。

Accept rate stability。某些 prompt 类型 accept rate 低（如复杂 reasoning prompt）。Production 监控 accept rate by prompt type。

Use case。在线 chat 服务 batch 多 1-4，spec decode 收益最大。Batch 32+ 收益降。

### 5. 常见追问 / 易错点
第一，accept rate 跟模型版本耦合。Draft model 跟 target model 配对训练。换 target 需要重训 draft。

第二，跟量化协同。Target + draft 都量化（FP8 / INT4），accept rate 略下降但仍 net win。

第3，复杂 prompt 收益少。Reasoning / code prompt accept rate 低（target 输出 diverse）。生产里按 prompt 类型 routing。

第四，CUDA Graph 集成。Spec decode 让每 forward shape 变（tree 验证 size 变），CUDA Graph capture 复杂。需要 capture 多个 tree size variants。

### 6. 实践建议
新业务起步：vLLM v1 + EAGLE-2 default，可观察 1.5-2x 加速。

进阶：EAGLE-3 + dynamic tree，accept rate 再提 10-20%。

跟量化组合：FP8 quant + EAGLE-3 = 总加速 3-5x（latency）。

监控：accept rate / 平均 accept length / 跟 baseline 速度对比。

### 7. 30 秒速答
- EAGLE-3: dynamic adaptive tree，按 prompt 难度调整
- Accept rate 2026 主流 75-85% (vs 2024 60-70%)
- 1-batch 2-3x 加速；32-batch 1.3-1.5x
- vLLM v1 一行配置启用，agent / chat 高频场景收益大

### 8. 自测 checklist
- [ ] 你能不能讲清 EAGLE-3 跟 EAGLE-2 的差异？
- [ ] 你能不能算 accept length 4 token / forward 的 throughput 收益？
- [ ] 你能不能识别 spec decode batch 大小的影响？
- [ ] 你能不能设计 spec decode 跟 quantization 的组合？

## Q28. KV Cache 量化算法 2026 演进（KIVI / QServe / SnapKV）

> 🔴 专家 · 2024 KV cache 量化是 FP8/INT8 朴素；2025-2026 KIVI / QServe / SnapKV 等让 KV cache INT2 / 自适应压缩 / sparse retention 上线——长上下文推理的关键 enabler。

### 1. 核心结论
KV cache 量化 2025-2026 主流方案：（1）**KIVI (Per-channel + Per-token quant for K, Per-token for V)**：K 用 per-channel + per-token 双维度 scale，V 用 per-token。INT2 量化质量接近 FP16。（2）**QServe**：4-bit weight + 8-bit activation + 4-bit KV 三合一极致量化。（3）**SnapKV**：not 量化而是 sparse retention —— 只保留最重要的 KV positions。（4）**Mixed precision KV**：K 用 FP8，V 用 INT4 非对称（K 更敏感）。生产部署：Hopper+ FP8 主流；Ampere INT8 主流；2026 INT4/INT2 进入主流。

### 2. 底层原理
KIVI insight：K 跟 V 量化敏感度不同（K 敏感 V 不敏感），且 K 不同 channel 量化敏感度不同。Per-channel × per-token grouping 平衡精度 + 压缩比。

QServe：完全量化（weight + activation + KV）。提供专门 W4A8KV4 kernel，端到端 4-bit 推理。质量损失 1-2%，速度 / 显存 1-2 个量级提升。

SnapKV：观察 attention pattern 通常稀疏（少数 token 主导）。保留高 attention score 的 KV，丢弃 low-score。Memory 大幅减少但质量保持。

Mixed precision：K=FP8, V=INT4 是常见 production 组合。K (进 softmax) 精度更高，V (线性 weighted sum) 精度可以低。

### 3. 关键机制 / 流程 / 数据结构
第一，per-channel scale 计算。每 layer K 的 d 维度每个 channel 一个 scale。Offline calibrate 或 online dynamic 算。

第二，量化 kernel。INT2/INT4 KV 量化 attention kernel 需要：dequant on-the-fly + softmax / matmul。FlashAttention 类 kernel 改造支持。

第3，SnapKV retention strategy。每 N step 评估 attention scores → top-K positions kept → 其余丢弃。Memory 大幅减少 (~70%)。

第四，vLLM / SGLang 集成。vLLM `--kv-cache-dtype fp8/int8` 主流支持。INT4/INT2 仍 experimental。SnapKV community plugin。

### 4. 工程权衡 / 性能影响
显存收益（70B 模型 128k context KV cache）：
- FP16: 140GB
- FP8: 70GB
- INT8: 35GB
- INT4: 18GB
- INT2 (KIVI): 9GB
- SnapKV (30% retention): 42GB

Throughput 收益。KV cache 减半 → batch 翻倍 → throughput 翻倍。Long context 推理 INT4/INT2 KV 几乎必备。

精度损失。FP8: ~0% 损失；INT8: <0.5%；INT4: 1-2%；INT2 (KIVI): 1-3%。Use case 决定容忍度。

### 5. 常见追问 / 易错点
第一，KV cache 量化跟 weight 量化独立。可以 weight FP16 + KV INT4，或 weight INT4 + KV FP8。Mixed combinations。

第二，calibration set。KV 量化（特别 SmoothQuant 类）需要 calibration set。用业务真实 prompts 比 generic dataset 效果好。

第三，跟 long context 协同。Long context inference 必备 KV 量化。Short context 收益小。

第四，跟 MLA 协同。MLA 已经把 KV 压缩到 latent。MLA + INT4 latent 量化 = 进一步压缩。

### 6. 实践建议
Hopper+ inference：FP8 KV default，long context 时上 INT4。

Ampere inference：INT8 KV default。

Long context (32k+) 业务：必备 INT4/INT8 KV + paged attention + offload。

监控：KV cache 量化精度（业务 metric vs FP16 baseline）+ memory usage。

### 7. 30 秒速答
- KIVI: K per-channel × per-token, V per-token, INT2 友好
- QServe: W4A8KV4 端到端 4-bit
- SnapKV: sparse retention 替代量化
- Mixed precision K=FP8 / V=INT4 production 主流

### 8. 自测 checklist
- [ ] 你能不能讲清 K 跟 V 量化敏感度差异？
- [ ] 你能不能算 70B 128k context 各种 KV 量化的显存？
- [ ] 你能不能识别 SnapKV 跟传统量化的区别？
- [ ] 你能不能设计长 context 推理的 KV 量化方案？
