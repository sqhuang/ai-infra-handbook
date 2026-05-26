# 事故复盘与故障教训

## 主题边界

本卷收录 25 个典型化、综合化的工程故障 case，每题用 6 节模板讲清楚现象、根因、定位、修复、教训、预防。涵盖 OOM、通信、存储/checkpoint、推理服务、平台/编排五类。所有 case 为综合化模型化情境，不指向任何具体公司、集群或时间点的真实事故。

本卷 6 节模板的语义与其它卷略有不同：
- §1 现象：故障表现、报警信号、用户感知
- §2 根因层级原理：触发故障的底层技术因果
- §3 定位手段与命令：使用的工具、命令、metric
- §4 修复路径与权衡：短期止血 vs 长期解决
- §5 同类变体 / 易错点：兄弟故障与面试追问
- §6 预防机制：监控、告警、流程、自动化

---

## Q1. 训练中途突发 CUDA OOM，但显存监控显示余量充足：梯度累计显存洪峰复盘

> 🔴 专家 · 现象往往是"训练跑了 3 个 step 才崩，跟显存监控曲线对不上"——这类 OOM 的关键不是稳态显存，而是反向那一瞬间的峰值。

### 1. 现象

某次大规模训练任务在第 3 个 epoch 中段突发 `CUDA out of memory` 抛错，单 rank 报 `Tried to allocate 2.4 GiB`，但 Prometheus 上 `nvidia_smi_memory_used` 在故障前 5 分钟稳定在 60 GB / 80 GB，留有 20 GB 余量。值班同学第一反应是"显存够啊为啥 OOM"。重启后 1–2 个 epoch 内必复现，定位到与 `gradient_accumulation_steps` 调大有关。

### 2. 根因层级原理

PyTorch 反向传播时，每一次 `loss.backward()` 都会把当前 micro-batch 的梯度累加到 `param.grad` 上。如果模型用了 `bf16` 主权重 + `fp32` 梯度（混合精度的常见配置），且没有显式指定 `set_to_none=True`，每个参数都常驻一份 `fp32` grad tensor。打开 grad accumulation 后，**反向阶段的瞬时峰值** = 已累计的 grad（常驻）+ 本 step 临时激活 + 本 step 临时反向中间张量。`nvidia-smi` 看到的"60 GB"是上一 step 结束后的稳态，**不是反向中那一瞬的峰值**。当 batch_size 或 seq_len 变化导致激活变大时，瞬时峰值就会越界。

### 3. 定位手段与命令

- 启用 PyTorch memory snapshot：训练 step 包一层 `torch.cuda.memory._record_memory_history(max_entries=100000)`，OOM 后 dump 到 `.pickle` 用 `pytorch.org/memory_viz` 离线看时间轴
- `torch.cuda.max_memory_allocated()` 比 `memory_allocated()` 更能反映峰值
- 打印 `torch.cuda.memory_summary(device, abbreviated=False)` 可看到 reserved / active / allocated 三层差距
- DCGM `DCGM_FI_DEV_FB_USED` 采样间隔降到 1s 看尖峰

### 4. 修复路径与权衡

- **短期止血**：减小 micro-batch 或 seq_len；激活 checkpointing（`torch.utils.checkpoint`）牺牲 ~30% 反向时间换 50%+ 激活显存
- **中期**：grad accumulation 步内手动 `optimizer.zero_grad(set_to_none=True)` 改为更细粒度的分桶；切到 `bf16` 全栈（grad 也 bf16，省一半）
- **长期**：上 FSDP2 或 ZeRO-2，把 grad 切到多 rank
- 权衡：激活 checkpointing 实现成本最低但拖训练吞吐；FSDP2 收益大但迁移成本高，且小集群（< 8 卡）反而因通信开销变慢

### 5. 同类变体 / 易错点

- 变体：`torch.compile` 编译后激活模式变化导致显存峰值漂移，重编译触发 OOM
- 变体：`accelerate` / `deepspeed` 配置里 `zero_stage` 与 `gradient_accumulation_steps` 交互复杂，误配会放大峰值
- 易错：只看 `nvidia-smi` 不看 PyTorch allocator 内部，被"reserved 但未 active"的显存误导
- 追问："为什么 reserved 比 allocated 大？"→ allocator 缓存碎片；用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解

### 6. 预防机制

- 训练脚本启动时打印一次预估峰值（`max_memory_allocated` after warmup step）写入 metric
- Pre-flight check：CI 跑一次 1-step dry run 在小卡上验证显存模型
- 在 launcher 里强制把 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` 作为基线
- Code review 守门：任何调大 grad_accum / seq_len / batch_size 的 PR 必须附 dry run 显存截图
- 报警：DCGM FB 利用率 > 92% 持续 1 min 报 warning，> 96% 报 critical

### 7. 30 秒速答

- 一句话核心：grad accumulation 期间反向那一瞬的显存峰值远高于稳态，nvidia-smi 看不到。
- 关键机制：常驻 fp32 grad + 临时激活 + 反向中间张量同时存在，micro-batch 或 seq_len 一变就越界。
- 修复 / 防御：FSDP2 切 grad + 激活 checkpointing + `expandable_segments` + dry-run 显存预估。
- 面试加分关键词：set_to_none、max_memory_allocated、memory snapshot、ZeRO-2 grad partition。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"显存稳态够但反向瞬时不够"这一核心症状？
- [ ] 你能不能解释为什么 grad accumulation 会放大反向峰值而非稳态？
- [ ] 你能不能举一个类似场景（例如 `torch.compile` 重编译导致激活模式漂移）？
- [ ] 你能不能说出至少 3 种防御手段（FSDP2、激活 checkpoint、expandable_segments、dry-run）？

---

## Q2. 长序列训练激活显存爆炸：Selective activation checkpointing 失效复盘

> 🔴 专家 · 把 seq_len 从 8K 拉到 32K，开了 `gradient_checkpointing` 还是炸——典型的"checkpoint 切的粒度在层级，但峰值在层内"的盲区，attention softmax 中间矩阵 O(L²) 就能把卡撑爆。

### 1. 现象

某次把训练序列长度从 8K 升到 32K 后，单 rank 反向阶段稳定 OOM。用户已经开了 `gradient_checkpointing=True`，"按理说激活只保留每层入口"。监控显示前向阶段显存平稳，反向第一个 step 内瞬时从 40 GB 飙到 79 GB。

### 2. 根因层级原理

`gradient_checkpointing` 默认是 **per-layer full recompute**：反向走到某层时，重新跑一遍该层前向以重建中间激活，再算梯度。但 attention 层内部如果用了 FlashAttention 或自定义算子，"该层"已经做了内部融合，激活峰值不在层入口而在 attention softmax 中间张量；同时 32K 序列的 attention 中间张量是 O(L²) 量级的，单层重算时 softmax logits 矩阵就能吃掉 20+ GB。换句话说：checkpoint 切的粒度在层级，但显存峰值在层内。

### 3. 定位手段与命令

- `torch.cuda.memory._record_memory_history()` 采到反向时间轴，看到峰值发生在某层 backward 内部而非层入口
- 直接读 `torch.profiler` 的 memory timeline，标注每个 op 的 alloc/free
- 对单层做"isolate"：写一个最小复现，仅跑一层前向 + 反向，看激活高度
- `nvitop` 实时观察反向时显存增长曲线斜率

### 4. 修复路径与权衡

- **短期**：换用 FlashAttention-2/3 或 PyTorch SDPA 的 `memory_efficient` backend，把 attention 内部峰值从 O(L²) 压到 O(L)
- **中期**：开 selective activation checkpointing，只对 MLP 做 checkpoint，attention 走 SDPA（attention 已是 memory-efficient 不需重算）
- **长期**：序列并行（Ulysses / Ring Attention），把 L 维切到多卡
- 权衡：FlashAttention 需要 SM ≥ 8.0；selective checkpoint 实现复杂度上去；序列并行需要改 model code 且只在 L 极大时才划算

### 5. 同类变体 / 易错点

- 变体：`use_reentrant=True`（PyTorch checkpoint 旧接口）有梯度图引用泄漏，长跑后激活慢慢涨；切到 `use_reentrant=False`
- 易错：以为开了 checkpoint 就万事大吉，没看实际单层峰值
- 易错：FlashAttention 包装的 mask 模式不对，回退到 dense path，悄悄退化
- 追问："序列并行和张量并行有什么区别？"→ 一个切 L 一个切 hidden

### 6. 预防机制

- 任何序列长度变更需先在小模型上跑一次 memory snapshot
- CI 报告"激活/参数比"金字塔指标，超阈值挂红
- 默认 attention backend 通过环境变量统一管控（`PYTORCH_SDPA_PRIORITY=flash,mem_efficient,math`），禁止业务代码直接 hardcode
- 观察 metric：反向 step 时长 / 前向 step 时长比例，比例突增提示 recompute 失控

### 7. 30 秒速答

- 一句话核心：层级 checkpointing 切不到层内 attention softmax 中间张量，O(L²) 矩阵把卡撑爆。
- 关键机制：reactive recompute 在层入口但峰值在层内；32K 序列下 softmax logits 单层吃 20+ GB。
- 修复 / 防御：FlashAttention / SDPA mem_efficient + selective checkpoint + 序列并行。
- 面试加分关键词：use_reentrant、SDPA backend、Ulysses、Ring Attention、O(L) vs O(L²)。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"开了 gradient checkpointing 仍然 OOM"这一症状？
- [ ] 你能不能解释 attention 内部 O(L²) 中间张量为何打穿 per-layer checkpoint？
- [ ] 你能不能举一个类似场景（如 long-context 微调或 RAG 拼超长上下文）？
- [ ] 你能不能说出 selective activation checkpointing 与 FlashAttention 各自的边界？

---

## Q3. Adam 优化器状态显存占用翻倍：从 fp32 master 到 8-bit 优化器迁移踩坑

> 🔴 专家 · 算账只算了模型权重就敢扩到 13B，结果连 step 1 都没跑起来——AdamW 自带 m/v/master 三份 fp32，每参数 18 bytes 这条公式不能忘。

### 1. 现象

某次扩模型从 7B 到 13B 后，加载 checkpoint 阶段就 OOM，连第一个 step 都没跑起来。`nvidia-smi` 显示加载时显存稳步爬到 80 GB 满载，进程退出。用户百思不解：13B × 2 bytes（bf16）= 26 GB，怎么会装不下 80 GB 的 H100？

### 2. 根因层级原理

Adam/AdamW 维护 first moment（m）+ second moment（v）+ master copy（fp32），每个参数 3 × 4 = 12 bytes。再叠加模型本身 bf16 权重（2 bytes）+ fp32 grad（4 bytes）= 18 bytes/param 总占用。13B × 18 ≈ 234 GB——单卡放不下，必须分布式。如果 Adam state 没切（DDP 模式或 ZeRO-1 未启用），单卡就会爆。即便切了，优化器状态加载阶段是先 `load_state_dict` 到主 rank，再 broadcast，主 rank 瞬时显存翻倍。

### 3. 定位手段与命令

- `torch.cuda.memory_summary()` 在 load 前后各打一次，对比 allocated 增量
- 启动加 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 看 ZeRO 切片实际形状
- 直接打印 `optimizer.state_dict()` 第一项 tensor 的 `numel * dtype.itemsize`，乘上参数数估算总量
- DCGM 采样观察 OOM 那一刻是哪个 rank 先爆

### 4. 修复路径与权衡

- **短期**：换 `bitsandbytes` 8-bit Adam（`bnb.optim.AdamW8bit`），状态从 8 bytes 压到 2 bytes/参数
- **中期**：切 ZeRO-2 或 FSDP2 的 `FULL_SHARD`，optimizer state 跨 rank 切分
- **长期**：CPU offload optimizer state（DeepSpeed ZeRO-Offload / FSDP2 `cpu_offload`）；显存换内存 + PCIe 带宽
- 权衡：8-bit Adam 在小学习率长训练下精度几乎无损但需要分块量化超参；CPU offload 拖每 step 时间 1.5–3×；ZeRO-2 通信量翻倍

### 5. 同类变体 / 易错点

- 变体：Lion / Adafactor 等优化器状态更小，但收敛性需要单独验证
- 变体：从单卡迁多卡时 optimizer state 切分逻辑没改，多卡 broadcast 仍按全量
- 易错：把"模型 fits 显存"误等同于"训练 fits 显存"，忽略 optimizer 6× 系数
- 追问："Adafactor 怎么省显存？"→ 用 row/column 因子化代替完整二阶矩

### 6. 预防机制

- Pre-flight 显存预算公式写到 launcher：`mem_per_param = sizeof(weight) + sizeof(grad) + optim_overhead`，启动前打印
- 默认 ZeRO-2 或 FSDP2 起步，DDP 仅小模型保留
- 优化器选型 PR 模板：附显存对比 + 收敛曲线对比
- 监控：`torch.cuda.memory_allocated()` 在 first step 后采集存档作为基线

### 7. 30 秒速答

- 一句话核心：AdamW 自带 m/v/master 三份 fp32，每参数 18 bytes，模型权重只是冰山一角。
- 关键机制：13B × 18 ≈ 234 GB，单卡放不下；rank 0 加载阶段还会瞬时翻倍。
- 修复 / 防御：bnb 8-bit Adam / ZeRO-2 / FSDP2 FULL_SHARD / CPU offload 多档可选。
- 面试加分关键词：bitsandbytes、Adafactor、ZeRO stages、optimizer state sharding。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"显存账只算了权重 OOM 在加载阶段"的核心症状？
- [ ] 你能不能解释每参数 18 bytes 这条公式的构成？
- [ ] 你能不能举一个类似场景（如从 7B 扩到 70B 没切 optimizer state）？
- [ ] 你能不能说出 8-bit Adam、ZeRO-2、CPU offload 三档的权衡？

---

## Q4. 推理服务 KV cache OOM：长 prompt 突发流量打爆显存

> 🔴 专家 · 工作日下午一批 8K 长 prompt 涌进来，QPS 才涨 1.3× 但 p99 直接从 800ms 飙到 30s+——这种"长 prompt 雪崩 + 反复 preempt"的链条，是 vLLM 部署绕不开的坑。

### 1. 现象

某线上推理服务（vLLM 部署 70B 模型）在工作日下午突然进入 5xx 高发期，p99 时延从 800ms 飙到 30s+。日志大量 `KV cache exhausted, request preempted` + `OOM, killing oldest request`。监控显示 `vllm:gpu_cache_usage_perc` 飙到 100% 持续 10+ 分钟。流量层 QPS 仅上涨 1.3×，与故障级联不成正比。

### 2. 根因层级原理

vLLM 的 PagedAttention 把 KV 缓存切成固定大小 block（默认 16 token/block），全卡 KV 缓存容量在启动时根据 `gpu_memory_utilization` 静态分配。当请求 prompt 长度分布从均值 1K 突变到一批 8K 长 prompt，单请求占用 block 数 8×，而 `max_num_seqs` 没变，总 block 池被瞬间耗尽。新请求进 waiting queue，老请求被 preempt（KV 被丢弃后续重算），形成"重算 → 占用 → preempt → 再重算"的雪崩，整体吞吐崩溃。

### 3. 定位手段与命令

- vLLM Prometheus metrics：`vllm:gpu_cache_usage_perc`、`vllm:num_preempted_total`、`vllm:waiting_lora_adapters` 同时画一张图
- 取访问日志统计 prompt 长度分布的 p50/p95/p99，对比故障前后
- `nvidia-smi` 看 GPU memory 稳定（KV 缓存是预分配的，OOM 不会反映在 nvidia-smi）
- 打开 vLLM `--disable-log-requests=False`，看具体哪些请求被 preempt

### 4. 修复路径与权衡

- **短期止血**：调大 `gpu_memory_utilization` 从 0.9 到 0.95；调小 `max_num_seqs` 控制并发；网关层加 prompt 长度限流
- **中期**：开 `enable_prefix_caching` 利用前缀复用减少 KV 重算；按 prompt 长度分流到不同实例（短 prompt 走高并发实例，长 prompt 走低并发）
- **长期**：上 vLLM V1 + 分块 prefill（chunked prefill），缓解长 prompt 占用；KV 缓存 offload 到 CPU/远端（牺牲 TPOT 换容量）
- 权衡：调大 `gpu_memory_utilization` 留给 activation 的余量变小，可能触发 activation OOM；分流方案需要网关侧 routing 改动

### 5. 同类变体 / 易错点

- 变体：MQA/GQA 模型 KV 占用本身已经压缩，长 prompt 下相对收益更明显但绝对量仍大
- 变体：推测解码（speculative decoding）开了之后小模型 KV 也占池
- 易错：以为 `gpu_memory_utilization=0.9` 是动态限制——实际是启动时一次性分配
- 追问："PagedAttention 和传统 KV 缓存区别？"→ 物理 block + 逻辑指针，避免内部碎片

### 6. 预防机制

- 监控：`kv_cache_usage` p95 设双阈值（80% warning / 95% critical）
- 灰度发布前压测覆盖长 prompt 场景（造一批 4K/8K/16K prompt 压测）
- 网关侧硬编码 `max_prompt_tokens` 上限，超长直接 4xx 拒绝而非进入服务
- 容量规划文档：基于 prompt 长度分布预估 block 池大小，留 30% headroom
- SRE rotation 季度演练：人为打 burst 长 prompt 流量验证降级路径

### 7. 30 秒速答

- 一句话核心：长 prompt 突增让 KV block 池被瞬间打满，preempt 与重算雪崩。
- 关键机制：vLLM block 池启动静态分配，单请求 block 数随 prompt 长度线性涨。
- 修复 / 防御：网关限 prompt 长度 + 分流长短 prompt + chunked prefill + prefix cache。
- 面试加分关键词：PagedAttention、gpu_memory_utilization、preempt、chunked prefill、KV offload。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"QPS 才涨 1.3× 但 p99 飙 30 倍"的核心症状？
- [ ] 你能不能解释 KV block 池静态分配 + 长 prompt 占用激增的雪崩机制？
- [ ] 你能不能举一个类似场景（如代码助手大 repo context 突发）？
- [ ] 你能不能说出网关限长、分流、chunked prefill 各自的代价？

---

## Q5. PyTorch CUDA allocator 碎片化导致"明明有 10GB 余量也 OOM"

> 🔴 专家 · 长跑 24h+ 后冒出来的"reserved 71 GB、free 8 GB，512 MB 都分不出来"——这是 caching allocator 经典外部碎片，`expandable_segments` 是第一把药。

### 1. 现象

长跑训练任务（连续运行 24h+）在某个 step 突然 OOM：`Tried to allocate 512 MiB but only 8.3 GiB free; allocator reserved 71.2 GiB; 8.7 GiB unreleasable`。短时间内 reserved 与 allocated 差距越拉越大，最终单次 512MB 分配请求都走不通。

### 2. 根因层级原理

PyTorch 默认 `caching allocator` 把释放的 tensor 内存留作缓存，下次同尺寸 tensor 直接复用避免 cudaMalloc 开销。但缓存按 size class 分桶（默认按 2MB 切分对齐），如果训练过程中频繁出现"先分配大 tensor，再分配小 tensor，再释放大 tensor"的顺序，大块在物理空间被小块拦腰切断，大尺寸新请求找不到连续段——即使总余量够也分不出来。这就是经典的外部碎片。动态形状（如变长 batch）、`torch.compile` recompile、bucket 切换都是触发源。

### 3. 定位手段与命令

- 设置 `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` 试一下原生 async allocator 是否能避免（适合较新 driver）
- 抓 memory snapshot：`torch.cuda.memory._dump_snapshot('snapshot.pickle')`，用 memory_viz 看碎片分布
- `torch.cuda.memory_stats()['reserved_bytes.all.current']` 与 `allocated_bytes.all.current` 差距即为缓存量；`num_alloc_retries` 非零说明已经发生过碎片重试
- 打印 `torch.cuda.memory_summary()` 看每个 size pool 的占用

### 4. 修复路径与权衡

- **短期**：设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，让 segment 可扩展，缓解碎片
- **中期**：调 `max_split_size_mb` 限制大块被切割（如 128 / 256 MB）；间隔性 `torch.cuda.empty_cache()`（注意这只清空 cache，不会回收 active）
- **长期**：固定 batch shape（pad 到 bucket 边界）+ 关掉动态 recompile；切 `cudaMallocAsync` backend（CUDA 11.4+ 支持）
- 权衡：`expandable_segments` 在某些老 kernel 下兼容性差；`empty_cache` 频繁调用会拖慢训练；固定 shape 会引入 padding 浪费

### 5. 同类变体 / 易错点

- 变体：多 stream 训练 / 推理混部时不同 stream 各自缓存导致碎片放大
- 变体：MIG 切片小卡上碎片相对总量比例更显著
- 易错：以为 `empty_cache()` 解决一切——其实只能释放未被引用的 cache 块
- 追问："`expandable_segments` 是怎么做到的？"→ 用虚拟地址保留 + 物理页按需 commit，类似 mmap

### 6. 预防机制

- 默认 launcher 注入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`
- 长跑任务启用 hourly memory snapshot 异步上报
- 监控：`torch.cuda.memory_stats()['num_alloc_retries']` 增量 > 10/min 报警
- 训练完成时 dump 一次 snapshot 进 artifact，供 review
- 长跑（> 12h）任务定期保存 checkpoint 后做 `torch.cuda.synchronize()` + `empty_cache()` 作为 housekeeping

### 7. 30 秒速答

- 一句话核心：caching allocator 外部碎片让"明明有余量"也分不出连续段。
- 关键机制：变长 batch / recompile / bucket 切换让大块被小块拦腰切断。
- 修复 / 防御：`expandable_segments:True`、`max_split_size_mb` 限割、固定 batch shape。
- 面试加分关键词：cudaMallocAsync、memory snapshot、num_alloc_retries、size pool。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"reserved 71G 但 512MB 分不出"的核心症状？
- [ ] 你能不能解释 caching allocator 缓存策略与外部碎片的因果？
- [ ] 你能不能举一个类似场景（如 multi-stream 推理混部）？
- [ ] 你能不能说出 expandable_segments 的实现原理（虚拟地址保留 + 按需 commit）？

---

## Q6. NCCL all-reduce hang：256 卡训练突然全卡 idle 复盘

> 🔴 专家 · SM 100%、功耗只有 idle 1.2 倍、不报错不打 log——分布式工程师看到这套指纹就知道是 NCCL spin-wait，全集群在等某一个 rank。

### 1. 现象

某 256 卡（32 节点 × 8 卡）训练任务，运行至第 4 小时全集群 GPU 利用率从 95% 跌到 0%，进程不退出、不报错、不打 log。`nvidia-smi` 显示所有卡 SM 占用 100% 但功耗只有 idle 水平的 1.2 倍——典型的 spin-wait。30 分钟后 SLURM 触发 walltime 警告，运维介入才发现 hang。

### 2. 根因层级原理

NCCL 的 collective（all-reduce / all-gather 等）按 ranks 间集合语义同步执行。每个 rank 在 collective 入口排好队后，内部 CUDA kernel 进入 busy-wait loop 等所有 rank 都到达。如果某个 rank 因为 (a) 某 op 异常 hang（例如 dataloader worker 死锁），(b) 某 GPU 因 ECC error 进入 fallback path，(c) NIC 链路 flap，导致它没投递自己的数据，则其它 rank 都卡在 busy-wait spin 上。`nvidia-smi` 看 SM 100% 但 GPU clock 是 lowest gear，因为只在跑 spin 指令。NCCL 默认无超时（旧版本）或 30 分钟超时（新版本），所以表现就是"看起来在算其实在等"。

### 3. 定位手段与命令

- 设 `NCCL_DEBUG=INFO`（或 `WARN`）+ `TORCH_NCCL_BLOCKING_WAIT=1` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` + `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`，触发 `dump_traceback` 看每个 rank 卡在哪
- `py-spy dump --pid <PID>` 抓 Python 调用栈
- `nsys profile` 在线接管已运行进程（受限场景才行），看 CUDA stream 状态
- 集群层：`dcgmi diag -r 3` 检 GPU 健康；NIC 看 `ibstat` 与 `ibping`；`nccl-tests/all_reduce_perf` 单独跑确认通信链路

### 4. 修复路径与权衡

- **短期**：定位是哪个 rank 没到 → 重启该节点或退出该 rank 让 SLURM 重排
- **中期**：开启 NCCL watchdog timeout（PyTorch 2.x 默认开），把 hang 转成 RuntimeError 让上层重启
- **长期**：训练 launcher 包一层 healthcheck，每 N step 跑一次 1-byte all-reduce 作为心跳；任何 rank 失联自动 abort + restart from checkpoint
- 权衡：超时设短了误杀长 collective（如启动期 broadcast 大权重），设长了响应慢；watchdog 触发的 abort 可能导致进程残留 GPU 占用

### 5. 同类变体 / 易错点

- 变体：dataloader 多进程死锁（pin_memory 与 fork 不兼容）只 hang dataloader rank
- 变体：grad accumulation 跨 step 触发的 collective mismatch（不同 rank 走不同分支）
- 易错：`NCCL_DEBUG=INFO` 输出量巨大 wraparound 看不到 root cause；用 `NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log` 落盘
- 追问："NCCL 怎么知道 rank 数？"→ 启动时通过 RDZV / 环境变量明确告知

### 6. 预防机制

- launcher 默认注入 `TORCH_NCCL_BLOCKING_WAIT=1`、`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600`、`TORCH_NCCL_TRACE_BUFFER_SIZE=20000`
- 训练前 pre-flight：`nccl-tests` 全节点跑一遍 baseline 带宽，低于阈值禁止启动
- 监控：每 rank 暴露 step 心跳到 Prometheus，集中 dashboard 看哪个 rank stale
- SLURM prolog 自动记录 NCCL 版本、IB driver 版本到 job log
- 季度演练：手动 kill -STOP 一个 rank 验证 watchdog 触发链路

### 7. 30 秒速答

- 一句话核心：NCCL collective busy-wait spin，看似 SM 100% 实则全集群在等一个 rank。
- 关键机制：某 rank dataloader 死锁 / GPU ECC / NIC flap 没投递数据，其它 rank 全卡 spin。
- 修复 / 防御：开 watchdog + heartbeat + trace buffer，hang 转 RuntimeError 自动重启。
- 面试加分关键词：TORCH_NCCL_BLOCKING_WAIT、HEARTBEAT_TIMEOUT、py-spy、flight recorder。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"GPU SM 100% 但不出 token、不报错"的核心症状？
- [ ] 你能不能解释 NCCL collective spin-wait 与 rank 失联的关系？
- [ ] 你能不能举一个类似场景（如 dataloader 死锁 / grad accum 跨 step collective mismatch）？
- [ ] 你能不能说出 4 个常用定位手段（NCCL_DEBUG、py-spy、heartbeat、dcgmi diag）？

---

## Q7. IB 与以太网混用，all-reduce 性能掉 10×：路由误配复盘

> 🔴 专家 · "IB 卡都插了为什么跨节点只有 4 GB/s"——新集群上线最常见的灵异事件，多半是 NCCL 选了 1 Gbps 管理网，或者只用了一张 mlx5。

### 1. 现象

某新上线的训练集群跑 nccl-tests 基准时，节点内 all-reduce 带宽 480 GB/s 正常，但跨节点（2 节点 × 8 卡）只有 4 GB/s，明显远低于 InfiniBand HDR 200 Gbps × 8 NIC 应有的 ~150 GB/s。训练任务能起，但每 step 慢 10 倍以上。

### 2. 根因层级原理

NCCL 在初始化时按一定规则选择网络后端：优先 IB（如果 `NCCL_IB_DISABLE=0` 且检测到 `mlx5_x` 设备）然后 NET/Socket。若节点上既有 IB HCA 又有以太网 NIC，且未通过 `NCCL_SOCKET_IFNAME` 限定接口，NCCL 可能选错——比如走了 management 的 1Gbps 以太网。或者更隐蔽：`NCCL_IB_HCA` 漏写导致只用一个 NIC（损失 8×），或者 IB SM 子网和路由没配 partition key（pkey），数据走了 fallback path。

### 3. 定位手段与命令

- `NCCL_DEBUG=INFO` 启动看 `NCCL INFO Using network: IB / Socket`，确认实际选定 backend
- `nccl-tests/all_reduce_perf -b 1G -e 1G -g 8` 在 2 节点跑出 algbw / busbw 对比 spec
- `ibstat` / `ibstatus` 看 LinkUp + 速率
- `ibdev2netdev` 确认 IB 设备与 net interface 的映射
- `ip addr` + `ethtool` 看以太网接口速率
- `NCCL_SOCKET_IFNAME=^docker,lo` 显式排除虚拟接口

### 4. 修复路径与权衡

- **短期**：环境变量强制：`NCCL_IB_HCA=mlx5_0,mlx5_1,...,mlx5_7`、`NCCL_IB_GID_INDEX=3`、`NCCL_SOCKET_IFNAME=ib0`（或 RoCEv2 场景下对应）
- **中期**：审视 SM/pkey 配置；多 rail 拓扑下开 `NCCL_IB_PCI_RELAXED_ORDERING=1`、`NCCL_NET_GDR_LEVEL=PHB`
- **长期**：网络运维侧固化 cluster-wide subnet manager 配置；image 里硬编码默认 NCCL 环境变量；起任务时 launcher 自动 detect 并打印 backend
- 权衡：硬编码环境变量影响通用性；多 rail 配置在小集群收益不明显

### 5. 同类变体 / 易错点

- 变体：RoCEv2 场景下 PFC / ECN 配置不当导致丢包重传，吞吐塌方
- 变体：跨 rack 路由跨多个 leaf-spine 跳，NCCL 不感知物理拓扑导致选边不优
- 易错：以为"插上 IB 卡就是 IB 通信"——SDK 没装 / OFED 版本对不上时回退 socket
- 追问："NVLS 在什么场景下生效？"→ NVL8/NVL72 内 + NCCL 2.20+ + 算法选 NVLS

### 6. 预防机制

- 集群 image baseline：OFED / NCCL / driver 三件套版本固定，禁止业务侧覆盖
- pre-flight：`nccl-tests` 跑两节点 baseline，busbw 低于 SOTA 80% 禁止开训
- launcher 启动后第一行 log 打印 `NCCL_DEBUG_SUBSYS=INIT,NET` 输出，监控系统抓关键字 `Using network`
- 网络运维侧周期性跑 `ib_send_bw` 全链路压测
- 文档化：每种机型的 `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` 模板

### 7. 30 秒速答

- 一句话核心：NCCL 走错网，跨节点全走 1 Gbps 管理网或漏用 NIC，性能掉 10×。
- 关键机制：未限定 `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` 时 NCCL 自由选择导致选错。
- 修复 / 防御：环境变量强制 NIC + pre-flight nccl-tests + image baseline 锁版本。
- 面试加分关键词：NCCL_IB_HCA、ibdev2netdev、busbw、NVLS、RoCEv2 PFC/ECN。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"跨节点 all-reduce 性能掉 10×"这一症状？
- [ ] 你能不能解释 NCCL 网络后端选择逻辑与误选场景？
- [ ] 你能不能举一个类似场景（如 RoCEv2 PFC 配错导致丢包重传）？
- [ ] 你能不能说出 pre-flight nccl-tests + image baseline 的防御组合？

---

## Q8. NVLink 拓扑不对称：相同模型同卡数训练吞吐差 30%

> 🔴 专家 · 两台同型号 8×H100，跑同样脚本一台 1.20s 一台 1.65s——不要怀疑代码，先看 `nvidia-smi nvlink -s`，多半某条链路 down 了让 ring 绕了远路。

### 1. 现象

两台同型号 8×H100 节点跑同一份训练脚本，节点 A 单 step 1.20 s，节点 B 单 step 1.65 s。GPU 型号、CUDA、驱动、NCCL 全一致，温度功耗都正常。差异在节点 B 的 NVLink ring 带宽测出 320 GB/s 而 A 为 450 GB/s。

### 2. 根因层级原理

H100 SXM 节点理想拓扑是 NVL8 全互联（每对 GPU 之间 4 条 NVLink，total 900 GB/s 双向）。但实际机器可能因为 (a) 某条 NVLink 物理故障被 disable（DCGM 报 link down），(b) NVSwitch 一颗芯片降级，(c) BIOS 配置切到了非对称拓扑（PCIe fallback），导致部分 GPU 间走 PCIe 而非 NVLink。NCCL 的 ring 算法会自动绕过坏链路，但重新构图后某些 hop 变长，整体 ring busbw 下降。

### 3. 定位手段与命令

- `nvidia-smi nvlink -s` 看每条链路状态（`Inactive` 或 `Disabled` 即为问题）
- `nvidia-smi topo -m` 看 GPU 间互联类型矩阵（`NV4` / `NV8` vs `PIX` / `PHB`）
- `dcgmi diag -r 4` 包含 NVLink 健康检查
- `nccl-tests/all_reduce_perf` 单节点 8 卡跑出 busbw 对比 spec（H100 ≈ 480 GB/s）
- `nvidia-smi nvlink --getthroughput` 实时看每条链路流量

### 4. 修复路径与权衡

- **短期**：把节点 B 标 drain，调度避开；任务重排到健康节点
- **中期**：联系硬件运维查 NVSwitch 健康、重做 ECC 检查、必要时重启节点（部分 NVLink 软错误重启可恢复）
- **长期**：硬件 RMA 走流程；image 启动时跑 NVLink 健康自检挂 metric
- 权衡：drain 节点损失算力；硬件 RMA 周期长；强行用降级节点拖慢整任务（gang scheduling 下整批跟着慢）

### 5. 同类变体 / 易错点

- 变体：NVL72 GB200 系统中跨机柜的 NVLink 拓扑变化对算法选择影响更大
- 变体：A100 NVSwitch 部分通道 ECC error 导致带宽缓慢退化（不一定 down）
- 易错：以为"卡数对就行"，忽视拓扑健康；DCGM 默认采集间隔 30s 看不到瞬时链路抖动
- 追问："为什么 ring algorithm 对 latency 不敏感对带宽敏感？"→ N-1 hops，每 hop 完整带宽

### 6. 预防机制

- 节点入集群 burn-in 必跑 nccl-tests baseline + DCGM diag 4 级
- 每节点暴露 `nvlink_link_count_active` metric，少于 spec 报警
- 调度器（SLURM）gres 维度加 `nvlink_health=ok` 标签，自动避开 degraded 节点
- 启动训练前 launcher 打印一次 `nvidia-smi topo -m`，diff 历史快照看是否退化
- 周度 cron：自动跑全集群单节点 nccl-tests，结果落数据库做 trend

### 7. 30 秒速答

- 一句话核心：NVLink 某条链路 down，ring 算法绕远路单节点吞吐掉 30%。
- 关键机制：NVSwitch 故障 / 链路 disable 让 NCCL 重构图，部分 hop 走 PCIe。
- 修复 / 防御：`nvidia-smi nvlink -s` 健康自检 + 调度 drain degraded 节点。
- 面试加分关键词：nvlink_link_count_active、DCGM diag、ring busbw、NVL72 拓扑。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"同型号节点吞吐差 30%"的核心症状？
- [ ] 你能不能解释 NVLink 一条链路 down 后 ring 算法的影响？
- [ ] 你能不能举一个类似场景（如 A100 NVSwitch 通道 ECC 退化）？
- [ ] 你能不能说出 burn-in 健康自检与调度避开的防御组合？

---

## Q9. 慢节点 straggler：1024 卡训练被 1 个 rank 拖慢 2×

> 🔴 专家 · 每张卡 SM 都跑到 95% 但整体吞吐只有理论值 55%——同步训练最坑的就是这种"看起来都在干活但被一个 rank 拖着走"，每天烧掉的钱都是真金白银。

### 1. 现象

1024 卡训练任务，监控显示集群整体吞吐只有理论值的 55%，但每个 rank 的 GPU SM 利用率显示 95%+。step time histogram 大部分 rank 在 1.0 s，少数（5–10 个）在 1.9 s，标准差异常大。任务跑得动但效率极差，每天浪费数千美元。

### 2. 根因层级原理

同步训练（DDP / FSDP / Megatron）每 step 由最慢 rank 决定全局节奏。straggler 来源众多：(a) 某 GPU 因为温度高触发 thermal throttling（clock 降频），(b) PCIe 链路降级（x16 退到 x8），(c) ECC 软错误触发频繁 retry，(d) noisy neighbor（同节点其它进程抢 host CPU），(e) NUMA 不亲和导致内存访问跨 socket。每 rank GPU SM 利用率 95% 没问题——只是这 95% 跑得比别人慢。

### 3. 定位手段与命令

- 上 HTA（Holistic Trace Analysis）跨 rank 对齐 trace，统计每个 rank 的 step time 分布，找尾部
- DCGM 采集每 GPU 的 `DCGM_FI_DEV_GPU_TEMP`、`DCGM_FI_DEV_SM_CLOCK`、`DCGM_FI_DEV_PCIE_LINK_GEN`、`DCGM_FI_DEV_PCIE_LINK_WIDTH`
- `lspci -vvv` 看 PCIe negotiated speed/width vs capability
- `numactl --hardware` + `nvidia-smi topo -m` 看 NUMA 与 GPU 关联，结合 CPU 亲和性
- 训练里嵌入轻量 timer：每 rank 上报 step time 到 Prometheus 看 p50/p99 differential

### 4. 修复路径与权衡

- **短期**：定位 straggler 节点 drain 出去重排；如果是 thermal 调高 fan curve 或检查机房送风
- **中期**：打开 GPU clock lock（`nvidia-smi -lgc max`）防止动态降频；NUMA 绑核（`numactl --cpunodebind`）
- **长期**：训练框架引入 micro-batch level adaptive load balancing（如 unequal pipeline schedule）；冗余 rank（如 ZeRO++ / Bamboo 之类研究方案）容忍 straggler
- 权衡：drain 节点损失部分算力；clock lock 提高功耗；冗余 rank 实现复杂；adaptive 在动态 workload 下不稳定

### 5. 同类变体 / 易错点

- 变体：bandwidth-bound 算法（如 all-gather）对 straggler 更敏感；compute-bound 反而能掩盖
- 变体：dataloader 慢导致部分 rank step time 偶发突增
- 易错：以为"所有卡 SM 都满了就没问题"——忽视绝对速度
- 追问："流水线并行（pipeline parallelism）下 straggler 影响有什么不同？"→ bubble 拉长，整体 PP 利用率塌方

### 6. 预防机制

- 训练 launcher 自带 step-time profiler，每 100 step 上报每 rank p99 step time
- 监控：每 rank step time 偏离全局 p50 超过 20% 持续 5 分钟报警
- pre-flight：节点入集群必跑单卡 microbenchmark（matmul、all-reduce）通过基线
- DCGM 健康守护：thermal / PCIe link / ECC error rate 异常自动 drain
- 调度器 epilog 上报 job 整体 throughput 到 metric，trend 异常触发审计

### 7. 30 秒速答

- 一句话核心：同步训练被一个慢 rank 拖累，每张卡都"满载"但整体吞吐塌一半。
- 关键机制：thermal throttle / PCIe 降级 / NUMA / ECC retry 让单 rank step time 翻倍。
- 修复 / 防御：每 rank step time 上报 Prometheus + DCGM 自动 drain + clock lock。
- 面试加分关键词：HTA、DCGM_FI_DEV_SM_CLOCK、PCIe link width、numactl。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"每张卡满载但整体 55%"的核心症状？
- [ ] 你能不能解释同步训练里最慢 rank 决定全局节奏这一同步语义？
- [ ] 你能不能举一个类似场景（如 noisy neighbor 抢 host CPU）？
- [ ] 你能不能说出 thermal、PCIe、NUMA、ECC 四类 straggler 来源？

---

## Q10. NCCL 版本偏差 + 节点镜像漂移，跨节点通信功能错乱

> 🔴 专家 · "16 节点能跑、64 节点必崩"，这种规模相关的稳定复现，几乎一定指向集群里某些节点的 NCCL/OFED/driver 版本悄悄漂了。

### 1. 现象

训练任务在测试环境（4 节点）跑通，扩到生产 64 节点后，第一个 step 还没完就出 `NCCL WARN ncclInternalError` + `peer mismatch`。回滚到 16 节点又能跑，扩 64 又坏，复现稳定但难定位。

### 2. 根因层级原理

NCCL 协议在不同小版本间 wire format 偶尔不兼容（特别是 2.18 → 2.20 NVLS 引入、2.22 引入新算法切换逻辑）。集群规模扩大后，不同节点 image 版本漂移概率上升：有的节点装的是 NCCL 2.18（包在 PyTorch wheel 里），有的节点系统装了 2.21（OFED 包带的），调度时随机分到一起。即使主版本一致，CUDA driver / OFED / `NCCL_PROTO`（LL/LL128/Simple）差异也可能让两端协议状态机走偏。

### 3. 定位手段与命令

- `NCCL_DEBUG=INFO` 在每节点启动时打印 NCCL 版本号，集中比对
- `python -c "import torch; print(torch.cuda.nccl.version())"` 看 PyTorch 内置版本
- `ldd $(python -c "import torch; print(torch.__file__)" | head)/lib/libtorch_cuda.so` 看实际链接的 NCCL 路径
- `dpkg -l | grep -i nccl` 或 `rpm -qa | grep nccl` 看系统包
- 启动时打印 `cat /proc/driver/nvidia/version`、OFED 版本、kernel 版本统一收集

### 4. 修复路径与权衡

- **短期**：把所有节点 NCCL 强制统一到 PyTorch 自带版本（`LD_LIBRARY_PATH` 干掉系统路径）
- **中期**：修 image 漂移流程，所有节点 image 重建 + 重灌；准入检查脚本作为 SLURM prolog
- **长期**：image 治理 / 不可变基础设施（每次更新走全量重建而非增量 patch）；SBOM 记录每节点 ML 组件版本，调度只挑同版本
- 权衡：强制 PyTorch 自带版本会失去某些系统优化（如 NVLS）；image 重建周期长；准入检查太严会拒掉有效节点

### 5. 同类变体 / 易错点

- 变体：CUDA driver 版本漂移导致 PTX JIT 编译路径不同，部分 kernel 行为差异
- 变体：OFED 与 NCCL 版本组合矩阵很大，某些组合官方未测试
- 易错：以为 `pip install` 的 PyTorch 自带 NCCL 就万事大吉——LD_LIBRARY_PATH 优先级会覆盖
- 追问："NVLS 是什么版本引入的？"→ NCCL 2.18 起，需要 driver R535+

### 6. 预防机制

- 集群 image 治理：所有节点 OS / driver / OFED / CUDA / NCCL 版本统一锁定，差异即报警
- SLURM prolog 收集每节点 6 件套版本号上报 metric，dashboard 看版本分布
- 启动 launcher 第一步：`nvidia-smi --query-gpu=driver_version --format=csv`、`echo $NCCL_VERSION`、`ibstat | grep Firmware`，diff 全集群打印
- pre-flight：跨 N 节点 nccl-tests 必通过且 busbw 在阈值内
- image 升级走灰度（5% → 25% → 100%），每阶段跑 baseline 训练对比指标

### 7. 30 秒速答

- 一句话核心：16 节点能跑 64 节点必崩，集群里某些节点 NCCL/OFED/driver 版本漂了。
- 关键机制：NCCL wire format 小版本不兼容 + image 漂移概率随规模上升。
- 修复 / 防御：image 治理 / 不可变基础设施 + SLURM prolog 收集版本 + LD_LIBRARY_PATH 干掉系统路径。
- 面试加分关键词：NCCL 2.18 NVLS、ldd 链接、image baseline、SBOM。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"规模相关的稳定复现 hang"这一症状指纹？
- [ ] 你能不能解释 NCCL 版本协议状态机与 wire format 的关系？
- [ ] 你能不能举一个类似场景（如 OFED 与 NCCL 组合矩阵踩坑）？
- [ ] 你能不能说出 image 治理与准入检查的防御组合？

---

## Q11. Checkpoint 静默损坏：恢复训练发现 loss 飞涨追溯文件 corruption

> 🔴 专家 · 节点崩了从最近 ckpt 恢复，loss 1.8 直接跳到 12+——这种"反序列化能成但权重是垃圾值"的静默损坏，背后多半是没 fsync、没 atomic rename 的老路子。

### 1. 现象

某次训练任务因节点故障重启，从最近一次 checkpoint（保存在 Lustre）恢复后，loss 从恢复前的 1.8 突然跳到 12+，几个 step 内变 NaN。回退到再上一个 checkpoint 同样问题。回退到三个之前的才能正常恢复，意味着最近 2 个 checkpoint 文件在不知不觉中已损坏。

### 2. 根因层级原理

Checkpoint 写入是一个非原子的、跨多文件的过程。常见 corruption 来源：(a) 写到一半节点崩溃，文件被截断但没有 fsync（POSIX 不保证 mtime 与文件完整性一致）；(b) 异步保存（FSDP2 / `torch.distributed.checkpoint` async API）的 staging buffer 还没全部 flush，主线程已经移走旧 ckpt；(c) Lustre / GPFS 在客户端 cache flush 失败时不向上层报错，文件 size 看起来对但内容是 0；(d) tar / zip 打包过程中 inode 变化文件丢页。这些情况下恢复时反序列化能成功，但权重某些层是垃圾值，loss 立即发散。

### 3. 定位手段与命令

- 写入侧：保存后立刻 `sha256sum` 整个目录写入 `.manifest`；恢复时校验
- 读出侧：`torch.load(map_location='cpu')` + 遍历 `state_dict()` 检查每个 tensor `is_finite().all()`、`norm()` 在合理范围
- 文件系统：`stat -c '%s %y' file` 比对预期 size；`lfs check` Lustre 文件健康
- `strings` / `xxd | head` 看文件头是否是合法 magic
- 历史 step 的 weight norm trend：如果某层权重 L2 norm 突然非线性跳变，是损坏证据

### 4. 修复路径与权衡

- **短期**：回退到更老 checkpoint；接受丢失的步数
- **中期**：保存流程改成"写到临时目录 → fsync → rename → 删旧"原子模式；保存后立刻读回校验（read-back validation）
- **长期**：上 `torch.distributed.checkpoint`（DCP）的 sharded format + per-rank 校验和；引入冗余 checkpoint（保留最近 K 个 + 每 N 个长期归档）
- 权衡：read-back 校验拖慢保存周期；冗余 checkpoint 占用存储 K×；DCP 切换需要改恢复路径

### 5. 同类变体 / 易错点

- 变体：保存被 `rsync` 拖到对象存储途中失败，目标端文件 size 0
- 变体：FSDP2 async save 与下一个 step 的 grad update 竞态导致部分 shard 是新版部分是旧版（mismatch）
- 易错：`torch.save` 默认不 fsync 的，靠 Python `with` 退出关闭只 flush 用户态 buffer
- 追问："为什么用 rename 是原子的？"→ POSIX 保证 same-FS rename 原子；跨 FS 的 mv 实际是 cp+rm 不原子

### 6. 预防机制

- 写入侧封装 `safe_save(state, path)`：写到 `.tmp` → `os.fsync(fileno)` → `os.rename`
- 每次 save 后异步 read-back + sha256 校验，写入 `.manifest.json`
- 训练 launcher 启动时校验最近 ckpt manifest，损坏自动 fallback 到上一个
- 监控：每个 ckpt 的 weight norm trend，跳变 > 3 sigma 报警
- 长期归档：每 N 个 ckpt copy 一份到对象存储 versioned bucket

### 7. 30 秒速答

- 一句话核心：ckpt 反序列化能成但权重是垃圾值，loss 跳变 / 飞涨追溯文件 corruption。
- 关键机制：写一半崩、没 fsync、async staging 与下一 step 竞态、FS 客户端 cache flush 失败。
- 修复 / 防御：`safe_save`：tmp + fsync + rename + sha256 manifest + read-back 校验。
- 面试加分关键词：原子 rename、torch.distributed.checkpoint、weight norm 监控、冗余 ckpt。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"恢复后 loss 飞涨"的核心症状？
- [ ] 你能不能解释非原子写入与 fsync 缺失的因果？
- [ ] 你能不能举一个类似场景（如 rsync 中断 / FSDP2 async 竞态）？
- [ ] 你能不能说出 tmp+fsync+rename+manifest 的防御链？

---

## Q12. Checkpoint 恢复耗时 1 小时：单 rank 串行加载 + 跨 NUMA 拷贝复盘

> 🔴 专家 · 175B 模型每次 reload 卡 60 分钟，GPU 全闲——这是 `torch.save` 单文件 + rank 0 broadcast 的老架构在大模型时代必然撞的墙，DCP 改造迟早要上。

### 1. 现象

某 175B 模型训练，每次因节点故障重启加载 checkpoint 耗时近 60 分钟，期间 GPU 全闲。一周内累计因 reload 浪费 ~10% 算力。监控显示加载阶段单节点 IO 吞吐 ~200 MB/s，远低于 Lustre / NVMe 标称值。

### 2. 根因层级原理

传统 `torch.save` 产出的是 single-file pickle，恢复时只能由 rank 0 读完后 broadcast，整个流程串行：read（受单 client 带宽限）→ deserialize → broadcast（受 NCCL 带宽限）→ 各 rank load_state_dict。175B × 2 bytes ≈ 350 GB，单 client Lustre 读取上限 1–2 GB/s 的话光 read 就 3–6 分钟，但 broadcast 阶段更慢——大 tensor 经过 host 内存（pinned 或非 pinned）+ 跨 NUMA + GPU peer copy，环节多瓶颈在某一段。再加上 deserialize 时反复 alloc/free 主存碎片化。

### 3. 定位手段与命令

- 加载阶段加细粒度 timer：`read_file`、`deserialize`、`broadcast`、`apply_to_module` 分别计时
- `iotop` / `iostat -x 1` 实时看磁盘 / 网络 IO
- `numastat -p $PID` 看跨 NUMA 内存分配
- `nvidia-smi dmon -s u` 看 GPU 等待时间
- Lustre 客户端：`lfs getstripe` 看条带配置，单文件单 stripe 时单 client 受限

### 4. 修复路径与权衡

- **短期**：把 ckpt 切成多个 shard，多 rank 并行读（手动按层切分）
- **中期**：切到 `torch.distributed.checkpoint`（DCP），原生 sharded + 多 rank 并行 IO；Lustre 上 `lfs setstripe -c 8` 让单文件多 OST 分布
- **长期**：异步加载 + checkpoint 流水化（边读边 apply 部分 layer）；用对象存储 multipart parallel get；FSDP2 集成 DCP 默认即多 rank 并行
- 权衡：DCP 改造需要重训现有 ckpt 或转换工具；Lustre 条带配置太宽对小文件反而劣化；异步加载工程复杂度高

### 5. 同类变体 / 易错点

- 变体：S3 单 object 单连接限速 ~50–100 MB/s，没用 multipart 时极慢
- 变体：恢复时 optimizer state 比 model 还大但同样串行加载
- 易错：以为 `map_location='cuda:0'` 就快——实际还是单 GPU 接收；要每 rank 直接 load 本 shard
- 追问："为什么 DCP 比 torch.save 快？"→ 每 rank 自己读自己的 shard，IO 并行 N×

### 6. 预防机制

- 训练默认上 DCP，所有新模型 day-1 用 sharded ckpt
- 启动时先打印预估恢复时间（基于 ckpt 大小 + 集群 IO 标称值），超阈值警告
- Lustre / GPFS 团队提供条带配置 best-practice 文档；存储路径分类（hot ckpt / cold archive）
- pre-flight：first epoch 结束后跑一次完整 save+load round-trip 验证耗时
- 监控：load 耗时 metric trend，超过历史 p95 1.5× 告警

### 7. 30 秒速答

- 一句话核心：单文件 ckpt + rank 0 broadcast 在 175B 量级注定卡死 60 分钟。
- 关键机制：read 受单 client 带宽限 + broadcast 跨 NUMA + alloc 碎片，串行链路太长。
- 修复 / 防御：切 DCP sharded + Lustre setstripe + 异步加载流水化。
- 面试加分关键词：torch.distributed.checkpoint、lfs setstripe、multipart parallel get。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"reload 60 分钟 GPU 全闲"的核心症状？
- [ ] 你能不能解释单文件 broadcast 的串行链路瓶颈在哪里？
- [ ] 你能不能举一个类似场景（如 S3 单 object 单连接限速）？
- [ ] 你能不能说出 DCP 比 torch.save 快的根本原因（每 rank 自读 shard）？

---

## Q13. S3 限速触发 SlowDown：分布式 checkpoint 上传雪崩

> 🔴 专家 · 1024 rank 同时往一个 prefix 推 multipart，瞬间几万 QPS——S3 一句 `503 SlowDown` 就让训练 hang 在 save 阶段，"无限扩展"是 prefix 维度的，不是瞬时的。

### 1. 现象

某次训练把 ckpt 同步上传 S3 作为长期归档，1024 rank 同时发起 multipart upload 后，AWS 返回大量 `503 SlowDown` + `RequestTimeTooSkewed`，部分 part 重试 5 次后失败，整个 ckpt 没传完，下一次 save 又叠加上来，最终训练 hang 在 save 阶段。

### 2. 根因层级原理

S3 对单 prefix 有 request rate limit（PUT/COPY/POST/DELETE 默认 3500 req/s/prefix，GET 5500 req/s/prefix）。1024 rank × multipart × 多 part = 瞬时几万 QPS，全打到同一 ckpt prefix 立即触发限速。S3 SDK 默认 retry 带指数退避，但所有 client 同步退避后又同时重试形成 thundering herd。叠加时钟漂移 `RequestTimeTooSkewed`（节点 NTP 偏离超过 15 分钟），更让重试失败率居高不下。

### 3. 定位手段与命令

- AWS S3 metrics：`5xxErrors` + `4xxErrors` + `AllRequests` 看请求曲线和错误率
- CloudTrail 看具体错误码分布
- 客户端 SDK 日志：`boto3` 的 retry 次数 metric
- `chronyc tracking` / `timedatectl` 看 NTP 偏移
- 抓包 `tcpdump host s3.amazonaws.com` 看 503 响应频率

### 4. 修复路径与权衡

- **短期**：减小并发上传数（aggregate 上传：rank 0 收集所有 shard 再串行/有限并发上传）；jitter retry 退避
- **中期**：prefix 散列化（路径加 hash 前缀分散到多 partition）；用 S3 Transfer Manager 自动管理 multipart concurrency
- **长期**：上 S3 Multi-Region Access Points 或专用 high-throughput bucket；换 R2/MinIO 等无固定限速的存储；本地落 + 后台异步推
- 权衡：rank 0 aggregate 让 rank 0 成为瓶颈；prefix 散列影响列表查询；专用 bucket 成本高

### 5. 同类变体 / 易错点

- 变体：GCS / Azure Blob 也有类似限速但门槛不同；混云时不能套同样并发
- 变体：multipart upload 未 abort 的残留 part 长期累积成本，需要 lifecycle 清理
- 易错：以为 S3 "无限扩展"——是 prefix 维度自动 scale，但短时突发仍被 throttle
- 追问："如何提高单 prefix 限速？"→ 实际无法直接提，要联系 AWS support 或拆 prefix

### 6. 预防机制

- 上传客户端默认开 jitter retry + bounded concurrency（如每节点 ≤ 4 connection）
- ckpt path 模板里加随机前缀：`s3://bucket/{rand4}/jobs/{jobid}/step{N}/`
- 监控 S3 5xx rate，> 1% 告警；客户端侧 `s3.upload_failure_rate`
- 训练框架提供 `local_first_then_async_upload` 模式（先落本地，后台异步推 S3）
- NTP 健康检查作为 SLURM prolog 必过项

### 7. 30 秒速答

- 一句话核心：1024 rank 同 prefix 推 multipart 瞬间几万 QPS 触发 503 SlowDown 雪崩。
- 关键机制：S3 单 prefix 限速 3500 PUT/s + 同步退避 + thundering herd。
- 修复 / 防御：prefix 散列 + bounded concurrency + jitter retry + 本地落异步推。
- 面试加分关键词：multi-part upload、Transfer Manager、RequestTimeTooSkewed、lifecycle 清理。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"训练 hang 在 save 阶段"的核心症状？
- [ ] 你能不能解释 S3 prefix 限速与 thundering herd 的因果？
- [ ] 你能不能举一个类似场景（如 GCS / Azure Blob 限速）？
- [ ] 你能不能说出 prefix 散列、bounded concurrency、本地落三种防御？

---

## Q14. Lustre 配额耗尽，训练任务静默写入失败

> 🔴 专家 · 凌晨配额满，训练任务还在跑只是 ckpt 全部静默失败——`torch.save` 抛的 `EDQUOT` 被一句 `except: log.warning` 吞掉，第二天才发现 10 小时无 ckpt。

### 1. 现象

某项目共享 Lustre 目录配额 50 TB，多个训练任务并发写 ckpt + log。某天凌晨配额满，新写入返回 `EDQUOT`，但训练框架未把 IO 错误当 fatal，task 继续跑只是 ckpt 全部静默失败。第二天早上发现过去 10 小时所有 save 都没成功，节点崩溃后无 ckpt 可恢复。

### 2. 根因层级原理

Lustre 配额有两类：block quota（容量）和 inode quota（文件数）。耗尽时新建/扩展操作返回 `EDQUOT`。问题在 (a) Python `open(..., 'wb')` + write + close 在某些情况下错误只在 close 时显现，而很多代码 `try: torch.save(...) except: log.warning(...)` 把它降级成 warning；(b) `torch.save` 内部 pickle 写入失败可能只丢 partial 文件而不抛异常；(c) 多任务共享配额，谁是凶手不易定位。

### 3. 定位手段与命令

- `lfs quota -u $USER /lustre/path` 看用户配额；`lfs quota -g $GROUP` 看组配额
- `df -h` 与 `lfs df -h` 看 OST 整体使用率
- `lfs find /lustre/path -size +1G` 找大文件
- `lfs find /lustre/path -type d -name 'step*' | xargs -I{} du -sh {}` 找占用大户
- 从 application log grep `EDQUOT` / `Disk quota exceeded` 关键字

### 4. 修复路径与权衡

- **短期**：清理临时文件、过期 ckpt 释放空间；联系存储管理员加配额
- **中期**：训练代码捕获 IO 异常并 fail-fast（不再静默 warning）；保存前先 statvfs 校验剩余空间；keep-last-K 策略自动回收旧 ckpt
- **长期**：分级存储：热 ckpt 在 Lustre，冷 ckpt 自动归档对象存储；预算治理（每项目配额 + 用量看板）
- 权衡：fail-fast 可能让任务因临时空间不足直接挂掉；自动归档增加复杂度；分级存储引入读取延迟

### 5. 同类变体 / 易错点

- 变体：inode 配额耗尽（小文件太多，例如 MoE 每 expert 一个文件）也触发同样症状
- 变体：GPFS / WekaFS 错误码不同（`EDQUOT` vs `ENOSPC`）但症状类似
- 易错：以为 `df -h` 看够用就行，忽视用户/组配额
- 追问："Lustre 元数据延迟为什么慢？"→ MDS 中心化节点；OST 数据本身分布

### 6. 预防机制

- 训练 launcher pre-flight：检查目标存储配额剩余 > 当前任务预估写入量 × 3
- 保存代码模板：用 `pathlib` + `os.statvfs` 校验空间；任何 IO 异常 raise 而非 warn
- 监控：用户/组配额使用率 > 85% 告警
- Cleanup cron：每天清理 `tmp/`、保留最近 K 个 ckpt（其它移冷库）
- 用量看板：每项目按 PI 维度展示存储使用 trend

### 7. 30 秒速答

- 一句话核心：配额满 `EDQUOT` 被 except 吞掉，10 小时静默无 ckpt。
- 关键机制：close 时才报错 + 框架 except 降级 warning + 多任务共享配额。
- 修复 / 防御：fail-fast + statvfs 预检 + keep-last-K + 用量看板。
- 面试加分关键词：lfs quota、inode quota、EDQUOT、分级存储归档。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"训练在跑但 ckpt 全部静默失败"的核心症状？
- [ ] 你能不能解释为什么 IO 错误会在 close 时才显现？
- [ ] 你能不能举一个类似场景（如 inode 配额耗尽 MoE 小文件）？
- [ ] 你能不能说出 fail-fast + 预检 + 用量看板的防御组合？

---

## Q15. 异步 checkpoint 与下一 step 竞态：FSDP2 async save 部分 shard 是旧版

> 🔴 专家 · 恢复后 loss 不是飞涨而是"跳一下慢慢恢复"——这是 async save 最阴险的形态：shard A 是 step N、shard B 是 step N+1，拼出来的 state 物理上存在但逻辑上不存在。

### 1. 现象

某 FSDP2 训练任务开了 `async_save=True`，原本期望保存与计算重叠隐藏 IO 时延。运行几天后偶发出现"恢复后 loss 跳变 + 慢慢恢复"现象（不像完全损坏，像是部分参数是 N-1 step 的）。复盘时发现保存目录里某些 shard 文件的 mtime 比其它晚 1 个 step time。

### 2. 根因层级原理

FSDP2 的 async save 工作机制：保存时把 sharded state 拷到 staging buffer，主流程立即返回继续训练；后台线程把 buffer 写入磁盘。问题在 (a) staging buffer 拷贝若不是真正的 deep copy（比如只是 record stream），下一 step 的 in-place update 会污染待写入数据；(b) 不同 shard 写入完成时间不同，某些 shard 已写完，某些还在 buffer 里被新 step 改掉；(c) atomic rename 是 per-shard 的，全局没有 commit 屏障。结果是恢复时拼出来的 state 是"step N 的 shard A + step N+1 的 shard B"——逻辑上不一致。

### 3. 定位手段与命令

- 检查 ckpt 目录下每个 shard 文件 mtime 是否一致：`find ckpt/ -name '*.distcp' -printf '%T@ %p\n' | sort`
- 在每个 shard metadata 里写入 `step_id`、`global_step`，恢复时校验全部一致
- 启用 PyTorch DCP 的 `async_save_blocking_wait` 选项，验证非异步路径能恢复正确
- 对比恢复后的 weight 与训练时的 in-memory weight 是否完全一致

### 4. 修复路径与权衡

- **短期**：关 `async_save=True`，回到 sync save（损失 IO 隐藏但保证一致性）
- **中期**：用真正 deep copy 的 staging（snapshot tensor 而非 reference）；引入 commit barrier，所有 shard 全部 fsync 后才更新 manifest 指向
- **长期**：双 buffer ping-pong，写期间禁止 in-place 改对应 buffer；用 `torch.distributed.checkpoint` 的官方 async API（v2.5+ 已修这类问题）
- 权衡：deep copy 要 2× 显存或 2× 主存；commit barrier 串行化降低 async 收益；版本升级回归测试成本

### 5. 同类变体 / 易错点

- 变体：DeepSpeed 的 universal checkpoint 也有类似 async 一致性边界
- 变体：optimizer state 与 model state 分文件保存，恢复时如果时间错位也是同款问题
- 易错：以为 mtime 一致就是一致——实际写入顺序依赖 OS 调度
- 追问："commit barrier 为什么能保证一致性？"→ 全局 happens-before：旧 manifest 指向 N-1，新 manifest 在所有 shard 落盘后原子翻转

### 6. 预防机制

- 默认 sync save 起步，async save 只在压测验证过的版本启用
- 每个 ckpt 写入完成后跑自动一致性校验（global_step 全 shard 一致 + manifest 完整）
- 框架版本升级走灰度，关注 release note 中 async save 相关 bugfix
- 监控：恢复后 first step loss 与 save 时 loss diff 超过 ε 报警
- 季度演练：人为在 async save 中 kill 进程，验证恢复行为符合预期

### 7. 30 秒速答

- 一句话核心：async save shard 写入时间错位，恢复出来部分参数是 step N-1 的"幽灵 state"。
- 关键机制：staging buffer 不是真 deep copy + 无 commit barrier，下一 step in-place update 污染。
- 修复 / 防御：默认 sync save + deep copy snapshot + 全 shard fsync 后翻 manifest + global_step 校验。
- 面试加分关键词：commit barrier、happens-before、global_step manifest、双 buffer。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"恢复后 loss 跳一下慢慢恢复"的特征？
- [ ] 你能不能解释 staging buffer 与 in-place update 的竞态机制？
- [ ] 你能不能举一个类似场景（如 optimizer state 与 model state 分文件时间错位）？
- [ ] 你能不能说出 deep copy + commit barrier + manifest 校验的防御链？

---

## Q16. vLLM continuous batching 死锁：waiting queue 堆积但 running queue 不前进

> 🔴 专家 · 进程不挂、health check 200，但 5 分钟没出 token——这是 scheduler 的活锁：preempt 释放的 block 立刻被新 prefill 抢走，原 seq 永远等不回来。

### 1. 现象

某线上 vLLM 推理服务突发 hang：所有新请求进入 `waiting` 状态后不再前进，已有 `running` 请求也不出 token，但服务进程不挂、health check 仍返回 200。`vllm:num_requests_running` 与 `vllm:num_requests_waiting` 同时非零且 5 分钟无变化。重启进程立即恢复，但每隔几小时复发。

### 2. 根因层级原理

vLLM 调度器在每个 step 决定哪些 seq 推理、哪些被 preempt。死锁场景：(a) 所有 running seq 都因 KV block 不足被标记为 preempt 候选，但 preempt 后释放的 block 又被新 prefill 请求立刻抢走，原 seq 永远等不到 block 回来；(b) prefix cache 引用计数泄漏，某 block "占用中"但实际没人用；(c) 自定义 logit processor 抛异常被吞，scheduler 状态不一致。结果是状态机进入"running 不能 decode、waiting 不能 prefill、preempt 释放被立刻抢"的活锁。

### 3. 定位手段与命令

- `py-spy dump --pid <vllm_pid>` 抓 Python 栈，看是不是卡在某个 lock
- vLLM metrics：`vllm:num_preempted_total` rate 异常高 + `gpu_cache_usage` 100% 是典型 fingerprint
- `curl http://localhost:8000/metrics` + 自定义 debug endpoint dump scheduler 内部状态
- 启动加 `--max-num-seqs=...` 与 `--max-num-batched-tokens=...` 显式上限，看是否避开
- 复现侧：录一段 prompt 长度分布回放，二分定位

### 4. 修复路径与权衡

- **短期**：自动 watchdog 监测 step 无进展超过 30 s 自动重启进程
- **中期**：升级 vLLM 到 V1（重写的 scheduler，活锁场景已 fix）；调小 `max_num_seqs` 留 KV block 余量；关掉 prefix cache 验证是否引用计数 bug
- **长期**：scheduler 加 priority aging（waiting 时间长的优先级提升）防饥饿；引入 fairness 算法
- 权衡：watchdog 重启会丢正在跑的请求；vLLM V1 升级回归测试成本；关 prefix cache 损失 cache hit 性能

### 5. 同类变体 / 易错点

- 变体：SGLang / TensorRT-LLM 的 continuous batching 调度有自己的 corner case
- 变体：推测解码（speculative decoding）引入 draft seq 后 KV 占用计算更复杂
- 易错：以为重启服务"治本"——实际只是把状态清空，根因在调度器逻辑
- 追问："continuous batching 与 static batching 区别？"→ token-level 而非 request-level，长短请求混排吞吐高

### 6. 预防机制

- watchdog：监测 `metrics_step_count` 增量，连续 N 秒不变自动重启
- 灰度：每个 vLLM 版本上生产前必跑 24h 压测覆盖长短 prompt 混合 + 高 preempt 场景
- 监控：`preempt_rate` p95 > 阈值 + `cache_usage` > 95% 持续 → 自动扩容 / 降级
- vLLM 升级走 canary（5% 流量先打），观察一周再全量
- 引入 chaos test：随机注入超长 prompt 验证调度稳定性

### 7. 30 秒速答

- 一句话核心：scheduler 活锁——preempt 释放的 block 立刻被新 prefill 抢走，running 永远不前进。
- 关键机制：所有 running 因 KV 不足被标 preempt，但释放后被新请求秒抢，状态机环停。
- 修复 / 防御：watchdog step-no-progress 重启 + 升级 vLLM V1 + 留 KV 余量 + priority aging。
- 面试加分关键词：num_preempted_total、prefix cache 引用计数、admission control、fairness。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"health 200 但 5 分钟不出 token"的活锁症状？
- [ ] 你能不能解释 preempt-prefill 互抢导致的状态机活锁？
- [ ] 你能不能举一个类似场景（如 prefix cache 引用计数泄漏）？
- [ ] 你能不能说出 watchdog 重启与 vLLM V1 升级两条修复路径？

---

## Q17. 推理服务 KV cache 内存泄漏：长跑后 OOM 与 cache hit rate 下滑

> 🔴 专家 · 服务跑了 7 天 OOM 周期从 24h 缩到 6h，cache hit 从 60% 跌到 20%——经典的 ref count 泄漏，看 `free_block_floor` 的下降趋势比看瞬时利用率更准。

### 1. 现象

推理服务部署 7 天后开始周期性 OOM 重启，每次重启间隔从初期 24h 缩短到 6h。`vllm:gpu_cache_usage` 长期居高不下，prefix cache hit rate 从 60% 慢慢降到 20%。重启后立即恢复，运行越久越糟。

### 2. 根因层级原理

KV block 通过引用计数管理生命周期：seq 引用 → ref++，seq 结束 → ref--，到 0 时回收。泄漏来源：(a) 异常路径未正确释放（请求超时被网关 abort 但 vLLM 内部 ref 没减）；(b) prefix cache 的 trie 节点逻辑 bug 导致某些 hash key 永远 alive；(c) LoRA 适配器卸载时漏释放对应 KV；(d) tool calling / function calling 请求结构嵌套带来的 ref 路径未覆盖。结果：可用 block 数随时间单调下降，cache hit 因 evict 越来越频繁也下滑。

### 3. 定位手段与命令

- 长跑监控 `vllm:num_total_blocks - vllm:num_free_blocks` trend，正常应在 [0, max] 周期波动；如果 floor 持续抬升即为泄漏
- vLLM `--enable-metrics` + 自定义 debug endpoint dump block ref 表
- `tracemalloc` 或 `memray` 抓 Python 侧泄漏（KV 是显存但元数据是 Python 对象）
- 对比有无 LoRA、有无 tool call、有无超时三种 traffic 的泄漏速率
- `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 看 vLLM 进程占用 trend

### 4. 修复路径与权衡

- **短期**：设 `max_lifetime` 周期性重启（蓝绿部署）；降级关 prefix cache
- **中期**：升级到修复对应泄漏的 vLLM 版本；自定义 LoRA 卸载路径手动释放 KV
- **长期**：引入 KV block 审计——周期性扫描 ref 表与活跃 seq 比对，发现孤儿 block 强制回收 + 报警
- 权衡：周期性重启会丢正在跑的请求需要 graceful drain；强制回收可能误杀正常 cache

### 5. 同类变体 / 易错点

- 变体：SGLang radix attention cache 也有类似引用计数 bug
- 变体：Triton Inference Server 的 model warmup buffer 没释放
- 易错：以为 OOM 是流量增长——其实是 floor 在涨，需要看 trend 而非瞬时
- 追问："prefix cache 怎么实现 trie？"→ 按 token id chunk hash，trie 节点指向 KV block

### 6. 预防机制

- 长跑健康指标：`free_block_floor`（每小时最低值），单调下降即泄漏
- 蓝绿部署默认开 `max_lifetime=72h` 强制周期性 rotate
- vLLM 升级前先在 staging 跑 7 天 longevity test
- 每个 vLLM 版本的 release note 关注 "memory leak" / "ref count" 关键字
- chaos：注入超时、客户端断连、LoRA 频繁切换的 traffic pattern 验证无泄漏

### 7. 30 秒速答

- 一句话核心：KV block 引用计数泄漏，free block floor 随时间单调下降，最终 OOM。
- 关键机制：异常路径 / prefix cache trie / LoRA 卸载漏减 ref，孤儿 block 永远不回收。
- 修复 / 防御：周期性蓝绿 max_lifetime + free_block_floor 监控 + 升级修复版本 + chaos test。
- 面试加分关键词：free_block_floor 趋势、tracemalloc/memray、graceful drain、ref 表审计。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"OOM 周期从 24h 缩到 6h"的趋势特征？
- [ ] 你能不能解释 ref count 泄漏与 cache hit 下滑的连带因果？
- [ ] 你能不能举一个类似场景（如 SGLang radix cache / Triton warmup buffer）？
- [ ] 你能不能说出 free_block_floor 单调下降这一关键 metric 的意义？

---

## Q18. 推理请求超时雪崩：上游重试放大流量 3× 击穿后端

> 🔴 专家 · 后端 QPS 3 分钟涨到 3× 但流量根本没那么多——客户端"超时即重试"的本能在 LLM 上特别危险，prefill 不可中断、KV 不可复用，重一次就翻倍。

### 1. 现象

某长 prompt 客户端请求 p99 时延突破 30 s 客户端超时，client 自动重试。3 分钟内后端 QPS 从 1× 涨到 3×，所有请求开始排队，p50 也突破 timeout，全链路 5xx。SRE 介入限流后服务恢复，但根因不是"流量真涨了"。

### 2. 根因层级原理

重试雪崩经典模型：客户端超时 → 重试 → 后端已 overload 处理慢 → 更多超时 → 更多重试 → 容量被同一请求多次占用。LLM 推理特别容易触发：(a) 长 prompt prefill 耗时本就接近 timeout 阈值，小波动就 trigger；(b) 重试请求与原请求 prompt 完全相同，prefix cache 命中度高但已经在跑的不会被合并；(c) 推理是有状态的（KV 缓存），重试不能复用已生成的部分，全部从头算，浪费翻倍；(d) 没有 hedging（受控并行）只有盲重试。

### 3. 定位手段与命令

- 客户端日志看每请求的 retry count
- 网关层（Envoy / nginx）retry policy 配置审计
- 后端 access log：按 `request_id` group 看是否同 ID 多次到达（如果 client 用幂等 key）
- Prometheus：`http_requests_total{status=~"5.."}` 与 `client_retry_total` 关联
- 抓一次故障 traffic 回放，对比有无重试

### 4. 修复路径与权衡

- **短期**：网关层禁用自动重试（或改成只在 5xx 且 idempotent 时重试 1 次）；服务端加 `Retry-After` header 引导客户端退避
- **中期**：服务端 admission control（队列长度超阈值直接 4xx 拒绝而非排队）；客户端 SDK 内置 jitter exp backoff
- **长期**：流量隔离——长 prompt 走独立 pool，避免拖垮短 prompt SLO；引入 hedged request（同时打 2 个并取先返回）只在受控并发下用
- 权衡：禁重试会让瞬时网络抖动表现为失败；admission control 提高 4xx 但保护后端；hedging 增加成本

### 5. 同类变体 / 易错点

- 变体：load balancer 健康检查超时把节点踢掉导致剩余节点更过载
- 变体：streaming 响应客户端断连后服务端继续生成浪费算力
- 易错：以为"加机器就行"——加机器在雪崩中也会被打满
- 追问："为什么 LLM 重试比 REST API 更危险？"→ 状态成本高、prefill 不可中断、单请求 GPU 占用大

### 6. 预防机制

- 网关 retry policy 严格规范化（禁默认无脑重试）
- 客户端 SDK 强制 jitter + max retry=2 + circuit breaker
- SLO 体系：TTFT / TPOT / p99 三指标分级，p99 超阈值自动降级（拒部分流量）
- 每季度跑混沌演练：人为注入慢响应观察重试雪崩防线
- 服务端 idle connection drain：客户端断连立即终止生成省算力

### 7. 30 秒速答

- 一句话核心：客户端超时盲重试在 LLM 上放大流量 3×，prefill 不可中断让代价更大。
- 关键机制：长 prompt 接近 timeout 阈值 + 重试不能复用 KV + 没有 hedging，雪崩成型。
- 修复 / 防御：禁默认重试 + admission control + 长短 prompt 流量隔离 + 客户端 jitter+circuit breaker。
- 面试加分关键词：Retry-After、hedged request、TTFT/TPOT SLO、idle connection drain。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"QPS 3× 但用户没多打"的雪崩特征？
- [ ] 你能不能解释 LLM 重试比 REST API 更危险的三个原因？
- [ ] 你能不能举一个类似场景（如 streaming 断连服务端继续生成）？
- [ ] 你能不能说出网关重试规范化 + admission control 的防御组合？

---

## Q19. FP8 量化推理数值漂移：上线后某些任务准确率掉 5%

> 🔴 专家 · 整体打分掉 0.3pp 通过了 PR，结果数学题和代码任务掉 4–6pp 被用户骂回来——FP8 的 outlier 不是均匀分布的，calibration 集没覆盖就会精准踩雷。

### 1. 现象

某模型从 BF16 量化到 FP8 (E4M3) 推理上线，整体 throughput 提升 1.8×，BLEU / MMLU 整体打分下降 0.3 pp 在可接受范围。但用户反馈数学题与代码生成任务正确率明显下降，离线复现显示这两类掉 4–6 pp。

### 2. 根因层级原理

FP8 (E4M3) 表示范围窄（max ≈ 448），需要 per-tensor 或 per-channel scale 量化。问题在 (a) calibration 数据集若主要是普通对话，没有覆盖数学/代码场景的极端 activation 分布，scale 选偏；(b) attention 中间 softmax 后 logits 的 outlier 在 FP8 下被 saturate；(c) 累加用 FP8 而非 FP32 累加器导致误差累积；(d) GEMM A/B 用 FP8 但 epilog 没用更高精度，多层叠加误差非线性放大。结果是对噪声敏感的任务（精确推理类）准确率塌方。

### 3. 定位手段与命令

- 分任务 eval：MMLU 各 subject、HumanEval、GSM8K 分别打分对比 BF16
- 中间层 activation distribution 抓样：layer-wise 看 FP8 与 BF16 的 cosine similarity，找漂移最严重的层
- `transformer_engine` profile，看哪些层用了 FP8 哪些 fallback BF16
- A/B 灰度：5% 流量 FP8、95% BF16，对每类请求分桶比对结果

### 4. 修复路径与权衡

- **短期**：回滚到 BF16；或部分关键层（attention softmax、最后 lm_head）保留 BF16 走 mixed-precision
- **中期**：扩展 calibration 集覆盖 math/code/long-context；用 SmoothQuant / AWQ 类技术处理 outlier
- **长期**：上 FP8 per-channel scale + Hadamard rotation outlier suppression；切到 NVFP4 / MXFP4（Blackwell 起精度更好）
- 权衡：mixed precision 工程复杂度高；扩展 calibration 集需要数据；新硬件需要换机；per-channel 比 per-tensor 慢但更准

### 5. 同类变体 / 易错点

- 变体：INT4 / INT8 量化的零点 / 对称量化选择不当
- 变体：长上下文 RoPE 位置编码下 attention score 在 FP8 下 underflow
- 易错：以为 "throughput 涨了 + 整体 score 没怎么掉" 就 OK，忽视分任务方差
- 追问："E4M3 与 E5M2 怎么选？"→ E4M3 forward（动态范围窄但精度高）、E5M2 backward（gradient 范围大）

### 6. 预防机制

- 量化上线必须分任务 eval（不只看综合分数），任一类掉 > 1pp 阻断上线
- Calibration 集 release 流程：数据多样性 audit
- 灰度比例≥ 7 天观察期，含异常样本回归
- 监控：在线 A/B 持续打分（影子流量同时跑 BF16 ref）
- 长期 trend：每 release 跑全套 eval suite 进数据库做 trend，回归即报警

### 7. 30 秒速答

- 一句话核心：FP8 整体打分通过但数学 / 代码任务掉 4–6pp，calibration 没覆盖 outlier 分布。
- 关键机制：E4M3 表示范围窄 + 关键层 outlier saturate + 累加用 FP8 精度损失叠加。
- 修复 / 防御：分任务 eval 阻断上线 + per-channel scale + SmoothQuant/AWQ + 关键层 BF16 mixed。
- 面试加分关键词：transformer_engine、E4M3 vs E5M2、Hadamard rotation、NVFP4/MXFP4。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"综合分通过但分任务掉 5%"的核心症状？
- [ ] 你能不能解释 FP8 outlier saturate 与累加精度的因果？
- [ ] 你能不能举一个类似场景（如 RoPE 长上下文 attention score underflow）？
- [ ] 你能不能说出 calibration、per-channel scale、mixed precision 三种防御？

---

## Q20. Autoscaler 抖动：流量小波动触发扩缩容震荡

> 🔴 专家 · HPA 按 GPU util 每 10 分钟扩缩一次，cold start 90s——LLM 推理副本不是 stateless 微服务，套通用 HPA 一定震荡，得换 `pending_tokens` 这类业务指标。

### 1. 现象

推理服务接 K8s HPA 按 GPU utilization 自动扩缩容。流量平稳时实例数应该稳定在 8 个，但实际监控显示在 6–12 之间频繁震荡，每 10 分钟一次扩缩，每次扩容耗时 90s（vLLM 加载模型慢），用户感受到周期性 p99 抖动。

### 2. 根因层级原理

HPA 默认按瞬时指标 + cooldown 决策。LLM 推理特殊：(a) 实例 cold start 90s（模型加载 + cuda graph 编译 + warmup），HPA cooldown 默认 5 分钟太短；(b) GPU utilization 是噪声大的指标——continuous batching 下利用率波动大但不代表过载；(c) 缩容时正在跑的请求被打断或要 graceful drain，drain 时间又比扩容长；(d) 多个副本同时被踢导致剩下的瞬时过载，触发扩容——形成震荡。

### 3. 定位手段与命令

- HPA 决策日志：`kubectl describe hpa <name>` + `kubectl get events`
- Prometheus 看决策指标 trend：`avg(gpu_utilization)` 与 `replicas` 时间对齐
- 模型加载耗时分布：`vllm:engine_load_seconds`
- 缩容时已存在的请求处理延迟：`graceful_drain_seconds` p99

### 4. 修复路径与权衡

- **短期**：调大 HPA cooldown（`scaleDown.stabilizationWindowSeconds=600`）；用更平滑的指标（5 分钟移动平均）
- **中期**：换 KEDA + 业务级指标（如 `queue_depth` 或 `pending_tokens`）做扩缩；按业务时段做 schedule-based scaling
- **长期**：predictive scaling（基于 traffic 时间序列预测）；warm pool 保留 N 个 ready-but-idle 副本快速激活；推理服务设计成快速冷启动（lazy load + cuda graph cache）
- 权衡：cooldown 大反应慢；KEDA 引入额外组件；warm pool 浪费成本；predictive 需要数据科学投入

### 5. 同类变体 / 易错点

- 变体：基于 request rate 扩缩对长尾 prompt 不公平（长 prompt 算力远大于平均）
- 变体：GPU MIG 切片 autoscaler 无法区分 slice 类型
- 易错：以为利用率高就需要扩容——其实 LLM 满载 90% 是 healthy
- 追问："为什么 cold start 这么慢？"→ 几十 GB 权重 IO + cuda graph capture + warmup forward

### 6. 预防机制

- 扩缩策略走 schedule + reactive 双轨：固定时段最低 baseline + 突发反应
- 自定义业务指标：`pending_tokens` / `tokens_per_second_capacity` 比 GPU util 准
- 模型 cold start 优化：常驻 warm pool ≥ 2，永远不缩到 0
- 监控：scale event 频率 trend，每小时 > 3 次报警
- 周度演练：人造流量波形验证扩缩稳定性

### 7. 30 秒速答

- 一句话核心：HPA 按 GPU util 套通用模板必然震荡，cold start 90s 远长于决策窗口。
- 关键机制：噪声大的瞬时指标 + 短 cooldown + drain 慢 + 缩容后剩余瞬时过载触发再扩。
- 修复 / 防御：KEDA + pending_tokens 业务指标 + 大 cooldown + warm pool 不缩到 0。
- 面试加分关键词：stabilizationWindow、predictive scaling、cuda graph cache、graceful drain。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"副本 6–12 之间每 10 分钟震荡"的核心症状？
- [ ] 你能不能解释 GPU util 为何不是 LLM 推理的好指标？
- [ ] 你能不能举一个类似场景（如 MIG 切片 autoscaler 失语义）？
- [ ] 你能不能说出 schedule + reactive 双轨 + warm pool 的防御组合？

---

## Q21. Spot 抢占雪崩：高优任务突发挤掉 100 个低优任务连锁失败

> 🔴 专家 · 推理流量突发，调度器一口气 preempt 100+ 训练任务，30s graceful 远不够 350GB ckpt 落盘——共享 ML 平台的"挤占即重伤"，是 SRE 与训练侧最容易踩的雷。

### 1. 现象

某共享 ML 平台同时跑训练（低优 spot）与推理（高优 reserved）。某次推理流量突发触发集群扩容，调度器为腾资源批量 preempt 低优任务。100+ 训练任务同时收到 SIGTERM，30 秒后被 SIGKILL，大部分没完成 graceful checkpoint。重新调度时又集中冲击调度器，集群 30 分钟内调度延迟从 2s 涨到 5 分钟。

### 2. 根因层级原理

调度器 preempt 决策通常按"释放最少资源满足请求"逻辑，但实现常见缺陷：(a) 没有 preempt rate limit，瞬间释放数百节点；(b) graceful period 默认值（30s）对 LLM 训练 ckpt 不够（保存 350 GB ckpt 需要分钟级）；(c) 被 preempt 的任务自动 requeue 全部进同一队列，竞争同样资源，formed thundering herd；(d) 集群没有缓冲池，preempt 完资源立即被新任务占满，没给原任务回来的机会。

### 3. 定位手段与命令

- 调度器日志（SLURM `slurmctld.log` / K8s `kube-scheduler` log）grep `preempt`
- 时间序列：被 preempt 任务数 / 重调度 latency / queue depth 三条线对齐
- `sacct -X --state=PREEMPTED --starttime=$T` 看具体事件
- 节点视角：`scontrol show node <n>` 看 reason 与 reboot count
- 训练 log：被 preempt 任务有没有完成 ckpt（看 `last_save_step` vs 退出 step）

### 4. 修复路径与权衡

- **短期**：调度器配 preempt rate limit（每分钟最多 N 节点）；增大 graceful period 到 5–10 分钟匹配 ckpt 时长
- **中期**：训练框架支持 SIGTERM 触发"快照后退出"逻辑，监听并尽量 best-effort save；requeue 走 priority aging 防新任务挤老任务
- **长期**：预留 buffer pool（5–10% 节点不调度新任务），preempt 时让原任务有缓冲；分集群部署（推理与训练物理隔离）
- 权衡：rate limit 让推理扩容慢；buffer pool 浪费 5–10% 资源；隔离丧失资源池化效率

### 5. 同类变体 / 易错点

- 变体：spot reclaim 通知时长（AWS 120s、GCP/Azure 30s）远短于 ckpt 时间
- 变体：MPS / MIG 共享 GPU 上 preempt 行为更复杂
- 易错：以为"会自动重排"就 OK——重排成功不等于无损失
- 追问："为什么 K8s preemption 比 SLURM 简单？"→ K8s pod 短暂、SLURM job 长且 stateful

### 6. 预防机制

- 训练框架默认每 N step + SIGTERM 触发都尝试 save；`atexit` 钩子保底
- launcher 的 entrypoint trap SIGTERM 转发给训练进程，给 graceful 时间
- 调度器配置必检项：preempt rate limit、graceful period ≥ 5 min、requeue priority aging
- 监控：preempt 频次 / ckpt 完成率 / requeue 后 step 0 重训量
- 季度 chaos：人造突发推理流量验证训练任务存活率

### 7. 30 秒速答

- 一句话核心：高优突发挤掉一批低优训练，30s graceful 远不够 350GB ckpt 保存。
- 关键机制：无 preempt rate limit + graceful period 短 + requeue thundering herd。
- 修复 / 防御：rate limit + 5–10 min graceful + buffer pool + SIGTERM 触发快照后退出。
- 面试加分关键词：priority aging、physical 隔离、atexit hook、spot reclaim 通知。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"100 个训练任务连锁失败"的核心症状？
- [ ] 你能不能解释 30s graceful 与 LLM ckpt 时长不匹配的根因？
- [ ] 你能不能举一个类似场景（如 AWS spot reclaim 120s 通知）？
- [ ] 你能不能说出 rate limit + graceful period + buffer pool 的防御组合？

---

## Q22. 误发 SIGTERM 致整集群训练任务连锁退出：cleanup script bug 复盘

> 🔴 专家 · 一行看似无害的 `pkill -f 'tmp_'` 在周末凌晨把全集群训练干掉——这种"运维好心做坏事"的事故，几乎都是 destructive 命令没 dry-run、没限定 uid 造成的。

### 1. 现象

某周末凌晨，平台运维提交一个看似无害的 cleanup PR：定期清理 `/tmp` 下过期文件。脚本通过 systemd timer 在每节点跑。脚本里 `pkill -TERM -f 'tmp_'` 这一行，意外把训练进程主目录在 `/tmp_train_xxx` 的 ~200 个 rank 全部杀掉。集群所有 LLM 训练任务退出，损失 ~6 小时算力。

### 2. 根因层级原理

`pkill -f` 匹配的是完整命令行（`/proc/PID/cmdline`）而非可执行名，正则 `tmp_` 会匹配任何 cmdline 含此字符串的进程。训练进程的工作目录或某 arg 含 `/tmp_train_xxx` 路径就被命中。这类问题的本质是：(a) destructive 命令的匹配范围审视不足；(b) 没有 dry-run / shellcheck / staging；(c) cleanup 脚本以 root 跑权限太大；(d) systemd timer 全节点同时执行，没分批。

### 3. 定位手段与命令

- `journalctl -u <cleanup-timer>` 看脚本执行历史
- 训练任务 log：所有 rank 同时收到 SIGTERM 是不正常的，正常 preempt 不会这么整齐
- `auditd` / `sysdig` 审计哪个进程发了 SIGTERM
- 时间序列对齐：cleanup script 执行时间 vs 训练任务退出时间
- `ps -ef` 模拟 + `pgrep -f 'tmp_'` 看会命中哪些进程（事后 dry-run）

### 4. 修复路径与权衡

- **短期**：禁用 cleanup timer；逐节点重启训练任务恢复
- **中期**：把 `pkill -f` 换成更精确的匹配（按 uid + cmdline 组合）；加 `--dry-run` 默认行为；改用 `find -mtime -delete` 处理文件而不杀进程
- **长期**：所有 destructive ops 必走 review + canary（先 1 个节点跑）+ 监控（杀进程数 metric，超阈值自动停）；建立 platform op runbook 与同行 review SOP
- 权衡：精确匹配可能漏杀；review SOP 拖慢运维节奏；canary 增加复杂度

### 5. 同类变体 / 易错点

- 变体：`rm -rf` 在变量为空时变成 `rm -rf /` 经典脚本灾难
- 变体：`find -exec rm` 与 `xargs rm` 的并发风险
- 易错：`pkill` 不加 `-u <uid>` 匹配所有用户的进程
- 追问："为什么 K8s 的 preStop hook 不会有这种问题？"→ K8s 通过 PID namespace 隔离

### 6. 预防机制

- 平台 op SOP：所有"批量"命令必须 (a) 默认 dry-run (b) 限定 uid/scope (c) 经同行 review (d) 先 canary 一个节点
- shellcheck / 静态扫描所有运维脚本进 CI
- audit 监控：每节点每分钟被 kill 的进程数 metric，异常突增报警
- 训练任务进程加 sentinel marker（特定环境变量），cleanup 脚本必须排除带此 marker 的
- 演练：故障注入演习（chaos monkey）每月跑一次，包含"误发 SIGTERM"场景

### 7. 30 秒速答

- 一句话核心：cleanup 脚本 `pkill -f 'tmp_'` 把训练进程当临时文件杀了。
- 关键机制：destructive 命令匹配范围审视不足 + 全节点同时执行 + root 权限 + 无 dry-run。
- 修复 / 防御：默认 dry-run + 限定 uid + 同行 review + canary 单节点 + 杀进程数监控。
- 面试加分关键词：systemd timer、auditd、sentinel marker、shellcheck CI。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"运维好心做坏事杀整集群"的核心症状？
- [ ] 你能不能解释 `pkill -f` 匹配规则与误伤路径？
- [ ] 你能不能举一个类似场景（如 `rm -rf $VAR` 变量为空灾难）？
- [ ] 你能不能说出 dry-run + uid 限定 + canary 三层防御？

---

## Q23. SLURM queue 饥饿：长任务永远排不上，运维不察觉

> 🟡 进阶 · "我 64 卡 7 天的任务排了 5 天还没起，但小任务一直在跑"——backfill window 太短 + 没 priority aging，长任务就被永远见缝插针的小任务压在队尾。

### 1. 现象

某 SLURM 集群设有 `short` / `long` / `priority` 三个 partition，研究人员反馈提交 64 卡 7 天的训练任务在 `long` 队列等了 5 天还没起，但小任务一直在跑。排查发现 backfill scheduler 配置不当，长任务永远抢不到时间窗。

### 2. 根因层级原理

SLURM backfill 调度器原理：在不影响最高优先级任务预定开始时间的前提下，把 idle 资源分给低优短任务利用率。问题是 (a) `backfill_window` 设得短（默认 1440 分钟，即 24h），看不到 7 天后的可用窗口，没法提前为长任务预留；(b) 短任务 walltime 估算不准（用户填 24h 实际跑 1h），调度器以保守估计排队，反而让短任务连续抢节点；(c) 长任务 priority 没有 aging（等待时间不增加优先级），永远被新提交压制；(d) `MaxNodes` 不限制，单用户大任务卡死 partition。

### 3. 定位手段与命令

- `sprio -j <jobid>` 看任务优先级各 component
- `scontrol show config | grep -i backfill` 看 backfill 配置
- `sshare -A <account>` 看 fairshare 状态
- `sacct --start=$T --format=JobID,Submit,Start,Elapsed,TimeLimit` 统计 wait time 分布
- 自定义脚本：每 partition 长任务的 wait time p99 trend

### 4. 修复路径与权衡

- **短期**：手动 `scontrol update job <id> Priority=...` 提高被饿死任务优先级
- **中期**：调 `bf_window` 到 7 天 + `bf_max_job_test` 增加；启用 priority aging（`PriorityWeightAge`）
- **长期**：拆 partition 按 walltime 分级（短/中/长）物理隔离；引入 reservation 给关键长任务预留；fairshare account 限制单用户/项目消耗
- 权衡：bf_window 大调度器开销大；partition 拆分降低池化效率；reservation 硬占用资源

### 5. 同类变体 / 易错点

- 变体：K8s gang scheduling / Volcano / Kueue 也有类似 long-job 饥饿
- 变体：preemptible 任务永远被抢回到队尾形成 livelock
- 易错：以为 "FIFO 公平"——SLURM 默认是按 priority 不是 FIFO
- 追问："backfill 与 priority scheduling 区别？"→ 一个见缝插针不影响主序，一个绝对优先

### 6. 预防机制

- 监控：每 partition 任务 wait time p99，超阈值（如 long > 24h）告警
- 调度配置作为 IaC（Ansible/Terraform 管理 slurm.conf），变更走 review
- 用户教育：walltime 不要膨胀填（影响 backfill 窗口估计）
- 周报：饥饿任务列表给平台管理员看
- 演练：人为提交极长任务验证 backfill 行为

### 7. 30 秒速答

- 一句话核心：长任务排了 5 天起不来，backfill window 太短 + 没 priority aging。
- 关键机制：backfill 看不到远期窗口 + 短任务 walltime 估算膨胀 + 长任务无优先级老化。
- 修复 / 防御：扩 bf_window + 启用 PriorityWeightAge + 按 walltime 拆 partition + fairshare 配额。
- 面试加分关键词：sprio、bf_max_job_test、Volcano gang scheduling、reservation。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"小任务在跑长任务永远排不上"的核心症状？
- [ ] 你能不能解释 backfill 调度看不到远期窗口的影响？
- [ ] 你能不能举一个类似场景（如 K8s Kueue 长任务饥饿）？
- [ ] 你能不能说出 bf_window + priority aging + partition 拆分的防御组合？

---

## Q24. GPU 资源声明不匹配：作业拿到 4 卡却只能用 2 卡

> 🟡 进阶 · `--gres=gpu:4` 申请到了，节点上 `nvidia-smi` 也是 4 张，但任务里 `CUDA_VISIBLE_DEVICES` 只有 2——SLURM gres、cgroup、CVD、容器 runtime 这条链上任一环出错都长这样。

### 1. 现象

某用户 SLURM 提交脚本写 `--gres=gpu:4`，启动后训练 launcher 只检测到 2 张 GPU，训练 hang 在 NCCL 初始化（rank 数与 GPU 数不一致）。`nvidia-smi` 在节点上显示 4 张卡，但 `CUDA_VISIBLE_DEVICES` 只暴露 2 张。

### 2. 根因层级原理

GPU 资源声明涉及多层：(a) SLURM gres 配置（`gres.conf` 描述节点有几张 GPU）；(b) cgroup / device cgroup 限制实际可见设备；(c) `CUDA_VISIBLE_DEVICES` 由 prolog 设置；(d) 容器场景叠加 NVIDIA Container Runtime 注入。任一层失配就出错。常见原因：节点 `gres.conf` 标错（说有 8 卡但实际 4 卡正常 + 4 卡 RMA 中），或 cgroup 配置 bug 漏暴露设备，或 prolog 脚本 set `CUDA_VISIBLE_DEVICES` 的 bash 解析错误。

### 3. 定位手段与命令

- 节点视角：`scontrol show node <n>` 看 gres 报告 vs 实际
- `cat /proc/$PID/status | grep Cpus_allowed` + `cat /sys/fs/cgroup/devices/.../devices.list` 看 cgroup 限制
- 任务内 `echo $CUDA_VISIBLE_DEVICES` + `nvidia-smi -L`
- SLURM prolog log：`/var/log/slurm/prolog.log` 看 set 过程
- `dcgmi diag -r 1` 看 GPU 健康，是否有卡 fault 被自动隔离

### 4. 修复路径与权衡

- **短期**：手动 `unset CUDA_VISIBLE_DEVICES` 或显式 set 全 0,1,2,3 验证；在另一节点重排
- **中期**：修 `gres.conf` 与实际硬件对齐；fix prolog 解析；统一容器镜像里 NVIDIA runtime 配置
- **长期**：节点入集群 burn-in 包含 GPU count check；SLURM gres 自动从 `nvidia-smi -L` 派生而非手填
- 权衡：自动派生需要节点稳定性；burn-in 增加上线时间

### 5. 同类变体 / 易错点

- 变体：MIG 模式下 SLURM 看到的是 MIG 实例数而非物理卡数
- 变体：K8s Device Plugin 注入逻辑与节点真实状态不一致
- 易错：以为 `nvidia-smi` 看到 4 张就是 4 张可用——cgroup 可能限制了可见性
- 追问："`CUDA_VISIBLE_DEVICES=0,1,2,3` 与 cgroup 谁优先？"→ cgroup 是底层硬限，CVD 是用户态过滤

### 6. 预防机制

- 节点入集群：burn-in 跑 `nvidia-smi -L` 与 `gres.conf` diff，不一致禁止 enable
- 训练 launcher 启动第一步 sanity check：`torch.cuda.device_count()` == `SLURM_GPUS_ON_NODE` 否则 fail-fast
- 监控：每节点 `gres_reported_gpus` vs `actual_visible_gpus` metric
- prolog 脚本进 CI（shellcheck + 集成测试）
- 节点 image 构建包含 NVIDIA runtime 版本固定

### 7. 30 秒速答

- 一句话核心：申请 4 卡只拿到 2 卡，SLURM gres / cgroup / CVD / runtime 链路某环漂了。
- 关键机制：多层 GPU 暴露机制相互独立，任一层失配即失语义。
- 修复 / 防御：launcher sanity check `device_count == SLURM_GPUS_ON_NODE` + 节点入集群 burn-in。
- 面试加分关键词：gres.conf 自派生、prolog log、NVIDIA Container Runtime、MIG 实例数。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"申请 4 卡只拿到 2 卡"的核心症状？
- [ ] 你能不能解释 gres、cgroup、CVD、runtime 四层之间的关系？
- [ ] 你能不能举一个类似场景（如 K8s Device Plugin 与节点状态不一致）？
- [ ] 你能不能说出 launcher fail-fast + burn-in + gres 自派生的防御组合？

---

## Q25. 节点失联检测延迟：30 分钟才发现节点死机训练全程在等

> 🟡 进阶 · 凌晨 3 点节点 kernel panic，调度器 15 分钟才标 down，NCCL watchdog 默认 30 分钟才触发——多层超时叠加，单次事故就吃掉半小时算力，每周来 2–3 次。

### 1. 现象

某 256 卡训练任务，某节点凌晨 3 点死机（kernel panic），整任务挂在 NCCL collective 等这个 rank。但调度器健康检查间隔 5 分钟、3 次失败才标 down，意味着 15+ 分钟才 mark down，再加上任务 reschedule 启动 10 分钟，总损失 30+ 分钟。每周这种事故发生 2–3 次。

### 2. 根因层级原理

健康检测延迟来源：(a) 默认 `SlurmdTimeout=300s`、`UnkillableStepTimeout=60s` 等过于保守；(b) ICMP ping 在 kernel panic 但网卡 NIC 还活的场景下仍能响应（伪健康）；(c) 节点维度健康只看 daemon 进程在不在，看不到 GPU 已经卡死；(d) 训练任务自身的 NCCL watchdog 默认 30 分钟 timeout，远大于实际可接受范围；(e) 没有跨层联动：调度器、监控、训练任务三方各自判断不汇总。

### 3. 定位手段与命令

- SLURM `slurmctld.log` 看 node down 的判定时刻
- 节点最后心跳：`scontrol show node <n>` 的 `BootTime` / `LastBusyTime`
- 训练 log：`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` 触发的 fingerprint
- 监控系统：节点 lost 事件与 hang 开始的时间差
- `dmesg` 抓 kernel panic 时间戳（需要 console log 或 IPMI 抓取）

### 4. 修复路径与权衡

- **短期**：调小 SLURM timeout（`SlurmdTimeout=120`）；调小 NCCL watchdog 到 5 分钟
- **中期**：多层健康联动——监控发现节点 GPU metric 5 分钟无更新主动 drain；训练任务嵌入 5 秒级心跳到 Prometheus，缺心跳即 raise
- **长期**：BMC / IPMI 带外监控（kernel panic 也能看到），自动 reset；redundant 节点（spare）随时顶替
- 权衡：timeout 小误报多（短网络抖动也触发）；BMC 集成复杂度高；redundant 节点成本

### 5. 同类变体 / 易错点

- 变体：节点假死（CPU 卡 100% 但响应 ping）比真死机更难检测
- 变体：网络分区（节点活但跨节点不通）调度器与节点都自认为正常
- 易错：以为 "ping 通就是活的"——ping 由 NIC 处理不依赖 CPU
- 追问："Kubernetes node not ready 怎么检测？"→ kubelet 心跳 + node-problem-detector

### 6. 预防机制

- 监控立体化：daemon 健康 + GPU metric 心跳 + 训练任务心跳 + BMC 带外，任一层异常都触发 drain
- 训练任务嵌入 cheap heartbeat（每 5 秒上报 step counter），平台侧 30 秒无更新即报警
- BMC / IPMI 接入监控系统（OpenBMC、Redfish）
- 节点 watchdog 自动 reset 死机节点（kernel softdog）
- 月度演练：人为 SysRq trigger panic 验证检测链路 e2e

### 7. 30 秒速答

- 一句话核心：节点 kernel panic 30 分钟后才被发现，单次事故吃掉半小时算力。
- 关键机制：多层 timeout 叠加（SlurmdTimeout + NCCL watchdog + ping 伪健康）。
- 修复 / 防御：5 秒级心跳 + GPU metric 缺失自动 drain + BMC 带外监控 + softdog 自动 reset。
- 面试加分关键词：TORCH_NCCL_HEARTBEAT_TIMEOUT、IPMI/Redfish、node-problem-detector、网络分区。

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清"30 分钟才发现节点死机"的核心症状？
- [ ] 你能不能解释多层 timeout 叠加导致的检测延迟？
- [ ] 你能不能举一个类似场景（如节点假死 ping 通 CPU 卡死）？
- [ ] 你能不能说出 daemon + GPU metric + 训练心跳 + BMC 四层联动的防御立体化？

---

## Q26. 比较：训练 NaN、通信 hang、推理 OOM 三类典型事故的定位路径与防御策略

> 🧭 综合 · 同样是"集群挂了"的报警，三类事故的指纹完全不同。能在脑里画出"现象 → 根因层 → 工具 → 防御"的对比矩阵，是 oncall 跨场景跳转的核心能力。

### 1. 现象

三类事故的报警信号差异巨大：
- 训练 NaN：loss 突然变 NaN / Inf，grad norm 飙升，监控曲线断崖式或台阶式跳变；rank 之间可能不同步（FSDP 下某 rank 先变 NaN）。
- 通信 hang：GPU SM 100% 但功耗低、不出 step、不报错；step counter 静止；rank 之间齐刷刷卡住。
- 推理 OOM：p99 延迟飙升 + KV cache usage 100% + preempt rate 高 + 5xx 涌现；进程不挂但服务降级。

三类的共同特征是"看上去挂了"但根因层完全不同。

### 2. 根因层级原理

| 维度 | 训练 NaN | 通信 hang | 推理 OOM |
|------|----------|-----------|----------|
| 主因层 | 数值 / 算法 / 数据 | 网络 / NCCL / 拓扑 | 资源调度 / 内存 |
| 触发尺度 | 单 step / 单 layer | rank / 节点 | 服务 / 池 |
| 时间特征 | 离散事件 | 持续 hang | 高负载持续 |
| 进程态 | 可能继续跑出 NaN 传播 | spin-wait 看似在跑 | 不挂只是退化 |
| 典型根因 | FP16 溢出、bad sample、学习率过大 | rank 失联、NIC flap、版本漂移 | 长 prompt、KV 泄漏、calibration 不对 |

### 3. 定位手段与命令对比

- 训练 NaN：`anomaly detection`、`torch.isnan().any()` 嵌入 step、按层 weight/grad norm trace、bad batch 回放
- 通信 hang：`py-spy dump`、`NCCL_DEBUG=INFO`、flight recorder、heartbeat 缺失检测、`dcgmi diag -r 3`
- 推理 OOM：`vllm:gpu_cache_usage`、`num_preempted_total`、`free_block_floor` trend、prompt 长度分布

共性工具：Prometheus 时间序列对齐、log aggregation、`/metrics` endpoint。差异工具：训练偏框架内 hook，通信偏系统层 trace，推理偏业务 metric。

### 4. 修复路径与权衡

- 训练 NaN：止血是回滚最近 ckpt + 跳过坏样本 / 降 lr / 切回 fp32；长期是 loss scaling + grad clipping + 数据清洗。
- 通信 hang：止血是定位失联 rank 重启节点；长期是 watchdog + heartbeat + image 治理。
- 推理 OOM：止血是网关限长 + 限并发 + 扩容；长期是 chunked prefill + 长短分流 + KV 卸载。

权衡上：训练复盘可"慢"（离线 bisect 即可），通信和推理必须秒级响应；训练 NaN 通常可恢复（回滚 ckpt），通信 hang 损失整段算力，推理 OOM 直接影响用户体验。

### 5. 同类变体 / 易错点

- 易错：把 "NCCL hang" 和 "training NaN propagated" 混淆——NaN 也可能让 collective 失败，但定位思路不同
- 易错：推理 OOM 看 nvidia-smi 没满（KV 是预分配）就放过
- 易错：把"all reduce 慢"当 hang，但其实是带宽劣化（busbw 退化 vs 完全卡住）
- 追问："这三类事故哪个最容易自动化恢复？"→ 训练 NaN（回滚 ckpt + 跳样本可脚本化）；推理 OOM 次之；通信 hang 最难（要定位单点）

### 6. 预防机制

立体化防御矩阵：
- 训练侧：loss watchdog + grad norm 红线 + dirty sample skip + checkpoint cadence
- 通信侧：launcher 注入 watchdog 参数 + nccl-tests pre-flight + image baseline + heartbeat
- 推理侧：prompt 长度限流 + KV 池容量规划 + autoscaler 业务指标 + chaos test
- 公共：跨层报警关联（同一 job 的训练 hang 与节点 ECC 关联看），平台级 SLO 看板

### 7. 30 秒速答

- 一句话核心：三类事故根因层不同——训练在数值/数据、通信在网络/拓扑、推理在资源/调度。
- 关键机制：训练是离散异常事件、通信是 spin-wait 整齐卡住、推理是持续退化 + 雪崩。
- 修复 / 防御：训练靠 anomaly detect + ckpt 回滚，通信靠 watchdog + heartbeat，推理靠限流 + 容量规划。
- 面试加分关键词：anomaly detection、flight recorder、admission control、立体化 SLO。

### 8. 自测 checklist

- [ ] 你能不能在白板上画出这三类事故的"现象 → 根因 → 工具 → 防御"矩阵？
- [ ] 你能不能解释为什么 nvidia-smi 在推理 OOM 中常常显示显存"够"？
- [ ] 你能不能举一个跨类事故的例子（如 NCCL hang 实际由 NaN 传播触发）？
- [ ] 你能不能说出三类事故各自最该上的一个监控指标？

---

## Q27. 场景：周五下午 5 点推理服务 P99 飙升 + region 5xx，30 分钟决策路径

> 🧭 综合 · 周五下午、要下班、用户投诉涌进来——这种"时间压力 + 信息缺失 + 决策可逆性差"的场景，最考验 oncall 的脑回路和团队协作。

### 1. 现象

oncall 收到告警：推理服务 P99 时延从 800 ms 飙到 12 s，us-west region 5xx 占比 8%（其它 region 正常），客户投诉开始进来。日程上 30 分钟后是周五全员下班；on-call 当前只有 2 个人在班，PM 在外地，资深架构师在飞机上。

### 2. 根因层级原理（30 分钟内可考虑的可能性）

按"高频 × 容易快速验证"排序：
1. 单 region 流量异常（被刷 / 突发长 prompt）→ 看 QPS + prompt 长度分布
2. 单 region 部分节点 GPU 健康问题（ECC、降频、链路）→ 看节点 health metric
3. 最近一次发布灰度异常（FP8 量化、调度器配置、Image 升级）→ 看发布 timeline
4. 上游依赖故障（向量库、模型路由、auth 服务）→ 看依赖 SLO
5. 跨 region 路由策略问题（traffic 错误分流）→ 看路由 weight 与实际流量
6. 底层基础设施（云厂商网络 / 存储）→ 看 cloud provider status

### 3. 定位手段与命令（建议时间盒）

- T+0 到 T+3 min：拉团队 war room（即使 2 人也建一个 channel），同步关键指标 dashboard 链接，明确"决策窗口 30 分钟"
- T+3 到 T+10 min：并行查 5 件事（每人 2.5 件）——
  - QPS / 长 prompt 比例 / 错误率分桶
  - 最近 24h 发布历史 + 灰度比例
  - 节点健康（DCGM + ibstat）
  - 上游依赖延迟
  - 路由策略最近改动
- T+10 到 T+15 min：根据上面拿到的证据收敛到 1–2 个 hypothesis
- T+15 到 T+25 min：执行"最可逆"动作先验证（回滚最近发布 / 切流到健康 region / 限流）
- T+25 到 T+30 min：评估是否解决；若没解决决定升级到 P0 拉资深人

### 4. 修复路径与权衡

按"风险 × 时效"分级动作：
- 立即可逆（低风险）：流量切走 us-west（traffic shift 到 us-east），网关限 prompt 长度，扩容副本
- 中等风险：回滚最近 release（先停止灰度推进，回到上一稳定版）；这要求发布有 1-click rollback
- 高风险：禁用某个功能模块（FP8 推理 → 切回 BF16），需要 PM 确认影响范围

决策原则：周五下午先止血（即使是粗暴 rollback），周一再 root cause 复盘；明确不在"无监督"状态下推不可逆变更。

### 5. 同类变体 / 易错点

- 易错：第一反应"加机器扩容"——雪崩中扩容也会被瞬间打满，应先限流再扩
- 易错：试图在线 debug 拉栈、看 trace 拖延决策——周五下午先回滚再 root cause
- 易错：不通知用户 / 不开 status page——透明度差让客户更不满
- 追问："如果回滚后还没好怎么办？"→ 升级到全员 incident，按 incident command system 走

### 6. 预防机制

- Runbook：oncall 必读"P0 30 分钟决策树"文档，明确止血优先 > root cause
- 一键 rollback：每个发布前自动 capture 稳定版 image + 配置 snapshot
- Region 隔离：流量层支持 1 min 内 traffic shift（如 weighted DNS / global load balancer）
- 上下班交接：周五下午高风险变更冻结窗口（"freeze Friday after 3pm"）
- 演练：季度 chaos 包括"周五下午雪崩"场景，验证 oncall 反应速度

### 7. 30 秒速答

- 一句话核心：周五下午先止血回滚，root cause 周一查；时间盒 30 分钟分三段。
- 关键机制：war room → 并行 5 路查 → 收敛 1-2 hypothesis → 可逆动作先做。
- 修复 / 防御：traffic shift + 限流 + 一键 rollback + Friday freeze + 30 分钟决策树 runbook。
- 面试加分关键词：incident command、blast radius、可逆性优先、status page、war room。

### 8. 自测 checklist

- [ ] 你能不能说出"止血先于 root cause"这一核心原则及其适用边界？
- [ ] 你能不能列出 30 分钟决策窗口里的 3 段时间盒分配？
- [ ] 你能不能举一个类似场景（如除夕夜 / 双十一 / 重大客户演示前夕）？
- [ ] 你能不能说出至少 3 个一键可逆动作（traffic shift / rollback / 限流）？

---

## Q28. 估算：万卡训练集群在 5% 年故障率下，端到端有效训练时间占比

> 🧭 综合 · 这种题在面试里是"看你有没有概率直觉 + 工程常识 + 算账能力"。结论比过程更有趣——万卡 30 天训练能拿到 75% 有效时间已经算运维很强。

### 1. 现象

某万卡集群（10000 个 GPU，按 8 卡/节点算 1250 个节点）跑一个 30 天训练任务。已知：单节点年故障率 5%，checkpoint 间隔 30 分钟，故障检测 + 调度恢复时间 15 分钟（含恢复 ckpt 时间）。问：端到端"有效训练时间 / 总时长"大约是多少？

### 2. 根因层级原理 / 估算模型

设：
- 节点年故障率 p = 5%/year/node = 0.05/8760h ≈ 5.7×10⁻⁶ /node/h
- 集群节点数 N = 1250
- 集群整体故障率 λ = N × p = 1250 × 5.7×10⁻⁶ ≈ 7.13×10⁻³ /h ≈ 0.171/day
- 即"集群层面平均每天约 0.17 次故障"，30 天约 5.1 次

每次故障损失时间 = 检测恢复时间 + 平均回滚损失
- 检测恢复 = 15 min
- 平均回滚 = ckpt 间隔的一半 = 15 min（假设故障均匀分布在 ckpt 间隔内）
- 单次故障损失 ≈ 30 min

30 天 = 43200 min；故障总损失 ≈ 5.1 × 30 ≈ 153 min；有效训练比例 ≈ (43200 - 153) / 43200 ≈ 99.6%。

但这只是一阶估算，实际还要叠加：
- ckpt 保存本身耗时（每 30 min 保存若耗 1 min，是 3.3% 持续 overhead）
- 网络抖动 / NCCL hang 误报（按经验每天 0.5 次小故障，每次 5 min ≈ 0.3% 损失）
- 慢节点 straggler（持续性 5–10% 的吞吐损失）
- 一次大故障（如机房断电）可能占 1–2% 损失

综合估算后端到端 effective_time / wall_time 落在 75%–85% 区间，是行业经验值。

### 3. 定位手段与命令（如何在实际系统里测量）

- `sacct` / `slurmctld` 看每个 job 的 wall time vs `effective_time`（除去重启段）
- 训练 log 里每 step 时间戳计算 `goodput = sum(step_time) / wall_time`
- 节点级 MTBF/MTTR 直接从 incident database 统计
- Checkpoint 保存耗时 trend 做 overhead
- 监控看板：`training_efficiency_ratio` 指标长期 trend

### 4. 修复路径与权衡（如何提升 effective ratio）

- 缩短 ckpt 间隔 → 减少回滚损失，但增加保存 overhead；最优解通常是 15–60 min（依模型大小）
- 异步 ckpt → 隐藏保存 overhead，但引入一致性风险（见 Q15）
- 弹性训练 / hot spare → 单 rank 故障不中断整任务，需要框架支持（Bamboo、PyTorch elastic）
- 故障预测 → 通过 DCGM trend 预判节点退化提前 drain
- 跨集群 federation → 故障 region 切到 backup（成本高）

工业界目标：千卡级 90%+、万卡级 75%–85%、十万卡级 60%–70%。

### 5. 同类变体 / 易错点

- 易错：直接用单节点 5% 当集群故障率——故障率随节点数线性放大
- 易错：忽略 ckpt 保存本身的 overhead
- 易错：忽略 straggler 持续性损失，只算硬故障
- 追问："故障率独立性假设合理吗？"→ 不完全合理（同机柜断电、同 image bug 会集体故障），实际更糟

### 6. 预防机制

- Effective training ratio 作为 platform team SLO 指标
- 每周复盘 top 3 损失源（按时间占比）
- 投入产出表：ckpt 间隔缩短 X min vs storage cost / overhead cost
- 弹性训练框架 day-1 评估
- 故障预测模型上线（ECC trend、PCIe link width trend、温度 trend）

### 7. 30 秒速答

- 一句话核心：万卡 30 天训练 effective ratio 现实区间 75%–85%，主要损失来自 ckpt overhead + straggler + 故障恢复。
- 关键机制：集群故障率 ≈ N × 单节点故障率，每次故障损失 ≈ 检测 + 回滚（ckpt 间隔一半）。
- 修复 / 防御：ckpt cadence 调优 + 异步保存 + 弹性训练 + 故障预测 + DCGM trend 预判。
- 面试加分关键词：MTBF/MTTR、goodput、elastic training、Bamboo、hot spare、SLO。

### 8. 自测 checklist

- [ ] 你能不能用一页纸算出"万卡 30 天 effective ratio"的一阶估算？
- [ ] 你能不能解释为什么 ckpt 间隔不是越短越好？
- [ ] 你能不能举一个类似场景（如十万卡推理集群可用率估算）？
- [ ] 你能不能说出 4 类损失来源（硬故障、ckpt overhead、straggler、误报）？

---

## Q29. 设计：100 人规模 ML 团队的事故复盘机制

> 🧭 综合 · "事故复盘机制"不是一份文档，而是从触发到沉淀的完整闭环。能否设计出适配 100 人规模、blameless、可扩展的机制，是衡量平台 / SRE leader 的核心维度。

### 1. 现象 / 需求

100 人 ML 团队（含训练、推理、平台、研究），每月 incident 数预计 20–40 次（含 P0/P1/P2 分级），需要一套机制：
- 事件触发后 oncall 知道怎么响应
- 处置过程被记录
- 事后做 blameless postmortem
- action item 被追踪到关闭
- 知识沉淀进可检索的知识库供新人和未来事故参考

要求：低开销（不能让 oncall 写 5000 字报告每次）、可扩展（团队增长不重构）、有度量（postmortem 完成率 / action item 关闭率可追踪）。

### 2. 根因层级原理（机制设计的核心思考）

一套好的复盘机制要满足：
- 触发触发器明确：什么级别的事故必须 postmortem（建议 P0/P1 必须，P2 可选）
- 角色清晰：incident commander、scribe、SME（subject matter expert）、postmortem owner
- Blameless 文化保障：不点名个人 / 不挂钩绩效，关注流程缺陷
- 模板标准化：现象 / timeline / root cause / 影响范围 / action items / lessons
- 追踪闭环：action item 进 Jira / Linear，有 owner 有 due date 有 review cadence
- 知识沉淀：搜索友好、tag 分类、相似事件链接

### 3. 定位手段与命令 / 具体机制流程

T0 触发：
- 监控系统报警 → PagerDuty → oncall 接手
- 严重度评估（P0/P1/P2/P3）由 oncall 初判 + 5 分钟内 incident commander 确认

T0–T+N（处置中）：
- 建 Slack / Teams 专属 channel `#incident-yyyy-mm-dd-shortname`
- Scribe（轮值）实时记录关键决策与时间戳
- 业务影响每 15 分钟 update status page

事后（24h 内）：
- Owner（默认 incident commander 或他指定）创建 postmortem 文档（模板填充）
- 24h 内填完 timeline + 初步 root cause

事后（5 工作日内）：
- 跨团队 review 会议（30–60 min），focus 在系统问题 + action item 而非个人
- Action item 必须 SMART（具体 / 可衡量 / 有 owner / 有 deadline）

事后（持续）：
- Action item 每周 review 进度
- 每月汇总进数据库：top 故障类型、top 复发模块、整体 MTTR/MTBF trend
- 季度 review：组织级 lessons + 关键模块改造投入

### 4. 修复路径与权衡（机制本身的 trade-off）

- Blameless vs 责任追溯：blameless 不代表无人改进——区分"个人疏忽"（培训）vs "系统缺陷"（工具/流程）；高频个人疏忽要私下 1:1
- 详尽度 vs 成本：100 人团队每月 30 次 P0/P1 全部深度复盘要消耗 60+ 人天，必须分级模板（深度 vs 轻量）
- 自动化 vs 灵活性：模板自动化（如自动抓 timeline、metric 截图）省时间，但限制叙事；保留 narrative section
- 共享 vs 隐私：postmortem 是否对外发布？建议内部全开放、外部按需脱敏发安全公告
- 强制 vs 自愿：P0/P1 强制 postmortem，P2 鼓励但不强制

### 5. 同类变体 / 易错点

- 易错：postmortem 只关注 "怎么不再发生"——同样重要的是"如何更快检测 + 更快修复"
- 易错：action item 写得太大没人做（"重构调度器"），要切到 1–2 周可完成的小单元
- 易错：没有跟进 cadence，action item 一年都不关闭
- 易错：knowledge base 没人维护变成废墟，要有 ownership + 季度 audit
- 追问："SRE 的 blameless 原则在小团队（10 人）适用吗？"→ 适用但更轻量，重点在文化而非流程

### 6. 预防机制（让机制本身不会失灵）

- 平台团队设专职 "incident program manager" 负责机制 owner（100 人规模能负担 0.3–0.5 FTE）
- Postmortem 模板版本化（Git 管理），每 6 个月迭代一次
- Metric 看板：postmortem 完成率、平均完成时长、action item 关闭率、复发事件数
- 季度组织级 review：top 3 反复发生的事故类型 → 重大投入决策
- 新人 onboarding 必读：近 6 个月 top 5 postmortem
- 工具集成：incident channel 自动生成 → postmortem doc 模板 → Jira 自动建 action item ticket

### 7. 30 秒速答

- 一句话核心：从触发 → 处置 → blameless 复盘 → action item 追踪 → 知识沉淀的完整闭环，分级模板减开销。
- 关键机制：incident commander / scribe / owner 角色清晰 + 24h 初稿 + 5 工作日 review + SMART action items。
- 修复 / 防御：blameless 文化 + 分级强制 + 自动模板 + metric 看板 + 季度组织级 review。
- 面试加分关键词：incident command system、blameless postmortem、SMART action item、MTTR/MTBF、knowledge base ownership。

### 8. 自测 checklist

- [ ] 你能不能画出这套机制从 T0 到 knowledge base 沉淀的完整时间线？
- [ ] 你能不能解释 "blameless" 不等于 "无责任"？
- [ ] 你能不能举一个 action item 写得不好（太大 / 没 owner）的例子并改写？
- [ ] 你能不能说出 4 个机制本身的健康度量指标？

---

