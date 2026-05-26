# 分布式训练与大模型并行

## 主题边界

本文聚焦分布式训练与大模型并行相关问题，重点覆盖 DDP、FSDP、ZeRO、通信与显存优化等工程实践主题。

## Q1. DDP 的梯度分桶（bucketing）机制是什么？bucket size 如何调优？

> 🟡 进阶 · 一个参数一个参数发 `AllReduce`，光启动开销就能把训练拖慢一倍。DDP 把梯度攒成桶一起发，让通信和反向重叠起来——桶大小调不好，要么碎片化、要么 overlap 没了，吞吐直接掉。

### 1. 核心结论

DDP 的 gradient bucketing 就是将多个参数的梯度按 bucket 聚合后统一触发 AllReduce，从而减少通信启动次数，并尽量把通信与反向计算重叠。bucket size 过小会导致通信碎片化、启动开销高；bucket size 过大又会削弱 overlap，通常需要结合网络带宽、模型层次结构和单卡 batch 大小做经验调优。

### 2. 底层原理

在 DDP 中，每个 rank 都保有完整模型副本。反向传播时，各参数梯度并不是等全部算完后再统一通信，而是按照预先划分的 bucket 归组。某个 bucket 内所有梯度都 ready 后，DDP 会立即对该 bucket 发起 AllReduce，并将结果回写为平均梯度。

这样做的关键目标有两个：
1. 降低通信次数，避免每个参数单独做一次 collective。
2. 尽早启动已 ready bucket 的通信，使后续层仍在反向计算时，前面 bucket 的通信已经进行，从而隐藏一部分通信时延。

### 3. 关键机制 / 流程 / 数据结构

DDP 通常会基于参数注册顺序与参数大小构造 Reducer 内部的梯度桶。典型流程如下：
1. 初始化时按参数顺序将参数对应的梯度张量打包到多个梯度桶，首个梯度桶通常较小（默认 1MB 左右），其余按 `bucket_cap_mb` 上限累加。
2. 反向传播中，每个参数梯度计算完成后会在 Reducer 里标记 ready，autograd hook 触发的顺序决定梯度桶是否能尽早启动通信。
3. 当某个梯度桶内所有梯度都 ready，Reducer 立刻对该梯度桶发起 async all-reduce，并把后续梯度继续累加到尚未触发的梯度桶。
4. all-reduce 完成后，将梯度桶内聚合并除以 world size 后的梯度写回对应 `.grad` 视图。
5. 所有梯度桶完成后，本轮梯度同步结束，训练线程才会继续 optimizer step。

实践里有几个与 PyTorch 2.x 相关的细节：
- `DistributedDataParallel(..., bucket_cap_mb=25, gradient_as_bucket_view=True, static_graph=True)` 是当前较常用的组合：`gradient_as_bucket_view=True` 让 `.grad` 直接落在梯度桶内存上，省去一次拷贝；`static_graph=True` 在模型每轮计算图固定时允许 Reducer 复用首轮学到的梯度 ready 顺序。
- 梯度桶划分依赖参数注册顺序，autograd 反向一般是按注册顺序的逆序 ready，所以默认实现会让较深层的梯度先 all-reduce，实现计算-通信重叠。
- 首轮迭代会执行一次 bucket rebuild，用真实 ready 顺序对桶内参数做再排序，后续轮次的重叠通常更稳定。
- 若开启 `torch.compile`，反向端由 AOTAutograd 生成，DDP 有一个 `DDPOptimizer` 会把编译后的反向图按 bucket 边界切分，避免整张图结束才 all-reduce 的退化。

### 4. 工程权衡 / 性能影响

bucket size 调优本质是在“通信启动开销”与“通信-计算重叠程度”之间折中：
- bucket 太小：AllReduce 次数变多，latency bound 明显，网络启动开销占比高。
- bucket 太大：必须等待更多梯度 ready 才能启动通信，导致 overlap 变差，尾部等待更长。
- 网络快、模型大、计算重时，可适当增大 bucket。
- 网络慢、层数深、希望更强 overlap 时，可适当减小 bucket。

常见经验是从默认 `bucket_cap_mb=25` 出发，结合 step time、通信占比、all-reduce 时间线和 NCCL profile 做对比。若在 Nsight Systems 或 `torch.profiler` 的 distributed view 里观察到大量小包通信（单次 all-reduce < 几 MB 却占很大比例），可增大 bucket；若反向计算结束后仍有长尾 all-reduce 未完成、暴露在关键路径上，则可尝试减小 bucket 或提前触发部分通信（例如减小模型首个 bucket 的容量）。

### 5. 常见追问 / 易错点

- bucket size 不是越大越好。很多人只关注带宽利用率，却忽略了 overlap 被破坏后的长尾问题。
- bucket 划分不只影响通信次数，也会影响显存峰值和梯度 ready 时机。
- 调优时不能只看单次 AllReduce 吞吐，还要看整步训练时间。
- 若模型存在动态控制流、unused parameters 或复杂 wrapping，bucket 的稳定性和收益可能下降。

### 6. 实践建议

建议先使用框架默认 `bucket_cap_mb=25` 与 `gradient_as_bucket_view=True` 做基线，再通过 `torch.profiler` 或 Nsight Systems 看 backward 与 all-reduce 的重叠情况。调优时一次只改一个量级，例如从 25MB 调到 50MB 或 10MB，避免过细碎搜索。多机训练优先关注跨机网络是否成为瓶颈；单机 NVLink 多卡则更多关注 kernel 与通信调度重叠是否充分。如果训练模型结构每轮固定、可以开 `static_graph=True`，通常能进一步减少 DDP 额外的动态检查开销。

### 7. 30 秒速答

- 一句话核心：把多个参数的梯度合并成 bucket 一起做 AllReduce，让通信与 backward 重叠以提高吞吐
- 关键机制：参数按反向顺序逆序分桶，bucket ready 即触发异步 AllReduce，与剩余 backward 并行
- 易踩坑 / 关键权衡：bucket_cap_mb 过大延迟启动 overlap，过小则碎片化通信，单机 NVLink 25 MB 起，跨机 50–100 MB
- 面试加分关键词：bucket_cap_mb / gradient_as_bucket_view / static_graph / overlap

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 DDP 梯度分桶的作用？
- [ ] 你能不能解释 参数按反向顺序逆序分桶，bucket ready 即触发异步 AllReduce，与剩余 backward 并行？
- [ ] 你能不能举一个 单机 8 卡 NVLink 训 13B 模型时调 bucket 大小 的具体场景？
- [ ] 你能不能说出 bucket 太小让 AllReduce 启动开销 dominate；bucket 太大让首个 AllReduce 等到 backward 末尾才启动 这种常见错误模式？

## Q2. DDP 的 find_unused_parameters 参数什么时候需要设置？性能影响？

> 🟡 进阶 · 模型里有动态分支、某些参数这一步没参与前向，DDP 会一直等它的梯度，直接 hang 住。`find_unused_parameters=True` 是救命稻草，但开了就要多扫一遍计算图，能不开就别开。

### 1. 核心结论

当模型某些参数在当前前向图中未参与 loss 计算、因此不会产生梯度时，才需要将 `find_unused_parameters=True`。它能避免 DDP 因等待某些永远不会 ready 的梯度而卡住，但会引入额外 autograd 图遍历与同步管理开销，因此在静态且所有参数都参与训练的模型中应尽量关闭。

### 2. 底层原理

DDP 默认假设每个需要训练的参数在每轮迭代都会参与反向传播，因此 Reducer 会等待这些参数对应的梯度 ready，再触发 bucket 通信。如果某些参数这轮没有出现在计算图里，DDP 就可能一直等不到对应梯度，最终表现为报错或挂起。

`find_unused_parameters=True` 时，DDP 会在反向阶段额外追踪 autograd 图，识别哪些参数没有连到最终 loss，从而将这些参数标记为 unused，避免 Reducer 错误等待。

### 3. 关键机制 / 流程 / 数据结构

典型适用场景包括：
1. 多分支网络中只激活部分 branch。
2. Mixture-of-Experts、条件路由、任务头按样本选择。
3. 训练时阶段性冻结部分子模块，但参数仍保留在 DDP 管理范围内。
4. 返回值结构复杂，导致部分输出未参与 loss。

其机制可以概括为：
- 反向开始后，DDP 从输出张量回溯 autograd 图。
- 找出本轮未被访问到的参数。
- 将这些参数从本轮 expected gradient set 中剔除。
- 让 bucket 仅等待实际会产生梯度的参数。

### 4. 工程权衡 / 性能影响

开启该选项的主要代价有：
- 每步需要额外遍历 autograd 图，增加 CPU 侧开销。
- Reducer 的状态管理更复杂，可能削弱 `static_graph=True` 带来的 bucket 重排与跳过 autograd 遍历等静态图优化收益。
- 对大模型或层数很多的网络，额外开销会更明显。
- 若其实不存在 unused parameters，却一直开启，通常会带来可观但无必要的性能损失。

因此，它更像是“保证正确性所需的兼容开关”，而不是通用推荐配置。若模型图静态、每轮都使用全部参数，关闭通常更快；需要注意 `find_unused_parameters=True` 与 `static_graph=True` 在语义上互斥，开启前者通常意味着放弃后者。

### 5. 常见追问 / 易错点

- “梯度为 0”不等于 unused parameter。unused 指的是根本没有参与本轮计算图，而不是参与了但梯度恰好为 0。
- 冻结参数若 `requires_grad=False`，通常不需要依赖该开关；真正的问题是参数仍被 DDP 视为待同步对象，但本轮不产出梯度。
- 某些返回值容器或 loss 拼装方式不当，也会让看似参与前向的分支在反向图中断开。
- 若模型是固定子图但通过代码路径写得很动态，也可能误以为必须开启，应先确认实际是否每轮都存在未使用参数。
- PyTorch 2.x 下打开 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 会在报错中列出具体未参与反向的参数名，比盲开 `find_unused_parameters` 更易定位问题根因。

### 6. 实践建议

只有在出现 unused parameter 报错、动态分支训练或确认存在条件计算时再开启该参数。定位时可先在小 batch 下跑若干步，结合 `TORCH_DISTRIBUTED_DEBUG` 日志或调试输出来确认哪些模块未参与反向。若后续将模型改造成静态图，应重新关闭并同步切到 `static_graph=True` 并复测吞吐，因为这类优化往往能直接改善 step time。

### 7. 30 秒速答

- 一句话核心：允许 DDP 容忍本轮未参与 backward 的参数，避免 Reducer 卡在等不到的梯度
- 关键机制：DDP 反向遍历 autograd 图标记 unused 参数，将其从 bucket expected set 剔除
- 易踩坑 / 关键权衡：多扫一遍 autograd 图带来 CPU 开销，且与 static_graph 互斥，能不开就别开
- 面试加分关键词：unused_parameters / static_graph / TORCH_DISTRIBUTED_DEBUG

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 find_unused_parameters 的作用？
- [ ] 你能不能解释 DDP 反向遍历 autograd 图标记 unused 参数，将其从 bucket expected set 剔除？
- [ ] 你能不能举一个 MoE 路由或多任务头按样本激活子模块 的具体场景？
- [ ] 你能不能说出 把 requires_grad=False 的冻结参数误认为需要开 find_unused 这种常见错误模式？

## Q3. FSDP 的分片（sharding）策略有哪些？FULL_SHARD vs SHARD_GRAD_OP？

> 🟡 进阶 · 7B 以上的模型一张卡塞不下完整的参数+梯度+优化器，DDP 直接 OOM。FSDP 把这三样都切片到各 rank，用通信换显存——`FULL_SHARD` 省得最狠，`SHARD_GRAD_OP` 省得少但通信也少，得看你的瓶颈是显存还是带宽。

### 1. 核心结论

FSDP1 的 sharding strategy 枚举包括 `FULL_SHARD`、`SHARD_GRAD_OP`、`HYBRID_SHARD`、`_HYBRID_SHARD_ZERO2`、`NO_SHARD`；PyTorch 2.4 起推出的 FSDP2（`torch.distributed.fsdp.fully_shard`）则用 per-parameter DTensor 分片 + `reshard_after_forward: bool | int` 参数替代了这组枚举——`reshard_after_forward=True` 等价 `FULL_SHARD`，`False` 等价 `SHARD_GRAD_OP`，整数值则可控制 reshard 到更小的 mesh（如只在节点内保持 unsharded）。核心区别仍在参数、梯度、优化器状态分别在什么阶段做分片或聚合：`FULL_SHARD` 对三者都彻底分片，显存最省；`SHARD_GRAD_OP` 则前向 all-gather 出参数后直到反向结束才 reshard，参数在中间阶段以完整形态驻留，通信更少但显存略高。

### 2. 底层原理

FSDP 的基本思想是：不要让每个 rank 长时间持有完整参数、完整梯度和完整优化器状态，而是在真正需要计算某个模块时短暂 all-gather 出完整参数，计算结束后再释放或重分片。这样能把模型状态的常驻显存占用从“完整副本”降到“分片副本”。

不同 sharding strategy 的差异，就是对三类状态的生命周期管理不同：
- 参数何时 all-gather 成完整形态、何时 reshard 回分片。
- 梯度在反向后是走 all-reduce 保留完整副本还是走 reduce-scatter 直接分片。
- 优化器更新时使用完整状态还是 sharded state。

### 3. 关键机制 / 流程 / 数据结构

常见策略可概括为：
- `NO_SHARD`：近似普通数据并行，不做参数状态分片，主要用于兼容或调试；FSDP2 官方已不再推荐，等价效果应直接用 DDP。
- `SHARD_GRAD_OP`：前向 all-gather 出参数后直到反向结束都保留 unsharded，梯度通过 reduce-scatter 分片，优化器状态也分片。
- `FULL_SHARD`：每个 FSDP unit 进入计算前 all-gather 参数，计算完立即 reshard；梯度 reduce-scatter；优化器状态保持分片。
- `HYBRID_SHARD`：节点内 `FULL_SHARD` + 节点间 replicate，等价 ZeRO-3 intra-node + DDP inter-node，跨机带宽紧张时常用。

二者对比如下：
1. `FULL_SHARD`（FSDP2 `reshard_after_forward=True`）
   - 显存收益最大。
   - 参数不长期以完整副本驻留。
   - 前向和反向都更依赖 all-gather/reshard。
2. `SHARD_GRAD_OP`（FSDP2 `reshard_after_forward=False`）
   - 相比 `FULL_SHARD`，省去了“反向前再次 all-gather”这一步。
   - 显存节省通常略弱。
   - 在某些场景下通信调度和执行路径更简单，性能可能更稳定。

### 4. 工程权衡 / 性能影响

`FULL_SHARD` 适合模型参数规模极大、显存压力最突出的场景，优点是显存节省最明显，缺点是通信更频繁、执行更复杂，对网络和 overlap 更敏感。

`SHARD_GRAD_OP` 更像中间态：
- 显存节省弱于 `FULL_SHARD`，但通常仍显著优于 DDP。
- 因反向前不再重新 all-gather，通信次数少一轮，某些模型上更容易获得稳定吞吐。
- 若训练已接近显存上限，往往还是要转向 `FULL_SHARD`。

简单理解：
- 显存最优先，倾向 `FULL_SHARD`。
- 性能稳定性、调试复杂度更优先，可先试 `SHARD_GRAD_OP`。
- 跨机带宽是主瓶颈，优先评估 `HYBRID_SHARD` 或 FSDP2 的整数 `reshard_after_forward`，把 all-gather 限制在节点内。

### 5. 常见追问 / 易错点

- 不要把 FSDP 的分片理解为“参数永远不完整存在”。实际上计算某层时通常仍要临时 all-gather。
- `FULL_SHARD` 并不总是更快，它只是更省显存；在网络较慢时可能更慢。
- 训练吞吐不只取决于 sharding strategy，还强依赖 wrap 粒度、prefetch、混合精度和激活重计算。
- 小模型上使用最激进分片可能得不偿失，因为通信固定成本会盖过显存收益。
- FSDP2 的 `fully_shard` 不再接受 `sharding_strategy` 枚举；迁移时把旧枚举映射到 `reshard_after_forward` 与 `mesh`，并用 `MixedPrecisionPolicy` / `OffloadPolicy` 替代原来的独立参数。

### 6. 实践建议

若首要目标是“单机/多机放下更大模型”，优先从 `FULL_SHARD` 起步；若模型能放下但希望降低优化复杂度或获得更稳吞吐，可对比 `SHARD_GRAD_OP`。新项目优先采用 FSDP2 的 `fully_shard`，它提供 per-parameter DTensor 分片、更规整的 composability（与 TP、PP、`torch.compile` 组合更干净），FSDP1 仅建议在老代码迁移期保留。评估时不要只看最大 batch size，还应同时记录 tokens/s、step time、显存峰值和网络利用率，才能判断是否真正划算。

### 7. 30 秒速答

- 一句话核心：FULL_SHARD 把参数/梯度/优化器状态都切分，SHARD_GRAD_OP 只切梯度和优化器状态
- 关键机制：forward 前 all-gather 参数，backward 后 reduce-scatter 梯度；FULL_SHARD 额外释放参数
- 易踩坑 / 关键权衡：FULL_SHARD 通信量翻倍但显存最省，单节点小模型 SHARD_GRAD_OP 更快
- 面试加分关键词：FULL_SHARD / SHARD_GRAD_OP / HYBRID_SHARD / NO_SHARD

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 FSDP 的 sharding 策略？
- [ ] 你能不能解释 forward 前 all-gather 参数，backward 后 reduce-scatter 梯度；FULL_SHARD 额外释放参数？
- [ ] 你能不能举一个 70B 模型 8 卡单机选 HYBRID_SHARD，多机选 FULL_SHARD 的具体场景？
- [ ] 你能不能说出 跨机大模型用 SHARD_GRAD_OP 导致单卡参数装不下而 OOM 这种常见错误模式？

## Q4. FSDP 的 auto_wrap_policy 如何配置？size_based vs module_based？

> 🟡 进阶 · FSDP 切片粒度太粗就一块 GPU 还是装不下，太细又会让 `all-gather` 调用次数爆炸。auto_wrap_policy 决定哪些 module 各自成为一个 shard 单元，配错了显存直接打回原形。

### 1. 核心结论

FSDP1 的 `auto_wrap_policy` 用来决定哪些子模块应作为独立 FSDP 单元被包裹，常用有 `size_based_auto_wrap_policy`（按参数量阈值 `min_num_params`）、`transformer_auto_wrap_policy` / `ModuleWrapPolicy`（按模块类型），以及把两者结合的 `CustomPolicy`。`size_based` 适合快速起步和通用模型；`module_based` 按模块类型或指定类名包裹，适合 Transformer 这类结构规则清晰、希望精确控制 wrap 边界的场景。FSDP2 的 `fully_shard` 不再需要 policy 对象，而是要求用户显式对每个子模块（如每一个 TransformerBlock）手动调用 `fully_shard`，由此获得更细粒度的控制与 composability。

### 2. 底层原理

FSDP 的包裹粒度直接影响参数 AllGather/Reshard 的时机、通信粒度、峰值显存和 overlap 效果。若整个大模型只包一层，单次 AllGather 过大，显存峰值和通信 burst 会偏高；若切得过碎，又会造成过多小粒度通信和调度开销。

`auto_wrap_policy` 的目的，就是自动选择合适子树作为 FSDP 单元，避免手工逐层 wrap 的繁琐与不稳定。

### 3. 关键机制 / 流程 / 数据结构

两类常见策略如下：

1. `size_based`
- 根据子模块参数量是否超过阈值决定是否 wrap。
- 常见配置是设定 `min_num_params`。
- 优点是简单、模型无关，适合先跑通。
- 缺点是它只看“大小”，不理解语义边界，可能把一个完整 block 切得不够理想。

2. `module_based`
- 按模块类型进行包裹，例如只包 `TransformerBlock`、`BertLayer`、`GPTJBlock` 这类重复结构。
- 优点是更符合模型天然计算边界，通常更利于稳定性能。
- 缺点是需要了解模型实现，迁移到新模型时需重新配置。

实践中通常遵循的流程是：
- 先识别模型中的重复大块模块。
- 优先尝试以 block 为单位做 `module_based` wrap。
- 若模型结构不规则，再退回 `size_based` 阈值法。

### 4. 工程权衡 / 性能影响

`size_based` 的优势是泛化强、上手快，但 wrap 边界可能与真实计算/通信边界不一致，导致：
- 某些层被切太碎，通信次数增多。
- 某些大层仍包得太粗，峰值显存偏高。

`module_based` 的优势是更可控：
- 更容易以 Transformer block 为单位建立稳定的 AllGather 与释放节奏。
- 便于和激活重计算、混合精度一起协同调优。
- 但若模块类选择不当，也可能造成 wrap 不均衡，出现某些 rank 负载偏重。

### 5. 常见追问 / 易错点

- auto wrap 不是包得越细越好。过细会让 FSDP unit 数量过多，通信调度成本上升。
- 也不是包得越粗越好。过粗会抬高单次 all-gather 峰值和显存压力。
- `size_based` 阈值不能脱离模型 hidden size 与 block 结构盲调。
- `module_based` 需要确认目标模块确实是重复计算单元，而不是只按文件中的类名机械选择。
- FSDP2 下不再存在“自动”wrap，漏写对某个子模块的 `fully_shard` 调用会让它被父模块吸收，容易出现“以为在分片实际没分片”的静默问题。

### 6. 实践建议

对于标准 Transformer，优先采用 `module_based`（FSDP1 的 `ModuleWrapPolicy({TransformerBlock})` 或 FSDP2 中对每个 block 显式 `fully_shard`），按 block 层级做 wrap，通常更容易得到可预测的显存和吞吐表现。若面对自定义模型、结构异构或快速验证场景，可先用 `size_based` 跑通，再通过 profiler 和显存曲线回调阈值。无论哪种策略，都建议避免把 embedding、lm head 与超大 block 混成同一 FSDP 单元，它们的规模和访问频率不同，共 wrap 容易同时抬高显存与通信开销。

### 7. 30 秒速答

- 一句话核心：决定 FSDP 用哪种粒度做单元化包裹（unit），影响显存峰值和通信粒度
- 关键机制：size_based 按参数量阈值切分；module_based 按 Transformer Block 类型切分，更易控制 all-gather 粒度
- 易踩坑 / 关键权衡：size 太小通信次数过多，size 太大显存峰值高；module_based 通常按 TransformerBlock 推荐
- 面试加分关键词：transformer_auto_wrap_policy / size_based_auto_wrap_policy / unit

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 auto_wrap_policy 的作用？
- [ ] 你能不能解释 size_based 按参数量阈值切分；module_based 按 Transformer Block 类型切分，更易控制 all-gather 粒度？
- [ ] 你能不能举一个 LLaMA Block 用 transformer_auto_wrap_policy 包到每个 decoder layer 的具体场景？
- [ ] 你能不能说出 把整个模型当一个 unit 包导致 all-gather 一次性拉回全部参数失去 FSDP 意义 这种常见错误模式？

## Q5. DeepSpeed 的 ZeRO-1/2/3 分别卸载（offload）了什么？显存节省和通信开销的 trade-off？

> 🟡 进阶 · ZeRO 是个阶梯：1 切优化器状态、2 加切梯度、3 连参数都切。每升一级显存省得更多，但通信也跟着翻——选哪一档不是越省越好，而是看你卡多卡少、互联带宽够不够。

### 1. 核心结论

ZeRO 的核心思想不是只做“offload 到 CPU/NVMe”，而是先把模型训练状态在数据并行组内做分片，再视配置决定是否进一步 offload。ZeRO-1 分片优化器状态，ZeRO-2 在此基础上再分片梯度，ZeRO-3 进一步分片参数本身。阶段越高，显存节省越大，但通信与运行时协调成本也越高。

### 2. 底层原理

传统数据并行中，每个 rank 都持有完整参数、完整梯度、完整优化器状态，导致显存占用随模型规模线性膨胀。ZeRO 将这些状态拆散到不同 rank 上，让每个进程只保留自己的那一份，从而显著降低冗余。

如果再叠加 CPU/NVMe offload，则部分状态不只是在 GPU 间分片，还会被迁移到更慢的存储层级，以换取更大的可训练模型规模。

### 3. 关键机制 / 流程 / 数据结构

三阶段差异可以概括为：

1. ZeRO-1
- 分片对象：优化器状态。
- 参数与梯度仍是每卡完整副本。
- 主要收益：显著降低 optimizer state 占用。
- 适合优化器状态很大、但参数本体尚能放下的场景。

2. ZeRO-2
- 分片对象：优化器状态 + 梯度。
- 参数仍是每卡完整副本。
- 相比 ZeRO-1，进一步降低反向后的梯度显存占用。
- 通信比 ZeRO-1 更复杂，因为梯度分片与归并更频繁。

3. ZeRO-3
- 分片对象：优化器状态 + 梯度 + 参数。
- 各 rank 不再长期持有完整参数。
- 前向/反向时需要按需收集参数分片参与计算。
- 显存收益最大，但运行时最依赖通信与调度。

```
   每个 rank 持有的训练状态  (P=参数, G=梯度, OS=优化器状态)
   N 个 DP rank 间分片情况:

   Baseline (DDP)  : [P  G  OS]   [P  G  OS]   [P  G  OS]   [P  G  OS]
                     全副本, 显存 ∝ 模型大小

   ZeRO-1          : [P  G  OS₀]  [P  G  OS₁]  [P  G  OS₂]  [P  G  OS₃]
                     只切 OS;  通信 ≈ DDP

   ZeRO-2          : [P  G₀ OS₀] [P  G₁ OS₁] [P  G₂ OS₂] [P  G₃ OS₃]
                     切 G + OS;  reduce-scatter 替代 all-reduce

   ZeRO-3 / FSDP   : [P₀ G₀ OS₀] [P₁ G₁ OS₁] [P₂ G₂ OS₂] [P₃ G₃ OS₃]
                     P 也切;  fwd/bwd 前 all-gather 临时物化, bwd 后 reduce-scatter
                     ★ 显存最省, 通信次数翻倍
```

若谈“offload 到哪里”，则通常是：
- optimizer offload：把优化器状态迁到 CPU/NVMe。
- parameter offload：把参数迁到 CPU/NVMe。
- 不同 ZeRO 阶段都可以和 offload 机制叠加，但最常见的重度场景在 ZeRO-2/3。

### 4. 工程权衡 / 性能影响

总体 trade-off 很明确：
- ZeRO-1：通信额外成本最小（仅在 optimizer step 前后多一次 broadcast/gather），显存收益也最有限。
- ZeRO-2：额外的 reduce-scatter 替换原本的 all-reduce，通信量相当，但 per-rank 梯度显存减半，通常是训练中显存收益和性能的常见折中。
- ZeRO-3：参数按 shard 按需 all-gather、反向后 reduce-scatter，显存节省最大，但通信次数在每个 FSDP/ZeRO 单元上翻倍，对网络和 overlap 更敏感。

若再叠加 CPU/NVMe offload：
- 可训练模型规模进一步提升，ZeRO-Infinity 可扩展到百亿到千亿级参数。
- 但 PCIe、CPU 内存带宽、NVMe 延迟都可能成为新瓶颈。
- 吞吐通常下降明显，更适合“能训起来优先”的场景，而非极致性能场景。

### 5. 常见追问 / 易错点

- 很多人把 ZeRO 和 offload 混为一谈。严格说，ZeRO 是分片策略，offload 是存储层级迁移策略，二者可组合但不是同义词。
- ZeRO 阶段越高并不必然越优，只是更省显存；当网络较慢时，stage 升高可能明显拖慢训练。
- ZeRO-2/3 的收益和代价与 batch size、梯度累积步数、网络拓扑强相关。
- 若模型本来就能稳定放进显存，盲目切到 ZeRO-3 可能会损失吞吐。

### 6. 实践建议

先从 ZeRO-1 或 ZeRO-2 起步：若瓶颈主要是 optimizer state，用 ZeRO-1；若梯度和 optimizer 一起顶满显存，用 ZeRO-2；只有当参数副本本身已放不下时，再考虑 ZeRO-3 或等价的 FSDP2 `FULL_SHARD`。若必须启用 CPU/NVMe offload，应同步评估主机内存容量、PCIe 带宽和实际 tokens/s，避免“显存省下来了但系统整体吞吐不可用”。在 PyTorch 生态中，ZeRO-3 与 FSDP `FULL_SHARD` 语义近似重叠，新项目若以 PyTorch 原生为主，优先评估 FSDP2 以减少第三方框架耦合。

### 7. 30 秒速答

- 一句话核心：ZeRO-1 切优化器状态，ZeRO-2 加切梯度，ZeRO-3 再加切参数
- 关键机制：级数越高显存越省，但 ZeRO-3 每步多 1.5× 通信（all-gather 参数 + reduce-scatter 梯度）
- 易踩坑 / 关键权衡：ZeRO-3 跨机通信压力大，单机内优先用 ZeRO-2 + offload
- 面试加分关键词：ZeRO-1/2/3 / Optimizer State / Gradient / Parameter Sharding / Offload

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 ZeRO-1/2/3 各自切分的状态？
- [ ] 你能不能解释 级数越高显存越省，但 ZeRO-3 每步多 1.5× 通信（all-gather 参数 + reduce-scatter 梯度）？
- [ ] 你能不能举一个 13B 模型单机 8 卡用 ZeRO-2，跨机 64 卡 70B 用 ZeRO-3 的具体场景？
- [ ] 你能不能说出 小模型盲目上 ZeRO-3 反而被通信拖慢 这种常见错误模式？

## Q6. 混合精度训练中的 loss scaling 在分布式场景下如何处理？

> 🟡 进阶 · fp16 的动态范围太窄，小梯度直接被截成 0；loss scaling 就是先把 loss 放大再 backward，最后再缩回去。分布式里如果各 rank 的 scale 不一致，梯度直接错乱，必须全局同步 overflow 状态。

### 1. 核心结论

分布式混合精度训练里，loss scaling 的核心目标没有变化，仍是避免 FP16 梯度下溢；变化在于 overflow 检测、梯度同步和 optimizer step 必须在所有 rank 上保持一致。实践中通常使用框架提供的动态 loss scaler，让每个 rank 基于本地梯度检查 inf/nan，再通过全局归约形成统一的 overflow 决策，确保所有进程要么一起跳过 step，要么一起更新参数。

### 2. 底层原理

FP16 的指数范围较窄，反向传播时小梯度容易下溢为 0。loss scaling 的做法是先把 loss 乘以一个较大 scale，使反向阶段产生的梯度整体放大；在真正做优化器更新前，再把梯度按同样 scale 缩回去。

分布式场景的关键点在于：
- 梯度是跨 rank 聚合的，不能某些 rank 先 unscale、某些 rank 仍处于 scaled 状态。
- 若某个 rank 出现 overflow，而其他 rank 没有，仍必须按照“全局 overflow”处理，否则参数会立刻失去一致性。
- 动态调整 scale 时，也应让所有 rank 采用同一 scale 演进策略。

### 3. 关键机制 / 流程 / 数据结构

典型流程如下：
1. 前向计算得到 loss。
2. 对 loss 乘以当前 scale，再执行 backward。
3. DDP/FSDP 对 scaled gradients 做同步；由于所有 rank 使用相同 scale，同步结果仍然正确。
4. 在 optimizer step 前执行 unscale，将梯度除以 scale。
5. 检查梯度中是否存在 inf/nan。
6. 对 overflow 标志做一次跨 rank 的 all-reduce，通常按逻辑或语义聚合。
7. 若任一 rank overflow，则所有 rank 都跳过 optimizer step，并降低 scale；否则共同执行 step，并按策略维持或增大 scale。

工程实现上常见两种模式：
- 框架托管模式：如 `torch.amp.GradScaler`（PyTorch 2.x 推荐写法，旧别名 `torch.cuda.amp.GradScaler` 仍可用）配合 DDP，由框架处理 unscale、检查和 step 跳过。
- 系统级托管模式：如 DeepSpeed、Megatron-LM 在 optimizer wrapper 或 distributed optimizer 中统一管理 loss scale 状态，并在 ZeRO/FSDP 的 reduce-scatter 前完成 unscale。

### 4. 工程权衡 / 性能影响

loss scaling 本身计算开销很小，但会带来几个工程约束：
- 需要增加 overflow 检查和一次额外状态同步，不过相较主通信通常不是瓶颈。
- scale 过大时会频繁 overflow，导致 step 被跳过，训练吞吐和收敛都会受影响。
- scale 过小时虽然更稳定，但保护不足，FP16 梯度下溢概率上升。
- BF16 通常不依赖 loss scaling，因为其指数范围更大，这也是很多大模型训练优先选择 BF16 的原因之一。

### 5. 常见追问 / 易错点

- 不是每个 rank 各自决定是否 `optimizer.step()`。分布式训练里 step 是否执行必须全局一致。
- 不要在梯度同步前让不同 rank 使用不同 scale，否则聚合结果不再具有可比性。
- 梯度累积场景下，通常应在真正 step 之前再统一 unscale 和 overflow 检查，而不是每个 micro-batch 都独立改写 scale。
- 如果使用梯度裁剪，正确顺序通常是先 unscale，再做 clip，再 step；否则裁剪阈值会被 scale 污染。

### 6. 实践建议

优先使用成熟框架的动态 loss scaler，不要手写各 rank 独立的缩放逻辑。若系统支持 BF16（Ampere/Hopper/Blackwell 原生支持），训练稳定性通常优于 FP16 + loss scaling，大模型训练默认应首选 BF16，仅在必须使用 FP16 的老硬件上再回到 loss scaling 路径。定位异常时重点看三个信号：是否频繁出现 skipped steps、scale 是否长期单向下降、overflow 是否总发生在固定层或固定数据批次；这些现象往往说明初始 scale、学习率或数值稳定性存在问题。

### 7. 30 秒速答

- 一句话核心：所有 rank 必须用同一个 scale 值，避免 rank 间梯度尺度不一致导致 reduce 后偏差
- 关键机制：GradScaler 在 unscale 前同步 inf/nan 检测结果，所有 rank 一起 skip 或者一起更新 scale
- 易踩坑 / 关键权衡：混合精度 fp16 才需要 loss scaling，bf16 动态范围足够通常不需要
- 面试加分关键词：GradScaler / fp16 / bf16 / dynamic_loss_scale / NaN sync

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 loss scaling 在分布式下的处理？
- [ ] 你能不能解释 GradScaler 在 unscale 前同步 inf/nan 检测结果，所有 rank 一起 skip 或者一起更新 scale？
- [ ] 你能不能举一个 A100 上 fp16 训 GPT 用 GradScaler，H100 上切 bf16 后关掉 的具体场景？
- [ ] 你能不能说出 某些 rank skip step、其他 rank 更新参数导致权重不一致 这种常见错误模式？

## Q7. 梯度累积（gradient accumulation）在 DDP 中的正确实现方式？

> 🟡 进阶 · 想要大 batch 但显存不够，就累积几个小 step 再 update。DDP 默认每次 backward 都触发 AllReduce，不用 `no_sync()` 包裹中间步骤的话，通信量直接翻 N 倍——白白浪费带宽。

### 1. 核心结论

DDP 中做梯度累积的关键不是“多次 backward”，而是“只在需要更新参数的那个 micro-step 上触发梯度同步”。标准做法是在前 `accum_steps - 1` 个 micro-batch 上关闭梯度 all-reduce，最后一个 micro-batch 再同步并执行 optimizer step。这样既能得到等价于大 batch 的梯度，又能避免每个 micro-batch 都做一次跨卡通信。

### 2. 底层原理

DDP 默认在每次 backward 过程中，当某个 bucket 内梯度 ready 时就触发 all-reduce。如果直接对每个 micro-batch 都调用 backward，DDP 也会对每次 backward 都做同步，这会让通信次数按累积步数成倍增加，而且语义上更接近“多次小 batch 更新前求平均”，而不是“单次大 batch 梯度更新”。

因此，梯度累积的正确语义是：
- 多个 micro-batch 的梯度先在本地参数 `.grad` 上累加。
- 在最后一个 micro-batch 才做一次跨 rank 同步。
- 同步完成后使用累计梯度执行一次 optimizer step。

### 3. 关键机制 / 流程 / 数据结构

在 PyTorch DDP 中，常见实现依赖 `no_sync()`：
1. 每轮 optimizer update 包含多个 micro-batch。
2. 对前 `accum_steps - 1` 个 micro-batch，使用 `with model.no_sync():` 执行 forward + backward。
3. 对最后一个 micro-batch，正常执行 forward + backward，此时 DDP 会对累计后的梯度触发同步。
4. 视需要将每个 micro-batch 的 loss 除以 `accum_steps`，保证总梯度尺度与大 batch 一致。
5. 在最终同步后的梯度上执行 gradient clipping、optimizer step、zero_grad。

一个关键细节是 loss 缩放：
- 若不对每个 micro-batch 的 loss 除以 `accum_steps`，则累计得到的梯度会比目标大 batch 梯度大 `accum_steps` 倍。
- 也可以不改 loss，而在学习率或梯度上做等价缩放，但前者更直观。

最小可运行骨架如下：

```python
for step, batch in enumerate(loader):
    is_sync_step = (step + 1) % accum_steps == 0
    ctx = contextlib.nullcontext() if is_sync_step else model.no_sync()
    with ctx:
        loss = model(batch) / accum_steps
        loss.backward()
    if is_sync_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### 4. 工程权衡 / 性能影响

梯度累积的收益是显著降低激活和 batch 维度带来的显存压力，从而用较小显存模拟更大有效 batch；代价主要包括：
- 单次参数更新前要执行更多次前向/反向，step latency 上升。
- optimizer step 次数减少后，学习率调度和日志统计都要按 update step 而非 micro-step 解释。
- 若没有 `no_sync()`，通信开销会显著放大，`accum_steps=8` 时通信次数会翻 8 倍，直接导致吞吐下降。
- 累积步数过大时，优化动态会更接近大 batch 训练，可能需要重新调学习率、warmup 和正则化。

### 5. 常见追问 / 易错点

- `optimizer.zero_grad()` 不应放在每个 micro-batch 前，否则前面累积的梯度会被清掉。
- `no_sync()` 只是在本轮 backward 上延迟 DDP 的同步，不是完全关闭 autograd。
- 最后一个 micro-batch 必须参与同步，否则各 rank 的 `.grad` 仍只是本地累计值。
- AMP 场景下，`scaler.step()` 和 `scaler.update()` 应只在完成一次有效 update 时调用一次。
- 如果同时使用 gradient clipping，应在最终累计完并且 unscale 之后再裁剪，而不是每个 micro-batch 单独裁剪。

### 6. 实践建议

优先把“micro-step”和“optimizer update step”两个概念在代码与日志中严格区分。实现时使用统一模板：`zero_grad -> 多次 micro-batch backward -> 最后一次同步 -> clip -> step -> update -> zero_grad`。验证正确性最直接的方法是用单卡大 batch 与多卡累积版本对比梯度范数和 loss 曲线；若差异明显，通常是 loss 缩放、`no_sync()` 使用位置或 zero_grad 时机有误。

### 7. 30 秒速答

- 一句话核心：累积步内用 model.no_sync() 关闭 AllReduce，只在最后一步触发同步
- 关键机制：no_sync() 让 backward 只累加本地梯度，最后一步退出 context 做一次 AllReduce
- 易踩坑 / 关键权衡：忘记 no_sync 会让每一个累积步都触发同步，吞吐折半甚至更差
- 面试加分关键词：no_sync / accumulation_steps / DDP

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 DDP 中梯度累积的正确写法？
- [ ] 你能不能解释 no_sync() 让 backward 只累加本地梯度，最后一步退出 context 做一次 AllReduce？
- [ ] 你能不能举一个 batch 32 但单卡只能放 8，用 4 步梯度累积 的具体场景？
- [ ] 你能不能说出 忘 no_sync 让通信成本随累积步线性增长 这种常见错误模式？

## Q8. 分布式 sampler 如何保证每个 epoch 的数据不重复？

> 🟢 基础 · 8 张卡都用同一个 DataLoader，每张卡看到一样的数据，等于训了个寂寞。`DistributedSampler` 给每个 rank 分一段不重叠的索引，还得记得每个 epoch 调 `set_epoch()`，不然 shuffle 模式都一样。

### 1. 核心结论

分布式训练里的 sampler 不能只做“把数据切成 N 份”，还必须保证所有 rank 在同一个 epoch 上共享同一套打乱结果，并且各自只取其中互不重叠的一段。典型做法是使用 `DistributedSampler` 基于全局随机种子和 epoch 共同生成一致的 shuffle 序列，再按 rank 切片分发；训练循环中必须在每个 epoch 开始前调用 `set_epoch(epoch)`，否则每轮顺序会重复。

### 2. 底层原理

如果每个 rank 独立调用本地随机打乱，那么不同进程看到的数据子集会出现重叠或遗漏，等价于全局样本分配失衡。`DistributedSampler` 的核心思路是：
- 所有 rank 使用相同 seed 和相同 epoch 生成同一组全局索引排列。
- 根据 `num_replicas` 和 `rank` 选择属于自己的索引子序列。
- 这样全局视角下，每个样本在一个 epoch 中只会分配给一个 rank。

当数据集大小不能被 world size 整除时，sampler 还需要决定是丢弃尾部样本还是补齐索引，这会影响“严格不重复”的定义范围。

### 3. 关键机制 / 流程 / 数据结构

`DistributedSampler` 的典型机制如下：
1. 根据 `seed + epoch` 初始化随机数生成器。
2. 对数据集索引 `[0, 1, ..., N-1]` 做一次全局 shuffle。
3. 若 `drop_last=False` 且 `N` 不能整除 world size，则通过重复部分索引补齐到可均分长度。
4. 每个 rank 取打乱后索引序列中自己的切片，如 `indices[rank : total_size : world_size]`。
5. DataLoader 根据这些索引读取样本。

这里有两个重要语义：
- `set_epoch(epoch)` 会改变 shuffle 序列，使不同 epoch 的样本顺序不同。
- “每个 epoch 数据不重复”通常指在 `drop_last=True` 或样本数可整除时，全局唯一分配；若启用补齐，尾部被复制的少数样本可能重复出现。

### 4. 工程权衡 / 性能影响

分布式 sampler 的主要权衡点包括：
- `drop_last=True`：避免补样本带来的重复，但会丢掉每轮尾部少量数据。
- `drop_last=False`：覆盖全数据集更完整，但为均分而补齐时会引入少量重复。
- 固定 seed 便于复现，但若忘记结合 epoch 改变，跨 epoch 顺序会完全一致。
- 自定义 sampler 若未正确处理 rank 切分和随机种子，很容易导致训练统计看起来正常、实际数据却重复采样。

### 5. 常见追问 / 易错点

- 最常见错误是忘记在每个 epoch 调用 `sampler.set_epoch(epoch)`，导致每轮 shuffle 结果相同。
- `shuffle=True` 与 `DistributedSampler` 一起使用时，通常应由 sampler 控制顺序，避免 DataLoader 再次本地打乱。
- “每卡不重复”不等于“全局不重复”。需要从所有 rank 合并后的样本集合判断。
- IterableDataset 或流式数据源不能直接套用普通 `DistributedSampler` 语义，通常要自定义按 shard 切分的数据读取逻辑。

### 6. 实践建议

若使用 map-style dataset，优先直接采用官方 `DistributedSampler`，并把 `set_epoch(epoch)` 写进标准训练模板（`for epoch in range(...): sampler.set_epoch(epoch); train_one_epoch(...)`）。对于数据量不能整除 world size 的任务，先明确是更关心样本覆盖率还是严格去重，再选择 `drop_last` 策略。流式/大文本语料通常改为“按 shard 预切分 + 每 rank 随机选择 shard”，配合 WebDataset / MosaicML StreamingDataset 等工具保证全局不重复。上线前可在小数据集上打印各 rank 一个 epoch 的样本 id，人工检查是否存在重叠、遗漏和跨 epoch 完全重复的问题。

### 7. 30 秒速答

- 一句话核心：按 rank 切分索引数组，每个 epoch 用 epoch 作 seed 重新 shuffle
- 关键机制：set_epoch(epoch) 同步所有 rank 的随机种子，确保打乱一致且不重复
- 易踩坑 / 关键权衡：忘记调 sampler.set_epoch 会让每个 epoch 的 shuffle 顺序完全相同
- 面试加分关键词：DistributedSampler / set_epoch / drop_last / shuffle

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 DistributedSampler 的不重复保证？
- [ ] 你能不能解释 set_epoch(epoch) 同步所有 rank 的随机种子，确保打乱一致且不重复？
- [ ] 你能不能举一个 多卡训练每个 epoch 起始处 sampler.set_epoch(epoch) 的具体场景？
- [ ] 你能不能说出 不调 set_epoch 导致每个 epoch 实际看到同样的数据顺序 这种常见错误模式？

## Q9. DDP 的 SyncBatchNorm 原理？什么时候必须用？

> 🟡 进阶 · 单卡 BN 算的是本卡 batch 的均值方差，多卡训练时每张卡的 batch 太小、统计量噪声大，模型精度直接掉。`SyncBN` 把所有卡的统计量同步成全局值——检测分割任务里 batch 一般很小，几乎是必选项。

### 1. 核心结论

SyncBatchNorm 的本质是把每张卡局部 batch 的均值、方差统计扩展成跨所有 rank 的全局统计，再用这组全局统计完成 BN 归一化。它适用于单卡 batch 很小、局部统计噪声太大而影响收敛的场景；如果每卡 batch 已足够大，或模型本就使用 LayerNorm / RMSNorm，则通常没有必要引入这类额外同步开销。

### 2. 底层原理

普通 BatchNorm 在单机单卡上使用当前 mini-batch 的均值与方差进行归一化。到了 DDP 场景，若仍按每卡本地 batch 独立计算 BN 统计，则每个 rank 上的归一化基准不同，相当于模型副本看到不同数据分布，容易在小 batch 下造成训练不稳定。

SyncBatchNorm 的解决方法是：
- 每个 rank 先计算本地特征的统计量，如局部均值、平方和、样本数。
- 通过 all-reduce 聚合得到全局总和与总样本数。
- 基于全局统计量计算统一均值和方差。
- 所有 rank 用同一组统计值完成归一化。

### 3. 关键机制 / 流程 / 数据结构

其典型流程可以概括为：
1. 前向时，每个 rank 对当前 channel 计算本地 `sum`、`sum of squares` 和 `count`。
2. 在进程组内执行 all-reduce，得到全局 `global_sum`、`global_sumsq`、`global_count`。
3. 按公式计算全局 `mean` 与 `var`。
4. 各 rank 使用统一的 `mean/var` 对本地激活做归一化。
5. running mean / running var 也据此更新，保证训练与推理统计口径一致。

因此，SyncBatchNorm 的同步发生在前向阶段，而不是像 DDP 梯度同步那样主要发生在反向阶段。它与 DDP 梯度 all-reduce 是两套不同目的的 collective，且反向还会再跑一次跨 rank 归约来传回 BN 统计相关的梯度。

### 4. 工程权衡 / 性能影响

使用 SyncBatchNorm 的代价比较明确：
- 每个 BN 层前向都会增加一次统计量同步，层数多时通信开销明显。
- 它会削弱前向阶段的并行独立性，降低吞吐。
- 在多机网络较慢时，BN 同步有时会成为隐藏很深的性能瓶颈。
- 但如果每卡 batch 很小，本地 BN 统计噪声过大，不同步带来的收敛损失往往比这些开销更严重。

### 5. 常见追问 / 易错点

- 不是“用了 DDP 就必须用 SyncBatchNorm”。只有当模型确实依赖 BN，且每卡 batch 小到局部统计不可靠时才通常需要。
- Transformer 类模型多数使用 LayerNorm/RMSNorm，因此通常不涉及 SyncBatchNorm。
- SyncBatchNorm 主要解决统计量一致性问题，不会替代梯度同步。
- 将模型转成 SyncBatchNorm 后，要求相关模块都运行在分布式环境下；若进程组配置不一致，容易出现同步异常或 hang。

### 6. 实践建议

如果模型是 ResNet、检测、分割等仍大量使用 BN 的视觉任务，并且每卡 batch 很小，应优先评估 SyncBatchNorm，一般通过 `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` 在 DDP 包裹前一键替换。若训练规模较大但网络慢，可同时比较 GroupNorm、FrozenBN 或增大单卡 batch 是否更划算。上线前建议在单卡 BN 与 SyncBN 之间对比收敛曲线和吞吐，只有当统计偏差确实影响精度时再承担额外同步成本。

### 7. 30 秒速答

- 一句话核心：把所有 rank 的 BN 统计量做跨卡同步，等效于 global batch 的 BN
- 关键机制：forward 时 all-gather 各 rank 的 mean/var，再回写归一化结果
- 易踩坑 / 关键权衡：通信开销大，仅 per-GPU batch < 16 且模型对 BN 敏感时才必须用
- 面试加分关键词：SyncBatchNorm / convert_sync_batchnorm / global batch stats

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 SyncBatchNorm 的作用？
- [ ] 你能不能解释 forward 时 all-gather 各 rank 的 mean/var，再回写归一化结果？
- [ ] 你能不能举一个 检测/分割模型 per-GPU batch=2 时必须开 SyncBN 的具体场景？
- [ ] 你能不能说出 大 batch LLM 用了 LayerNorm 还盲目调 convert_sync_batchnorm 这种常见错误模式？

## Q10. 如何排查分布式训练中的 hang 问题？NCCL_DEBUG=INFO 的输出如何解读？

> 🔴 专家 · 训练突然不动、几个小时没有输出但 GPU 还显示 100% 利用率——这是分布式工程师的噩梦。`NCCL_DEBUG=INFO` 加上 timeout 配置是第一把刀，能告诉你卡在哪个 collective 上、是哪两个 rank 没对上。

### 1. 核心结论

排查分布式训练 hang 的核心不是盲目重启，而是先判断“卡在计算、卡在通信、还是卡在进程同步语义不一致”。最常见根因包括：某些 rank 没有进入同一 collective、batch 数不一致、某个 rank 提前报错退出、网络接口配置错误、NCCL 初始化拓扑异常。`NCCL_DEBUG=INFO` 的价值在于告诉你 communicator 如何建立、使用了哪些网卡与传输路径、哪个 collective 停在了什么阶段，从而把 hang 从“黑盒”变成可定位的通信时序问题。

### 2. 底层原理

分布式训练依赖所有 rank 在相同顺序上参与同一组 collective。只要有一个 rank 少调或多调一次 AllReduce / Broadcast / Barrier，其余 rank 就会一直等待。NCCL 本身并不知道你的上层语义是否正确，它只会忠实执行通信队列；因此 hang 往往不是“网络坏了”这么简单，更多是程序控制流、数据长度或异常处理破坏了 collective 对齐。

从系统层面看，hang 常见发生在三类阶段：
- 初始化阶段：rank 无法互相发现、建链失败、网卡或端口不可达。
- 训练中阶段：某个 collective 等不到全部参与者。
- 退出阶段：某些 rank 已结束，另一些 rank 仍在 barrier 或 dataloader worker 清理。

### 3. 关键机制 / 流程 / 数据结构

实战排查通常按以下顺序推进：
1. 先确认所有 rank 的日志是否都推进到同一训练步、同一 micro-step。
2. 检查是否存在数据长度不一致，例如某 rank 提前耗尽 dataloader、异常跳出循环或 OOM 后静默失败。
3. 打开 `NCCL_DEBUG=INFO`，必要时结合 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 看 collective 调用栈与 bucket 信息。
4. 观察日志里的 communicator 初始化信息，确认 world size、rank、local rank、设备映射和网络接口一致。
5. 定位最后一个成功完成的 collective，以及第一个没有所有 rank 都打印完成信息的 collective。
6. 若怀疑网络层问题，再进一步看 socket/IB 连接、网卡选择、超时和重试日志。

`NCCL_DEBUG=INFO` 常见输出可这样理解：
- `Bootstrap` / `NET/Socket` / `NET/IB`：表示 NCCL 正在建立初始连接与选择传输层，重点看是否选中了预期网卡和链路。
- `Connected all rings/trees`：说明 ring/tree 拓扑建立成功，初始化阶段基本完成。
- `Channel` / `Ring` / `Tree`：展示通信拓扑映射，可帮助判断 GPU、NIC、跨节点路径是否合理。
- `opCount`：表示 collective 操作序号。若某些 rank 停在同一 `opCount`，通常说明都在等待该次操作完成；若不同 rank 的 `opCount` 不一致，常说明程序语义已经跑偏。
- `AllReduce` / `Broadcast` / `ReduceScatter` / `AllGather`：表明正在执行的 collective 类型，可对应到 DDP、FSDP 或 checkpoint 加载阶段。
- `Abort`、`Async error`、`connection closed`：通常不是 hang 本身，而是某个 rank 已先出错，其他 rank 随后感知到连接断开。

### 4. 工程权衡 / 性能影响

打开详细调试日志会显著增加输出量，并可能扰动训练时序，因此通常用于复现问题的短跑而非常态生产配置。但它的收益很高：
- 能快速区分“初始化建链问题”和“训练中 collective 不对齐”。
- 能识别是否错误选择了慢网卡、容器内虚拟网卡或未启用 IB。
- 能通过 `opCount` 对齐各 rank 状态，缩小问题发生的代码区间。

日志只能告诉你“通信停在哪里”，不一定直接告诉你“为什么某个 rank 没来”。后者还要结合上层训练日志、异常栈、dataloader 状态和资源监控综合判断。

### 5. 常见追问 / 易错点

- hang 不一定是 NCCL bug，更多时候是某个 rank 提前 OOM、数据集长度不一致、条件分支导致 collective 次序不同。
- 看到某 rank 没输出后续 NCCL 日志，不代表问题一定在该 rank 的通信层，也可能它早就在前面的 Python 逻辑或 CUDA kernel 上卡住了。
- `barrier()` 不是万能排查工具；乱加 barrier 反而可能引入新的等待点。
- 若只看单个 rank 日志，很容易误判。必须横向对比所有 rank 在同一时间窗口的最后日志位置。
- dataloader worker 死锁、文件系统阻塞、CPU 线程耗尽也可能表现为“像通信 hang”，不能只盯 NCCL。

### 6. 实践建议

建议建立一套固定排查模板：先收集所有 rank 的最后 100 行训练日志，再收集开启 `NCCL_DEBUG=INFO` 后的短复现日志，对齐 `opCount` 与 step 编号，判断是哪一个 collective 首次失配。PyTorch 2.3+ 引入的 Flight Recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE` / `TORCH_NCCL_DUMP_ON_TIMEOUT=1`）会在 watchdog 超时时自动 dump 每个 rank 最近若干条 collective 调用栈与 seq id，是定位“哪个 rank 少发了一次 collective”最直接的工具，应在生产作业模板中默认启用。若是初始化 hang，优先核对 `MASTER_ADDR`、`MASTER_PORT`、网卡环境变量和容器网络；若是训练中 hang，优先检查 dataloader 长度、`no_sync()`/梯度累积逻辑、条件分支和异常处理。必要时启用更短超时、最小化 batch 和最小 world size 复现，把问题从全量训练任务缩小到一个可稳定重现的最小用例。

### 7. 30 秒速答

- 一句话核心：结合 NCCL_DEBUG=INFO、py-spy/gstack 和 nccl wait timeout 日志定位卡在哪个 collective
- 关键机制：hang 通常是 rank 间 collective 不对齐（顺序/形状/参与方），NCCL 默认 30 min timeout 才报错
- 易踩坑 / 关键权衡：开 NCCL_ASYNC_ERROR_HANDLING + 较小 timeout，能快速 fail 而不是默默挂
- 面试加分关键词：NCCL_DEBUG=INFO / TORCH_NCCL_BLOCKING_WAIT / py-spy / Flight Recorder

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 分布式 hang 排查思路？
- [ ] 你能不能解释 hang 通常是 rank 间 collective 不对齐（顺序/形状/参与方），NCCL 默认 30 min timeout 才报错？
- [ ] 你能不能举一个 部分 rank 跳过某层导致 AllReduce 数量不一致 的具体场景？
- [ ] 你能不能说出 只在 rank 0 加日志，其他 rank 卡死时反而看不到现场 这种常见错误模式？

## Q11. DDP 的 torchrun 和 mp.spawn 启动方式的区别？

> 🟢 基础 · `mp.spawn` 是老办法，主进程 fork 子进程，调试方便但故障恢复一塌糊涂；`torchrun` 是新标配，每个 rank 都是独立进程，挂了可以 elastic 重启。多机训练几乎一定要走 torchrun。

### 1. 核心结论

`torchrun` 是 PyTorch 官方推荐的分布式 launcher，负责启动多进程、注入 `RANK` / `WORLD_SIZE` / `LOCAL_RANK` 等环境变量并处理多机 rendezvous，适合标准 DDP / FSDP 训练。`mp.spawn` 只是 Python 侧的进程创建工具，通常需要脚本自己计算 rank、设置设备并调用 `init_process_group`，更适合单机实验或需要高度自定义进程生命周期的场景。

### 2. 底层原理

DDP 运行的前提是每个训练进程都知道自己的全局 rank、本地 rank、world size 以及 master 地址。`torchrun` 把这些“进程发现与环境注入”的工作放在训练脚本之外，由 launcher 统一完成；脚本只需读取环境变量并初始化进程组。`mp.spawn` 则是在脚本内部启动多个子进程，子进程拿到的通常只是本地序号 `i`，全局 rank、节点编号和 rendezvous 参数仍需用户手工传入和维护。

### 3. 关键机制 / 流程 / 数据结构

1. `torchrun` 的典型流程：
   - 外部命令行指定 `--nproc-per-node`、`--nnodes`、`--node_rank`、`--master_addr`、`--master_port` 或 rendezvous 参数。
   - launcher 为每个进程设置环境变量，如 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`。
   - 脚本中按 `LOCAL_RANK` 绑定 GPU，并调用 `init_process_group(init_method="env://")`。
2. `mp.spawn` 的典型流程：
   - 主进程先知道本机要拉起多少 worker。
   - `spawn(fn, nprocs=n)` 将本地序号传给子进程。
   - 子进程内部自行计算 `global_rank = node_rank * nprocs + local_rank`，再设置 CUDA device、master 地址和 world size，最后初始化进程组。
3. 多机场景下，`torchrun` 天然围绕 rendezvous 设计；`mp.spawn` 更像“本机内启动器”，跨节点协调需要自行补齐。

### 4. 工程权衡 / 性能影响

两者在训练阶段的算子性能通常没有本质差别，差别主要体现在启动与运维复杂度上：
- `torchrun`：
  - 多机配置标准化，和大多数教程、作业系统、容器编排更兼容。
  - 更适合统一日志管理、失败处理和标准化部署。
  - 代码更简洁，不易把 rank / world size 写错。
- `mp.spawn`：
  - 单机脚本内集成更灵活，便于做自定义控制流或研究性实验。
  - 但样板代码更多，多机时更易出现环境变量、端口和 rank 映射错误。
  - 若脚本里已提前初始化 CUDA 上下文，进程派生方式处理不当还可能带来额外问题。

### 5. 常见追问 / 易错点

- `mp.spawn` 不会自动提供 `LOCAL_RANK` 等环境变量；很多示例代码直接读取这些变量，只适用于 `torchrun`。
- 旧版 `python -m torch.distributed.launch` 已被 `torchrun` 取代，PyTorch 2.x 文档明确标记为 deprecated，新项目不应再使用。
- `torchrun` 的弹性模式（`--max-restarts`、`--rdzv-backend=c10d`）是 elastic training 的标准接口，若上层使用 Kubernetes + Kueue / Volcano，通常通过 torchrun + c10d rendezvous 实现节点重入。
- 启动方式不同，不代表训练要改成“一进程多卡”；DDP 仍通常是一张 GPU 对应一个进程。
- 多机时如果各节点用 `mp.spawn`，但 global rank 计算或 master 配置不一致，常直接表现为初始化 hang。

### 6. 实践建议

标准 DDP / FSDP 训练默认优先使用 `torchrun`，尤其是多机多卡和需要交给调度系统管理的任务。只有在单机研究脚本、需要把子进程创建深度嵌入 Python 代码时，再考虑 `mp.spawn`。无论哪种方式，都建议把“设备绑定、rank 计算、进程组初始化”封装成统一入口，避免训练逻辑与启动细节耦合。

### 7. 30 秒速答

- 一句话核心：torchrun 是生产推荐，支持 elastic、rendezvous 和故障重启；mp.spawn 仅适合单机调试
- 关键机制：torchrun 由外部 launcher 管理 worker 生命周期；mp.spawn 在 Python 进程内 fork
- 易踩坑 / 关键权衡：mp.spawn 配合 CUDA 容易遇到 fork-after-cuda-init 问题，必须用 spawn 模式
- 面试加分关键词：torchrun / torch.distributed.launch / mp.spawn / elastic

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 torchrun 与 mp.spawn 的区别？
- [ ] 你能不能解释 torchrun 由外部 launcher 管理 worker 生命周期；mp.spawn 在 Python 进程内 fork？
- [ ] 你能不能举一个 多机训练统一用 torchrun + rendezvous_backend=c10d 的具体场景？
- [ ] 你能不能说出 生产环境用 mp.spawn 无法 elastic 自动恢复 这种常见错误模式？

## Q12. 多机多卡训练时，如何设置 NCCL_SOCKET_IFNAME 和 NCCL_IB_DISABLE？

> 🟡 进阶 · 机器有好几张网卡，NCCL 默认挑一张可能挑到管理网，10Gb 拖死 100Gb IB。`NCCL_SOCKET_IFNAME` 让你显式指定走哪张卡，配错了通信带宽直接掉一个数量级，多机训练几乎跑不动。

### 1. 核心结论

`NCCL_SOCKET_IFNAME` 用来告诉 NCCL 选择哪张网络接口做 socket bootstrap 与 TCP 通信；`NCCL_IB_DISABLE` 用来控制是否禁用 InfiniBand / RDMA verbs 通道。纯以太网集群通常应显式指定正确的 `eth` / `bond` / `ens` 接口，并设置 `NCCL_IB_DISABLE=1`；具备 IB 或 RoCE 的集群通常保持 `NCCL_IB_DISABLE=0`，必要时再配合 `NCCL_IB_HCA` 精确选择 HCA。

### 2. 底层原理

NCCL 在多机训练里需要先让各 rank 互相发现并建立连接，bootstrap 常依赖 socket；真正的数据面可能走 socket / TCP，也可能走 IB / RDMA。若接口选错，例如选到 `lo`、`docker0`、容器虚拟网卡或不可达的副网卡，常见结果就是初始化慢、建链失败或训练 hang。`NCCL_IB_DISABLE` 是在“启用高性能 RDMA 通道”与“强制退回 socket”之间切换。

### 3. 关键机制 / 流程 / 数据结构

1. 先盘点节点网络：确认训练流量应走哪张物理接口，避免把管理网、回环网卡或虚拟网卡选进来。
2. 设置 `NCCL_SOCKET_IFNAME`：
   - 可写具体接口或前缀，如 `eth0`、`bond0`、`ens`。
   - 也可用排除语法，例如 `^lo,docker,veth`，避免 NCCL误选虚拟接口。
3. 纯以太网集群：
   - 让所有节点都使用同一类可达接口。
   - 设置 `NCCL_IB_DISABLE=1`，避免 NCCL 尝试不存在或未配置好的 IB 通道。
4. IB / RoCE 集群：
   - 通常设置 `NCCL_IB_DISABLE=0`，允许 NCCL 使用 RDMA。
   - 若机器上有多张 HCA，可进一步用 `NCCL_IB_HCA=mlx5_0,mlx5_1` 等方式约束设备。
   - 多 NIC 机型（H100/GB200）通常把所有 HCA 都列上，NCCL 会做多 rail 并行，实测带宽接近累加。
5. 用 `NCCL_DEBUG=INFO` 验证：
   - 看 `NET/Socket` 是否选择了预期接口。
   - 看 `NET/IB` 是否成功启用 verbs 链路，而不是悄悄退回 TCP。

### 4. 工程权衡 / 性能影响

配置正确时，IB / RoCE 往往能显著降低延迟、提高跨机带宽；但它也更依赖驱动、固件、子网管理和容器网络配置。强制 `NCCL_IB_DISABLE=1` 的兼容性更好，但跨机性能通常明显弱于 RDMA。`NCCL_SOCKET_IFNAME` 看似只是选网卡，实则会直接影响 bootstrap 成功率、回退路径和最终吞吐；若误选慢网卡或跨子网接口，问题常表现为“能跑但很慢”或“偶发 hang”。

### 5. 常见追问 / 易错点

- `NCCL_SOCKET_IFNAME` 不只在纯 socket 模式下有意义；即便启用了 IB，bootstrap 和部分回退路径仍依赖可达的 socket 接口。
- RoCE 虽然跑在以太网上，但通常仍属于 verbs / RDMA 路径，不能简单理解为“以太网就该设 `NCCL_IB_DISABLE=1`”。
- 不同节点接口名字可以不同，但必须都指向同一张可互通的训练网络；否则容易出现单向可达或路由不一致。
- 不要让 NCCL 自动从大量虚拟接口中猜网卡，容器环境里这类误选非常常见。

### 6. 实践建议

建议把这两个变量固化到启动脚本或作业模板中，而不是依赖人工临时导出。新集群上线时，先用 `nccl-tests` 的 `all_reduce_perf` / `all_gather_perf` 配合 `NCCL_DEBUG=INFO` 验证链路带宽与拓扑，再跑正式任务。若网络环境复杂，除了 `NCCL_SOCKET_IFNAME` 与 `NCCL_IB_DISABLE`，通常还应一并梳理 `NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`（RoCEv2 常见坑）、`NCCL_NVLS_ENABLE`（NCCL 2.22+ NVLink SHARP）、容器网卡映射和防火墙策略。

### 7. 30 秒速答

- 一句话核心：前者指定 NCCL 使用的 TCP/IP 网卡，后者强制关闭 IB 走 socket
- 关键机制：多 NIC 集群必须显式指定 IFNAME，否则 NCCL 可能误选管理网或 docker bridge
- 易踩坑 / 关键权衡：IB_DISABLE=1 是 debug 工具不是生产配置，会把训练性能砍掉 10×
- 面试加分关键词：NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_DISABLE

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 NCCL_SOCKET_IFNAME 与 NCCL_IB_DISABLE？
- [ ] 你能不能解释 多 NIC 集群必须显式指定 IFNAME，否则 NCCL 可能误选管理网或 docker bridge？
- [ ] 你能不能举一个 A100 集群把 IFNAME 设成 ib0 或 bond0，HCA 设成 mlx5_0 的具体场景？
- [ ] 你能不能说出 生产开 IB_DISABLE=1 跑慢了一倍还以为是模型问题 这种常见错误模式？

## Q13. 梯度压缩（gradient compression）的方法有哪些？fp16 vs bf16 vs 1bit Adam？

> 🔴 专家 · 跨数据中心训练时通信带宽是真贵，fp32 梯度全量传输根本扛不住。fp16/bf16 压一半、1bit Adam 极限压到 1/32，但每种方案都有精度风险——选错了模型直接发散。

### 1. 核心结论

梯度压缩常见方法包括低精度通信、量化、稀疏化、误差反馈和延迟同步等。`fp16` / `bf16` 属于风险最低、最常见的“轻量压缩”，可把通信体积相对 fp32 直接减半；其中 bf16 数值稳定性通常更好。`1bit Adam` 属于更激进的通信压缩型优化器，带宽节省更大，但实现复杂、约束更多，通常只在跨机通信已成为主瓶颈时才值得引入。

### 2. 底层原理

梯度压缩的目标是减少分布式同步时需要传输的数据量，常见切入点有三类：
- 减少每个元素的比特数，例如从 fp32 降到 fp16 / bf16 / int8 / 1bit。
- 减少需要发送的元素个数，例如只发 top-k 大梯度或阈值以上梯度。
- 减少同步频率，例如 local SGD、延迟同步或分层同步。

压缩收益来自更低的通信开销，但代价通常是量化噪声、残差积累、收敛变慢或实现复杂度上升，因此本质是在“带宽”与“优化精度”之间做折中。

### 3. 关键机制 / 流程 / 数据结构

常见方法可以概括为：
1. 低精度通信：
   - `fp16`：把梯度或通信 bucket 转成 fp16，同步后再转回或在 master weight 上更新。
   - `bf16`：同样是 16 bit 传输，但指数范围更大，通常更不容易溢出或下溢。
2. 量化通信：
   - 8bit / 4bit / block-wise quantization：按块记录 scale，再做量化与反量化。
   - signSGD / 1bit 类方法：只传符号位或极低比特表示，通常要配合残差补偿。
3. 稀疏化：
   - top-k、threshold、random-k，只同步少量重要梯度。
   - 未发送部分保存在 residual buffer 中，等待后续补偿。
4. 误差反馈：
   - 把本轮压缩误差累积到下一轮输入，降低长期偏差。
5. 延迟或局部同步：
   - 多步本地更新后再同步，进一步减少通信频率，但会引入额外 stale effect。

若专门比较三者：
- `fp16`：通信体积约为 fp32 的一半，生态成熟，但数值范围较窄，常需 loss scaling。
- `bf16`：通信体积同样减半，但指数范围接近 fp32，通常比 fp16 更稳。
- `1bit Adam`：可将跨节点通信进一步压到极低比特，常配合 warmup、误差反馈和特定优化器流程，不属于“直接替换 dtype”那么简单。

### 4. 工程权衡 / 性能影响

从工程可用性看，三者大致是“越激进，收益越依赖场景”：
- `fp16`
  - 优点：成熟、硬件支持广、改造成本低。
  - 缺点：易受溢出 / 下溢影响，数值稳定性弱于 bf16。
- `bf16`
  - 优点：与 fp16 同样节省带宽，但通常更稳定，已成为现代大模型训练的主流选择之一。
  - 缺点：尾数位更少，老硬件支持有限。
- `1bit Adam`
  - 优点：当跨机网络成为主瓶颈时，通信节省可能远超 16 bit 方案。
  - 缺点：实现和调参复杂，常依赖特定框架支持；若训练本来不是 communication bound，实际收益有限。

因此，`fp16` / `bf16` 更像默认工程手段，`1bit Adam` 更像特定瓶颈下的专项优化。

### 5. 常见追问 / 易错点

- 混合精度训练不完全等于梯度压缩，但在分布式场景下，低精度梯度通信确实会直接减少带宽占用。
- bf16 并不是“比 fp16 更高精度”；它的主要优势是指数范围更大，因此训练稳定性通常更好。
- 1bit 类方法若没有 residual / error feedback，收敛质量往往会明显恶化。
- 压缩不是越强越好；如果训练主要受计算或数据加载限制，激进压缩可能只增加系统复杂度而几乎不提速。

### 6. 实践建议

默认优先级通常是：先确认是否真的被通信卡住，再优先尝试 bf16，其次是 fp16 + 稳定的 loss scaling；只有在大规模多机训练中确认网络已成为主要瓶颈，且框架对 1bit Adam、0/1 Adam、Lamb 一类压缩优化器有成熟支持时，才建议引入更激进的压缩方案。近年还出现了 FP8 通信（DeepSeek-V3、NCCL 2.22+）、PowerSGD（rank-k 低秩近似，PyTorch `ddp_comm_hooks` 原生支持）等实践，可作为中间档选择。评估时不要只看带宽利用率，还要同时看 step time、loss 曲线和最终收敛质量。

### 7. 30 秒速答

- 一句话核心：fp16/bf16 半精度通信、1-bit Adam、PowerSGD 等低秩压缩
- 关键机制：降低每个 element 字节数或秩，减小 AllReduce 总流量
- 易踩坑 / 关键权衡：1bit/PowerSGD 收敛敏感，需 warmup + error feedback；bf16 是默认推荐
- 面试加分关键词：bf16 / fp16 / 1-bit Adam / PowerSGD / Error Feedback

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 梯度压缩的常见方法？
- [ ] 你能不能解释 降低每个 element 字节数或秩，减小 AllReduce 总流量？
- [ ] 你能不能举一个 bandwidth-bound 跨机训练把 reduce 通信换成 bf16 的具体场景？
- [ ] 你能不能说出 梯度直接低 bit 压缩没加 error feedback 导致发散 这种常见错误模式？

## Q14. 异步训练（如 Hogwild!）在工业界为什么很少用？

> 🟡 进阶 · 论文里 async SGD 听起来很美——不用等其他 worker、扩展性极好。但工业界几乎没人用：stale gradient 会让 Adam 类优化器收敛性彻底崩盘，复现实验和调参都没法做，得不偿失。

### 1. 核心结论

异步训练之所以在工业界较少成为主流方案，核心原因是它用更弱的一致性换取更少的同步等待，但现代深度学习更在乎稳定收敛、可复现性和系统可控性。对于大多数 dense GPU 训练任务，参数陈旧、优化噪声、调试困难和工程复杂度带来的代价，往往超过它节省的那部分同步成本。

### 2. 底层原理

Hogwild! 这类方法的基本假设是：多个 worker 可以在几乎无锁的情况下并发更新共享参数，冲突较少时仍能收敛。这个假设更适合稀疏、CPU、共享内存式的优化问题。工业界主流的大模型训练则通常具有以下特征：
- 参数和梯度高度稠密，更新冲突远比稀疏场景严重。
- 训练运行在 GPU / 多机集群上，不是简单共享内存模型。
- 常使用 momentum、Adam、loss scaling、BN 统计等对一致性更敏感的机制。

一旦 worker 基于不同版本参数计算梯度，再异步写回，就会引入明显的 stale gradient 问题，优化路径也更难分析。

### 3. 关键机制 / 流程 / 数据结构

异步训练常见机制大致如下：
1. 每个 worker 拉取某一时刻的参数副本。
2. 本地完成若干步前向 / 反向。
3. 不等待其他 worker，直接把梯度或参数更新推回共享参数或参数服务器。
4. 其他 worker 可能已经基于更新前的旧参数继续计算。

这会带来几个直接后果：
- 同一时刻不同 worker 使用的模型版本不一致。
- 梯度对应的参数点不一致，更新噪声变大。
- 若再叠加 Adam 一类状态化优化器，`exp_avg`、`exp_avg_sq` 也会出现版本漂移。
- 在 GPU 集群中，collective、checkpoint、混合精度和故障恢复都更难定义一致语义。

### 4. 工程权衡 / 性能影响

异步训练并非完全没有优势，它在某些场景下确实有：
- 更弱的 barrier，同步等待更少。
- 对 straggler 有一定容忍度。
- 在部分在线学习、推荐或强化学习系统里，可换来更高系统利用率。

但对主流工业训练任务，其主要问题更突出：
- 吞吐提升不一定转化为更短的 time-to-target，因收敛步数可能明显增加。
- 结果波动大、复现实验困难，不利于大规模迭代和回归测试。
- 和现代 GPU 通信栈、collective 优化、同步 checkpoint 体系不够契合。
- 故障恢复与断点续训复杂，因为系统中不存在一个清晰的一致更新边界。

### 5. 常见追问 / 易错点

- Hogwild! 最初针对的是稀疏 CPU 场景，不能直接外推到稠密大模型训练。
- “设备利用率更高”不等于“整体训练更快”，真正要看的是达到目标精度所需总时间。
- 工业界不是完全不用异步思想；它在参数服务器、在线学习、actor-learner 架构中仍有局部应用，但很少作为通用大模型训练主路径。
- 梯度累积、流水并行和异步训练不是一回事，前两者通常仍可以保持全局优化语义一致。

### 6. 实践建议

对于大多数监督学习和大模型训练，优先采用同步数据并行，再通过梯度桶、通信重叠、梯度累积和更快互联去优化性能。若系统确实受 straggler 或超大规模在线更新限制，应优先评估 bounded staleness（SSP）、局部异步或解耦数据采集，而不是直接采用完全异步参数更新。RLHF 场景里 actor / learner 分离（如 OpenRLHF、veRL）也是“采样异步 + 学习同步”的混合，而非 Hogwild! 那种完全无锁参数更新。任何异步方案都应以 time-to-target、收敛方差和恢复复杂度为主指标做 A/B 验证。

### 7. 30 秒速答

- 一句话核心：staleness 让收敛不可控，且现代同步 SGD + 高速互联已经足够快
- 关键机制：Hogwild! 等无锁更新让不同 worker 看到不一致权重，理论收敛性弱
- 易踩坑 / 关键权衡：同步训练在 BF16 + NVLink/IB 下通信成本低，异步带来的吞吐收益已不明显
- 面试加分关键词：Hogwild! / Stale Gradient / ASGD / Sync vs Async SGD

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 异步训练为何工业界很少用？
- [ ] 你能不能解释 Hogwild! 等无锁更新让不同 worker 看到不一致权重，理论收敛性弱？
- [ ] 你能不能举一个 参数服务器架构的 CTR 模型才偶尔用异步 的具体场景？
- [ ] 你能不能说出 LLM 训练盲目尝试异步反而踩 staleness 收敛坑 这种常见错误模式？

## Q15. 如何实现自定义的分布式优化器？继承 torch.optim.Optimizer 的注意事项？

> 🔴 专家 · 自己写优化器时分布式接口最容易踩坑：state_dict 没分片、step 里偷偷做了 host 同步、param_group 顺序在各 rank 不一致——任何一个都会让 resume 后 loss 抽风。

### 1. 核心结论

自定义分布式优化器通常不只是“继承 `torch.optim.Optimizer` 并重写 `step()`”，而是要同时设计参数分片、梯度通信、优化器状态布局和 checkpoint 语义。`Optimizer` 只提供 param group、state 和 `state_dict` 的基础框架；真正的难点在于所有 rank 必须以一致顺序管理同一组参数，并在合适时机执行 `reduce_scatter`、本地更新和 `all_gather` / `broadcast`。

### 2. 底层原理

普通优化器默认假设完整参数、完整梯度和完整状态都在本地进程中。分布式优化器则希望避免“每个 rank 都维护一整份 optimizer state”，常见做法是：
- 先把梯度按参数或 flat buffer 切分。
- 通过 `reduce_scatter` 把梯度分发给负责该 shard 的 rank。
- 由 owner rank 只更新自己持有的参数分片及其状态，如 `exp_avg`、`exp_avg_sq`。
- 在下次前向前，再按需要把更新后的参数分片 `all_gather` 成可计算视图。

这类流程和 ZeRO / FSDP 的 optimizer state sharding 思路相近，只是把控制权放到了自定义实现中。

### 3. 关键机制 / 流程 / 数据结构

一个可落地的实现通常包含以下部分：
1. 初始化阶段：
   - 保证所有 rank 的参数遍历顺序完全一致。
   - 调用 `super().__init__(params, defaults)` 正确建立 param groups。
   - 为每个参数或 flat buffer 生成 shard 元数据，如 owner rank、offset、长度、dtype、device。
2. 反向后的通信阶段：
   - 将梯度按 bucket 或 flat buffer 聚合。
   - 执行 `reduce_scatter` 或等价通信，让 owner rank 收到自己那一段聚合梯度。
   - 若使用 AMP，还要保证 overflow 判定、unscale 和梯度裁剪在全局语义上一致。
3. 本地优化阶段：
   - 只在 owner rank 上懒初始化本地状态，如 `state[p]['exp_avg']`、`state[p]['exp_avg_sq']`、`state[p]['step']`。
   - 在 `torch.no_grad()` 语义下更新本地参数分片。
4. 参数再物化阶段：
   - 若前向需要完整参数，则对更新后的分片执行 `all_gather`。
   - 若系统本身已由 FSDP / ZeRO 管理参数视图，则应与其参数物化时机配合，而不是重复 gather。
5. checkpoint 阶段：
   - 明确 `state_dict` 是返回本地分片状态，还是先聚合成完整状态再导出。
   - 恢复时必须保证参数顺序、分片规则和进程组拓扑兼容。

### 4. 工程权衡 / 性能影响

自定义分布式优化器的收益主要在于：可以把 optimizer state 显存降下来，并且根据模型结构定制通信与计算 overlap。代价则非常明显：
- 实现复杂度远高于普通优化器，collective 次序稍有不一致就可能 hang。
- `state_dict`、断点续训、参数重排和混合精度处理都更容易出错。
- 若模型和网络拓扑并没有特殊需求，手写实现往往难以超过成熟框架的稳定性。

因此，它更适合确实存在定制通信模式、特殊分片需求或研究型优化器设计的场景，而不是通用首选方案。

### 5. 常见追问 / 易错点

- `Optimizer.state` 是按参数对象 identity 建索引的，初始化后不要把参数张量替换成新的 Python 对象，否则状态会丢失或错绑。
- 所有 rank 的参数顺序必须一致；哪怕只是模块注册顺序不同，也可能导致 shard 映射和通信结果错位。
- `step()` 不能假设每个参数都有梯度，应正确处理 `grad is None`、冻结参数以及不支持的 sparse grad。
- 若使用混合精度，master weight、unscale、overflow 跳步和梯度裁剪必须在所有 rank 上保持一致语义。
- `state_dict()` 的语义要提前定义清楚：是 local shard checkpoint 还是 full checkpoint；否则恢复流程会非常混乱。
- 如果不支持 closure，应显式限制而不是隐式忽略，以免和通用训练循环接口冲突。

### 6. 实践建议

更稳妥的路径通常是先基于现有优化器做 wrapper，或直接扩展 ZeRO、FSDP2 optimizer state dict、`torch.distributed.optim.ZeroRedundancyOptimizer`、Megatron-LM `DistributedOptimizer`、Apex `DistributedFusedAdam` 一类成熟实现，而不是从零手写全部分布式逻辑——这些实现已把 reduce-scatter / all-gather 与 optimizer step 的 overlap、FP8/BF16 master weight、断点续训等细节打磨过多轮。开发时先验证单进程数值等价，再做 2 rank、4 rank 的梯度一致性测试、checkpoint round-trip 测试和故障注入测试。只有当 profiling 已明确表明现有方案在显存或通信上无法满足需求时，才值得维护一套自定义分布式优化器。

### 7. 30 秒速答

- 一句话核心：state_dict 必须可分片，step 内显式 AllReduce/ReduceScatter，注意 param_group 的设备一致性
- 关键机制：继承 torch.optim.Optimizer 后重写 step，并实现 sharded state_dict 钩子供 FSDP 调用
- 易踩坑 / 关键权衡：不小心在 step 里做了 .item() 同步，会让所有 rank 串行化
- 面试加分关键词：Optimizer / state_dict / FSDP optim_state_dict / fused step

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 自定义分布式优化器要点？
- [ ] 你能不能解释 继承 torch.optim.Optimizer 后重写 step，并实现 sharded state_dict 钩子供 FSDP 调用？
- [ ] 你能不能举一个 LAMB/Lion 在 FSDP 下需实现 sharded state 的具体场景？
- [ ] 你能不能说出 step 里 host-device 同步导致 GPU 空转 这种常见错误模式？

## Q16. Tensor Parallelism（TP）的 fused attention 实现细节？

> 🔴 专家 · attention 里 Q/K/V 一切就涉及多次 all-reduce，朴素拆法通信开销吞掉所有收益。Megatron 的做法是把 QKV 在 head 维拆开、output projection 留到最后才同步——这是大模型推理框架里几乎人人抄的标准实现。

### 1. 核心结论

TP 下的 fused attention 通常不是把整个 attention 在所有卡上做一次大通信，而是先按 head 或 hidden shard 切分，使每个 rank 本地完成自己那一部分 QKV 投影与 attention 核心计算，再只在必要位置做通信。最常见的融合点包括 QKV 线性层融合、scale-mask-softmax 融合、attention dropout 融合，以及与 FlashAttention 风格 tile 计算结合；真正需要跨 rank 的往往是输入/输出投影阶段，而不是每一步 attention score 计算。

### 2. 底层原理

在经典多头注意力中，若 hidden size 为 `h`、头数为 `n_heads`，TP 常把 `n_heads` 或对应的 hidden 维均匀切到多个 rank。只要每个 attention head 完整落在单个 rank 上，那么该 head 内部的 `QK^T`、softmax、`PV` 都可以完全本地完成，不需要在 score 矩阵级别跨卡通信。

fused attention 的目标有两层：
1. 减少 kernel launch 次数，把原本分离的 reshape、bias、transpose、scale、mask、softmax、dropout、matmul 串成更少的 kernel。
2. 避免中间张量完整落地到 HBM，通过寄存器 / shared memory / SRAM tile 直接衔接前后算子。

因此，TP 与 fused attention 的组合，是在“张量切分后尽量本地算完”和“本地算子链尽量融合”两个层面同时优化。

### 3. 关键机制 / 流程 / 数据结构

典型实现流程可概括为：
1. 输入 `X` 先进入 column-parallel QKV projection，每个 rank 只生成自己负责的一组 `Q/K/V` 分片。
2. 在 fused kernel 中完成 bias add、reshape、transpose，得到本地 head 视图。
3. 对本地 head 执行 `QK^T`，并把 scale、causal mask、padding mask、softmax 以及可选 dropout 融合到同一计算流水。
4. 继续在本地完成 `softmax(QK^T) @ V`，生成本 rank 的 context。
5. 输出投影通常接 row-parallel linear：各 rank 先本地做一部分 GEMM，再对 hidden 维做一次 all-reduce 或 reduce-scatter，恢复后续层所需布局。

几个实现细节很关键：
- 若采用 grouped-query attention（GQA）或 multi-query attention（MQA），`num_q_heads`、`num_kv_heads` 与 TP 大小之间的整除关系必须对齐，否则需要先在 TP 组内 all-gather KV 再做 attention；Llama 3、Mistral、DeepSeek-V3 等模型在 TP=8 时对 KV head 数的选择就是这一约束的直接体现。
- FlashAttention 类实现（FA2/FA3）按 block streaming 方式重算局部 softmax 统计量 `(m, l)`，使 `L x L` attention matrix 不再落 HBM；FA3 进一步在 Hopper 上引入 warp specialization 与 FP8 支持，与 fused QKV/output projection 组合后端到端收益更显著。
- 训练态还要处理 dropout RNG、causal mask、变长序列和 packed sequence（`cu_seqlens`、`seq_idx`），这些路径都会影响 fuse 的边界；某些 mask（如 ALiBi、sliding window、prefix-LM）需要定制 kernel 而不能直接复用 SDPA 通用实现。

### 4. 工程权衡 / 性能影响

fused attention 在 TP 下的主要收益是显存访问减少、kernel launch 更少、局部算子链更紧凑，因此通常能显著改善 attention 的 latency 和吞吐。但它也带来几个约束：
- 融合越深，对张量 layout、dtype、sequence length 对齐和 mask 形式越敏感。
- TP shard 方式若使一个 head 被切碎到多个 rank，就会破坏“attention 本地完成”的前提，通信会明显增加。
- FlashAttention 风格 kernel 对硬件代际、shared memory 大小和 head dim 有较强适配要求。
- 训练中的变长 batch、KV 缓存、prefix mask 等特殊路径，往往会让部分 fuse 回退到较慢实现。

### 5. 常见追问 / 易错点

- 很多人以为 TP attention 一定需要在 `QK^T` 时做 all-reduce；实际上只要按 head 切分，score 计算通常是本地的。
- fused attention 不等于 FlashAttention。前者强调算子融合，后者更强调 IO-aware tile 计算；二者常组合，但不是同义词。
- 输出投影后的通信位置容易和 QKV 投影混淆。常见做法是 QKV 用 column parallel，输出投影用 row parallel。
- 若 sequence parallel 同时开启，attention 前后的 layout 变换与 reduce-scatter / all-gather 关系会更复杂，不能只看单层公式。

### 6. 实践建议

优先采用成熟框架已有的 fused attention 路径（Megatron-LM `te.Attention`、TransformerEngine、xFormers、PyTorch SDPA + FA 后端），而不是自己手写跨 kernel 拼接。调优时重点看三类指标：attention kernel 时间、HBM 带宽占用、输出投影处的通信尾部。若 profile 显示 attention 本体已很快，但层尾 all-reduce 很长，问题往往不在 fused kernel，而在 TP 切分粒度、并行组大小或网络拓扑；此时更应考虑叠加 sequence parallel 把 layernorm/dropout 的激活按 token 维分摊，或评估 async TP 让 all-gather/reduce-scatter 与后续 GEMM 重叠。

### 7. 30 秒速答

- 一句话核心：QKV 投影按 head 维切分到不同 rank，attention 内部计算无需通信，仅在 output projection 做 AllReduce
- 关键机制：QKV 是 column-parallel，attention 算完后 output 投影 row-parallel + AllReduce
- 易踩坑 / 关键权衡：num_heads 必须能被 TP size 整除，否则切不均匀
- 面试加分关键词：column_parallel / row_parallel / AllReduce / FlashAttention TP

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 TP 下 fused attention 的关键？
- [ ] 你能不能解释 QKV 是 column-parallel，attention 算完后 output 投影 row-parallel + AllReduce？
- [ ] 你能不能举一个 LLaMA 70B 用 TP=8，每 rank 处理 8 个 head 的具体场景？
- [ ] 你能不能说出 num_heads=64 但 TP=12 导致无法均分 这种常见错误模式？

## Q17. Megatron-LM 的 column parallel 和 row parallel 的矩阵划分策略？

> 🔴 专家 · 一个 MLP 两个 Linear，怎么切才能让中间结果不用通信？Megatron 给的答案是先 column 切（输出维），后 row 切（输入维），中间不需要同步——这套 `f / g` 算子正是大模型 TP 的灵魂。

### 1. 核心结论

Megatron-LM 的张量并行把线性层 `Y = XW + b` 拆成两类经典策略：column parallel 按 `W` 的输出维切分，row parallel 按 `W` 的输入维切分。前者适合产生“可继续分片流动”的中间激活，如 QKV 或 MLP 第一层；后者适合把前一层分片结果汇总回统一语义，如 attention output projection 或 MLP 第二层。

### 2. 底层原理

设 `X` 形状为 `[B, H_in]`，权重 `W` 形状为 `[H_in, H_out]`。
- column parallel：把 `W` 按列切成 `W_1, W_2, ..., W_p`，即每个 rank 负责一部分 `H_out`。
- row parallel：把 `W` 按行切成 `W_1, W_2, ..., W_p`，即每个 rank 负责一部分 `H_in`。

两种切法的核心区别在于通信位置：
- column parallel 让所有 rank 读相同输入 `X`，各自产生不同输出分片，通常本层末尾不急着聚合。
- row parallel 让输入 `X` 本身就是分片的，各 rank 先算部分结果，再把局部结果求和，才能得到完整输出。

### 3. 关键机制 / 流程 / 数据结构

1. Column Parallel Linear
- 划分方式：`W = [W_1, W_2, ..., W_p]`，沿输出维切分。
- 本地计算：`Y_i = X W_i`。
- 输出形态：得到 `Y = [Y_1, Y_2, ..., Y_p]` 的分片视图。
- 通信特点：若下一层也接受分片输入，则可暂不 all-gather；若后续需要完整 hidden，再显式 gather。

2. Row Parallel Linear
- 划分方式：`W^T = [W_1^T, W_2^T, ..., W_p^T]`，等价于 `W` 沿输入维切分。
- 输入要求：`X` 也按同样维度切成 `X_1, X_2, ..., X_p`。
- 本地计算：`Z_i = X_i W_i`。
- 输出恢复：执行 all-reduce 或等价求和，得到完整输出 `Y = Σ Z_i`。

Megatron 常见搭配是：
- attention 中的 QKV projection 用 column parallel；
- attention 输出投影用 row parallel；
- MLP 第一层扩维线性用 column parallel；
- MLP 第二层回缩线性用 row parallel。

通信对称关系可以这样记忆：forward 的 column parallel 入口需要 identity（或依赖上游已是完整视图），其反向是 all-reduce；forward 的 row parallel 出口需要 all-reduce，其反向才是 identity。Megatron 用 `f` / `g` 两个自定义 autograd 函数封装这一对通信算子，开启 sequence parallel 后，`f` 的 all-reduce 被拆成 reduce-scatter + all-gather，从而在 token 维把 layernorm/dropout 激活摊薄。

```
   MLP block (TP=N) — 中间层无需通信
                      ┌────────────┐
       X (完整)  ──f──▶│ ColumnPar  │  W₁ 按输出维切
   ┌──────────┐       │  Y_i = X·W₁_i │
   │ fwd: id  │       └─────┬──────┘
   │ bwd: AR  │             │ Y_i (分片, 各 rank 持有不同列)
   └──────────┘             ▼
                      ┌────────────┐
                      │  GeLU      │  逐元素, 在分片上本地做
                      └─────┬──────┘
                            │ Z_i (仍分片)
                            ▼
                      ┌────────────┐
                      │ RowParallel│  W₂ 按输入维切
                      │  out_i = Z_i·W₂_i │
                      └─────┬──────┘
                            │ partial sum
                            ▼  ┌──────────┐
       Y (完整)  ◀──g──── Σ    │ fwd: AR  │
                               │ bwd: id  │
                               └──────────┘
   ★ 中间 GeLU 不需要任何 collective; 通信只在两端
```

### 4. 工程权衡 / 性能影响

这种搭配的好处是把通信压到层与层之间最自然的位置：
- column parallel 让扩维或多头展开阶段少一次不必要聚合。
- row parallel 则在语义上需要把多个 partial sum 合并时再做 collective。

但代价也很明确：
- column parallel 要么要求输入广播可见，要么要求前一层已对齐 layout。
- row parallel 的 all-reduce 是硬同步点，若网络慢，尾部会明显拉长。
- 参数初始化、bias 处理、state dict 保存都要考虑 shard 语义，否则单卡恢复与多卡恢复容易不一致。bias 常放在 column parallel 这一侧并按 shard 存，因为 row parallel 侧若每个 rank 都加一份完整 bias，再 all-reduce 会把 bias 累加 `N_tp` 次。

### 5. 常见追问 / 易错点

- column parallel 不是按 batch 维切，而是按输出 hidden 维切。
- row parallel 不是简单地把权重切开后各算各的；它要求输入也按同样规则分片，否则矩阵乘法维度不对齐。
- 很多人把“all-gather 输出”和“all-reduce 求和”混为一谈。column parallel 常对应 gather 拼接，row parallel 常对应 reduce 求和。
- 这两种线性层通常需要成对出现，否则中间激活 layout 会越来越难管理。

### 6. 实践建议

阅读 Megatron 代码时，优先沿着一条完整 block 跟踪 tensor layout，而不是孤立看某个 linear 类。工程调优时重点关注两个位置：column-parallel 后是否真的避免了无谓 gather，row-parallel 后的 all-reduce 是否成为长尾。若 profile 显示通信占比过高，先检查 TP 大小是否过大，再看 hidden size 是否足以摊薄通信开销。

### 7. 30 秒速答

- 一句话核心：column-parallel 按输出维切（A 分列），row-parallel 按输入维切（A 分行）
- 关键机制：column 之后是 row 可以把两次通信合并为一次 AllReduce；MLP 的 up + down 就是这种 pattern
- 易踩坑 / 关键权衡：单独的 column-parallel 输出需 all-gather，需配套 row-parallel 形成对偶才省通信
- 面试加分关键词：ColumnParallelLinear / RowParallelLinear / Megatron-Core

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 column vs row parallel 矩阵划分？
- [ ] 你能不能解释 column 之后是 row 可以把两次通信合并为一次 AllReduce；MLP 的 up + down 就是这种 pattern？
- [ ] 你能不能举一个 MLP h→4h 用 column，4h→h 用 row，中间无通信 的具体场景？
- [ ] 你能不能说出 把两层都用 column-parallel 导致每层都要 all-gather 这种常见错误模式？

## Q18. Pipeline Parallelism（PP）的 bubble 问题如何量化？GPipe vs PipeDream？

> 🟡 进阶 · PP 把模型分段塞到不同卡上，但流水线开头和结尾必然有一段时间某些卡在干等——这就是 bubble。bubble 占比直接决定 PP 能不能跑出有用的吞吐，调度算法选 GPipe 还是 1F1B 是核心。

### 1. 核心结论

PP 的 bubble 本质是流水线填充和排空阶段的空转时间。量化时通常把一次 mini-batch 拆成 `m` 个 micro-batch、流水线深度记为 `p`，则 bubble 占比常近似写成 `O((p-1)/m)`；更精确地，对 flush 型 schedule 可写成 `bubble_fraction ≈ (p-1)/(m+p-1)`。GPipe 通过增加 micro-batch 数降低 bubble，但需要等整批前向完成后再统一反向；PipeDream 采用更激进的流水调度，让不同 stage 更快进入稳态，但会引入权重陈旧问题。

### 2. 底层原理

假设有 `p` 个 pipeline stage。无论采用何种调度，第一批 micro-batch 进入时，后面的 stage 还没有工作；最后一批 backward 结束前，前面的 stage 也会逐渐空下来。这些 fill/drain 空隙就是 bubble。

如果一次 mini-batch 只有很少的 micro-batch，那么大量时间都会花在“等流水线灌满”和“等流水线排空”上；只有当 `m` 足够大时，稳态阶段占比才会上升，bubble 才会下降。这也是 PP 总强调要把 global batch 再切成足够多 micro-batch 的原因。

### 3. 关键机制 / 流程 / 数据结构

常见量化方法是把流水线执行离散成时隙：
1. fill 阶段大约有 `p-1` 个空转时隙。
2. steady-state 阶段处理 `m` 个 micro-batch。
3. drain 阶段再产生约 `p-1` 个空转时隙。

对 flush 型 schedule，可把总开销近似看成“有效工作 + bubble”，于是得到：
- bubble slots 约为 `2(p-1)`；
- 总时隙约为 `2m + 2(p-1)`；
- 归一化后 bubble 占比约为 `(p-1)/(m+p-1)`。

两类代表性调度可这样对比：
- GPipe：先完成整批 micro-batch 的 forward，再统一执行 backward，语义简单、无权重陈旧，但需要保存 `m` 个 micro-batch 的完整激活栈，显存峰值最高，且每个 mini-batch 都要经历完整 fill/drain。
- PipeDream：更早进入交替执行 forward/backward 的稳态，可把设备利用率拉高，并降低激活驻留压力；但若不 flush，不同 micro-batch 的 forward/backward 可能使用不同版本权重，需要版本化权重或 staleness 补偿。
- 1F1B（flush 型）：介于两者之间，是 Megatron-LM / PyTorch `torch.distributed.pipelining` 的默认调度，稳态下每个 stage 在一次 forward 后紧跟一次 backward，in-flight 激活数被限制在约 `p` 份（相对 GPipe 的 `m` 份），bubble 近似仍是 `(p-1)/(m+p-1)`，但显存显著优于 GPipe。

### 4. 工程权衡 / 性能影响

GPipe 的优点是实现与收敛语义更干净，适合先跑通；缺点是：
- `m` 不够大时 bubble 明显；
- 需要为整批已前向未反向的 micro-batch 保存激活，显存压力大。

PipeDream 的优点是：
- 更容易把 pipeline 保持在接近满载的稳态；
- 激活生命周期更短，显存更友好。

代价则是：
- 有权重 staleness 管理复杂度；
- checkpoint、重计算、数值对齐和调试难度更高。

### 5. 常见追问 / 易错点

- bubble 不等于通信时间。它是 stage 因调度结构而空闲的时间，哪怕通信无限快也仍然存在。
- `m` 越大 bubble 越小，但不会免费；micro-batch 太碎会增加 kernel launch、调度和通信次数。
- GPipe 和 PipeDream 的核心差别不只是“一个快一个慢”，而是是否允许更激进的前后向重叠以及是否接受权重陈旧。
- 若每个 stage 计算量本身严重不均衡，bubble 公式只能给理想近似，真正瓶颈会被最慢 stage 主导。

### 6. 实践建议

评估 PP 时，先用简单公式估算 `m` 是否足以压低 bubble，再用 profiler 观察真实各 stage idle 时间。若优先追求训练语义稳定和实现可控，先上 GPipe 或 flush 型 1F1B；只有在确认 PP 已成为主瓶颈、且团队能处理权重版本与恢复逻辑时，再考虑更激进的 PipeDream 风格调度。

### 7. 30 秒速答

- 一句话核心：GPipe bubble ratio = (P-1)/(M+P-1)，micro-batch M 越大、stage 数 P 越小 bubble 越小
- 关键机制：bubble 来自 pipeline 启动和清空阶段，stage 之间数据未填满时 GPU 空转
- 易踩坑 / 关键权衡：PipeDream 1F1B 通过提前 backward 减少 in-flight 激活，但 bubble 比例不变
- 面试加分关键词：GPipe / PipeDream / 1F1B / bubble ratio

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 PP bubble 的量化？
- [ ] 你能不能解释 bubble 来自 pipeline 启动和清空阶段，stage 之间数据未填满时 GPU 空转？
- [ ] 你能不能举一个 P=4 M=32 时 bubble≈8.6%，P=8 M=32 时 bubble≈18% 的具体场景？
- [ ] 你能不能说出 把 PP stage 数加得太多让 bubble 吃掉吞吐 这种常见错误模式？

## Q19. interleaved pipeline（如 Megatron 的 1F1B）如何减少 bubble？

> 🔴 专家 · 朴素 PP 的 bubble 比例和 stage 数成正比，model 切得越细 bubble 越大。1F1B 让 forward 和 backward 交替进行、interleaved 再把每张卡的工作切成多块——bubble 能压到 1/v，这是 175B 训练的标配。

### 1. 核心结论

interleaved pipeline 的核心不是改变 PP 基本原理，而是把每个物理 stage 再切成多个更小的 virtual stage，并用更细粒度的 1F1B 调度把这些块交错执行。这样可让同样 `p` 张卡在一个 mini-batch 内拥有更密的工作时隙，bubble 通常会随 virtual chunk 数增加而下降，近似可理解为把原本按物理 stage 计算的空转摊薄到更多细粒度步骤上。

### 2. 底层原理

普通 1F1B 中，每张卡只负责一个连续 stage。即便进入稳态，fill/drain 仍以“整张卡对应一个大 stage”为粒度发生。interleaving 的做法是把每张卡上的模型再切成 `v` 个 chunk，使一张卡在时间线上交替执行 chunk0、chunk1、chunk2 的前向和反向。

这样做后，逻辑上的流水线深度变细了：
- 同样的物理设备数下，可形成更细的依赖链；
- 前后向可以更早穿插；
- 原本因单个 stage 过粗产生的等待，被拆散到多个更短片段。

因此，bubble 占比常可近似看作相对非 interleaved 情况再下降一个与 `v` 相关的比例因子。Megatron-LM 原论文给出的近似是：interleaved 1F1B 的 bubble 约为 `(p-1)/(v × m)`，即在 `v` 路 virtual chunk 下相较 `(p-1)/m` 再缩小 `v` 倍。实践中 `m` 必须是 `p` 的整数倍，且 `num_layers` 需能被 `p × v` 整除，否则 chunk 切分会退化。

### 3. 关键机制 / 流程 / 数据结构

Megatron 风格 interleaved 1F1B 的关键机制包括：
1. 把每个物理 stage 切成多个 model chunk，每个 chunk 有自己的参数与前后向顺序。
2. 调度器不再只按“设备号”推进，而是按“设备号 + virtual chunk id”生成更细粒度时序。
3. warmup 后，每张卡在不同 micro-batch 上交替执行不同 chunk 的 forward/backward，形成更密集的 1F1B 流。
4. 点对点通信也从“大块激活传递”变成更频繁但更小粒度的 send/recv。

从效果上看：
- 非 interleaved：每卡一个 stage，bubble 由粗粒度 fill/drain 决定。
- interleaved：每卡多个虚拟 stage，steady-state 更早出现，末尾空转更少。

```
   普通 1F1B (p=4, m=4):  bubble ≈ (p-1)/(m+p-1) = 3/7
   time ─▶
   GPU0 │ F1 F2 F3 F4 B1 B2 B3 B4 │
   GPU1 │ ░  F1 F2 F3 F4 B1 B2 B3 B4 │
   GPU2 │ ░  ░  F1 F2 F3 F4 B1 B2 B3 B4 │
   GPU3 │ ░  ░  ░  F1 B1 F2 B2 F3 B3 F4 B4 │   ░ = bubble

   Interleaved 1F1B (v=2 chunks/stage): bubble ≈ (p-1)/(v·m) → 减半
   每卡轮流执行 chunk0/chunk1 的 F/B, 流水线"看起来"更深、更密
   GPU0 │ F1ᵃ F2ᵃ F1ᵇ F3ᵃ F2ᵇ F4ᵃ ... B1ᵇ B1ᵃ ... │
   GPU1 │  ░  F1ᵃ F2ᵃ F1ᵇ F3ᵃ F2ᵇ ...             │
        通信次数 ↑, 但每卡 idle 间隙被 chunk 切片填满
```


### 4. 工程权衡 / 性能影响

interleaving 的收益主要是更高设备利用率和更低 bubble，但代价并不小：
- 调度复杂度明显上升，micro-batch 数通常需要满足更严格的整除条件。
- 通信次数增加，虽然单次消息更小，但若网络延迟高，收益可能被抵消。
- 每卡要管理多个 chunk 的参数、激活和重计算状态，实现更复杂。
- 若每个 chunk 切得过碎，kernel launch 和框架调度开销会抬高。

因此，它适合 stage 较深、bubble 已明显可见、并且单个 stage 计算足够重的场景；对小模型或低并行度任务，收益可能有限。

### 5. 常见追问 / 易错点

- 1F1B 本身主要解决激活驻留问题，interleaved 1F1B 才进一步通过虚拟 stage 细化来降低 bubble；两者不要混为一谈。
- virtual stage 越多不代表越好。切太碎后，通信和调度开销会反噬收益。
- interleaving 不会消除 stage load imbalance；若 chunk 切分不均，某些卡仍会成为瓶颈。
- 微批数不足时，interleaving 也很难发挥作用，因为流水线根本没有足够样本进入稳态。

### 6. 实践建议

启用 interleaved pipeline 前，先确认普通 1F1B 的 profile 中确实存在明显 idle gap。实践中优先选择少量 virtual chunk 做对比实验，例如 2-way interleaving（Megatron 里对应 `--num-layers-per-virtual-pipeline-stage` 约等于 `num_layers / (p * 2)`），而不是一开始就切得很细。评估指标不要只看平均吞吐，还要看 P2P 通信次数、每步 launch 数、显存峰值以及调度器是否引入新的长尾。PyTorch `torch.distributed.pipelining` 在 2.4+ 中也提供了 `Interleaved1F1B` / `LoopedBFS` schedule，可作为非 Megatron 栈的落地选项。

### 7. 30 秒速答

- 一句话核心：把每个 rank 上的多个 layer 切成多个 virtual stage，让 schedule 更紧凑
- 关键机制：每个 device 持有 v 个 chunk，bubble ratio 缩小为 (P-1)/(v·M + P - 1)
- 易踩坑 / 关键权衡：v 越大显存压力越大，且通信次数线性增加
- 面试加分关键词：virtual_pipeline_model_parallel_size / interleaved 1F1B / Megatron

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 interleaved 1F1B 减少 bubble？
- [ ] 你能不能解释 每个 device 持有 v 个 chunk，bubble ratio 缩小为 (P-1)/(v·M + P - 1)？
- [ ] 你能不能举一个 Megatron-LM 175B 设 virtual_pp=2 把 bubble 从 18% 降到 9% 的具体场景？
- [ ] 你能不能说出 virtual_pp 调太大导致 stage 间通信开销吃掉收益 这种常见错误模式？

## Q20. 激活重计算（activation checkpointing）在 PP 中的特殊处理？

> 🔴 专家 · PP 里激活值得在卡上存到 backward，micro-batch 多了显存就爆。checkpointing 用计算换显存，但在 PP 里得跟流水线调度配合好——什么时候 recompute、recompute 哪些层，做得不好反而把 bubble 撑大。

### 1. 核心结论

激活重计算在 PP 中不能只按“单卡重算一层”来理解，因为一个 micro-batch 的 forward 和 backward 被时间上拉开，中间还夹着多个其他 micro-batch。PP 下的特殊点在于：stage 边界激活、in-flight micro-batch 元数据、send/recv buffer、RNG 状态以及调度顺序都必须与重计算策略一起设计；否则即使单层重算逻辑正确，也可能在流水线里出现显存泄漏、数值不一致或时序错乱。

### 2. 底层原理

单卡 checkpointing 的基本思想是前向时少存中间激活、反向时重跑部分 forward 来换显存。到了 PP 场景，每个 stage 只持有局部子图，但要同时服务多个处于不同阶段的 micro-batch。于是会多出两个约束：
- 某个 micro-batch 的 stage 输入必须在未来 backward 到来时仍可恢复。
- 跨 stage 发送出去的边界激活不能像普通中间激活那样随意丢弃，因为下游 stage 的 forward/反向依赖它们的时序一致性。

这意味着 PP 中真正可自由重算的，主要是“stage 内部中间层激活”；而 stage 边界张量及其调度元信息通常需要更谨慎管理。

### 3. 关键机制 / 流程 / 数据结构

典型处理方式包括：
1. 以 stage 或 block 为单位做 checkpoint，只保留每个 micro-batch 进入该 stage 的输入边界张量。
2. stage 内部的大量中间激活在 forward 后立即释放，等对应 backward 到来时再基于边界输入重跑一次局部 forward。
3. 对 1F1B / interleaved 调度，要按“micro-batch id + stage id + chunk id”维护激活栈或队列，确保 backward 能取到正确版本的边界输入。
4. 若前向中包含 dropout、随机路由或 fused kernel，必须保存并回放一致的 RNG 状态，否则重算结果与原 forward 不一致。
5. PipeDream 风格若存在权重版本化，还要保证重算使用与原始 forward 对应的权重版本，而不是当前最新权重。

因此，PP 下 checkpointing 实际上不只是“存不存激活”，而是“为延迟到来的 backward 保存最小可重建状态”。

Megatron-LM 提供了更细粒度的“selective activation recomputation”，只对 attention 中间张量（softmax 输出、dropout mask、`QK^T`）这类“计算便宜但存储昂贵”的张量重算，而保留线性层输出。PyTorch 侧可通过 `torch.utils.checkpoint.checkpoint(use_reentrant=False)` 或 `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` 搭配 FSDP2，对每个 Transformer block 粒度开启。

### 4. 工程权衡 / 性能影响

PP 中使用激活重计算的收益通常很大，因为每个 stage 可能同时挂着多个 in-flight micro-batch，激活峰值远高于单卡非流水训练。通过重算，可显著降低这一部分显存占用，换取更大 micro-batch 或更深 pipeline。

代价则包括：
- 每个 stage 的局部 forward 需要被重复执行，计算开销上升。
- 调度越复杂，checkpoint 元数据管理越难，尤其在 interleaving 下更明显。
- 如果边界激活保留过多，显存收益会被侵蚀；保留过少，又会导致无法正确重建 backward 所需状态。
- 某些 fused op、随机 op 或自定义 CUDA kernel 可能不完全支持可重放重算路径。

### 5. 常见追问 / 易错点

- 不是所有激活都能一刀切丢掉。跨 stage 边界输入、通信句柄和与调度绑定的 micro-batch 元数据通常必须保留。
- 在 PP 里，checkpoint 粒度过粗会让单次重算太贵；粒度过细又会增加调度和 Python 框架开销。
- 忘记保存 RNG 状态是常见坑，表现为开启 checkpoint 后 loss 抖动或与不开启时数值对不上。
- 若使用非 flush 式 PipeDream，重算还要考虑历史权重版本，否则 backward 对应的前向语义会错位。

### 6. 实践建议

优先从“按 Transformer block 做 stage 内 checkpoint”起步，再逐步观察边界激活、P2P buffer 和重算开销的占比。上线前至少验证三件事：开启与关闭 checkpoint 的数值一致性、不同 micro-batch 调度下无激活泄漏、以及重算后吞吐下降是否在可接受范围内。若 PP 已配合 interleaving 使用，建议把 checkpoint 元数据索引显式绑定到 `micro-batch/chunk/stage`，避免隐式栈顺序导致难排查的错配。

### 7. 30 秒速答

- 一句话核心：每个 stage 独立 checkpoint，需在 stage 边界把 saved tensor 留住以便反向重算
- 关键机制：forward 阶段不保存中间激活，backward 阶段重算；PP 下还要叠 micro-batch 维度
- 易踩坑 / 关键权衡：checkpoint 边界和 PP stage 边界不对齐会重复存激活，反而更耗显存
- 面试加分关键词：activation checkpoint / recompute_granularity / selective_recompute

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 PP 中的 activation checkpointing？
- [ ] 你能不能解释 forward 阶段不保存中间激活，backward 阶段重算；PP 下还要叠 micro-batch 维度？
- [ ] 你能不能举一个 Megatron 用 selective_recompute 只重算 attention 部分 的具体场景？
- [ ] 你能不能说出 开 full recompute 让计算量上升 30%+，得不偿失 这种常见错误模式？

## Q21. 如何平衡 TP、PP、DP 的维度划分？以 175B 模型为例

> 🔴 专家 · 1024 卡训 175B，TP/PP/DP 三个维度怎么乘就是个工程艺术。TP 一般限定单机 8 卡（NVLink 内）、PP 跨机但不超过节点数、DP 吃剩下的——比例错了，要么 OOM 要么吞吐打骨折。

### 1. 核心结论

平衡 TP、PP、DP 的原则不是平均分配，而是先用 TP/PP 解决“单卡放不下”和单层算子过大问题，再把剩余卡数尽量留给 DP 提升吞吐。以 175B GPT 类模型为例，常见思路是优先让 TP 落在单机高速互联域内，PP 用来切开层深，DP 作为最外层复制维度；例如在 1024 张 GPU 上，可把 `TP=8, PP=16, DP=8` 作为一类合理起点，而不是唯一答案。

### 2. 底层原理

三种并行维度解决的是不同瓶颈：
- TP 主要切 hidden 维或 attention/MLP 权重，降低单层参数与 GEMM 尺寸带来的单卡显存压力，但会在每层引入通信。
- PP 主要切层深，把模型按 block 分到不同 stage，降低单卡参数常驻量，但会引入 bubble 与 stage 间 P2P。
- DP 不切模型结构，只复制模型副本换取数据吞吐，通信主要集中在梯度同步。

因此，三者平衡的实质，是在“每层通信开销”“流水线空转”“数据并行扩展效率”之间找一个乘积最优点，而不是只盯某一个维度做大。

### 3. 关键机制 / 流程 / 数据结构

以 GPT-3 175B 常见配置为例，可按以下顺序确定并行度：
1. 先根据 hidden size、attention head 数和单层 GEMM 尺寸确定 TP 值，通常选择 2、4、8 这类能整除 head 数、且最好落在单节点 NVLink 域内的值。
2. 再根据总层数切 PP，使每个 stage 持有接近数量的 Transformer blocks；175B 约 96 层时，`PP=16` 意味着每个 stage 约 6 层，便于做较均匀切分。
3. 最后由总卡数除以 `TP * PP` 得到 DP，例如 1024 卡下 `DP = 1024 / (8*16) = 8`。
4. 若显存仍紧张，再叠加激活重计算、sequence parallel、ZeRO optimizer state sharding 等辅助策略，而不是先盲目继续增大 TP。

一个经验判断是：
- TP 过大时，每层 all-reduce / all-gather 过多；
- PP 过大时，micro-batch 数要求变大，bubble 与调度复杂度上升；
- DP 过小时，整体吞吐和全局 batch 扩展能力不足。

### 4. 工程权衡 / 性能影响

对 175B 这类模型，常见权衡如下：
- TP 增大：单卡显存与单层算子压力下降，但层内通信更频繁，对网络拓扑更敏感。
- PP 增大：模型更容易放下，但 stage 数变多后 bubble 更明显，micro-batch 必须增多。
- DP 增大：吞吐扩展最好，但前提是模型已能在 `TP*PP` 下稳定运行，否则只是复制更多“放不下的副本”。

常把 TP 控制在单机内，把 PP 扩展到跨机，再让 DP 作为最外层复制维度。原因是 TP 通信频率最高，最依赖低延迟互联；PP 主要是相邻 stage 的激活传递，相对更适合跨节点组织。

### 5. 常见追问 / 易错点

- 不是 `TP * PP * DP` 越平均越好。通信模式不同，维度间代价不对称。
- TP 不应随意设成不能整除 attention head 数或 hidden shard 不规则的值，否则实现复杂且易退化。
- PP 切分不能只按层数平均，embedding、final norm、lm head 往往会破坏均衡。
- 很多人先把 DP 拉满再补 TP/PP，这在超大模型上通常顺序是反的：应先满足可训练性，再谈扩展吞吐。

### 6. 实践建议

对超大模型，推荐按“先单层放下、再整模放下、最后扩吞吐”的顺序调参：先确定 TP，再确定 PP，最后用剩余资源扩 DP。以 175B 为例，优先尝试单机内 `TP=8`，再从 `PP=8` 或 `PP=16` 开始试配，最后根据总卡数和网络情况确定 DP。评估时同时记录显存峰值、tokens/s、pipeline bubble 和 TP 通信占比，不要只看单步时间。

从 2024–2025 实际配置可以看到两类典型：Meta 训练 Llama 3 405B 在 16K H100 上使用 `TP=8, PP=16, CP=2, DP=64` 的 4D 并行（引入 context parallelism），而 DeepSeek-V3 670B MoE 在 2K H800 上使用 `TP=1, PP=16, EP=64, DP=?` 并辅以 DualPipe 调度与 FP8 计算——这两者都说明“175B 的经典三轴”已经演化成 TP/PP/DP + CP/EP 的混合配置，175B 只是其中较温和的一个起点。

### 7. 30 秒速答

- 一句话核心：TP 限制在单机 NVLink 内（≤8），PP 用于跨机分层，DP 填满剩余 rank
- 关键机制：TP 通信密集吃 NVLink，PP 通信稀疏可走 IB，DP 通信粒度大
- 易踩坑 / 关键权衡：TP 跨机性能崩塌；PP stage 数 > 16 后 bubble 主导
- 面试加分关键词：TP=8 NVLink / PP cross-node / DP 大粒度

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 TP/PP/DP 维度划分原则？
- [ ] 你能不能解释 TP 通信密集吃 NVLink，PP 通信稀疏可走 IB，DP 通信粒度大？
- [ ] 你能不能举一个 175B 模型 1024 卡：TP=8 PP=8 DP=16 的具体场景？
- [ ] 你能不能说出 TP=16 跨机让 AllReduce 成为瓶颈 这种常见错误模式？

## Q22. torch.distributed.pipeline.sync.Pipe 的使用限制？

> 🟡 进阶 · 官方的 `Pipe` 看起来很方便，几行代码就能切流水。但它只支持单机、不能和 DDP 嵌套、模型必须是 nn.Sequential——稍微复杂一点的模型就用不了，工业上几乎都换成 Megatron 或 PiPPy。

### 1. 核心结论

`torch.distributed.pipeline.sync.Pipe` 更适合教学或规则较强的顺序模型，不适合复杂生产级大模型训练。它的核心限制是：模型必须基本满足顺序执行语义，切分边界较刚性，对动态控制流、复杂多输入输出、跨 stage 残差依赖和与其他并行策略的深度组合支持都比较有限。PyTorch 2.4 起，官方已把这套基于 torchgpipe 的 API 标记为 legacy，新的生产级落地应迁移到 `torch.distributed.pipelining`（由 PiPPy 项目合并而来），后者原生支持跨节点 PP、多种 schedule（GPipe / 1F1B / Interleaved1F1B / LoopedBFS）以及与 FSDP2、TP、DDP 的组合。

### 2. 底层原理

`Pipe` 的设计前提是把一个大模型拆成若干顺序分段，并通过 micro-batch 驱动流水线执行。为了让调度器知道“下一段是什么、输入输出如何传递”，模型图必须足够静态、线性、可分段。若模型存在分支、共享权重、跨段跳连或运行时改变路径的逻辑，调度器就很难稳定构造正确的 forward/backward 时序。

### 3. 关键机制 / 流程 / 数据结构

其常见限制可归纳为：
1. 更偏向 `nn.Sequential` 或可线性展开的模块结构，不擅长一般 DAG。
2. 每个 partition 通常要求模块与参数驻留在预期设备上，跨设备自由混排会增加出错概率。
3. 对多输入、多输出、复杂 Python 容器嵌套的支持有限，实际工程中常需要额外包装。
4. 对 tied weights、跨 stage 残差引用、动态 shape/动态分支支持不如专用大模型框架成熟。
5. 与 TP、FSDP、ZeRO、interleaved schedule 等复杂并行策略的组合空间较窄，常见用法仍是“纯 PP + 少量 DDP 包裹”。
6. 调度主要围绕同步 flush 语义，灵活性不如 Megatron、DeepSpeed 的专用流水并行实现。

### 4. 工程权衡 / 性能影响

`Pipe` 的优点是 API 相对直接、语义清晰，适合把顺序模型快速改造成 PP。缺点是：
- 图结构越复杂，改造成本越高。
- 由于调度与切分策略较保守，性能上往往不如专用训练系统。
- 生态协同较弱，和激活重计算、optimizer state sharding、sequence parallel 等组合时可调空间有限。
- 对真实大模型训练常见的异构 stage、权重共享和自定义通信路径支持不足。

### 5. 常见追问 / 易错点

- `Pipe` 不是“任意 PyTorch 模型自动变流水线”的通用解。
- 即使模型表面上是顺序的，只要中间有跨段引用、共享模块或条件分支，也可能破坏可分段性。
- `chunks` 增大不一定更快；若每个 micro-batch 太小，kernel 效率和通信开销会恶化。
- 很多人把 `Pipe` 和 Megatron/DeepSpeed 的 PP 等价看待，但两者在调度能力和工程成熟度上差异很大。

### 6. 实践建议

若只是验证 PP 基本概念，`Pipe` 足够好用；若目标是训练几十亿到百亿级以上 Transformer，在原生 PyTorch 栈内应优先迁移到 `torch.distributed.pipelining`：用 `pipeline()` 对 `nn.Module` 做 trace-based 切分，再通过 `ScheduleGPipe` / `Schedule1F1B` / `ScheduleInterleaved1F1B` 等 `_PipelineSchedule` 驱动；再往上层还可以选 Megatron-LM、TorchTitan、DeepSpeed 等整套训练栈，以获得 TP/PP/DP 组合的成熟实现。使用 legacy `Pipe` 或迁移前，先检查模型是否可近似线性分段，再在小 batch、小 world size 上验证数值与时序正确性，避免一开始就叠加其他并行策略。

### 7. 30 秒速答

- 一句话核心：单机内 PP 工具，要求 nn.Sequential、不支持 tied weight 和复杂 forward 结构
- 关键机制：基于 RPC + chunks 切分输入，micro-batch 串行调度，不是生产级
- 易踩坑 / 关键权衡：生产 PP 推荐用 Megatron-Core PipelineSchedule 或 TorchTitan
- 面试加分关键词：torchpipe / Sequential / RPC PP

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 torch.distributed.pipeline.sync.Pipe 限制？
- [ ] 你能不能解释 基于 RPC + chunks 切分输入，micro-batch 串行调度，不是生产级？
- [ ] 你能不能举一个 prototype 阶段用 torchpipe 跑 toy 模型 的具体场景？
- [ ] 你能不能说出 拿 torch.distributed.pipeline 上百卡训 LLM 而非用 Megatron 这种常见错误模式？

## Q23. 模型并行中的 all-gather 和 reduce-scatter 通信模式？

> 🟡 进阶 · 一个 all-reduce 等于 reduce-scatter + all-gather，这是分布式训练里最基础的恒等式。FSDP/ZeRO 把这两步拆开，分别和 forward/backward 重叠——理解这两个原语，分布式训练的通信图就能一眼看懂。

### 1. 核心结论

在模型并行里，all-gather 和 reduce-scatter 往往是一对互补操作：all-gather 用于把各 rank 的分片张量重新物化为完整视图，reduce-scatter 用于把各 rank 的局部结果先做规约再分片回各自 owner。前者偏“扩展可见性”，后者偏“规约并落回分片状态”；很多系统会用 reduce-scatter 替代 all-reduce + split，以同时降低显存峰值和通信后处理开销。

### 2. 底层原理

模型并行中，参数、激活或梯度常以 shard 形式分布在多个 rank 上。计算链路里会交替出现两类需求：
- 某一步算子需要完整张量语义，例如下一层需要完整 hidden 或某模块需要完整参数视图，这时要 all-gather。
- 某一步算子产生的是各 rank 的局部部分和，最终只需让每个 rank 保留规约后的一个 shard，这时用 reduce-scatter 更合适。

从数学上看，all-gather 是“拼接/收集”，reduce-scatter 是“先求和再切片”。

### 3. 关键机制 / 流程 / 数据结构

典型使用场景如下：
1. 参数物化：FSDP/ZeRO-3 在前向前对参数分片做 all-gather，临时恢复完整参数。
2. 激活拼接：某些 TP/sequence parallel 路径需要把 hidden shard all-gather 成完整张量，供后续非分片算子使用。
3. 梯度分发：反向后各 rank 持有局部梯度贡献时，可用 reduce-scatter 直接得到“已规约且已分片”的梯度。
4. Row-parallel 输出：局部 matmul 结果先规约，再按目标布局切分给各 rank，常可用 reduce-scatter 替代更粗的 all-reduce。

可把它们理解为：
- all-gather：`[shard_0, shard_1, ..., shard_p] -> 每个 rank 都拿到 full tensor`
- reduce-scatter：`每个 rank 的 partial tensor -> 全局按元素规约 -> 再切成 shard_i 发回 rank i`

一个关键恒等式：`all-reduce = reduce-scatter + all-gather`。FSDP / FSDP2 / ZeRO-3 正是利用这一分解在一步训练里复用通信：前向 `all-gather` 参数、反向先 `reduce-scatter` 梯度到 owner rank，再按需要做后续分片更新，避免在同一个 step 内做两次完整 all-reduce。NCCL ring 算法在理论带宽上也正是用 `(N-1)` 步 reduce-scatter + `(N-1)` 步 all-gather 实现 all-reduce。

```
   起点: 每个 rank 持有 partial 张量 (各列代表 chunk a/b/c/d)
     R0[a₀ b₀ c₀ d₀]   R1[a₁ b₁ c₁ d₁]   R2[a₂ b₂ c₂ d₂]   R3[a₃ b₃ c₃ d₃]

       ┌─────────── reduce-scatter ───────────┐
       ▼                                       ▼
     R0[Σa  ·   ·   · ] R1[ ·  Σb  ·   · ] R2[ ·   ·  Σc  · ] R3[ ·   ·   ·  Σd]
       (每 rank 只持有自己的 owner chunk, 已规约)
       │
       │ FSDP/ZeRO-3 在此停下: 梯度只回到 owner, 直接给本地 optimizer 用
       │
       ▼  ┌─────────── all-gather ───────────┐
     R0[Σa Σb Σc Σd]  R1[Σa Σb Σc Σd]  R2[Σa Σb Σc Σd]  R3[Σa Σb Σc Σd]
       (合起来等价于一次完整 all-reduce)
```

FSDP 把这两半分别和 fwd/bwd 重叠：fwd 前 all-gather 参数、bwd 后 reduce-scatter 梯度。

### 4. 工程权衡 / 性能影响

两者的性能差异主要体现在目标形态上：
- all-gather 后每个 rank 持有完整结果，显存峰值更高，但有利于后续完整算子直接消费。
- reduce-scatter 后每个 rank 只保留一份 shard，更省显存，也更利于继续分片计算。
- 如果后续最终仍只需要 shard，先 all-reduce 再本地切片通常不如 reduce-scatter 直接。
- 如果后续必须完整访问，reduce-scatter 之后还要再 gather，可能得不偿失。

因此，优选哪一种，取决于通信后的消费者到底需要 full tensor 还是 sharded tensor。

### 5. 常见追问 / 易错点

- reduce-scatter 不只是“scatter”，它包含全局规约语义；和简单按块发送完全不同。
- all-gather 不是总比 all-reduce 更贵或更便宜，关键要看前后数据布局和是否会引入额外 reshape/split。
- 很多人只从带宽看 collective，忽略了通信后张量是否会让显存峰值上升。
- 在 TP/FSDP 中，真正影响性能的不仅是 collective 类型，还包括是否能与计算 overlap。

### 6. 实践建议

应优先围绕“下一个算子需要什么布局”来选 collective，而不是凭经验固定用 all-reduce。若后续还能继续在 shard 上算，优先考虑 reduce-scatter；若必须完整消费，再用 all-gather。profile 时同时看 collective 时间、通信后显存峰值以及是否出现 layout 转换长尾，因为很多性能损失并不在 collective 本身，而在其后的数据重排。

### 7. 30 秒速答

- 一句话核心：all-gather 收集分片到所有 rank，reduce-scatter 求和并分片，组合等价 all-reduce
- 关键机制：Ring all-gather 通信量 (P-1)·M/P，与 reduce-scatter 相同，组合为 2(P-1)·M/P 等于 all-reduce
- 易踩坑 / 关键权衡：如果上游已分片，单独用 all-gather 比 all-reduce 省一半带宽
- 面试加分关键词：all_gather / reduce_scatter / ring all-reduce

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 all-gather 与 reduce-scatter 的对偶？
- [ ] 你能不能解释 Ring all-gather 通信量 (P-1)·M/P，与 reduce-scatter 相同，组合为 2(P-1)·M/P 等于 all-reduce？
- [ ] 你能不能举一个 FSDP forward all-gather 参数，backward reduce-scatter 梯度 的具体场景？
- [ ] 你能不能说出 参数已经分片仍调 all-reduce 浪费一半带宽 这种常见错误模式？

## Q24. 流水线并行中的 micro-batch 大小如何影响吞吐？

> 🟡 进阶 · micro-batch 切得越多 bubble 越小，但每段计算变短、kernel 启动开销占比变大、激活值还得多存好几份。这是个典型的双向 trade-off，必须配合 `global batch / pipeline depth` 实测找最优。

### 1. 核心结论

PP 中 micro-batch 大小影响吞吐的方式不是单调的：micro-batch 太小，会让每次前后向算子太碎、P2P 次数增多、kernel 效率下降；micro-batch 太大，又会减少一个 mini-batch 内的 micro-batch 数，导致 bubble 变大、激活峰值升高。实际最优点通常出现在“单个 micro-batch 已能跑出较好算子效率，同时 micro-batch 数仍足够压低 bubble”的中间区域。

### 2. 底层原理

设全局 batch 固定，则 micro-batch 大小越小，能切出的 micro-batch 数越多，流水线更容易进入稳态，bubble 占比下降；但每个 micro-batch 的 GEMM、attention 和 layernorm 工作量也更小，GPU 更容易变成 launch-bound 或 latency-bound。反之，micro-batch 大小越大，单次算子更饱满，但流水线并行度下降，fill/drain 空转更难摊薄。

因此，micro-batch 大小同时决定两件事：
- 单个 micro-step 的算子效率。
- 一个 mini-batch 内可供流水线调度的颗粒度。

### 3. 关键机制 / 流程 / 数据结构

影响链路通常如下：
1. 固定 global batch 时，`micro-batch size` 变小会让 `num_micro_batches` 变大。
2. `num_micro_batches` 变大通常有利于降低 bubble。
3. 但更小的 micro-batch 会让每个 stage 上的 matmul、attention、通信包都更碎。
4. 更碎的执行意味着 launch 数增加、P2P 次数增加、某些 kernel 难以跑满。
5. 同时，micro-batch size 变大又会抬高每个 in-flight micro-batch 的激活占用，限制可并发数量。

所以吞吐的真实决定因素往往不是 micro-batch size 单独一个量，而是它与 pipeline 深度、梯度累积、global batch 的联动关系。

### 4. 工程权衡 / 性能影响

常见规律是：
- 太小的 micro-batch：bubble 低，但算子效率差、通信频繁、调度开销高。
- 太大的 micro-batch：算子效率高，但 micro-batch 数不足，bubble 明显。
- 在深 PP 下，通常更需要足够多的 micro-batch 去摊薄 bubble，因此不宜把 micro-batch 调得过大。
- 在浅 PP 或单 stage 计算很重时，可以适当放大 micro-batch，以换取更高单卡效率。

另外，micro-batch size 还直接影响激活显存与 checkpointing 开销，因此吞吐优化往往受显存上限约束。

### 5. 常见追问 / 易错点

- 不是 micro-batch 越小越好。很多人只看到 bubble 降低，却忽略 kernel 已经被切得太碎。
- 也不是只看单卡算子效率就把 micro-batch 调大；PP 下必须同时看流水线空转。
- `micro-batch size` 和 `num_micro_batches` 经常被混用，但二者不是同一个量。
- 若全局 batch、梯度累积和 DP 同时变化，不能把吞吐变化简单归因到 micro-batch size 一个变量。

### 6. 实践建议

调优时建议先固定 `PP`、`DP` 与全局 batch，再扫描少量候选 micro-batch 大小，观察三个指标：tokens/s、bubble 占比、单 stage kernel 利用率。经验上，先确保 `num_micro_batches` 明显大于 `PP` 深度——Megatron 常用 `m ≈ 4p` 起步、`m ≥ p` 是硬下限；interleaved schedule 下 `m` 还需是 `p` 的整数倍。在此范围内寻找能让 attention/GEMM 保持高效率的较大 micro-batch；若显存吃紧，优先结合激活重计算或 sequence parallel，而不是简单把 micro-batch 无限缩小。

### 7. 30 秒速答

- 一句话核心：micro-batch 越多 bubble 越小，但太多会让 stage 内 batch 太小损失算力
- 关键机制：micro-batch 数 M 决定 bubble = (P-1)/(M+P-1)，但单 micro-batch 内 GPU 利用率取决于 batch size
- 易踩坑 / 关键权衡：micro-batch size 过小让矩阵乘掉到内存带宽 bound 区
- 面试加分关键词：micro_batch_size / global_batch_size / num_micro_batches

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 micro-batch 大小对 PP 吞吐影响？
- [ ] 你能不能解释 micro-batch 数 M 决定 bubble = (P-1)/(M+P-1)，但单 micro-batch 内 GPU 利用率取决于 batch size？
- [ ] 你能不能举一个 P=8 global_batch=1024，micro=1 时 M=128，bubble≈5% 的具体场景？
- [ ] 你能不能说出 为减 bubble 把 micro 切到 1，GEMM 跑不满 这种常见错误模式？

## Q25. 如何处理 PP 中的负载不均衡？recompute 和 no-recompute 层的分配？

> 🔴 专家 · PP 里 embedding 层和最后的 lm_head 都比中间层重，平均分段会让首尾的 stage 慢一拍，整条流水线被拖死。分段时要按 FLOPs 来均衡，再用 recompute 微调每段显存——这是工程化大模型训练的细活。

### 1. 核心结论

PP 负载均衡不能只按“每个 stage 分同样层数”处理，而应按每层的有效时间成本切分，尤其要把 recompute 层在 backward 中额外引入的一次 forward 代价算进去。实践上，recompute 层应更少分配给本来就重的 stage；显存余量大的 stage 可多放 no-recompute 层，显存紧张的 stage 可多放 recompute 层，但最终目标始终是平衡 stage time，而不是平衡层数或显存单指标。

### 2. 底层原理

PP 的整体吞吐由最慢 stage 决定。若某些 stage 除了正常 forward/backward，还承担 embedding、lm head、cross entropy 或更多 recompute block，那么其单 micro-batch 时长会更长，其他 stage 即使很空闲也无法提升系统吞吐。

activation recompute 会把某些层的成本从近似 `F + B` 变成接近 `F + B + F_recompute`。因此，同样一层 Transformer block，在 recompute 与 no-recompute 下的“时间权重”并不相同，stage 切分时必须区别对待。

### 3. 关键机制 / 流程 / 数据结构

处理流程通常如下：
1. 先 profile 单层或单类模块的 forward/backward/recompute 时间，而不是只统计参数量。
2. 为每层建立近似成本模型，例如：
   - no-recompute 层成本约为 `F + B`
   - recompute 层成本约为 `F + B + F`
3. 再把 embedding、final norm、lm head、loss 等特殊模块单独计价，因为它们常明显偏重。
4. 分 stage 时按“总成本接近”做切分，而不是按“层数接近”做切分。
5. 若某个 stage 显存富余，可把部分层改为 no-recompute，以换取更低时延；若某个 stage 显存紧张，则可增加 recompute，但应同步减少该 stage 的层数。

对混合分配，可粗略理解为：
- 重 stage：少放 recompute 层，多放 no-recompute 层或直接减少 block 数。
- 轻 stage：可以承接更多 recompute 层，帮整体显存达标。

首尾 stage 的不均衡来源也要单列：embedding lookup（首）和 lm head + cross entropy（尾）往往带来远超单个 Transformer block 的成本，且在 tie embeddings / lm head 权重共享时，还会出现跨 stage 的隐式梯度同步。Megatron 的做法是把 embedding 与最后一层 layernorm/lm head 当作半层预算计入首尾 stage，并在必要时减少首尾 stage 的 block 数以平衡。

### 4. 工程权衡 / 性能影响

这种调法本质是在三件事之间折中：
- 吞吐：希望各 stage 用时接近，减少 straggler。
- 显存：希望重层或长序列阶段通过 recompute 降峰值。
- 实现复杂度：混合 recompute 策略会增加配置和调试成本。

若只追求最低显存，把所有 stage 都设成 recompute，往往会把某些本来就重的 stage 进一步拉长；若只追求最快，把所有层都 no-recompute，又可能根本放不下。真正有效的方案通常是非均匀配置。

### 5. 常见追问 / 易错点

- 平衡参数量不等于平衡运行时间。embedding/lm head 常让首尾 stage 明显失衡。
- recompute 不是“白拿显存优化”，它会真实增加计算时间。
- 很多人把同样数量的 recompute 层均匀撒到所有 stage，看起来公平，实际可能加重瓶颈 stage。
- interleaved pipeline 下还要把 chunk 粒度纳入考虑，否则物理 stage 看似平衡，virtual stage 仍可能失衡。

### 6. 实践建议

建议先做一次 stage 级 profile，得到每个 stage 的 forward、backward、idle 与 recompute 时间占比，再决定是否重分层。若首尾 stage 因 embedding 或 lm head 偏重，可优先减少其 block 数，并把更多 recompute 层放到中间较轻的 stage。配置完成后，重点验证 steady-state 下各 stage 的 micro-step 时长是否接近；如果仍存在长尾，再继续调层数、recompute 比例或 virtual chunk 切分，而不是只盯显存峰值。

### 7. 30 秒速答

- 一句话核心：给计算重的 stage 少分层，给 embedding/lm_head stage 多分层；recompute 也按 stage 微调
- 关键机制：第一个和最后一个 stage 通常额外承担 embedding/loss，需把中间层适当多分给它们
- 易踩坑 / 关键权衡：均匀分层会让首尾 stage 慢一拍，全 pipeline 跟着等
- 面试加分关键词：num_layers_in_first_stage / num_layers_in_last_stage / selective recompute

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 PP 负载不均衡处理？
- [ ] 你能不能解释 第一个和最后一个 stage 通常额外承担 embedding/loss，需把中间层适当多分给它们？
- [ ] 你能不能举一个 LLaMA 70B PP=8 把第一/最后 stage 各少分 1 层 的具体场景？
- [ ] 你能不能说出 均匀分层后 last stage 跑 loss 时其他 stage 空等 这种常见错误模式？

## Q26. 3D 并行（3D parallelism）的通信复杂度分析？

> 🔴 专家 · TP/PP/DP 三个维度各自的通信量、频率、跨节点与否完全不同，叠加之后总通信成本不是简单加法。算清楚每一层在哪个维度通信、用多少带宽，是判断 3D 配置合不合理的唯一办法。

### 1. 核心结论

3D 并行通常指 DP、TP、PP 的组合。它的通信复杂度不能只写成一个总公式，而要按三个维度分别分析：DP 主要承担梯度或优化器状态同步，TP 主要承担层内张量重分布，PP 主要承担相邻 stage 的激活传递。最常见的结论是：TP 对低延迟最敏感，DP 对总带宽最敏感，PP 的通信量通常相对线性但会受 micro-batch 数影响；3D 并行的瓶颈往往来自三者中最难被 overlap 的那一部分，而不是总字节数最大者。

### 2. 底层原理

设总卡数为 `N = N_dp × N_tp × N_pp`。一次训练 step 中，通信大致分三类：
- DP 组内：梯度 all-reduce，或 ZeRO/FSDP 路径中的 reduce-scatter + all-gather。
- TP 组内：attention/MLP 线性层前后的 all-reduce、all-gather、reduce-scatter。
- PP 组间：相邻 stage 间发送激活与反向梯度。

若以每层 hidden state 大小、参数量和 micro-batch 数记账，可得到一个直观结论：DP 通信更像“按参数规模”计费，PP 通信更像“按激活规模”计费，TP 通信则同时受参数切分方式与层内数据布局影响，因此在大模型中最容易成为细粒度高频通信源。

### 3. 关键机制 / 流程 / 数据结构

可用分项方式理解复杂度：
1. DP：
   - 经典 DDP 近似每步做一次与梯度总量同规模的全局规约。
   - 若用 classic ring all-reduce，单 rank 需要发送和接收的总字节数约为 `2*(N_dp-1)/N_dp * P`，其中 `P` 是完整参数或梯度的总字节数，因此按字节量看它仍是 `O(P)` 量级；不要把它理解成字节数会被参与 rank 数简单均摊，真正随参与者数量增长更明显的是 latency 项和分轮次数。
2. TP：
   - 每层可能发生一次或多次 collective，通信量和该层激活/输出 shard 大小相关。
   - 若按 head 或 hidden 切分，常见是每层 `O(A / N_tp)` 量级的若干次 all-reduce 或 all-gather，其中 `A` 表示该层相关激活字节数。
3. PP：
   - 相邻 stage 之间每个 micro-batch 至少传一次 forward activation、一次 backward gradient。
   - 若激活大小为 `M`，`m` 个 micro-batch，则总量近似 `O(m × M)`，但只发生在邻接 stage。
4. 组合后：
   - 总通信不是简单相加后再平均，因为三类通信位于不同时间轴，重叠能力也不同。

### 4. 工程权衡 / 性能影响

3D 并行的收益是能把“显存放不下”和“单维并行扩不动”两个问题拆开处理；代价是通信面更复杂：
- `N_tp` 增大：层内通信频率升高，适合单机高速互联，不适合跨慢网络扩太大。
- `N_pp` 增大：单卡模型负载下降，但 bubble 和 P2P 次数上升。
- `N_dp` 增大：吞吐扩展通常最好，但前提是 DP 同步仍可被网络承受。

因此，大模型集群常采用“TP 限制在单机内，PP 跨机切层，DP 作为最外层复制”的布局，以降低最敏感的 TP 延迟开销。

### 5. 常见追问 / 易错点

- 不能把 3D 并行的通信只等价成一次 all-reduce；不同维度的 collective 完全不同。
- 通信复杂度不只看字节数，还要看调用次数和是否落在关键路径上。
- TP 规模不是越大越好，很多场景下先被 latency 打爆，而不是先被带宽打满。
- PP 看似只做点对点，但如果 micro-batch 太碎，消息数会非常多。

### 6. 实践建议

做 3D 并行规划时，先按层估算 TP collective 次数，再按参数总量估算 DP 同步体积，最后按 micro-batch 数估算 PP 激活传输。调参顺序建议先满足显存与单层可计算性，再最小化关键路径通信：通常先定 TP，再定 PP，最后扩 DP。profile 时必须分别看 TP、DP、PP 三条时间线，否则很容易误判瓶颈。

注意 2024–2026 的工业栈已普遍从“3D”扩到 4D/5D：在 TP/PP/DP 之上再叠 context parallelism（CP，沿 seq 维切 attention，如 Ring/Ulysses）、expert parallelism（EP，MoE 专属）。TorchTitan、Megatron-Core、NeMo、DeepSpeed-Ulysses 都以 `DeviceMesh` 抽象表达这些正交维度，使通信复杂度分析也要扩展为按每个 mesh 维度分别建模，而不是停留在经典 `N = N_dp × N_tp × N_pp`。

### 7. 30 秒速答

- 一句话核心：TP 通信量∝活动激活，PP∝stage 间 activation，DP∝参数 × 频率
- 关键机制：TP 每步多次 AllReduce，PP 每 micro-batch send/recv，DP 每 step 一次 AllReduce/ReduceScatter
- 易踩坑 / 关键权衡：TP 通信粒度最细必须放 NVLink 内；DP 粒度最大放最外层
- 面试加分关键词：3D parallelism / communication volume / hierarchical comm

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 3D 并行的通信复杂度？
- [ ] 你能不能解释 TP 每步多次 AllReduce，PP 每 micro-batch send/recv，DP 每 step 一次 AllReduce/ReduceScatter？
- [ ] 你能不能举一个 175B 1024 卡 TP=8 PP=8 DP=16 的通信 budget 拆解 的具体场景？
- [ ] 你能不能说出 把 DP 放最内层让跨机 AllReduce 频率拉满 这种常见错误模式？

## Q27. 序列并行（Sequence Parallelism）在长文本训练中的应用？

> 🔴 专家 · 长文本训练时 LayerNorm 和 dropout 的激活值与 seq_len 成正比，TP 不切 seq 这部分激活全副本存。SP 把 seq 维也切了——是 32k+ 上下文训练能跑起来的关键拼图。

### 1. 核心结论

Sequence Parallelism（SP）的核心思想是把原本在 TP 组内按 hidden 维切分的部分计算，改为按 sequence 维分摊到不同 rank，从而降低长文本训练中的激活显存和部分逐 token 算子的重复开销。它特别适合长上下文 Transformer，因为当序列长度远大于 hidden 切分收益时，单纯 TP 往往先被激活显存和 layernorm/dropout 等非矩阵乘阶段拖住。在 2024–2026 的生态里，需要区分三个层次：Megatron-style SP（只在 TP 组内分摊 LN/dropout）、DeepSpeed-Ulysses（用 all-to-all 把 seq/head 维互换，让 attention 本体按 head 本地算）、以及 Ring Attention / Context Parallelism（沿 seq 维做分块 attention，用 P2P 环把 KV 在 rank 间流转）。

### 2. 底层原理

在普通 TP 中，某些逐 token 操作如 layernorm、dropout、residual add 往往需要每个 rank 持有完整 sequence 上的本地 shard 视图，因此长序列下激活显存仍然很重。Sequence Parallelism 会在 TP 组内把 token 维度拆开，让每个 rank 只保留部分 token 的激活，再在需要完整 hidden 语义的位置做 all-gather 或 reduce-scatter。

这样做的直接效果是：当 `seq_len` 增长时，激活常驻成本更接近按 `1 / N_tp` 缩减，而不是所有 rank 都完整保存整段序列的某些中间结果。

### 3. 关键机制 / 流程 / 数据结构

典型流程如下：
1. 在线性层或 attention 某些边界之后，将 hidden 表示在 TP 组内做 reduce-scatter，按 sequence 维分发。
2. layernorm、dropout、residual add 等逐 token 操作在本地 sequence shard 上完成。
3. 当后续算子需要特定布局时，再通过 all-gather 恢复所需张量视图。
4. 若与 FlashAttention、激活重计算联用，通常还会进一步减少长序列训练的显存峰值。

在长文本场景中，它主要帮助三类问题：
- 降低非 GEMM 激活显存。
- 缓解 layernorm/dropout 等算子在长序列下的冗余存储。
- 让更长 `seq_len` 能在既定 TP 配置下跑起来。

长上下文真正走到 100K+ 时，单独的 Megatron SP 往往不够，需要进一步引入 CP：
- Ring Attention 让每个 rank 只持有 `S/N` 的 Q/K/V，通过环形 P2P 轮转 KV，使 attention 计算与通信重叠，但会把一次 GEMM 拆成更多小 GEMM，算术强度下降。
- DeepSpeed-Ulysses 借 all-to-all 在 `seq ↔ head` 维互换，attention 内部仍是“单 rank 完整 seq、部分 head”，单机内效率高；但并行度受 head 数上限约束。
- xDiT / Unified Sequence Parallel（USP）、Llama 3 的 CP=2 以及 TorchTitan 的 `CP mesh` 都是两者混合方案，典型做法是 Ulysses 分 head、Ring 分 seq。

### 4. 工程权衡 / 性能影响

Sequence Parallelism 的主要收益在长序列下更明显：
- `seq_len` 越长，显存收益通常越大。
- 对 attention 本体的 FLOPs 减少有限，但对激活峰值与某些算子布局更有帮助。
- 它通常会引入额外的 layout 变换和 collective，因此短序列时未必划算。
- 若网络较慢或 TP 组跨节点，额外通信可能抵消收益。

因此，它更像长上下文训练中的显存/吞吐联合优化，而不是任何长度下都应开启的默认选项。

### 5. 常见追问 / 易错点

- Sequence Parallelism 不是把 attention 的二次复杂度变成线性；它主要优化存储与部分并行布局。
- 它通常建立在 TP 之上，不是独立替代 TP 的新并行维度。
- 短序列、小模型场景下收益可能很弱，甚至因额外 collective 变慢。
- 若实现没有处理好 layout 转换，profile 中看到的瓶颈可能从 layernorm 变成通信尾部。

### 6. 实践建议

当训练目标进入 8K、32K 甚至更长上下文时，可优先评估 Sequence Parallelism 与激活重计算、FlashAttention 的组合。验证时同时记录激活峰值、tokens/s 和 TP 通信时间，而不是只看能否放下更长序列。若 TP 已跨机，需特别谨慎评估新增 collective 是否会把收益吃掉。

### 7. 30 秒速答

- 一句话核心：把 LayerNorm/Dropout 的 sequence 维切分到不同 rank，进一步省激活显存
- 关键机制：在 TP 区间外把 sequence 维切分，AllReduce 换成 reduce-scatter+all-gather
- 易踩坑 / 关键权衡：必须和 TP 配合，SP 单独用没有意义
- 面试加分关键词：sequence_parallel / Megatron SP / activation memory

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Sequence Parallelism 的作用？
- [ ] 你能不能解释 在 TP 区间外把 sequence 维切分，AllReduce 换成 reduce-scatter+all-gather？
- [ ] 你能不能举一个 Megatron 训 100k 序列 SP=TP=8 显著降激活 的具体场景？
- [ ] 你能不能说出 开 SP 但 sequence 维不能整除 TP size 这种常见错误模式？

## Q28. Expert Parallelism 在 MoE 模型中的 all-to-all 通信优化？

> 🔴 专家 · MoE 把 token 路由到不同专家，跨卡 all-to-all 是它最贵的一步——通信量随专家数线性涨，不优化的话训练时间一半都耗在这上面。Tutel/FasterMoE 都是冲着 all-to-all 去的。

### 1. 核心结论

MoE 的 Expert Parallelism 中，all-to-all 往往是系统瓶颈，因为 token 需要按路由结果跨 rank 发送到对应 expert，再把输出返回原 rank。优化主线通常不是单一“更快的 all-to-all”，而是同时减少待发送 token 数、提高消息规整度、增强通信与专家计算重叠，并尽量避免严重负载倾斜。

### 2. 底层原理

MoE 前向中，router 会为每个 token 选出 top-k expert。若 expert 分散在 EP 组的不同 rank 上，就需要先把 token 按目标 expert 重新分桶，再做 all-to-all 发送。反向阶段同理还要把梯度按相反路径返回，因此一次 MoE 层通常包含成对的 dispatch/combine 通信。

通信成本主要受三件事支配：
- token 分布是否均匀。
- 每个消息块是否足够大且连续。
- 通信能否与本地 expert GEMM 重叠。

### 3. 关键机制 / 流程 / 数据结构

常见优化手段包括：
1. 路由侧约束：
   - 使用 capacity factor、aux loss、load balancing loss，减少热点 expert。
   - 合理设置 top-1 / top-2 路由，避免无谓扩大通信量。
2. 打包与布局优化：
   - 先按 expert id 做排序/分桶，把 token 打成连续 buffer 再 all-to-all。
   - 尽量使用 fused dispatch/combine kernel，减少 gather/scatter 的碎片拷贝。
3. 通信与计算重叠：
   - 一边接收已到达 expert 的 token，一边启动本地 expert 前向/反向。
   - 对不同 expert chunk 做流水处理，而不是等整批 token 到齐再算。
4. 拓扑与分组优化：
   - 尽量让 EP 组落在高速互联域内。
   - 对多节点环境，可采用层级式 all-to-all 或节点内优先聚合。
5. 稀疏性控制：
   - 控制 dropped token 策略、capacity 溢出回退策略，避免极端长尾。

### 4. 工程权衡 / 性能影响

这些优化都带有明显 trade-off：
- 更强的 load balancing 有利于降低 all-to-all 长尾，但可能伤害路由质量。
- top-2 往往提升模型质量与鲁棒性，但通信量通常明显高于 top-1。
- 更激进的 token 排序和 fused packing 能提高带宽利用率，但实现复杂、调试难。
- capacity 设得太紧会减少通信峰值，却可能增加 token drop，影响收敛。

因此，MoE 优化应同时看吞吐与模型质量，不能只从网络视角做局部最优。

### 5. 常见追问 / 易错点

- all-to-all 的问题不只是网络慢，很多时候是路由不均导致少数 rank 成为 straggler。
- token 数量相同不代表消息代价相同；碎片化 buffer 会严重拉低有效带宽。
- top-k 路由收益不能脱离 capacity 和负载均衡损失一起看。
- 如果 expert 很小、计算很轻，all-to-all 更容易完全暴露在关键路径上。

### 6. 实践建议

优化 EP 时先看三个 profile 指标：dispatch all-to-all 时间、combine all-to-all 时间、每个 expert 的 token 数直方图。若分布极不均，优先调 router 和 capacity；若分布均匀但通信仍慢，优先优化 packing/fusion 与 EP 组拓扑。多机场景建议优先把 expert 组限制在节点内或高速域内，再考虑更大范围扩展。

2025 年以来的工程标配是直接评估 DeepSeek 开源的 DeepEP：它提供两套 kernel——高吞吐 training/prefill 版在 NVLink 与 RDMA 之间做异构 domain forwarding，仅用 20 SM 就可打满 IB+NVLink；低延迟 decode 版完全走 RDMA，并用 hook-based 通信算法把 dispatch/combine 与 expert GEMM 纯 overlap，不再占用 SM。它原生支持 FP8 通信与 DeepSeek-V3 group-limited gating，是目前大规模 EP 训练（SGLang/LMSYS 96 卡 H100 部署、DeepSeek-V3 自研 2K H800 集群）上的事实参考实现。

### 7. 30 秒速答

- 一句话核心：MoE 把 token 按 router 选择跨 EP 发送，all-to-all 是关键通信原语
- 关键机制：dispatch all-to-all → expert compute → combine all-to-all，token 分布越均匀越好
- 易踩坑 / 关键权衡：负载不均的 expert 让 all-to-all 长尾，需 capacity factor + aux loss 控制
- 面试加分关键词：all_to_all / Tutel / FasterMoE / expert_parallel

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 EP 的 all-to-all 优化？
- [ ] 你能不能解释 dispatch all-to-all → expert compute → combine all-to-all，token 分布越均匀越好？
- [ ] 你能不能举一个 DeepSeek-V3 671B 用 EP=64 + capacity_factor=1.5 的具体场景？
- [ ] 你能不能说出 不开 token-drop 让 all-to-all 等待最热 expert 这种常见错误模式？

## Q29. 零气泡流水线（Zero Bubble）的最新进展？

> 🔴 专家 · 1F1B 已经把 bubble 压到很小，但还能更狠——把 backward 拆成 weight grad 和 input grad 两段分别调度，理论 bubble 可以做到 0。这是 2024 年 PP 调度的最新方向，也是开源框架在追的目标。

### 1. 核心结论

零气泡流水线的主线进展，是从传统 1F1B 只关注 stage 间排队，发展到进一步拆分 backward 内部依赖，把对关键路径更敏感的梯度计算优先执行，并把权重梯度、参数更新、通信等尽量挪到气泡区或非关键路径。换句话说，“Zero Bubble”并不意味着物理上绝对没有任何空闲，而是通过更细粒度调度，把原本暴露的空转尽可能转化为有用工作。

### 2. 底层原理

传统流水线里，bubble 来自 fill/drain 以及 stage 间负载不均。进一步观察 backward 可发现：并非所有反向子任务都同样阻塞下一 stage。以 Transformer 为例，输入梯度相关计算通常更影响下游继续反传，而权重梯度部分在依赖图上往往更容易后置。因此近几年的零气泡思路，核心就是把 backward 拆成更细的 dgrad / wgrad 或等价关键子阶段，优先调度关键路径部分。

### 3. 关键机制 / 流程 / 数据结构

近年的主流方向可概括为：
1. 更细粒度的 backward 拆分：
   - 将反向拆成 B（输入梯度 dgrad）与 W（权重梯度 wgrad）两段；dgrad 留在关键路径，wgrad 被推迟到气泡区。
   - Qi et al. 2024（ICLR）提出的 ZB-H1 / ZB-H2 / ZB-V 就是这一分解下的三种手工 schedule：ZB-H1 在不增加 1F1B 峰值内存的前提下把 bubble 砍到约 1/3；ZB-H2 允许更高内存换取真正“零气泡”；ZB-V 在 V 型排布下逼近 ZB-H2 的吞吐但内存只有一半。

```
   传统 1F1B: backward 是不可分整体, 关键路径上拖住下游
   time ─▶
   GPUk-1│ Fᵢ        Bᵢ                          │
   GPUk  │   Fᵢ        Bᵢ          ░░ idle ░░    │   B 阻塞下一 stage
   GPUk+1│     Fᵢ        Bᵢ                      │

   Zero-Bubble: B 拆成 B(dgrad, 关键路径) + W(wgrad, 可后置)
   GPUk-1│ Fᵢ      Bᵢ   Wᵢ                       │
   GPUk  │   Fᵢ      Bᵢ   Fⱼ Bⱼ Wᵢ Wⱼ ...        │  Wᵢ 填进原 bubble
   GPUk+1│     Fᵢ      Bᵢ   Wᵢ                   │
            ▲                ▲
            │ dgrad 立刻发给上游, 不等 wgrad
            └ 关键路径只有 F + dgrad
   ★ wgrad 不阻塞下游 backward, 被挤进气泡区
```
2. 关键路径感知调度：
   - 不再把一个 micro-batch 的 backward 当成不可分整体。
   - 让非关键路径工作填充原本 bubble 区间。
3. 与 interleaving 结合：
   - 通过 virtual pipeline stage 提高可调度颗粒度。
   - 在更多小 chunk 上做零气泡式排程。
4. 与通信/优化器重叠结合：
   - 把梯度同步、参数更新、prefetch 尽量压到不阻塞前向/反向主链的位置。
   - DeepSeek-V3 的 DualPipe 是这一方向的代表：双向 pipeline + 一对 forward/backward chunk 内部的计算–通信 overlap，使 MoE 训练中的 all-to-all 几乎完全被隐藏。
5. 调度搜索与自动化：
   - 从手工 schedule 走向 cost model 或搜索式 schedule，针对不同 stage time 自动选更优排程。
   - TorchTitan 与 PyTorch `torch.distributed.pipelining` 也已在 2024–2025 合入 ZB schedule 的实验实现。

### 4. 工程权衡 / 性能影响

零气泡方法通常能降低 PP 的空转占比，尤其在 stage 较多、backward 可拆分且负载不完全均衡时收益明显。但其代价也很直接：
- 调度器更复杂，对 runtime 与依赖管理要求更高。
- 需要更精细的 kernel、通信、optimizer 顺序控制。
- 若 stage 本身已高度均衡、micro-batch 足够多，新增复杂度未必带来大收益。
- 某些实现会增加暂存状态或改变权重更新时间点，需要谨慎验证数值一致性。

### 5. 常见追问 / 易错点

- “零气泡”不是永远 100% 利用率；它是尽量消除关键路径上的空泡。
- 不是所有 backward 都能随意拆分，必须满足依赖正确性。
- 只改 schedule 不解决 stage 极端失衡时，收益会有限。
- 若没有足够的 runtime 可观测性，零气泡调度很难稳定落地到生产系统。

### 6. 实践建议

若当前 PP 已明显受 bubble 影响，可先从更现实的手段入手：增加 micro-batch、做 stage 重平衡、启用 interleaving；只有在这些手段已接近上限时，再评估零气泡调度。验证时不要只看吞吐提升，还要检查 optimizer step 时机、loss 对齐、checkpoint 恢复语义是否保持一致。对于生产系统，优先选择已有成熟实现或论文复现较稳定的方案，而不是一次性手写全新调度器。

### 7. 30 秒速答

- 一句话核心：把 backward 拆成 dW（权重梯度）和 dX（输入梯度），dW 可后置填补 bubble
- 关键机制：ZB-1F1B 让 dW 在 idle slot 执行，bubble 比例从 (P-1)/(M+P-1) 降到接近 0
- 易踩坑 / 关键权衡：需要重排 backward 计算图，框架支持有限（Megatron-Core / TorchTitan 部分支持）
- 面试加分关键词：Zero Bubble / ZB-1F1B / dW post-processing

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Zero Bubble PP 的核心？
- [ ] 你能不能解释 ZB-1F1B 让 dW 在 idle slot 执行，bubble 比例从 (P-1)/(M+P-1) 降到接近 0？
- [ ] 你能不能举一个 Megatron-Core 2024 起支持 zero-bubble schedule 的具体场景？
- [ ] 你能不能说出 假定开 ZB 就一定加速，实际依赖 stage 间通信能否 overlap dW 这种常见错误模式？

## Q30. 如何 profile 分布式训练的通信开销？torch.profiler 的 distributed view？

> 🟡 进阶 · 训练慢但说不上来慢在哪——大概率是通信被 overlap 没盖住。torch.profiler 的 distributed view 能告诉你每个 collective 实际跑了多久、和哪个 kernel 重叠了，没这玩意瞎调相当于盲人摸象。

### 1. 核心结论

分布式训练的通信 profile 不能只看单卡 kernel 时间，必须把 collective、P2P、等待空洞和各 rank 之间的时间对齐一起看。`torch.profiler` 的价值在于能把 PyTorch 算子、CUDA kernel 与分布式通信事件放到统一 trace 中，并通过 distributed view 观察不同 rank 的 step 对齐、collective 对应关系和 straggler 位置；但它通常仍需要结合 NCCL 日志与 Nsight Systems 交叉验证。

### 2. 底层原理

通信开销常以三种形式出现：
- collective 本身耗时，如 all-reduce、all-gather、reduce-scatter、all-to-all。
- 显式或隐式同步带来的等待时间。
- 由于 rank 不均衡导致的“别人没算完，所以我在等”的空洞。

单 rank 视角只能看到“我很慢”，distributed view 才能回答“是谁先慢、谁在等谁、哪次 collective 没对齐”。这也是它对分布式调优更有价值的原因。

### 3. 关键机制 / 流程 / 数据结构

一个常见工作流如下：
1. 只采集稳定窗口：
   - 跳过 warmup、编译、缓存建立阶段。
   - 只抓若干个稳定 step。
2. 在所有 rank 上开启 `torch.profiler`：
   - 记录 CPU/CUDA activity。
   - 必要时开启 `record_shapes`、`with_stack`，但要控制开销。
3. 导出 trace 并在 TensorBoard/trace viewer 中查看：
   - 先看每个 rank 的 step 是否对齐。
   - 再看 NCCL collective 或 P2P 事件是否形成长尾。
4. 使用 distributed view：
   - 对比各 rank 的时间线，定位 straggler rank。
   - 观察某个 collective 是否所有 rank 几乎同时进入、同时退出。
   - 检查通信与 backward/forward 是否发生有效 overlap。
5. 联合其他信号：
   - 配合 `NCCL_DEBUG=INFO`、Nsight Systems、网络监控，确认是框架调度、通信库还是链路本身导致问题。

### 4. 工程权衡 / 性能影响

`torch.profiler` 的优点是离 PyTorch 语义近，能快速把问题定位到“哪层、哪次 collective、哪个 rank”。缺点是：
- 大规模多 rank trace 非常重，采集窗口必须严格控制。
- 开启过多选项会显著扰动时间线。
- distributed view 擅长看关联关系，但对底层链路拥塞、NIC 利用率等问题仍不如系统级工具直接。

因此，它最适合作为第一层分布式定位入口，而不是唯一工具。

### 5. 常见追问 / 易错点

- 只看 rank0 trace 往往不够，分布式问题常发生在最慢 rank 或某个特定节点。
- collective 时间长不一定是网络差，也可能是前序计算失衡导致迟到。
- profiling 时若保留过多 Python 栈或 shape 信息，trace 本身就可能拖慢训练。
- 必须保证所有 rank 的 profiling 窗口大体一致，否则 distributed view 的对齐会失真。

### 6. 实践建议

建议采用“三层定位法”：先用 `torch.profiler` distributed view 看哪个阶段、哪个 rank 出现长尾；再用 NCCL 日志（`NCCL_DEBUG=INFO` 或 PyTorch 2.3+ 的 Flight Recorder `TORCH_NCCL_TRACE_BUFFER_SIZE`）确认 collective 类型、顺序和是否有异常等待；最后用 Nsight Systems 或网络侧指标验证底层原因。对上百卡的 trace，建议用 Meta 开源的 Holistic Trace Analysis（HTA）做离线聚合分析，自动抽出 idle time breakdown、collective 时间矩阵和 straggler rank，比逐张打开 Chrome trace / Perfetto 更可规模化。做优化前后对比时，固定 batch、并行配置和采集窗口，否则很难得到可信结论。

### 7. 30 秒速答

- 一句话核心：用 torch.profiler 的 distributed view + Nsight Systems trace NCCL kernel
- 关键机制：profiler 标注 NCCL kernel 时间线，distributed view 聚合各 rank collective 时长
- 易踩坑 / 关键权衡：只看单 rank profiler 看不出 straggler，要多 rank 合并 timeline
- 面试加分关键词：torch.profiler / nsys / NCCL kernel / straggler

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 profile 分布式通信开销？
- [ ] 你能不能解释 profiler 标注 NCCL kernel 时间线，distributed view 聚合各 rank collective 时长？
- [ ] 你能不能举一个 发现某 rank 的 AllReduce 比别人长 20%，定位 NIC 问题 的具体场景？
- [ ] 你能不能说出 只在 rank 0 跑 profiler 漏掉 straggler 这种常见错误模式？

## Q31. Megatron-LM 的代码结构？megatron/core 的新设计？

> 🔴 专家 · Megatron 老版本一坨 monolithic 代码，想抠出 TP 算子单独用都难。`megatron/core` 重新分层之后，TP/PP/分布式优化器都成了独立可组合模块——读懂这套结构，自研框架时基本不用从零开始。

### 1. 核心结论

Megatron-LM 早期代码更多围绕训练脚本、模型定义和并行工具展开，而 `megatron/core` 的新设计则把并行原语、模型组件、优化器与分布式状态管理抽离成更稳定的库层。可以把它理解为：旧结构偏“训练工程仓库”，新结构偏“可复用分布式训练内核”，目的是让 GPT/BERT/T5 等不同上层模型共享一套 TP、PP、SP、分布式 checkpoint 与训练调度基础设施。

### 2. 底层原理

早期 Megatron-LM 的很多能力与具体模型脚本、参数解析和训练循环耦合较紧，例如张量并行层、pipeline schedule、optimizer glue code 常直接服务于 GPT 训练入口。随着功能从 TP/PP 扩展到 distributed optimizer、sequence parallel、transformer engine、distributed checkpointing、MoE 等，代码若继续堆在单体脚本中，会导致复用困难、边界不清和测试成本升高。

`megatron/core` 的设计目标就是把“并行训练通用机制”下沉为独立层，包括：并行进程组管理、并行线性层与 attention/MLP 组件、pipeline schedule、分布式优化器、状态保存恢复等。上层训练仓库只负责把这些能力装配到具体模型与数据流程中。

### 3. 关键机制 / 流程 / 数据结构

从代码分层看，到 2025 年的 Megatron-Core（MCore v0.11+，`megatron/core/` 目录）通常可粗分为几层：
1. 入口与训练编排：参数解析、训练循环、数据集构建、日志与启动脚本（仍在外层 `megatron/training/` 与 `pretrain_*.py`）。
2. 模型层：`megatron/core/models/gpt` / `bert` / `t5` / `mamba` / `multimodal`，定义 `GPTModel`、`MambaModel` 等任务相关组合。
3. `megatron/core/` 核心层：
   - `parallel_state`：维护 TP/PP/DP/CP/EP 等 process group 与 rank 映射，`get_tensor_model_parallel_group()` 等查询 API。
   - `tensor_parallel`：`ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding`、`vocab_parallel_cross_entropy` 等并行算子，含 sequence parallel all-gather/reduce-scatter 变体。
   - `pipeline_parallel/schedules.py`：`forward_backward_pipelining_without_interleaving`（1F1B）、`forward_backward_pipelining_with_interleaving`（Interleaved 1F1B）、`forward_backward_no_pipelining`，以及 P2P 通信封装。
   - `transformer`：`TransformerBlock` / `TransformerLayer` + `ModuleSpec` / `submodules` 抽象，支持通过 spec 组合 attention、MLP、norm、MoE 实现（GPT spec、DeepSeek spec 等）。
   - `distributed`：`DistributedDataParallel`（带梯度桶）与 `distributed/fsdp/` 下的 Megatron-FSDP 实现。
   - `optimizer`：`DistributedOptimizer`（ZeRO-1 风格）、`ChainedOptimizer`、参数分桶与 optimizer state 管理。
   - `dist_checkpointing`：sharded state dict、并行维度感知的保存与恢复，支持 load 时重分片。
   - `inference` / `export`：在线推理引擎与 TensorRT-LLM 导出路径（MCore 原生训推一致化的入口）。
4. 外部集成层：Transformer Engine（FP8/FP4、cuDNN FlashAttention backend）、FlashAttention、CUDA fused kernel、Apex `DistributedFusedAdam` 等。

新设计的一个代表性变化是大量采用 `ModuleSpec` / `submodules` 式抽象：上层通过 spec 对象选择 block 组成（例如切换到 DeepSeek 式 MLA + fine-grained MoE），而不是把实现硬编码在单个模型类里，从而让同一套核心层服务 Llama、DeepSeek、Mixtral、Mamba 混合等多种模型变体。

### 4. 工程权衡 / 性能影响

这种重构的主要收益是可维护性、可测试性和跨模型复用性提升，也更容易与 NeMo、Megatron-Core 作为独立库集成。代价是抽象层级更多，阅读门槛比早期单体代码高；同时高度模块化后，调用链会更长，调试时需要先弄清“配置对象 -> spec -> module 实例 -> 并行 wrapper”的装配路径。

性能上，`megatron/core` 本身不是“因为抽象而更快”，真正的收益来自它更容易稳定承载 fused kernel、sequence parallel、distributed optimizer 和 checkpointing 等优化能力，并使这些能力在不同模型中复用。

### 5. 常见追问 / 易错点

- 不要把 `megatron/core` 理解成“只是把文件搬了位置”，它本质是在做并行训练内核层的解耦。
- `megatron/core` 不等于只做张量并行，它还覆盖 pipeline、optimizer、checkpoint 等核心组件。
- 阅读代码时，很多逻辑不在单个模型类里，而是散落在并行状态、模块 spec、训练调度器和外部 kernel 封装之间。
- 新设计强调库化，但并不意味着上层训练仓库可以完全脱离具体任务脚本；很多数据与训练策略仍在应用层。

### 6. 实践建议

阅读 Megatron-LM 时，建议按“parallel_state -> tensor_parallel -> transformer -> pipeline_parallel -> optimizer -> dist_checkpointing”的顺序建立心智模型，再回看 GPT 训练入口如何装配这些组件。若目的是二次开发，优先在 `megatron/core` 的模块边界上扩展，而不是直接在训练脚本里堆 patch；若目的是定位性能问题，则先确认问题属于并行切分、调度、kernel 还是 checkpoint 路径。

### 7. 30 秒速答

- 一句话核心：把 Megatron-LM 的并行原语抽出来做成可插拔库（megatron.core），便于上层框架接入
- 关键机制：提供 ColumnParallelLinear / VocabParallelEmbedding / PipelineSchedule 等模块化组件
- 易踩坑 / 关键权衡：Megatron-LM 是 reference，Megatron-Core 是库；新项目优先用 core
- 面试加分关键词：megatron-core / TransformerEngine / PipelineSchedule

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Megatron-Core 设计？
- [ ] 你能不能解释 提供 ColumnParallelLinear / VocabParallelEmbedding / PipelineSchedule 等模块化组件？
- [ ] 你能不能举一个 NeMo / TorchTitan 直接复用 megatron-core 的具体场景？
- [ ] 你能不能说出 还在用旧 megatron-LM 单体代码而非 core 抽象 这种常见错误模式？

## Q32. DeepSpeed 的 ZeRO-Infinity 和 ZeRO-Offload 的代码入口？

> 🟡 进阶 · 显存不够时把优化器状态卸到 CPU 内存（Offload）甚至 NVMe（Infinity），听起来魔法但实际入口就是 ds_config 里几行 JSON。看懂这些字段触发的代码路径，才知道为啥同样配置不同模型差几倍速度。

### 1. 核心结论

从实现视角看，DeepSpeed 的 ZeRO-Offload 和 ZeRO-Infinity 都建立在 ZeRO Stage 2/3 的参数、梯度和 optimizer state 分片框架之上，区别主要在于 offload 层次与数据搬运调度复杂度。代码入口通常先从 `deepspeed.initialize()` 和 `runtime/engine.py` 进入，再落到 `runtime/zero` 目录中的 stage 实现、offload 配置解析与 swapper/partition 管理逻辑；其中 ZeRO-Infinity 相比 ZeRO-Offload 更强调 CPU/NVMe 分层搬运与异步预取。

### 2. 底层原理

ZeRO-Offload 的核心目标是把一部分 optimizer state、gradient 或 parameter 从 GPU 挪到 CPU，以显著降低显存占用；ZeRO-Infinity 则进一步把可驻留介质扩展到 NVMe，并引入更复杂的预取、换入换出与 buffer 管理机制，让“超出 CPU 内存友好范围”的超大模型也能训练。

因此，从代码组织上，二者不会各自完全独立一套训练引擎，而是共享 ZeRO 分片框架，再在 offload 设备、tile 化访问、异步 I/O 与 swap 策略上分化。

### 3. 关键机制 / 流程 / 数据结构

定位代码时，通常可以按以下路径阅读（对齐 DeepSpeed 0.18 主线）：
1. 顶层入口：
   - `deepspeed.initialize()`
   - `deepspeed/runtime/engine.py`
   这里负责读取 JSON 配置（`DeepSpeedZeroConfig` / `DeepSpeedZeroOffloadParamConfig` / `DeepSpeedZeroOffloadOptimizerConfig`），并根据 `zero_optimization.stage` 与 `offload_*` 字段决定装配哪类 ZeRO 实现。
2. ZeRO 主体实现：
   - `deepspeed/runtime/zero/stage_1_and_2.py`
   - `deepspeed/runtime/zero/stage3.py`
   Stage 1/2 主要处理 optimizer state / gradient 分片；Stage 3 进一步处理参数分片，是 ZeRO-Infinity 的基础。
3. Offload 配置与实现：
   - `deepspeed/runtime/zero/offload_config.py`：解析 `device`（`cpu` / `nvme`）、`pin_memory`、`buffer_size` 等字段。
   - `deepspeed/runtime/zero/parameter_offload.py`：`DeepSpeedZeRoOffload`，安装 pre/post-forward hook，把参数 gather/release 托管给协调器。
   - `deepspeed/runtime/zero/partitioned_param_coordinator.py`：`PartitionedParameterCoordinator`，负责参数 fetch / release / prefetch 的状态机（`AVAILABLE` / `INFLIGHT` / `NOT_AVAILABLE`）。
   - `deepspeed/runtime/zero/partition_parameters.py`：`Init()` 上下文、`all_gather_coalesced` 等分片原语。
4. Infinity 特有路径：
   - `deepspeed/runtime/swap_tensor/`：`partitioned_param_swapper.py`、`optimizer_swapper.py`、`async_swapper.py`，底层由 `deepspeed.ops.aio` 基于 libaio 提供异步 NVMe I/O。
   - buffer 管理、`contiguous_gradients`、tile 化访问与预取深度均在这一层调。
5. 运行时流程：
   - forward 前由协调器按 trace 预取并 all-gather 或换入参数。
   - backward 后 reduce-scatter 梯度、释放全量参数视图。
   - optimizer step 时在 CPU/NVMe 上更新状态（配合 `DeepSpeedCPUAdam`），并按需回写。

面试里如果只问“代码入口”，回答时抓住三点即可：`initialize/engine` 是总入口，`runtime/zero/stage3.py` + `partitioned_param_coordinator.py` 是 ZeRO-3/Infinity 主线，`runtime/swap_tensor/` 与 `ops/aio/` 是 NVMe 分层内存的关键入口。

### 4. 工程权衡 / 性能影响

ZeRO-Offload 的收益是实现相对成熟，适合“显存不足但 CPU 内存与 PCIe 尚可”的场景；其代价是 host-device 带宽与 CPU 端调度可能成为瓶颈。ZeRO-Infinity 能进一步突破显存和内存上限，但性能更依赖 NVMe 吞吐、异步 I/O 深度、预取命中率和 tile 设计，不当配置时很容易从“显存节省”变成“吞吐崩塌”。

从代码复杂度上，Infinity 的运行时明显更难读，因为它不只是状态放在哪个设备上，而是何时搬、搬多少、是否可重叠、是否命中预取窗口。

### 5. 常见追问 / 易错点

- ZeRO-Offload 不等于 ZeRO-Infinity；后者通常可视为更强的分层 offload 体系，而不是简单“CPU offload + NVMe”。
- 不是所有 offload 都在独立文件名里直观出现，很多逻辑嵌在 stage3 参数生命周期管理中。
- 看代码时不要只盯 `stage3.py`，很多关键行为由参数协调器、swapper、AIO 配置与 partition 模块共同完成。
- 面试回答“入口”时不需要背出每个文件名，但要讲清调用链层次：engine -> zero stage -> offload/swapper。

### 6. 实践建议

若要阅读或二次修改 DeepSpeed，建议先用最小 ZeRO-2/3 配置跑通，再分别打开 `offload_optimizer`、`offload_param` 和 NVMe 配置，结合日志观察参数何时 gather、释放和 swap。定位性能问题时，优先区分瓶颈属于通信、CPU 计算、PCIe 还是 NVMe I/O；否则即使找到了入口文件，也很难理解真实性能退化原因。

### 7. 30 秒速答

- 一句话核心：deepspeed config 中 zero_optimization.stage=3 + offload_param / offload_optimizer 配 cpu 或 nvme
- 关键机制：参数和优化器分片下沉到 CPU 内存或 NVMe，按需 prefetch 回 GPU
- 易踩坑 / 关键权衡：NVMe offload 受 PCIe + 盘 IO 限制，吞吐显著下降
- 面试加分关键词：zero_optimization / offload_param / offload_optimizer / nvme_path

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 ZeRO-Infinity / Offload 代码入口？
- [ ] 你能不能解释 参数和优化器分片下沉到 CPU 内存或 NVMe，按需 prefetch 回 GPU？
- [ ] 你能不能举一个 单机 8×A100 80G 用 NVMe offload 训 175B 的具体场景？
- [ ] 你能不能说出 盲目开 NVMe offload 但 SSD 是消费级，IO 瓶颈拖垮训练 这种常见错误模式？

## Q33. Colossal-AI 的 Gemini 和 PatrickStar 的异同？

> 🔴 专家 · 两个都是 chunk 化的异构内存管理器，但抢的位置不一样：PatrickStar 偏研究、Gemini 在 Colossal-AI 里成了产品。理解它们的 chunk 调度策略，能帮你判断什么时候用 Colossal、什么时候直接走 DeepSpeed。

### 1. 核心结论

Gemini 和 PatrickStar 都是在“大模型参数分块管理 + 异构内存调度”思路下解决显存不足问题的系统，但二者定位与实现风格不同。PatrickStar 更强调类似运行时内存管理器的 chunk 驱动式参数调度，关注训练期间参数/梯度/状态在 CPU 与 GPU 间的生命周期；Gemini 则是 Colossal-AI 体系中更紧密结合 booster、chunk manager 与并行训练栈的统一内存管理方案，工程集成度更高，也更偏向与 ZeRO/FSDP 风格训练体验融合。

### 2. 底层原理

这两者都基于一个共同事实：大模型训练时，参数、梯度和 optimizer state 并不需要在整个 step 全程同时常驻 GPU。只要能在 forward/backward/step 的关键时刻把“当前要用的数据”放到 GPU，把“不立即使用的数据”迁回 CPU 或以 chunk 形式托管，就能显著降低峰值显存。

PatrickStar 更像是先提出“chunk 是一等公民”的设计，把参数映射到 chunk，再由运行时预测访问模式并做迁移与复用。Gemini 继承并发展了这类思想，但更强调与 Colossal-AI 的 chunk-based placement policy、搜索策略和训练封装协同工作，使用户能以更统一的 API 使用异构内存优化。

### 3. 关键机制 / 流程 / 数据结构

两者的共性主要有：
1. 都以 chunk 而非单参数为主要管理粒度，降低碎片和调度开销。
2. 都需要追踪张量状态，如 hold、compute、release、offload 等生命周期。
3. 都依赖访问时机预测或至少阶段性感知，在 forward/backward 前后做迁移。
4. 都试图把 CPU 内存作为 GPU 显存的扩展层。

差异可概括为：
- PatrickStar：
  - 更突出运行时 memory manager 风格。
  - 强调参数 chunk 注册、访问顺序预测、动态置换与训练期状态机。
  - 论文和实现中更常从“通用 chunk-based training runtime”角度描述。
- Gemini：
  - 属于 Colossal-AI 训练系统的一部分，与 booster、plugin、并行策略、optimizer 集成更紧。
  - 提供更工程化的 placement policy 与 chunk manager 设计，强调易用性和系统一体化。
  - 在用户视角上，通常比单独接触 PatrickStar 更贴近“像 ZeRO/FSDP 一样可直接启用”。

从演进关系看，可以认为 Gemini 吸收了 PatrickStar 一类异构内存思想，并将其产品化、框架化到 Colossal-AI 的整体分布式训练体系中。

### 4. 工程权衡 / 性能影响

PatrickStar 的优点是设计思想清晰，便于理解 chunk 级内存调度；缺点是若脱离完整训练框架，落地门槛与兼容成本会更高。Gemini 的优势在于与 Colossal-AI 其他并行能力协同更好，用户启用成本更低；但它也意味着你更受框架整体架构约束，调试时需要理解 plugin、chunk manager 和 runtime policy 的配合。

性能上，两者都受 host-device 带宽、参数访问预测准确性、chunk 大小与碎片率影响。若预取不准或 chunk 过大，会增加无效搬运；若 chunk 过小，又会引入管理和传输碎片化开销。

### 5. 常见追问 / 易错点

- Gemini 不是简单重命名版 PatrickStar，二者有思想继承，但工程定位不同。
- 不要把它们等同于纯 ZeRO；它们更强调 chunk 驱动的异构内存运行时，而不只是 optimizer/gradient/state 分片。
- 显存节省不代表训练一定更快，很多场景只是“能训起来”，吞吐可能明显低于纯 GPU 常驻方案。
- 讨论差异时，最好从“抽象层级、框架集成度、运行时策略”三个维度回答，而不是只说谁支持 CPU offload。
- 2024–2026 的生态现状是：FSDP2 `OffloadPolicy` + DeepSpeed ZeRO-Infinity 已成为异构内存训练的主流，Gemini / PatrickStar 更多是历史参照与思想源头，新建项目不推荐从 Gemini 起步。

### 6. 实践建议

面试中建议先讲共性：都用 chunk 和异构内存扩展有效显存；再讲差异：PatrickStar 更像运行时内存管理方案，Gemini 更像 Colossal-AI 中系统化集成后的实现。工程落地时，应先确认目标是“极限省显存”还是“省显存同时保留较好吞吐”，再决定是否采用此类方案；同时务必结合 PCIe/NUMA/CPU 内存带宽做验证。对 2025 年的新项目，更实际的选择是 FSDP2 `fully_shard(..., offload_policy=CPUOffloadPolicy(...))` 或 DeepSpeed ZeRO-3 + NVMe offload，它们在社区活跃度、与 Transformer Engine/FP8 的兼容性上均明显优于 Gemini/PatrickStar。

### 7. 30 秒速答

- 一句话核心：两者都是基于异构内存（GPU+CPU）的 chunk 管理，Gemini 是 Colossal-AI 当前主线
- 关键机制：Gemini 用 chunk-based 内存管理，PatrickStar 早期方案已被 Gemini 替代
- 易踩坑 / 关键权衡：Colossal-AI 文档以 Gemini 为准，PatrickStar 知识陈旧不要在面试主推
- 面试加分关键词：Gemini / chunk / heterogeneous memory / Colossal-AI

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Gemini vs PatrickStar？
- [ ] 你能不能解释 Gemini 用 chunk-based 内存管理，PatrickStar 早期方案已被 Gemini 替代？
- [ ] 你能不能举一个 单卡 24G 训 13B 时用 Gemini 自动 offload 的具体场景？
- [ ] 你能不能说出 把 PatrickStar 当主流方案讨论而忽视 Gemini 这种常见错误模式？

## Q34. FairScale 的 FullyShardedDataParallel（FSDP）实现细节？

> 🔴 专家 · 现在 PyTorch 里的 FSDP 其实是 FairScale 这套实现迁过来的——读懂 FairScale 的 wrap、flatten、reshard hook 链，就理解了 FSDP 现在为什么长这个样子，以及哪些 bug 是历史包袱。

### 1. 核心结论

FairScale 的 FSDP 可以视为 PyTorch 原生 FSDP 的重要前身之一，其核心思想是把参数展平并分片存储，在需要计算时短暂 all-gather 全量参数，计算后再释放或重分片，从而显著降低单卡参数常驻显存。其实现重点不只是“分片”本身，而是参数 flatten、autograd hook、前后向中的 gather/free、状态字典处理以及混合精度/CPU offload 等机制如何配合。

### 2. 底层原理

普通 DDP 下，每个 rank 长期持有一份完整参数、完整梯度和大部分 optimizer state；而 FSDP 将参数打平为 flat parameter 后，仅让每个 rank 持有其中一段 shard。forward 前需要把当前模块的完整参数 all-gather 出来，forward/backward 使用全量视图完成计算，随后再在适当时机丢弃非本地 shard，并在反向阶段对梯度做 reduce-scatter 或等价聚合。

这种设计利用了一个关键事实：某个子模块的完整参数只在该模块前后向的局部时间窗口内必需，因此无需整步训练都常驻。

### 3. 关键机制 / 流程 / 数据结构

FairScale FSDP 的典型实现细节包括：
1. 参数 flatten：
   - 将多个原始参数拼成 flat parameter，减少小 tensor 管理开销。
   - 维护原参数到 flat buffer 的 metadata，用于 view 恢复与 state dict 映射。
2. shard 管理：
   - 初始化时把 flat parameter 按 rank 切片，每个 rank 持有本地 shard。
   - 非本地 shard 平时不常驻本卡。
3. forward 前处理：
   - 注册 pre-forward hook 或在模块入口显式触发 all-gather。
   - 重建该模块当前所需的 full parameter view。
4. backward 后处理：
   - 利用 autograd hook 在梯度 ready 后进行 reduce-scatter、释放 full param 或切回 shard 形态。
   - 及时释放中间全量参数，控制峰值显存。
5. 嵌套与 auto wrap：
   - FSDP 常对 transformer block 级别做分层 wrap，使 gather/free 粒度更细。
   - 嵌套层次会影响 overlap、显存峰值和通信次数。
6. 状态保存恢复：
   - 需要在 full state dict、local state dict、sharded state dict 之间做映射。
   - optimizer state 也要能按 flat parameter 与 shard 对齐。

这套机制的难点在于：参数在不同阶段既可能是 shard、本地 flat buffer，也可能是临时 full view，框架必须保证 autograd、optimizer 和 checkpoint 都看到一致语义。

### 4. 工程权衡 / 性能影响

FairScale FSDP 的最大收益是显著降低参数常驻显存，使超大模型训练成为可能；代价是每层或每个 wrapped 单元都要付出 all-gather / reduce-scatter 通信成本。wrap 粒度太细会导致通信碎片化，太粗又会增大瞬时显存峰值并削弱 overlap。

此外，flatten 带来更好的内存管理和更少的元数据开销，但也增加了参数别名、state dict 转换和调试难度。CPU offload 虽能进一步省显存，却可能显著拖慢训练。

### 5. 常见追问 / 易错点

- FSDP 不是简单“把 DDP 参数切一下”，它依赖参数生命周期重写、hook 与状态管理配合。
- flatten parameter 是实现关键，不只是优化小细节；没有它，参数管理和通信组织会更复杂。
- auto wrap 不存在通用最优值，必须结合层大小、网络带宽和显存预算调。
- FairScale FSDP 与 PyTorch 原生 FSDP 思路相近，但接口、实现细节和后续演进并不完全一致，回答时不要混为一谈。
- FairScale 仓库从 2023 年起已基本不再活跃更新，生产推荐直接使用 PyTorch 2.4+ 原生 FSDP2（`torch.distributed.fsdp.fully_shard`）。FSDP2 用 `DTensor` + per-parameter sharding 取代了 `FlatParameter`，不再需要打平，参数/梯度/optimizer state 均按原始 tensor 粒度保留 `DTensor` 视图，state dict 处理因此比 FairScale 时代简单很多。

### 6. 实践建议

如果面试问“实现细节”，建议按“flat parameter -> shard -> forward all-gather -> backward reduce-scatter -> free full param -> checkpoint/state dict”这条主线回答 FairScale/FSDP1 的路径，再补一句“FSDP2 已改为 per-parameter DTensor 分片，不再 flatten”。优先以 block 级 wrap 建立基线，再调混合精度、激活重计算和 CPU offload；排查性能时要重点看 all-gather 长尾、wrap 粒度和是否出现过多 full parameter 驻留。

### 7. 30 秒速答

- 一句话核心：FairScale 是 FSDP 的早期参考实现，PyTorch FSDP / FSDP2 已替代它
- 关键机制：FairScale FSDP 是 ZeRO-3 风格分片，2024 后维护减少
- 易踩坑 / 关键权衡：新项目直接用 torch.distributed.fsdp（FSDP2 / FSDP1），不要再选 FairScale
- 面试加分关键词：fairscale / FSDP1 / FSDP2 / DTensor

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 FairScale FSDP 历史？
- [ ] 你能不能解释 FairScale FSDP 是 ZeRO-3 风格分片，2024 后维护减少？
- [ ] 你能不能举一个 历史代码迁移：fairscale.FullyShardedDataParallel → torch FSDP 的具体场景？
- [ ] 你能不能说出 还在新项目里用 fairscale 而错过 FSDP2 的 DTensor 改进 这种常见错误模式？

## Q35. Hugging Face 的 Accelerate 如何简化分布式训练？

> 🟢 基础 · 同一份训练脚本要支持单卡、DDP、FSDP、DeepSpeed，不用 Accelerate 你就得写一堆 if/else。它的核心是把各种 backend 抽成统一接口——理解它的封装哪里轻、哪里漏抽象，能少踩很多坑。

### 1. 核心结论

Accelerate 的核心价值不是发明新的并行算法，而是把设备放置、进程启动、DDP/FSDP/DeepSpeed/混合精度等训练样板代码统一封装起来，让用户用接近单机单卡的 PyTorch 写法迁移到多卡、多机甚至多后端训练。它通过 `Accelerator` 统一管理 model、optimizer、dataloader、backward、gather 和日志行为，从而显著降低分布式训练接入门槛。

### 2. 底层原理

传统 PyTorch 分布式训练常需要用户自己处理：初始化 process group、设置 local rank、模型 `.to(device)`、包一层 DDP/FSDP、不同 rank 的 dataloader sampler、梯度同步、混合精度和 checkpoint 逻辑。Accelerate 的思路是把这些样板逻辑收敛到一个运行时对象中，根据配置自动选择后端并调整训练流程。

因此，用户代码主要表达“训练语义”，而不是“我现在在哪个 rank、该不该打印、该不该同步”。框架通过环境探测与配置文件，把底层后端差异屏蔽掉。

### 3. 关键机制 / 流程 / 数据结构

Accelerate 的常见工作流如下（对齐 Accelerate 1.x 主线）：
1. 通过 `accelerate config` 生成运行配置：
   - 指定单机/多机、GPU 数、混合精度（`fp16` / `bf16` / `fp8` via TransformerEngine 或 MS-AMP）、DeepSpeed / FSDP / FSDP2 / Megatron-LM / TorchTitan 等后端。
2. 在代码中构造 `Accelerator`：
   - 运行时对象内部持有设备、进程信息、分布式类型与插件配置（`FullyShardedDataParallelPlugin`、`DeepSpeedPlugin`、`MegatronLMPlugin` 等）。
3. 使用 `accelerator.prepare(...)`：
   - 自动处理 model、optimizer、scheduler、dataloader 的设备放置和 wrapper。
   - 根据配置决定是否包 DDP、FSDP1/FSDP2（Accelerate 1.0+ 已原生支持 `fsdp_version=2`）或接入 DeepSpeed engine。
4. 训练阶段：
   - 用 `accelerator.backward(loss)` 统一反传入口。
   - 用 `with accelerator.accumulate(model):` 上下文处理梯度累积，内部会在非同步步关闭 DDP/FSDP 的梯度 all-reduce。
   - 用 `accelerator.gather_for_metrics()`、`pad_across_processes()`、`unwrap_model()` 等 API 处理多进程张量聚合和保存（`gather_for_metrics` 会自动丢弃 dataloader 为对齐 batch 补的 padding 样本）。
5. 启动阶段：
   - `accelerate launch` 负责进程拉起与环境变量设置，减少手写 `torchrun` 参数和 rank 解析代码；底层仍基于 `torchrun` / `torch.distributed.run` 的 elastic rendezvous。

可以把它理解为“轻量训练运行时 + 多后端适配层”，而不是完整 Trainer 替代品。

### 4. 工程权衡 / 性能影响

Accelerate 的最大优势是开发效率高、迁移成本低，尤其适合研究代码、原型验证和需要在 DDP/FSDP/DeepSpeed 间切换的场景。缺点是抽象层会隐藏部分底层细节，遇到复杂性能问题时，最终还是要回到具体后端的语义中排查；另外，某些高度定制训练循环下，直接写原生分布式代码可能更可控。

性能上，Accelerate 本身通常不是主要瓶颈，关键还是底层选用的 DDP/FSDP/DeepSpeed 配置是否合理。但若用户误以为“用了 Accelerate 就自动最优”，往往会忽略 bucket、wrap policy、ZeRO 阶段、sampler 与 checkpoint 等真正影响性能的因素。

### 5. 常见追问 / 易错点

- Accelerate 是封装层，不是新的通信后端；真正执行分布式的是 DDP/FSDP/DeepSpeed 等。
- `prepare()` 之后对象语义可能已变化，例如模型被 wrapper 包裹，保存和取底层模块时要用 `unwrap_model()`。
- 它能简化启动和训练样板，但不能替代对底层并行策略的理解。
- 与 `transformers.Trainer` 的关系要分清：Accelerate 更底层、更通用，Trainer 可以建立在它之上。

### 6. 实践建议

如果目标是快速把单卡 PyTorch 代码扩展到多卡，优先考虑 Accelerate；若后续需要极致性能调优，再逐步下沉到 FSDP 或 DeepSpeed 的原生配置层。编写代码时，尽量统一通过 `accelerator` 提供的 API 处理 backward、保存、日志和 gather，避免一半走封装、一半手写分布式逻辑，导致语义混乱。

### 7. 30 秒速答

- 一句话核心：封装 DDP/FSDP/DeepSpeed 启动逻辑，统一 launcher 和配置文件
- 关键机制：accelerate config + accelerate launch，运行时按 backend 路由到对应实现
- 易踩坑 / 关键权衡：Accelerate 不提供新并行算法，只是 wrapper；性能仍靠底层 backend
- 面试加分关键词：accelerate / launch / config / backend

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Accelerate 的作用？
- [ ] 你能不能解释 accelerate config + accelerate launch，运行时按 backend 路由到对应实现？
- [ ] 你能不能举一个 HF Trainer 默认基于 accelerate 启动 multi-GPU 的具体场景？
- [ ] 你能不能说出 把 accelerate 当算法库期望它自带 ZeRO/FSDP 优化 这种常见错误模式？

## Q36. torch.distributed.fsdp 的 limit_all_gathers 参数作用？

> 🔴 专家 · FSDP 默认会 prefetch 多层的 all-gather 来抢通信带宽，但 prefetch 太狠的话临时显存峰值能把卡撑爆。`limit_all_gathers=True` 是一个保险——bandwidth 和显存的 trade-off，正确配置才能稳定训练。

### 1. 核心结论

`limit_all_gathers` 的核心作用是限制 FSDP 前向阶段过早、过多地发起参数 all-gather，避免多个模块的 full parameter 同时在途或同时驻留，从而抑制显存峰值、减少 allocator 抖动，并让 all-gather 调度更可控。它是一个 rate limiter，优先换取更稳定的内存行为，而不是单纯追求最激进的通信预取。

### 2. 底层原理

FSDP 在使用参数分片时，模块真正执行计算前，往往要先把当前模块所需的 full parameter all-gather 回来。若框架为了追求 overlap，允许 CPU 线程连续为后续多个模块提前发起 all-gather，就可能出现“当前模块还没算完，后面多个模块的 full param 已经提前到位”的情况。

这样虽然可能增加通信与计算重叠，但也会带来两个副作用：一是瞬时显存峰值变高，因为多个模块的 full parameter 同时存在；二是 allocator 更容易出现碎片、重试和抖动。`limit_all_gathers=True` 时，FSDP 会对这种预取节奏做限制，只允许有限数量的 all-gather 在前向路径上超前进行。

### 3. 关键机制 / 流程 / 数据结构

可以把它理解为“限制 in-flight full parameter 的数量”。典型影响路径如下：
1. 模块即将执行 forward 时，需要进入 pre-forward unshard/all-gather。
2. 若未限制，CPU 线程可能继续为后续 wrapped module 提前提交 all-gather。
3. 若开启 `limit_all_gathers`，运行时会在合适位置阻塞或放慢 CPU 侧提交节奏。
4. 这样可避免多个模块 full parameter 同时堆积在显存中。
5. 当前模块计算完成并可 reshard/release 后，后续模块再继续推进。

因此，它主要影响的是“all-gather 发起时机与并发度”，而不是参数分片语义本身。

### 4. 工程权衡 / 性能影响

开启后，优点是显存峰值更稳、CUDA allocator 压力更小、发生 OOM 或 `cudaMalloc retry` 的概率更低，尤其在大模型、细粒度 wrap、激活重计算叠加时更有价值。缺点是预取窗口变小，某些场景下通信-计算 overlap 会略受影响，吞吐未必达到最激进配置下的上限。

因此它是典型的“以内存稳定性换一部分调度自由度”的参数。若训练本来就接近显存边界，通常优先保留该限制；若显存余量充足且 profile 显示 all-gather 长尾明显，再考虑评估关闭后的收益。

### 5. 常见追问 / 易错点

- 它不是限制 collective 的总次数，而是限制 all-gather 的超前并发与提交节奏。
- 它主要作用在前向参数 unshard/all-gather 路径，不是梯度 reduce-scatter 的开关。
- 开启后看到 CPU 线程短暂等待，通常是预期行为，不一定代表 GPU 空转严重。
- 关闭后吞吐不一定更高；若引入更高显存峰值和 allocator 抖动，最终 step time 反而可能变差。
- FSDP2（`torch.distributed.fsdp.fully_shard`，PyTorch 2.4+）已不再暴露 `limit_all_gathers` 这个参数；等价的节流能力由内部的 collective stream 管理与 `reshard_after_forward` 语义承担，用户层更常通过 `set_reshard_after_forward(int)` / `set_modules_to_forward_prefetch` 控制 prefetch 深度。

### 6. 实践建议

建议把 `limit_all_gathers=True` 作为 FSDP1 大模型训练的保守默认值，先确保训练稳定，再结合 profiler 观察 full parameter 生命周期、显存峰值和 all-gather 长尾。若要关闭，最好与 auto wrap 粒度、混合精度、激活重计算一起联合评估，而不要只看单轮平均吞吐。迁移到 FSDP2 的项目无需再查 `limit_all_gathers`，但应在 profiler 中观察同一指标（显存峰值 + in-flight all-gather 数量），再调 `reshard_after_forward` 与 `forward_prefetch`。

### 7. 30 秒速答

- 一句话核心：限制同时 in-flight 的 all-gather 数量，控制显存峰值
- 关键机制：默认会 prefetch 下一层参数，limit=True 则严格按一层一拉，减少峰值
- 易踩坑 / 关键权衡：开了会牺牲一些 overlap，显存紧张时才用
- 面试加分关键词：limit_all_gathers / prefetch / FSDP memory

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 limit_all_gathers 的作用？
- [ ] 你能不能解释 默认会 prefetch 下一层参数，limit=True 则严格按一层一拉，减少峰值？
- [ ] 你能不能举一个 边界 OOM 模型开 limit_all_gathers=True 换显存 的具体场景？
- [ ] 你能不能说出 盲目开 limit 让 prefetch 失效拖慢训练 这种常见错误模式？

## Q37. 大模型训练的 checkpoint 格式？sharded checkpoint 的合并？

> 🟡 进阶 · 175B 的 checkpoint 单文件几百 GB，每个 rank 各存一份还要 rank0 合并，I/O 是真噩梦。sharded checkpoint 让每个 rank 写自己的那片，加载时再按 mesh 重新组装——大模型必备技能。

### 1. 核心结论

大模型训练的 checkpoint 不只是模型权重，还应包含 optimizer state、lr scheduler、AMP scaler、随机数状态、全局步数与数据消费进度等恢复所需状态。格式上通常分为 full checkpoint 与 sharded checkpoint；后者把参数或状态按 DP/FSDP/ZeRO 的分片方式分别保存，适合超大模型训练。所谓“合并 sharded checkpoint”，本质是依据元数据把各 shard 恢复成统一参数视图，必要时再导出为单文件或推理友好的格式。

### 2. 底层原理

单文件 full checkpoint 的优点是简单直观，但对百亿、千亿参数模型往往不可行：保存慢、单文件过大、单机内存不够、恢复时 gather 成本高。因此训练系统更常按并行切分结果保存：每个 rank 只写自己的参数 shard、optimizer state shard 和相关 metadata。

这样做的关键在于“数据本体 + 元数据描述”同时存在。数据本体是每个 rank 的 shard 文件，元数据则记录参数名、全局 shape、dtype、切分维度、offset、world size、并行拓扑等信息。没有这些元数据，就无法正确合并。

### 3. 关键机制 / 流程 / 数据结构

训练场景中常见 checkpoint 内容包括：
1. 模型权重：`state_dict`、flat parameter shard 或 safetensors/pt 文件。
2. optimizer state：如 Adam 的 `exp_avg`、`exp_avg_sq`，通常也是分片保存。
3. 训练进度：global step、consumed samples、epoch、grad accumulation 位置。
4. 调度与数值状态：lr scheduler、AMP grad scaler。
5. 随机状态：Python、NumPy、Torch CPU/CUDA、各 rank RNG。
6. 数据恢复信息：sampler epoch、shuffle seed、dataloader offset。

sharded checkpoint 的合并流程通常是：
1. 读取 manifest / metadata，建立“参数名 -> 全局张量布局”的映射。
2. 遍历各 shard 文件，按 offset 或分片维度把局部块放回正确位置。
3. 若训练时使用 flat parameter（FSDP1 时代），需要先还原 flat buffer，再切回原参数视图；FSDP2 / DCP 直接以 `DTensor` 粒度记录 placement，不需要这一步。
4. 必要时做 dtype 转换、去 wrapper 前缀或导出到目标格式。
5. 若只为推理导出，通常只合并模型权重；若要完整恢复训练，还需同时重建 optimizer/scheduler/RNG 等状态。

PyTorch 生态当前主流是 `torch.distributed.checkpoint`（DCP）：`dcp.save` / `dcp.load` 支持多 rank 并行写入、load-time resharding（同一份 ckpt 可在不同 TP/PP/DP 拓扑上加载），`dcp.async_save` 把 GPU→CPU staging 与落盘解耦，仅 staging 阶段阻塞训练主线。离线导出为单文件时使用 `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`（反向是 `torch_save_to_dcp`）。Megatron-Core 的 `dist_checkpointing` 与 DeepSpeed 的 Universal Checkpoint 思路类似，也都支持按目标拓扑重分片加载。

### 4. 工程权衡 / 性能影响

full checkpoint 的优势是迁移和加载简单，适合推理发布、小模型训练或跨框架交换；缺点是保存和恢复代价大。sharded checkpoint 的优势是可扩展、保存时无需全量聚合，更适合 FSDP、ZeRO、Megatron 等大模型训练；代价是格式更复杂、跨框架可移植性更差、离线合并步骤更重。

合并成本往往不只是 I/O，还包括 CPU 内存峰值、网络搬运和格式转换。尤其 optimizer state 体量通常远大于权重本身，若只是做推理导出，没必要把训练状态一并合并。

### 5. 常见追问 / 易错点

- checkpoint 不是只有模型参数；缺少 optimizer、scheduler、RNG 和数据进度，严格意义上不能无损 resume。
- sharded checkpoint 不是把文件简单 `cat` 起来，必须按 metadata 重建全局布局。
- FSDP、ZeRO、Megatron 的 shard 语义并不完全相同，不能假设不同框架的分片天然兼容。
- 若训练中使用 flat parameter、张量并行或 vocab parallel，合并时尤其要注意参数命名和切分维度。

### 6. 实践建议

训练时优先采用框架原生的 sharded checkpoint 机制，并定期做“保存 -> 重新加载 -> 继续训练”演练。若需要发布模型，单独提供一条离线 merge/export 路径，把训练态 checkpoint 转成推理态权重。应明确区分“用于 resume 的 checkpoint”和“用于部署的导出权重”，避免二者混用。

### 7. 30 秒速答

- 一句话核心：每个 rank 保存自己的 shard，离线工具合并成 full state_dict
- 关键机制：torch.distributed.checkpoint（DCP）支持并行写入，重启时按当前并行度重切
- 易踩坑 / 关键权衡：checkpoint 形状要带 metadata，否则换 TP/PP 维度后无法 resume
- 面试加分关键词：torch.distributed.checkpoint / DCP / sharded_state_dict

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 sharded checkpoint？
- [ ] 你能不能解释 torch.distributed.checkpoint（DCP）支持并行写入，重启时按当前并行度重切？
- [ ] 你能不能举一个 FSDP2 默认 DCP 写盘，比单 rank 集中保存快 8× 的具体场景？
- [ ] 你能不能说出 切回 TP=4 加载 TP=8 存的 ckpt 失败 这种常见错误模式？

## Q38. 训练恢复的 consistency 问题？如何确保 resume 后的 loss 一致？

> 🟡 进阶 · 半路重启之后 loss 突然飘高几个点，问题大概率出在 RNG 状态、DataLoader 位置、optimizer state 三件套没存全。要做到 bit-exact resume，每个细节都不能漏——否则花几百万 GPU-hour 训出来的曲线根本无法复现。

### 1. 核心结论

resume 后想让 loss 与中断前严格一致，必须恢复的不只是模型权重，而是“整个训练状态机”：optimizer、scheduler、AMP scaler、随机数状态、数据读取位置、sampler/shuffle 状态、全局步数、梯度累积位置等都要一致。否则即使权重一样，后续看到的 batch、dropout mask、学习率或梯度缩放不同，loss 也会马上漂移。

### 2. 底层原理

训练过程是一个强状态依赖系统。某一步的 loss 不只由当前参数决定，还取决于：当前 batch 是什么、dropout 随机掩码是什么、学习率是多少、是否处于梯度累积中间步、AMP 缩放器当前 scale 是多少、数据增强随机种子是什么。

因此，“能继续训”与“resume 后逐步 loss 完全一致”是两个不同目标。前者只要求大体恢复；后者则要求恢复到中断点前几乎相同的执行轨迹。分布式训练下这个问题更难，因为每个 rank 还涉及独立 RNG、sampler 切分和跨 rank 同步顺序。

### 3. 关键机制 / 流程 / 数据结构

要尽可能保证 resume 一致，通常需要同时保存并恢复以下状态：
1. 模型状态：参数、buffer、EMA（若有）。
2. optimizer 状态：动量、一二阶矩、master weights（混合精度时 fp32 主副本）。
3. scheduler 状态：当前 step、warmup/decay 进度（`lr_scheduler.state_dict()`）。
4. AMP 状态：`torch.amp.GradScaler` 的 scale 与增长/回退计数（`scaler.state_dict()`）。
5. RNG 状态：Python（`random.getstate()`）、NumPy（`np.random.get_state()`）、Torch CPU（`torch.get_rng_state()`）、每张 GPU 的 CUDA RNG（`torch.cuda.get_rng_state_all()`），且各 rank 分别保存。
6. 数据状态：sampler epoch、shuffle seed、数据游标、已消费样本数、dataloader 恢复位置（PyTorch 2.4+ 的 `torchdata.StatefulDataLoader` / `StatefulDistributedSampler` 提供了 `state_dict()` / `load_state_dict()` 官方接口，IterableDataset 也可实现 `state_dict` 协议）。
7. 训练控制状态：global step、micro step、grad accumulation index、当前 epoch。

此外还要注意恢复流程：
1. 在 step 边界统一保存，并确保所有 rank 看到同一个 checkpoint 版本。
2. resume 时先恢复分布式拓扑与进程组，再恢复模型和 optimizer。
3. 之后恢复 RNG 与数据游标，确保下一个 batch 与中断前预期一致。
4. 若使用 IterableDataset 或流式数据，必须显式保存消费偏移，否则很难严格回放。

### 4. 工程权衡 / 性能影响

保存越多状态，resume 一致性越好，但 checkpoint 更大、保存更慢、实现更复杂。严格一致通常还需要 deterministic 设置，这会带来额外性能损失，例如限制某些高性能 kernel、关闭非确定性算法、稳定数据顺序等。

在很多工业训练中，目标往往不是“逐步 bitwise 一致”，而是“统计意义上一致且不会破坏收敛”。但在调试数值问题、验证训练稳定性或做 ablation 时，强一致恢复非常重要。

### 5. 常见追问 / 易错点

- 只恢复模型权重而不恢复 optimizer，loss 往往立刻变化，尤其对 Adam 类优化器影响很大。
- 只恢复 global step 但不恢复 dataloader/sampler 位置，会让后续 batch 顺序不同。
- 分布式下只保存 rank0 的 RNG 或 sampler 状态是不够的，各 rank 都可能不同。
- 若 checkpoint 不是在 step 边界原子写入，可能出现部分 rank 已更新、部分 rank 未更新的“不一致快照”。

### 6. 实践建议

如果目标是严格复现实验，建议固定随机种子（`torch.manual_seed` + `torch.cuda.manual_seed_all`）、启用必要的 deterministic 配置（`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`）、在 step 边界加 barrier 后保存，并把 RNG、dataloader offset、grad accumulation 位置都纳入 checkpoint。最好建立自动化校验：中断训练后 resume，比较连续若干步的 batch id、lr、loss 和 grad norm 是否一致；若只能做到近似一致，也应明确允许的误差范围。

### 7. 30 秒速答

- 一句话核心：必须恢复 RNG、dataloader epoch、optimizer state，并保证并行拓扑一致
- 关键机制：seed → sampler.set_epoch → optimizer.state_dict → scheduler.state_dict 全链路恢复
- 易踩坑 / 关键权衡：忘记保存 dataloader index 会让 resume 后跳过或重复数据
- 面试加分关键词：state_dict / RNG / dataloader_state / deterministic

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 resume 后 loss 不一致？
- [ ] 你能不能解释 seed → sampler.set_epoch → optimizer.state_dict → scheduler.state_dict 全链路恢复？
- [ ] 你能不能举一个 resume 后前几步 loss 和 baseline 对不上，多半 RNG 没恢复 的具体场景？
- [ ] 你能不能说出 只存 model+optimizer 不存 RNG 与 sampler 状态 这种常见错误模式？

## Q39. 大模型训练的 data pipeline 优化？WebDataset、tfrecord？

> 🟡 进阶 · 训练 trillion-token 数据集时 GPU 经常等数据，瓶颈往往不在算力而在数据加载。WebDataset/tfrecord 这些 shard 化、流式格式比小文件快几个量级，data pipeline 不优化、买再多 H100 也喂不饱。

### 1. 核心结论

大模型训练的数据瓶颈常不在模型算子，而在“小文件过多、远端存储吞吐不足、解码开销重、shuffle 代价高、CPU 预处理跟不上”。优化 data pipeline 的核心思路是把随机小 I/O 变成顺序大吞吐，把昂贵解码与训练主线程解耦，并让分片、shuffle、prefetch、cache 与分布式采样协同工作。WebDataset 和 TFRecord 都是在这种背景下常见的样本封装格式，本质目标都是提升顺序读取效率和大规模分布式可扩展性。

### 2. 底层原理

若直接从海量原始小文件读取，训练会遭遇元数据查询多、文件打开关闭频繁、对象存储请求放大、worker 间竞争严重等问题。将样本打包为较大的 shard 后，可以把读路径变成近似顺序流式访问，从而显著降低 IOPS 压力。

与此同时，大模型训练通常还叠加 tokenizer、图像解码、数据增强、样本过滤与拼接等 CPU 密集操作。如果这些步骤与 GPU 训练串行执行，GPU 很容易因“等数据”而空转。因此通常需要把 I/O、decode、transform、host-to-device copy 做成流水线。

### 3. 关键机制 / 流程 / 数据结构

常见优化手段包括：
1. 数据分片：将大量样本打包成大小适中的 shard，减少小文件随机访问。
2. 多级 shuffle：先做 shard-level shuffle，再做 shard 内 sample-level shuffle，兼顾随机性与吞吐。
3. 异步预取：利用 dataloader worker、prefetch queue、后台下载线程提前准备下一批数据。
4. 本地缓存：把热点 shard 缓存到本地 NVMe，降低远端对象存储抖动。
5. 解码优化：图像/文本预处理并行化，必要时把部分 decode 或 resize 下沉到更高效实现。
6. 传输优化：`pin_memory`、persistent workers、合理 batch collation，降低 host-device copy 开销。

两类常见格式的特点可概括为：
- WebDataset：
  - 常把样本打成 tar shards，适合顺序流式读取。
  - 对对象存储友好，PyTorch 生态使用方便，常与 IterableDataset 结合。
  - 适合超大规模图文训练，因为 shard 容易做按节点切分、缓存和重试。
- TFRecord：
  - 是 record-oriented 的二进制样本容器，单条样本边界清晰，TensorFlow 生态非常成熟。
  - 在大规模生产链路里常与索引、压缩、预取结合使用。
  - 在 PyTorch 中也能用，但通常需要额外 reader 或桥接组件，生态便利性不如 WebDataset 直接。

2024–2026 的补充选项：
- MosaicML StreamingDataset（`streaming` 库）：基于 MDS 格式 + shard 级 remote→local 缓存，内置 deterministic resume（按 `global step` 还原样本指针）与跨节点分布式采样，是 Databricks / LLM 预训练常见选择。
- HuggingFace `datasets` + Parquet：Parquet 列式格式天然适合行式读取与部分列采样，`load_dataset(..., streaming=True)` 走 IterableDataset 流式接口，适合 text-only 与小图场景。
- NVIDIA DALI：把 decode / resize / normalize 下沉到 GPU 或 CPU+GPU 混合流水线，能明显缓解图像解码瓶颈，常与 WebDataset 或 TFRecord 搭配。

### 4. 工程权衡 / 性能影响

shard 太小，仍会有大量打开文件和调度开销；太大则会降低 shuffle 粒度，失败恢复也更重。缓存可以提升稳定性，但会增加本地盘占用和缓存失效管理成本。更激进的数据增强或在线 tokenize 虽然灵活，却可能把瓶颈从存储转移到 CPU。

WebDataset 的优势是简单、顺序读取友好、和对象存储结合自然；缺点是若 tar 设计不合理，细粒度随机访问不方便。TFRecord 的优势是工业化程度高、记录边界明确；缺点是在非 TensorFlow 训练栈中常需要额外集成工作。没有绝对最优格式，关键是看团队主框架、存储系统与数据生产链路。

### 5. 常见追问 / 易错点

- data pipeline 优化不等于只调 `num_workers`，真正瓶颈可能在存储、解码或 shuffle 设计。
- 顺序吞吐与样本随机性是折中关系，不能只追求一个指标。
- 远端对象存储带宽够，不代表请求模式就合理；小对象风暴仍会拖垮吞吐。
- 格式选型要看整个生态链路，而不是只看训练代码是否能读。

### 6. 实践建议

建议先做端到端 profile，拆出“下载/I/O、解码、tokenize、collate、H2D、GPU 空转”各阶段占比，再决定是换格式、加缓存还是重构流水线。若团队以 PyTorch 为主且训练数据主要来自对象存储，WebDataset 往往是较自然的起点；若已有成熟的 TF 数据生产体系，TFRecord 也完全可用，但要提前评估与现有训练框架的集成成本。

### 7. 30 秒速答

- 一句话核心：WebDataset/MosaicML streaming/tfrecord 解决海量小文件 + 随机访问难题
- 关键机制：把数据打包成 tar/shard，sequential read + per-worker shuffle，避免 metadata 风暴
- 易踩坑 / 关键权衡：ckpt 训练 1T token 时 metadata IO 会拖垮 lustre/NFS，必须 shard
- 面试加分关键词：WebDataset / MosaicML streaming / tfrecord / shard

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 data pipeline 优化？
- [ ] 你能不能解释 把数据打包成 tar/shard，sequential read + per-worker shuffle，避免 metadata 风暴？
- [ ] 你能不能举一个 LLaMA 训练用 mmap shard + nproc_per_node 个 dataloader worker 的具体场景？
- [ ] 你能不能说出 直接读 1 亿个小 jsonl 让 NFS metadata 服务挂掉 这种常见错误模式？

## Q40. 多模态大模型的训练基础设施挑战？图文数据的加载？

> 🟡 进阶 · 多模态训练一个 batch 里图像和文本长度都不一样、解码 CPU 开销大、shard 不均衡就会有 straggler。光把 LLM 训练那套搬过来根本不够用——data pipeline 和 batching 都得专门重写。

### 1. 核心结论

多模态大模型训练的基础设施难点，往往不只是“模型更大”，而是“数据形态更复杂”：文本是变长 token，图像是高解码成本、变分辨率、强预处理依赖的数据，二者还要保持样本配对、过滤质量与跨模态对齐。图文数据加载的核心目标是同时解决吞吐、随机性、配对一致性与 batch 形状稳定性，避免 GPU 被解码、拼接和不均衡样本拖慢。

### 2. 底层原理

相较纯文本训练，多模态 pipeline 多了至少三类额外成本：
1. 图像文件读取与解码成本远高于纯文本 token 读取。
2. 图像分辨率、patch 数、文本长度都可能变化，导致 batch 尺寸波动大。
3. 样本经常带有 caption、对话、OCR、region 或多图关联信息，schema 更复杂，数据清洗与配对错误更常见。

因此，多模态训练常见瓶颈不是单点，而是存储、解码、tokenize、image processor、collate、dynamic batching 共同形成的串联长链路。

### 3. 关键机制 / 流程 / 数据结构

图文训练常见的数据加载流程是：
1. 样本封装：每条样本包含 image bytes/路径、text、metadata、可能的多图字段或任务标签。
2. 分片存储：把图文样本打成 shard，便于顺序读取和分布式切分。
3. worker 侧处理：
   - 读取图像字节并解码。
   - 做 resize、crop、normalize 或视觉 tokenizer 预处理。
   - 对文本做 tokenize、模板拼接与特殊 token 注入。
4. batch 组装：
   - 对文本做 padding 或 packing。
   - 对图像按固定分辨率、动态分桶或 patch 数约束做对齐。
   - 生成 attention mask、position id、image-grid metadata 等辅助结构。
5. 分布式分发：确保各 rank 获取互不重叠且统计均衡的数据分片。

通常还需要额外机制：
- 图文配对校验，避免图片与文本错位。
- 按分辨率/文本长度做 bucketing，减少 padding 浪费。
- 本地缓存与异步预取，降低图像 decode 抖动。
- 坏样本隔离、重试和统计，避免单个损坏文件拖垮整个训练。
- 视觉 token 数量差异大（Qwen2-VL / InternVL 类的 native-resolution、动态 tile 或 NaViT / AnyRes patch 打包会让每条样本视觉 token 在数百到数千之间跳动），因此 batch 维度往往是“variable vision tokens + variable text tokens”的双变量，通常要用 sequence packing（把多条样本拼成固定长度）+ `FlashAttention` varlen 接口 / `BlockMask` 做 per-sample attention mask，才能保持吞吐。

### 4. 工程权衡 / 性能影响

若图像预处理放在训练时在线完成，灵活性高，但 CPU 和 I/O 压力大；若做离线预处理，可显著提速，但会牺牲变换灵活性并增加数据膨胀。固定图像分辨率能简化 batch 与算子实现，但会损失部分信息；动态分辨率或多尺度训练更高效利用样本，却会增加 batch 调度复杂度。

多模态训练还更容易出现负载不均：某些 batch 图像大、文本长、解码慢，导致 step time 抖动明显。因此仅看平均吞吐不够，还要关注 tail latency 和慢 worker。

### 5. 常见追问 / 易错点

- 多模态数据问题往往先表现为“GPU 利用率低”，但根因可能在图像 decode、样本配对或 batch 形状波动。
- 不能把图文样本简单视为“图片文件 + 一段文本”；真实生产中经常带多图、多段文本和复杂 metadata。
- 只做全局随机 shuffle 不一定够，还要保证不同模态来源、分辨率和任务类型的采样分布合理。
- 坏样本处理必须工程化，否则分布式训练里单个损坏样本就可能导致整个作业失败。

### 6. 实践建议

工程落地时，建议优先把图像读取/解码、文本 tokenize、batch collate 分阶段 profile，确认瓶颈是在存储、CPU 还是 batch 组织。数据格式上尽量采用适合顺序流式读取的 shard 化方案，并把图像字节、文本与 metadata 一起封装，减少跨文件查找。若训练规模很大，可考虑分辨率分桶、本地 NVMe 缓存、异步 decode 和坏样本黑名单机制，先把吞吐与稳定性做实，再追求更复杂的数据混训策略。

### 7. 30 秒速答

- 一句话核心：图文/音视频数据 IO 不均，token 长度差异大，需要 packing 和异构 batch
- 关键机制：图像解码慢，文本快，需 prefetch + sample packing 让 GPU 不空闲
- 易踩坑 / 关键权衡：不 pack 让 padding token 占 30%+ FLOPs
- 面试加分关键词：multimodal / sample packing / image decode / variable seq len

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 多模态训练 infra 挑战？
- [ ] 你能不能解释 图像解码慢，文本快，需 prefetch + sample packing 让 GPU 不空闲？
- [ ] 你能不能举一个 VLM 训练把 image embedding 预算好缓存到 shard 的具体场景？
- [ ] 你能不能说出 原图直接喂 GPU，CPU 解码成瓶颈 这种常见错误模式？

## Q41. RLHF 训练的 infra 设计？PPO 的分布式实现？

> 🔴 专家 · RLHF 同时跑 actor、critic、reward、ref 四个模型，还要 rollout 生成、计算 advantage、训练更新交替——分布式调度比纯 SFT 难一个数量级。infra 设计不好整套 pipeline 利用率能掉到 10%。

### 1. 核心结论

RLHF 训练基础设施的核心，不是单纯把 PPO 跑起来，而是把“采样、打分、训练、参数同步、样本缓存”做成稳定流水线。工业实现通常会拆成 actor、reference model、reward model、critic、learner 与 rollout buffer 多类角色；PPO 的分布式难点则在于在线采样带来的强耦合、高方差与版本一致性约束，需要通过异步生成、批量打分、微批训练和受控参数刷新来维持吞吐与稳定性。

### 2. 底层原理

RLHF 中的 PPO 与监督学习最大区别在于：训练样本不是静态语料，而是当前策略模型在线生成的 rollout。一次迭代通常先由 actor 按 prompt 采样 response，再由 reward model 打分，并结合 reference model 的 KL 约束得到 reward；critic 估计 value，最后 learner 基于 advantage 做 PPO 更新。

因此 infra 需要同时满足三件事：
1. 高吞吐生成，因为 rollout 往往比反向训练更耗时。
2. 策略一致性，避免 learner 已更新很多步，而 actor 还在用过旧权重采样。
3. 数据闭环，保证 prompt、response、logprob、value、reward、mask、sequence length 等字段能完整回流到训练端。

### 3. 关键机制 / 流程 / 数据结构

一个常见分布式 PPO 流程是：
1. prompt dispatcher 将样本分发到 actor 集群。
2. actor 用当前 policy 生成 response，同时记录 old logprob、token mask、长度信息。
3. reward/reference 服务对整条序列或末尾 answer 打分，计算 reward 与 KL penalty。
4. critic 计算 value，拼出 trajectory 或 sequence-level batch。
5. rollout buffer 按 token 级或 sequence 级存储 `input_ids / attention_mask / old_logprobs / rewards / values / advantages / returns`。
6. learner 读取 buffer，多轮 mini-batch PPO update，更新 policy 与 value 网络。
7. learner 定期将新权重广播或 checkpoint 到 actor/reward/critic 侧需要同步的组件。

常见的并行拆法包括：
- 生成侧做数据并行或张量并行，提高解码吞吐（2024–2026 的事实标准是用 vLLM / SGLang 作为 rollout 引擎，吃连续批处理 + PagedAttention + prefix cache）。
- learner 侧做 DDP/FSDP/ZeRO/Megatron，承接 PPO、GRPO、RLOO、REINFORCE++ 等算法。
- reward model 与 reference model 采用独立推理服务，避免与 learner 争抢显存。
- 通过参数版本号管理 rollout 所属 policy version，避免混入不兼容样本。
- **Hybrid engine（colocate）路线**：veRL（HybridFlow）、OpenRLHF hybrid engine、NeMo-Aligner 等把 actor 训练与 rollout vLLM 引擎放到同一组 GPU，依赖 weight resharding / `update_weights_from_tensor` 在 FSDP/Megatron 分片与 vLLM TP 布局之间转换；优点是生成阶段能占用全部卡，缺点是每次同步都要刷新 vLLM KV 缓存。
- **Disaggregated（分离）路线**：actor、critic、reward、rollout 分别占独立 GPU 池，靠 Ray + NCCL 广播权重，适合大集群和异构硬件，但总体 MFU 低。

### 4. 工程权衡 / 性能影响

若完全同步：每轮 rollout 后再统一训练，逻辑简单但 GPU 利用率低。若完全异步：吞吐高，但 actor 采样的策略可能严重滞后，导致 off-policy 程度上升，PPO 稳定性变差。通常需要折中：例如限制 actor 落后 learner 的版本窗口、使用固定长度的 rollout buffer、设置较短的参数刷新周期。

另一个权衡是“生成成本 vs 训练成本”。大模型 RLHF 往往生成更贵，因此系统瓶颈常在 decoding、KV 缓存占用与 reward 打分，而不是 PPO 反向本身。若 rollout 侧没有做连续批处理、prefix cache、动态 batching，训练侧再快也会被前端饿死。

### 5. 常见追问 / 易错点

- PPO 在 RLHF 中通常不是环境交互式 RL，而是基于静态 prompt 的 sequence optimization，数据结构与传统控制任务不同。
- advantage、return、old logprob 必须与采样时的策略版本一致，不能在 learner 侧用新模型重算替代。
- reward model、reference model 不一定需要每步同步；reference 常固定，reward 也经常独立冻结。
- 很多训练不稳定并非 PPO 公式错，而是 rollout 截断、padding mask、KL 计算范围或 eos 处理不一致。

### 6. 实践建议

工程实现上建议先把系统拆成“生成平面”和“学习平面”，分别做容量规划与监控。优先保证 rollout schema 完整、版本可追踪、buffer 可恢复，再调 PPO 超参。若生成明显成为瓶颈（典型 RLHF 里 rollout 占 80%+ 时间），可优先切到 vLLM/SGLang + hybrid engine；若 learner 成为瓶颈，再引入 FSDP2、ZeRO 或 sequence packing。对 2025 年的新项目，推荐直接从 veRL 或 OpenRLHF 的既有 recipe（PPO / GRPO / DAPO）起步，而不是自研 actor-learner 框架；要跑 DeepSeek-R1 风格的 reasoning RL 时，GRPO + 无 critic 的路径比传统 PPO 更简洁，也是 2025 年多数开源复现的默认选择。

### 7. 30 秒速答

- 一句话核心：actor / ref / reward / critic 四模型并存，需要 colocate 或 disaggregate
- 关键机制：rollout 阶段跑推理生成样本，learn 阶段做 PPO 更新，actor 权重周期性同步给 rollout
- 易踩坑 / 关键权衡：rollout 和 learn 资源比例不均会让一方空转
- 面试加分关键词：PPO / actor-critic / rollout / vLLM / OpenRLHF / verl

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 RLHF / PPO infra 设计？
- [ ] 你能不能解释 rollout 阶段跑推理生成样本，learn 阶段做 PPO 更新，actor 权重周期性同步给 rollout？
- [ ] 你能不能举一个 OpenRLHF 用 vLLM 做 rollout + DeepSpeed 做 learn 的具体场景？
- [ ] 你能不能说出 rollout 不开 KV cache 让生成阶段比 learn 还慢 这种常见错误模式？

## Q42. MoE 模型的 all-to-all 通信优化？Tutel、FasterMoE？

> 🔴 专家 · MoE 训练里 all-to-all 是单点瓶颈，token 不均衡时还会有少数 expert 被打爆。Tutel 做了动态调度、FasterMoE 做 hierarchy 通信——这俩都是为了让 MoE 真正跑出 dense model 的吞吐。

### 1. 核心结论

MoE 的核心瓶颈往往不是 expert MLP 本身，而是 token dispatch 与 combine 阶段的 all-to-all。优化目标是减少无效 token 搬运、降低小包通信开销、改善 load balance，并让路由、打包、通信、专家计算尽可能流水化。Tutel、FasterMoE 这类系统都是围绕 dispatcher、路由布局与并行映射做优化，以降低 expert parallel 的通信常数项。

### 2. 底层原理

MoE 前向通常先由 gate 为每个 token 选择 top-k experts，然后把属于不同 expert 的 token 从原 rank 发送到持有对应 expert 的 rank；专家计算完成后，再把输出送回原位置。这个过程天然形成两次 all-to-all 或等价的 token exchange。

难点在于：
1. token 到 expert 的映射高度动态，每步负载可能不同。
2. 单个 expert token 数常不均匀，容易出现 straggler。
3. 每个 token payload 不大但数量多，容易形成大量小消息和 packing 开销。
4. top-2 路由会进一步放大通信体积与 combine 成本。

### 3. 关键机制 / 流程 / 数据结构

MoE 通信优化常见手段包括：
1. token 分桶与连续布局：先按目标 expert 或目标 rank 做排序/前缀和，把 token 打成连续 buffer，减少 scatter-gather 碎片。
2. fused dispatch/combine：把索引计算、pack、unpack、combine 尽量融合，减少 kernel launch 与中间拷贝。
3. capacity factor 与溢出控制：限制单个 expert 接收 token 上限，避免极端倾斜拖垮尾延迟。
4. group-local routing：尽量先在节点内或子组内完成 dispatch，减少跨机 all-to-all。
5. 通信与计算重叠：部分 expert 已收到 token 后可先开始 MLP，不必等所有 peer 到齐。

从系统实现上看：
- Tutel 强调高效的 dispatcher、针对 MoE 的通信布局优化，以及与 PyTorch/Megatron 等框架的集成便利性。
- FasterMoE 更强调从路由、通信到执行调度的系统级加速，包括减少 token 迁移冗余、改善多机场景下的负载均衡与执行效率。
- 2024–2026 的补充：**Megablocks** 把 MoE MLP 改写成 block-sparse GEMM，绕开 capacity factor / token drop 的离散化问题；**DeepEP**（DeepSeek 开源，Q73 已详谈）把 NVLink ↔ RDMA 异构 domain forwarding、20 SM 饱和、FP8 dispatch、hook-based 纯 overlap 做成了事实标准，Megatron-Core MoE 与 SGLang EP 推理路径均已对接。

三者共同点是都试图把“动态稀疏计算”转成更适合 GPU 和网络的连续批处理问题。

### 4. 工程权衡 / 性能影响

更激进的路由优化和 fused kernel 能降低延迟，但实现复杂、可维护性差，也更容易受特定 CUDA/NCCL/拓扑版本影响。capacity factor 设得小，可以控制最坏时延与显存，但会带来 token drop 或 auxiliary loss 压力；设得大，吞吐更稳，但长尾和显存峰值更高。

另外，all-to-all 对网络拓扑极其敏感。单机 NVLink 下 MoE 可能扩展性很好，跨机后若 InfiniBand 布局差、EP 组跨节点过多，性能会急剧恶化。因此 expert parallel 的分组通常要优先考虑机内局部性。

### 5. 常见追问 / 易错点

- MoE 不是天然“参数大但计算便宜”，通信设计不好时反而比 dense 更慢。
- top-k 路由的主要代价不只在 gate，而在 dispatch/combine 的搬运和不均衡。
- 只看平均每个 expert 的 token 数不够，还要看 tail expert、节点间流量和跨机比例。
- Tutel、FasterMoE 解决的是系统效率问题，不会自动修复糟糕的负载均衡或不合理的并行拓扑。

### 6. 实践建议

实际部署时，建议先做三类 profile：gate 后的 token 分布、all-to-all 时间占比、各 expert 的尾延迟。若跨机 all-to-all 成为瓶颈，优先重排 EP 组与节点映射，而不是先微调 kernel。对生产训练，通常要同时监控 capacity overflow、aux loss、drop rate 和 expert 利用率；只有模型质量与系统吞吐一起看，MoE 优化才有意义。

### 7. 30 秒速答

- 一句话核心：Tutel/FasterMoE 用 grouped GEMM + 流水重叠 dispatch/combine
- 关键机制：把 expert compute 拆 stream，与 all-to-all 通信并行；token capacity factor 控负载
- 易踩坑 / 关键权衡：all-to-all 跨机带宽是 MoE 训练的硬瓶颈
- 面试加分关键词：Tutel / FasterMoE / grouped GEMM / capacity factor

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 MoE all-to-all 优化？
- [ ] 你能不能解释 把 expert compute 拆 stream，与 all-to-all 通信并行；token capacity factor 控负载？
- [ ] 你能不能举一个 DeepSeek-V3 用 DeepEP 自研 all-to-all 库做 H800 集群优化 的具体场景？
- [ ] 你能不能说出 不做 token-drop 让 hottest expert 拖慢全机 这种常见错误模式？

## Q43. 长上下文训练（100K+ tokens）的显存优化？Ring Attention？

> 🔴 专家 · 100K 上下文光 attention 的中间 score 矩阵就 100K×100K，单卡根本存不下。Ring Attention 把 KV 在卡之间环形传递，让每张卡只持有一段——这是 Gemini/Claude 这类长上下文模型能训出来的核心技术之一。

### 1. 核心结论

100K+ tokens 训练的核心矛盾，是 attention 的激活、KV、中间 buffer 与通信都会随序列长度急剧膨胀。显存优化通常不能只靠单一技巧，而要组合使用 FlashAttention/块化 attention、激活重计算、序列并行、上下文并行、paged buffer 和更细粒度的 micro-batch。Ring Attention 的价值在于把超长序列上的注意力计算拆到多个设备上，以环式传递 K/V 或局部状态，减少单卡常驻上下文压力，同时维持全局注意力语义。

### 2. 底层原理

长上下文训练下，显存开销主要来自：
1. attention score 与 softmax 相关中间量。
2. Q/K/V 激活与反向所需缓存。
3. 更长序列带来的 optimizer state 以外的临时 buffer 膨胀。
4. 序列切分后新增的通信状态与同步开销。

标准全注意力若直接物化完整 `S x S` 相关矩阵，100K 级别几乎不可接受。因此优化方向通常是两类：一类是通过 IO-aware/块化算法避免显式保存大矩阵；另一类是把序列维度切到多卡，让每张卡只保有部分 token 上下文。

### 3. 关键机制 / 流程 / 数据结构

长上下文训练常见显存手段包括：
1. FlashAttention 或 blockwise attention：按 tile 流式计算，避免显式存储完整 attention matrix。
2. 激活重计算：不保存部分中间激活，反向时重算。
3. sequence/context parallelism：把序列维切到多卡，降低单卡激活与 KV 占用。
4. selective recompute：只重算 attention 或 MLP 中最占显存的部分。
5. 更小 micro-batch + grad accumulation：用吞吐换显存。
6. packed sequence 与长度分桶：减少 padding 浪费。

Ring Attention 可概括为一种“沿设备环传递上下文块”的分布式注意力实现。典型思路是：
- 每个 rank 持有本地一段 query/key/value。
- 本地 query 固定，远端 K/V 分块按 ring 顺序依次传入。
- 每次只对当前块做局部注意力更新，并维护 running max / running sum 等归一化状态（复用 FlashAttention 的 online softmax）。
- 环跑完一圈后，本地 query 已聚合所有上下文贡献，但不需要任何时刻在单卡上持有全量 K/V。

其本质是把“全量上下文常驻显存”改成“分块流式访问 + 分布式归约”。

2024–2026 的主流实现分三类，面试时可串起来讲：
- Megatron-SP：沿序列维度对 LayerNorm / Dropout / residual 等逐元素算子分片，attention 内部仍保留全 Q/K/V，适合序列到 4K–32K。
- DeepSpeed-Ulysses：对 Q/K/V 的 head 维做 all-to-all，把序列分片换成 head 分片后再跑 attention，单步延迟短但对 `num_heads` 有整除约束。
- Ring / Llama-3 Context Parallel / USP（Unified SP）：沿序列维度环形传递 K/V，可与 Ulysses 组合成“内 head-parallel + 外 ring-parallel”两层，是 Llama 3 1M、Gemini long-ctx、TorchTitan CP 路线的主流选择。
-  causal mask 下的负载不均衡问题（后段 query 计算量远大于前段）通常用 zigzag / load-balanced Ring Attention 的 chunk 重排来修。

### 4. 工程权衡 / 性能影响

Ring Attention 能显著降低单卡显存压力，但会引入额外跨卡通信，并使实现更依赖拓扑与 overlap。若互联较慢，环式传输可能把瓶颈从显存转成通信。与此同时，过度依赖 checkpointing 虽然省显存，却会显著增加反向重算时间；micro-batch 过小还会降低张量核利用率。

因此长上下文优化是在“显存峰值、通信体积、重算开销、吞吐稳定性”之间做四方折中，而不是单纯追求最大可训练长度。

### 5. 常见追问 / 易错点

- 长上下文问题不只在 attention，embedding、loss mask、position encoding、optimizer step 前后的临时 buffer 也会放大。
- FlashAttention 解决的是 attention 内部存储问题，不等于自动解决 100K 序列的整体显存问题。
- Ring Attention 不是免费扩展；如果设备间带宽差，性能可能不升反降。
- 训练与推理的长上下文瓶颈不同，训练更受反向和激活影响，不能简单套用推理优化结论。

### 6. 实践建议

建议先拆账：分别统计参数、优化器、激活、attention workspace、KV 与通信 buffer 的显存占比，再决定是否引入 context parallel 或 Ring Attention。若只是从 8K 提升到 32K，通常 FlashAttention + checkpointing 已足够；若目标是 100K+，则应尽早从并行维度设计入手，规划序列切分、网络拓扑与通信重叠。上线前要重点压测长序列尾部 case，因为很多实现只在平均长度下稳定。

### 7. 30 秒速答

- 一句话核心：把 KV 沿 sequence 维切到不同 rank，环形传递做局部 attention 累积
- 关键机制：每个 rank 持有一段 KV，环上滚动一圈完成全局 attention，激活显存与 seq 解耦
- 易踩坑 / 关键权衡：环上通信成为新瓶颈，需要 IB + double buffering
- 面试加分关键词：Ring Attention / Context Parallel / CP / sequence sharding

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 长上下文 Ring Attention？
- [ ] 你能不能解释 每个 rank 持有一段 KV，环上滚动一圈完成全局 attention，激活显存与 seq 解耦？
- [ ] 你能不能举一个 100K context Llama 用 CP=8 + Ring Attention 的具体场景？
- [ ] 你能不能说出 只切 sequence 不切 KV 让单 rank 仍要装全 KV 这种常见错误模式？

## Q44. 训练任务的 fault tolerance？elastic training 的实现？

> 🟡 进阶 · 上千卡训练时一晚上挂掉几张是常态——一个 GPU ECC 错误整个 job 全停，是不可接受的浪费。`torchelastic` 让 worker 退出后重新组队继续训练，这是大规模训练的最低保命线。

### 1. 核心结论

大模型训练的 fault tolerance 重点，不只是“定期存 checkpoint”，而是要保证节点故障、作业重启、成员变更后，训练能以可接受的损失继续推进。elastic training 则进一步允许 world size 或 rank 成员在运行中变化，并通过 rendezvous、重组进程组、重建 dataloader 和恢复 checkpoint 来继续训练。它解决的是长作业在不稳定集群上的可持续运行问题。

### 2. 底层原理

在大规模集群中，单机故障、网络闪断、抢占式实例回收都很常见。若训练框架要求固定 world size 且任何 rank 失败都导致全局失败，那么长周期训练的成功率会快速下降。因此系统需要两层能力：
1. checkpoint/restart：作业失败后从最近保存点恢复。
2. membership elasticity：允许 worker 数变化后重新组成训练拓扑。

elastic training 的核心是把“训练状态”和“进程组成员关系”解耦。成员变化时，系统先通过 rendezvous 服务重新分配 rank/world size，再让训练脚本重新初始化通信组，并从兼容的 checkpoint 继续。

### 3. 关键机制 / 流程 / 数据结构

典型 fault tolerance 设计包含：
1. 周期性 checkpoint：保存模型、优化器、lr scheduler、随机数状态、sampler 进度、global step。
2. 原子提交：先写临时目录，再 rename 或写 success marker，避免半写入快照。
3. 心跳与健康检查：检测 rank 卡死、节点失联、网络异常。
4. 失败分类：区分可恢复错误和需要人工介入的错误。
5. 自动重启控制器：由调度器或 launcher 拉起新 worker。

elastic training 额外需要：
- rendezvous backend：用于 worker 注册、屏障和新 rank 分配（PyTorch 默认 `c10d`，也可用 `etcd-v2`）。
- restart epoch/step 协议：明确重启后从哪个逻辑位置恢复。
- world-size-aware checkpoint/load：如按 DP shard 重分片或重新映射 optimizer state（依赖 DCP 的 load-time resharding 能力）。
- dataloader 重建：确保成员变化后样本切分仍正确。

在 PyTorch 生态中，`torchrun --nnodes=MIN:MAX --max-restarts=... --rdzv-backend=c10d` 的 elastic 模式就是把这些机制封装到了 launcher 和 worker 生命周期管理里。2024–2026 的新增路径：
- **torchft**（PyTorch 官方 fault-tolerant 训练库，2024 Meta 开源）在 DDP/FSDP 之上加入了 per-step 失败隔离，故障 rank 可被“就地跳过”而不整体重启，适合 H100/GB200 级别的大规模训练。
- **Google / MegaScale / ByteDance**（MegaScale NSDI '24）论文工程经验里，restart 时间从小时级压到分钟级的关键往往是 in-memory redundant checkpoint + 异步持久化，而不只是更频繁落盘。

### 4. 工程权衡 / 性能影响

更频繁的 checkpoint 能降低故障损失，但会增加存储带宽占用与训练暂停时间。更强的 elastic 能提高作业存活率，但会显著增加实现复杂度，尤其是 FSDP/ZeRO/MoE 这类带分片状态的训练，成员变化后的状态重映射并不便宜。

此外，elastic 后 world size 改变往往还会影响 global batch size、梯度噪声和学习率策略。若只是简单恢复训练而不修正这些量，收敛曲线可能漂移。因此 fault tolerance 设计既是系统问题，也是训练一致性问题。

### 5. 常见追问 / 易错点

- “有 checkpoint”不等于“可容错”，若快照不完整、不可原子恢复或不能跨 world size 加载，真正出故障时仍会失败。
- 只恢复模型参数而不恢复 dataloader/sampler 状态，会造成数据重复或跳样。
- elastic world size 变化后，global batch、lr、grad accumulation 常需要联动调整。
- 某些 hang 并不是单机故障，而是部分 rank 集体等待；这类问题需要健康检查和 timeout 机制配合排查。

### 6. 实践建议

建议把 fault tolerance 分三层建设：先做可靠 checkpoint，再做自动重启，最后再做 elastic membership。checkpoint 方案要通过真实演练验证：故意 kill 若干 rank、模拟对象存储写失败、验证恢复后的 batch id 和 loss 轨迹。若业务允许，优先保证固定 world size restart 的稳定性；只有在抢占式资源或超长任务场景下，再投入 elastic training 的复杂实现。

### 7. 30 秒速答

- 一句话核心：torchelastic / TorchX 在 worker 失联时自动 rendezvous 重组并 resume
- 关键机制：基于 etcd/c10d backend 做成员关系管理，failure 时重启 healthy worker 集合
- 易踩坑 / 关键权衡：elastic 必须配合 checkpoint 频率足够高，否则恢复时回滚太多 step
- 面试加分关键词：torchelastic / rendezvous / etcd / c10d / TorchX

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 elastic training？
- [ ] 你能不能解释 基于 etcd/c10d backend 做成员关系管理，failure 时重启 healthy worker 集合？
- [ ] 你能不能举一个 千卡训练每 30min 存 ckpt，节点挂掉自动剔除剩余继续 的具体场景？
- [ ] 你能不能说出 启用 elastic 但 ckpt 间隔 1 天，恢复时丢一天进度 这种常见错误模式？

## Q45. 大模型训练的 experiment tracking？Weights & Biases、MLflow？

> 🟢 基础 · 训了一周的 run 配置写在哪、loss 曲线跟谁比、超参怎么扫——没 tracking 工具全得靠人脑记，团队稍大就乱。WandB/MLflow 不是花架子，是大模型时代的实验科学基础设施。

### 1. 核心结论

大模型训练的 experiment tracking 不是简单记几条 loss 曲线，而是要把“代码版本、配置、数据版本、环境、指标、checkpoint、告警、产物”串成可追溯闭环。Weights & Biases（W&B）更偏向面向训练过程的实时观测与协作分析，MLflow 更强调实验记录、参数与模型产物管理。两者都能用，但选型应看团队更需要在线可视化协作，还是统一的实验/模型注册治理。

### 2. 底层原理

大模型实验复杂度高，单次训练通常涉及数百个超参与多种外部依赖。如果没有统一 tracking，后果往往不是“看不到图”，而是无法回答以下问题：这次结果对应哪份代码？用了哪版数据？是否启用了某个 fused kernel？checkpoint 是在哪个 step 保存的？异常发生前 GPU 利用率如何？

因此 experiment tracking 的本质是为训练系统建立元数据平面，把运行期信号与离线产物统一索引。

### 3. 关键机制 / 流程 / 数据结构

一套完整 tracking 通常包含：
1. run metadata：实验名、git commit、分支、镜像版本、机器拓扑、随机种子、启动命令。
2. config logging：模型结构、优化器、并行策略、数据配置、环境变量。
3. metrics logging：train/val loss、lr、grad norm、throughput、tokens/s、GPU util、显存、通信耗时。
4. artifact 管理：checkpoint、评测结果、配置快照、profile 文件。
5. lineage：run 与数据集、模型版本、下游评测任务之间的关联。

两类常见系统的侧重点通常是：
- W&B：实时面板、丰富图表、对比实验、在线协作与告警能力较强，适合训练过程观测与团队共享；W&B Artifacts 可与 registry/血缘绑定。
- MLflow：实验记录、参数管理、artifact 存储、model registry 更通用，适合与企业内部平台或部署链路集成；MLflow 2.x 新增了 LLM-specific 的 `mlflow.evaluate` 与 Prompt Engineering UI。

2024–2026 可考虑的其他替代：
- TensorBoard + 自建对象存储：最小依赖，适合单团队。
- Aim（开源、本地优先）、Neptune、ClearML：介于 W&B 与 MLflow 之间，私有化部署友好。
- Weights & Biases Weave / Arize / LangSmith：侧重 LLM 应用层的 trace/eval，而不是训练指标。

在大模型训练里，还常把 tracking 与调度平台、日志系统、profiling 系统联动，例如把 job id、cluster id、checkpoint uri、Nsight/HTA trace 与 Flight Recorder 链接统一挂到同一条 run 下。

### 4. 工程权衡 / 性能影响

记录越全面，复现与排障越容易，但日志量、网络开销和接入复杂度也会增加。若每 step 高频上传大量标量、直方图或模型样本，可能反过来拖慢训练主进程，尤其是在多 rank 同时上报时更明显。因此通常需要控制采样频率，并将“每步关键指标”和“低频大对象产物”分层管理。

另外，托管式平台接入方便，但会带来合规、网络出口与成本问题；自建方案更可控，却需要额外维护存储、鉴权和查询能力。

### 5. 常见追问 / 易错点

- experiment tracking 不应只记录 rank0 loss，还应记录吞吐、数据延迟、OOM、重启次数等系统指标。
- 没有数据版本和代码版本的实验记录，复现价值很有限。
- 多机训练若每个 rank 都无节制打点，常会造成日志风暴；通常需要聚合后再报。
- W&B 与 MLflow 不是互斥关系，很多团队会用一个做过程观测，另一个做模型与产物治理。

### 6. 实践建议

实践中建议先定义最小必需字段：代码版本、配置快照、数据版本、核心训练指标、checkpoint 地址、异常事件，再逐步扩展。多机训练优先采用 rank0 汇总或分层上报，避免对训练热路径造成干扰。若团队强调研究协作与在线分析，W&B 往往更顺手；若已有内部 MLOps 平台或需要更强 registry 能力，MLflow 更容易融入统一治理体系。

### 7. 30 秒速答

- 一句话核心：W&B/MLflow 记录 hyper-param、metric、artifact，方便对比和重现
- 关键机制：run/experiment 层次结构，metric 时间序列写入，artifact 版本化 ckpt/dataset
- 易踩坑 / 关键权衡：高频 log 会拖慢训练，建议 N step 一次 reduce
- 面试加分关键词：wandb / mlflow / artifact / hyperparameter sweep

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 experiment tracking？
- [ ] 你能不能解释 run/experiment 层次结构，metric 时间序列写入，artifact 版本化 ckpt/dataset？
- [ ] 你能不能举一个 大模型预训练每 100 step log loss/lr，每 1k step log gradient norm 的具体场景？
- [ ] 你能不能说出 每 step log 几十个 metric 让 IO 成为瓶颈 这种常见错误模式？

## Q46. NCCL_ALGO / NCCL_NCHANNELS / NCCL_BUFFSIZE 怎么调？各算法适用场景？

> 🔴 专家 · 同样的硬件、同样的模型，NCCL 参数调不调有时差 20% 吞吐。`NCCL_ALGO=Ring/Tree/CollNet` 适合不同 message size，`NCHANNELS` 决定并发流——不懂这些，集群买回来发挥不出 80% 性能。

### 1. 核心结论

`NCCL_ALGO`、`NCCL_NCHANNELS`、`NCCL_BUFFSIZE` 是 NCCL 三个最常被工程师摸到的旋钮：算法决定 collective 在拓扑上走哪种通信模式（Ring / Tree / CollNet / NVLS），channel 数决定并行管道数（GPU 上同一 collective 切成几条 ring/tree 同时跑），buffer size 决定每个 channel 一次 staging 的字节数。它们的默认值是 NCCL 在启动时按拓扑与 message size 探测出来的，绝大多数情况下不要动；只有在「特定 message 区间命中算法切换边界」「PCIe / Ethernet 拓扑下默认参数明显跑不到带宽上限」「跨机 all-reduce 比单机 NVLink 慢一个数量级以上」这类异常时才值得调，调之前必须先有 `nccl-tests` 的 baseline 数字。

### 2. 底层原理

NCCL 把 collective 拆成「算法」与「协议」两个独立维度。算法决定通信图的形态：

- **Ring**：所有 rank 连成一圈，每步把数据切成 N 份依次传递；带宽利用率高，跨步同步少，message 大时几乎能跑满 NVLink/IB 带宽；缺点是 latency 随 rank 数线性增长。
- **Tree**：把 rank 组织成二叉/双向树，reduce 时沿树上行、broadcast 时沿树下行；latency 是 `O(log N)` 而非 `O(N)`，小 message 比 ring 显著快。
- **CollNet**：把 reduction offload 到带 SHARP 能力的 IB switch（在网计算）；只在 IB SHARP 拓扑命中时启用。
- **NVLS（NVLink SHARP）**：NCCL 2.17+ 在 H100/H200 NVSwitch 上用 multicast + 在 switch 内 reduce 实现的 in-network primitive，进一步降低 NVLink 域内 all-reduce 与 reduce-scatter 的字节数。NCCL 2.19+ 已默认在 H100 NVL8/NVL72 启用，由 `NCCL_NVLS_ENABLE` 控制。

协议（`NCCL_PROTO`）有 `LL` / `LL128` / `Simple` 三档，决定每个 chunk 的同步开销：LL 用 8B flag-data pair 让 GPU 之间通过 polling 直接看到对端 ready，延迟最低但带宽最差；LL128 是 NVLink-friendly 的 128B 变体；Simple 走完整 fence + memcpy，message 大时带宽最高。

### 3. 关键机制 / 流程 / 数据结构

- `NCCL_ALGO`：取值 `Tree`、`Ring`、`CollNet`、`NVLS`、`NVLSTree`，可用逗号枚举多个让 NCCL 在其中按 message size 选择，例如 `NCCL_ALGO=Ring,Tree`。NCCL 内部有一张 `tuning table`（基于 GPU 型号、互联类型、rank 数），按 message size 决定算法切换点。手动指定单一算法相当于关掉这张表的自动选择。
- `NCCL_NCHANNELS`（也写作 `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS`）：每个 collective 启动多少条并行 ring/tree。一般默认 `2~32`，由拓扑决定。channel 多 → 更多 SM 投入通信、更高聚合带宽，但占用 SM 也更多，会与计算 kernel 抢资源。H100 上常见默认 `NCHANNELS=8~32`，PCIe 机器上常见 `NCHANNELS=2~4`。
- `NCCL_BUFFSIZE`：每条 channel 的 FIFO buffer 大小，默认 4 MB。对超大 message（GB 级 all-reduce）增大到 8/16 MB 可减少 chunk 切分次数；但显存占用 = `NCHANNELS × BUFFSIZE × 2`（双缓冲），盲目调大会吃掉激活预算。

PyTorch 侧观测调优结果通常用三件套：`torch.profiler` distributed view 看 collective 时间线、Flight Recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE=2000`）记 collective metadata、`nccl-tests`（`all_reduce_perf -b 8 -e 8G -f 2 -g 8`）量基准带宽。

### 4. 工程权衡 / 性能影响

调这三个变量的真正决策图谱：

- **小 message dominated（< 1 MB，例如梯度 bucketing 后的小 bucket、PP 控制流）**：强制 `NCCL_ALGO=Tree` + `NCCL_PROTO=LL` 通常比默认快；channel 数可以减小到 `1~2`，避免启动多 channel 的 fixed cost。
- **大 message dominated（> 100 MB，例如 ZeRO-3 参数 all-gather、FSDP reshard）**：`NCCL_ALGO=Ring` + `NCCL_PROTO=Simple` + `NCCL_BUFFSIZE=8388608`（8 MB）通常更稳；H100 上加 `NCCL_NVLS_ENABLE=1` 在 NVLink 域内能再省 ~30% 字节数。
- **PCIe-only 多机**：默认 ring 容易把 PCIe 打满拥塞，建议 `NCCL_NCHANNELS=2`、`NCCL_P2P_LEVEL=NVL` 关掉跨 NUMA P2P。
- **IB SHARP 集群**：`NCCL_ALGO=CollNet` + `NCCL_COLLNET_ENABLE=1`，但要先确认 SHARP daemon (`sharpd`) 起来、`sharp_smx_ucx_check` 通过。
- 改 `NCHANNELS` 会跟计算 kernel 抢 SM，长期监控 SM 利用率，否则可能 collective 快了但训练 step 反而变慢。

### 5. 常见追问 / 易错点

- `NCCL_DEBUG=INFO` 启动时打印的 `Channel 00 : 0 1 2 3 ...` 就是当前算法选择与 ring 拓扑，调参后必看是否真的换了算法。
- 别同时改三个变量，每次只动一个，配 `nccl-tests` 比对 algbw/busbw 才能定位收益来源。
- `NCCL_BUFFSIZE` 太大会显著增加显存与 page-locked host memory，多机训练时会先看到 OOM 而不是性能问题。
- 不同 NCCL 版本（2.18 / 2.22 / 2.27）默认 tuning table 差很多，复现别人的"环境变量魔法"前先确认 NCCL 版本号。
- `NCCL_ALGO=NVLS` 在非 NVL8/NVL72 拓扑上会 fallback 到 ring，不会报错但也不会更快。

### 6. 实践建议

调参流程建议固定为：先跑 `all_reduce_perf` 量出当前默认带宽，再用 `NCCL_DEBUG=INFO` 确认现在走的算法/协议；只有在 default 显著低于 nominal（例如 NVLink 4.0 实测 < 200 GB/s busbw）时才动手调。生产集群最好把 `NCCL_ALGO` / `NCCL_PROTO` 写进 launch script 而不是 ad-hoc export，避免不同 rank 拿到不同环境变量导致 collective hang。NCCL 2.22+ 起 H100 集群默认启用 NVLS，多数情况下不需要再手动指定算法；真的要手动调时优先调 `NCCL_NCHANNELS` 和 `NCCL_BUFFSIZE`，因为它们对显存和 SM 占用的影响最直接、可解释。

### 7. 30 秒速答

- 一句话核心：ALGO 选 ring/tree/NVLS，NCHANNELS 调并行通道数，BUFFSIZE 调单次通信缓冲
- 关键机制：小消息 tree 延迟低，大消息 ring/NVLS 带宽高；NCHANNELS 越多并发越高但占资源
- 易踩坑 / 关键权衡：盲目把 NCHANNELS 拉满会争抢 SM，反而变慢
- 面试加分关键词：NCCL_ALGO / NCCL_NCHANNELS / NCCL_BUFFSIZE / NVLS

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 NCCL_ALGO / NCHANNELS / BUFFSIZE？
- [ ] 你能不能解释 小消息 tree 延迟低，大消息 ring/NVLS 带宽高；NCHANNELS 越多并发越高但占资源？
- [ ] 你能不能举一个 H100 NVL 集群 NCCL 2.20+ 默认 NVLS，大消息加速 2× 的具体场景？
- [ ] 你能不能说出 跨机大消息选 tree，实际带宽掉到一半 这种常见错误模式？

## Q47. IB SHARP（in-network reduction）实战怎么用？命中条件是什么？

> 🔴 专家 · SHARP 让 InfiniBand 交换机直接帮你做 reduce，理论上 all-reduce 时间砍半。但命中条件挺挑——交换机型号、subnet manager、message size 都对得上才生效，配错了就退回 Ring，调优时要会查日志判断有没有命中。

### 1. 核心结论

IB SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）是 NVIDIA Quantum InfiniBand switch 内置的「在网计算」能力，让 reduction 算子（all-reduce / reduce / barrier 等）的归约动作直接在 switch ASIC 内完成，而不再把数据拉回每个 rank 的 GPU。命中后 all-reduce 的网络字节数从 `2(N-1)/N × M` 降到约 `M`，跨机大 message all-reduce 实测可降低 30%–50% 通信时间。命中条件相当苛刻：必须是 Quantum/Quantum-2 IB switch + 启用 SHARPv2 license + `sharpd` 启动 + NCCL 2.7+ 编译时 `--with-sharp` + `NCCL_COLLNET_ENABLE=1` + collective 大小落在 SHARP tree 配置内。任何一项不满足都会静默 fallback 到 ring，性能数字不会差但也不会赚到。

### 2. 底层原理

传统 ring all-reduce 在 N rank 时每个 rank 要发送/接收 `2(N-1)/N × M` 字节；其本质是数据走两遍网络（reduce-scatter 一遍 + all-gather 一遍）。SHARP 把这件事改成树形归约：

1. 每个 GPU 把 partial gradient 通过 IB QP 发往 SHARP tree 的叶子；
2. switch ASIC 在硬件 reduction unit 里把多个子树的数据相加；
3. 根节点得到 reduced 结果后沿同一棵树多播回所有叶子。

这样每个 rank 实际只发出 `M` 字节、收到 `M` 字节，且 reduction 不消耗 GPU 算力或 GPU↔NIC PCIe 带宽。SHARPv2 进一步支持 streaming aggregation（不必整 buffer 到齐才开始 reduction）和更大的 datatype 支持（FP16/BF16/FP32 都可在 switch 内累加）。

### 3. 关键机制 / 流程 / 数据结构

部署前必须确认：

- **硬件**：Mellanox/NVIDIA Quantum (HDR 200G) 或 Quantum-2 (NDR 400G) switch；EDR 100G Switch-IB 2 不支持 SHARPv2，只支持有限的 SHARPv1。
- **License**：SHARP 是付费 feature，`sharp_cmd license_show` 必须返回 active；二手 switch 经常没 license。
- **Subnet manager**：UFM (Unified Fabric Manager) 或开源 OpenSM + SHARP plugin，负责构建 SHARP tree。
- **`sharpd` daemon**：运行在每个 compute node，负责 NCCL ↔ switch 之间的控制平面；用 `systemctl status sharpd` 验证。
- **NCCL build**：发行版自带的 `libnccl.so` 不一定带 SHARP 支持；`strings libnccl.so | grep -i sharp` 应能看到符号；自编需要 `--with-sharp=$SHARP_DIR`。
- **运行环境**：`NCCL_COLLNET_ENABLE=1`、`NCCL_ALGO=CollNet,Ring`（让 NCCL 在 message size 命中时自选 CollNet）、`SHARP_COLL_ENABLE_SAT=1` 启用 streaming aggregation。

启动后用 `NCCL_DEBUG=INFO` 看到 `NCCL INFO Connected CollNet` 与 `using SHARP` 字样才算真的命中。

### 4. 工程权衡 / 性能影响

实测收益强依赖 message size 和 N（rank 数）：

- **N 大、message 中等（10 MB – 100 MB）**：收益最显著，跨机 all-reduce 时间常能降到 ring 的 60%–70%。
- **N 小（< 16 GPU）或 message 极小（< 1 MB）**：SHARP 的固定控制开销使收益接近零；NCCL 默认 tuning table 也会自动 fallback 到 Tree/Ring。
- **message 很大（> 1 GB）**：SHARP 受 switch reduction unit 数量限制，可能不如纯 ring 的 NIC 带宽利用；这种 case 通常需要把 collective 切成多个子 collective 让 SHARP tree 流水。
- 与 NVLS（NVLink SHARP）正交：NVLS 在节点内 NVSwitch 上做 in-network reduction，SHARP 在节点间 IB switch 上做；H100 NVL8 + IB Quantum-2 集群里两者可以同时启用，NCCL 会先走 NVLS reduce-scatter，再走 SHARP all-reduce 跨机。

### 5. 常见追问 / 易错点

- 「我们买了 Quantum switch 一定有 SHARP」——错。SHARP 是 license-gated feature，`sharp_cmd license_show` 不通过就用不了，云厂商租赁 IB 集群时尤其要先确认。
- `sharpd` 起不来通常是 PKey、subnet manager 或 firmware 不匹配，而不是 NCCL 配置问题；先在裸机用 `sharp_be` 跑 `hello world` 验证再回 NCCL 调试。
- 多租户集群里 SHARP tree 资源是有限的（通常每 switch 几十棵 tree），高峰时新作业会拿不到 tree 而静默 fallback。
- ETH/RoCE 集群即使带宽相同也跑不了 SHARP，SHARP 只在原生 IB 上工作；GPUDirect RDMA over RoCE 的 in-network reduction 有 NVIDIA 「Multi-cast Aggregation」但生态尚不成熟。
- 命中后 NCCL profile 中 collective duration 显著下降但 NIC 计数器（`mlx5_perf` 看到的 PCIe TX bytes）也同步下降——这是 SHARP 真正生效的反向证据。

### 6. 实践建议

落地路径建议：先在 2 个节点 16 GPU 上用 `nccl-tests`（`all_reduce_perf -b 8M -e 256M -f 2 -g 8`）量出无 SHARP（`NCCL_COLLNET_ENABLE=0`）vs 有 SHARP（=1）的 busbw 对比，确认 message size 在 16 MB – 128 MB 区间收益 ≥ 20% 再扩到全集群。生产作业里把 `NCCL_COLLNET_ENABLE=1` 写进 launch script，但保持 `NCCL_ALGO=CollNet,Ring,Tree` 三选项，让 NCCL 按 message 自动选；不要硬指定 `NCCL_ALGO=CollNet`，否则小 message collective 会变慢。监控上要把 `sharp_cmd jobs_info` 加到 dashboard，作业拿不到 SHARP tree 时会从 dashboard 看到 `using SHARP=false` 而不必在训练日志里翻。

### 7. 30 秒速答

- 一句话核心：Mellanox switch 支持 SHARP，固件版本和 NCCL 版本对齐，集合通信落到 switch 做 reduce
- 关键机制：SHARP 让 reduce 在 IB switch 上做，跳过 host 一层 reduce，延迟显著降
- 易踩坑 / 关键权衡：需要全机柜 SHARP-capable switch + 对应固件，少一台都不行
- 面试加分关键词：SHARP / NCCL_COLLNET_ENABLE / Mellanox / InfiniBand

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 IB SHARP 的命中条件？
- [ ] 你能不能解释 SHARP 让 reduce 在 IB switch 上做，跳过 host 一层 reduce，延迟显著降？
- [ ] 你能不能举一个 DGX SuperPOD 默认开 SHARP，跨机 AllReduce 延迟降 30% 的具体场景？
- [ ] 你能不能说出 只升 NCCL 不升 switch 固件，SHARP 没生效自己以为开了 这种常见错误模式？

## Q48. Async TP（PyTorch 2.5+）的启用条件、性能收益与陷阱？

> 🔴 专家 · PyTorch 2.5 新加的 Async TP 把 TP 通信和 matmul 计算重叠，理论上能省一大块通信时间。但启用条件很挑、对 NVLink 拓扑也敏感，不满足条件反而比同步版还慢——是个新工具，得看清楚再用。

### 1. 核心结论

Async Tensor Parallel（也叫 SymmetricMemory Async TP）是 PyTorch 2.5+ 引入、由 TorchTitan 在 Llama 3 训练里验证的「把 TP 中 GEMM 与 all-gather/reduce-scatter 重叠到细粒度」的优化。它把传统 `RowwiseParallel` / `ColwiseParallel` 的 sequential pattern（先通信再计算或先计算再通信）改写成 stream 间 pipeline，在 NVLink/NVSwitch 域内可让 TP 通信几乎完全隐藏到 GEMM 时间里。启用条件：PyTorch 2.5+、TP rank 全在同一个 NVLink 域（NVL8/NVL72）、用 `parallelize_module(... PrepareModuleInput/PrepareModuleOutput + SequenceParallel)` 装好后再调 `torch.distributed._symmetric_memory.enable_symm_mem_for_group(tp_group)`，并在 `torch.compile` 下让 inductor 自动 codegen async-collective 形态。收益典型在 H100 NVL8 上 TP 部分通信时间降低 60%–80%，整步训练 step 提速 5%–15%；陷阱主要是跨节点 TP 不能用、动态形状会触发 recompile、与 PP/CP 组合时 mesh 必须对齐。

### 2. 底层原理

传统 TP 的 `f`/`g` 算子对（Megatron 范式）有两个同步点：column parallel 后向走 all-reduce（或 sequence parallel 时走 reduce-scatter）、row parallel 前向走 all-reduce（或 SP 时走 all-gather）。这两个 collective 必须等 GEMM 全部算完才能开始，否则没数据可发；同样 GEMM 必须等 collective 完成才能拿到完整 input。Async TP 做的事是：

1. 把一个大 GEMM 在 K 维度（输入维）切成多个 chunk；
2. 当第一个 chunk 算完后，立刻把它的 partial sum 通过 SymmetricMemory（NVLink 共享内存原语）发给同 group 的其他 rank，同时启动下一个 chunk 的 GEMM；
3. 远端 rank 收到 partial sum 后做 in-place 累加；
4. 全部 chunk 处理完后整个 all-reduce/reduce-scatter 已经完成，无需独立 collective。

效果上是把「N 步 GEMM + 1 次 collective」改成「N 步 (GEMM + partial collective)」，且 partial collective 走的是 NVLink 直读直写而非 NCCL kernel 队列，启动开销几乎为零。

### 3. 关键机制 / 流程 / 数据结构

启用流程（PyTorch 2.5+，参考 `torch.distributed._symmetric_memory` 与 TorchTitan）：

```python
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]
parallelize_module(model, tp_mesh, {
    "attn.wq": ColwiseParallel(),
    "attn.wo": RowwiseParallel(),
    ...
})
enable_symm_mem_for_group(tp_mesh.get_group().group_name)
torch._inductor.config.symmetric_memory.enable_async_tp = True
model = torch.compile(model)
```

inductor 在 fusion pass 里识别 `all_gather + matmul` / `matmul + reduce_scatter` 模式并 codegen 成 async 实现。运行时会在 NVLink 域内分配 SymmetricMemory（每个 rank 把同一段 VA 映射到对端 GPU），collective 退化为对 SymmetricMemory 的 `put`/`get` + barrier。

### 4. 工程权衡 / 性能影响

收益曲线：

- **NVLink 域内 TP（H100 NVL8、TP=8）**：典型收益 5%–15% step time。Llama 3 训练里 TP 通信占比从 ~25% 降到 ~5%。
- **跨节点 TP（TP > NVLink 域大小）**：收益接近零甚至负值，因为 SymmetricMemory 不支持 IB，inductor 会 fallback 到 NCCL；这种场景下不要开 async TP，让 NCCL 走标准 ring。
- **小 TP（TP=2/4）**：收益边际化，async overhead 和原 collective 时间相当。
- **与 SP 组合**：必须开 SP 才有 async TP（async TP 本质就是把 SP 的 all-gather/reduce-scatter 异步化）；纯 TP 无 SP 时没有 collective 可异步。
- **与 FSDP2 组合**：FSDP2 的 reshard collective 不在 async TP 范围内，FSDP2 自己有 forward prefetch 机制；二者正交叠加。

### 5. 常见追问 / 易错点

- `enable_symm_mem_for_group` 没调或调晚了（在 `parallelize_module` 之前），inductor 找不到 SymmetricMemory handle，pass 静默不触发。可以打 `TORCH_LOGS="+inductor"` 看是否 codegen 了 `_symm_mem.async_all_gather`。
- 必须 `torch.compile`，eager mode 下 async TP 完全不工作（没有 inductor pass）。
- 输入形状每步变化（动态 batch / seqlen）会触发 recompile，async TP 收益被编译时间吃掉；用 `torch._dynamo.config.cache_size_limit` 调大或固定 shape。
- mesh 名字必须与 `enable_symm_mem_for_group` 传入的 group_name 一致；不一致时 collective 仍会执行但走 NCCL fallback，性能没收益但代码不报错。
- async TP 与 CUDA Graph 不兼容（截至 PyTorch 2.6），用 `torch.compile(mode="reduce-overhead")` 时要关掉 async TP 否则崩。
- 多机训练里 mesh 必须把 TP 维度限制在节点内：`mesh_shape=(num_nodes, tp_per_node)` 而不是 `(global_tp,)`。

### 6. 实践建议

落地建议：先在单节点 8 GPU 跑 baseline TP=8 + SP，量出 TP 通信占比；只有 ≥ 15% 时才值得开 async TP。开启后用 Nsight Systems 看 timeline 应该能看到 GEMM kernel 与 SymmetricMemory put/get 完全交错；如果还是看到一段独立的 NCCL all-reduce/reduce-scatter，说明 inductor pass 没命中。生产配置稳健做法：用 `mesh_dim_names=("dp","pp","tp")`，把 TP 维度限制在 NVLink 域内（H100 NVL8 上 TP ≤ 8，GB200 NVL72 上 TP ≤ 72），跨节点用 PP/DP/CP 而不是 TP。Llama 3 405B 参考配置是 TP=8/PP=16/CP=2/DP=64，TP=8 全在节点内正好用上 async TP；这是 2024–2026 业界主流做法。

### 7. 30 秒速答

- 一句话核心：PyTorch 2.5+ 的 micro-pipelined TP，把 AllGather/ReduceScatter 与 GEMM 重叠
- 关键机制：DTensor + SymmetricMemory + Triton kernel 实现 fused comm-compute
- 易踩坑 / 关键权衡：只在 H100 + NVLink/NVSwitch + 特定 shape 下生效，调试门槛高
- 面试加分关键词：async TP / DTensor / SymmetricMemory / micro-pipelined

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Async TP？
- [ ] 你能不能解释 DTensor + SymmetricMemory + Triton kernel 实现 fused comm-compute？
- [ ] 你能不能举一个 TorchTitan 在 H100 上开 async TP，70B MFU +5% 的具体场景？
- [ ] 你能不能说出 A100 上盲试 async TP 没收益反而踩 bug 这种常见错误模式？

## Q49. DDP / FSDP1 / FSDP2 怎么写公平的吞吐 benchmark？

> 🔴 专家 · "FSDP 比 DDP 慢 30%" 这种话十有八九是 benchmark 写得不公平——batch、warmup、sync 时机、显存压力都得对齐才能比。基线立不住，后面所有优化决策都是空中楼阁。

### 1. 核心结论

公平 benchmark 的核心是「同模型、同 global batch、同精度、同 NCCL 版本、warmup 后稳态测量、并隔离掉 dataloader/checkpoint/logging 等非训练开销」。常见错误是直接拿 step time 比，但 DDP/FSDP1/FSDP2 默认配置在显存占用、micro-batch 上限、reshard 时机等多个维度都不同，必须先固定独立变量。推荐方法：固定 global tokens/step、micro-batch 选各方案各自能稳跑的最大值（让 MFU 接近上限）、用 `torch.profiler` 量出 forward/backward/optimizer/comm 四段时间、跑 ≥ 100 步丢弃前 20 步、报 `tokens/sec/GPU`、`MFU`、`peak memory` 三件套；任何只报 step time 的对比都是耍流氓。

### 2. 底层原理

三种 DP 在内存与通信模式上有本质差异：

- **DDP**：每 rank 完整模型副本，反向 all-reduce 梯度，optimizer 本地执行。通信量 `2(N-1)/N × |params|` 每步，显存 = `(params + grads + optimizer_state)`，单卡装不下时直接 OOM。
- **FSDP1**：参数/梯度/优化器状态全分片；前向 all-gather 参数、计算、reshard、反向再 all-gather 参数 + reduce-scatter 梯度。通信量比 DDP 翻倍（all-gather + reduce-scatter ≈ 2× all-reduce），但显存能跑 N 倍大的模型。FSDP1 用 `FlatParameter` 把多个 sub-param 打包成 1D buffer，wrap policy 决定打包粒度。
- **FSDP2**：与 FSDP1 内存模型相同，但用 per-parameter `DTensor` 分片替代 `FlatParameter`，不再需要 wrap policy（手动 `fully_shard` 每个 module），mixed precision/offload 通过 `MixedPrecisionPolicy`/`OffloadPolicy` 在 module 级声明。通信量与 FSDP1 相同，但 reshard 时机更细（per-module 而非 per-flat-buffer），与 TP/PP 组合更顺。

吞吐对比的「公平」必须落在：相同 global batch（同样的优化轨迹）、相同精度（BF16 / FP8 设置一致）、相同序列长度、相同模型尺寸、相同 dataloader（最好用 dummy data 排除 IO 干扰）。

### 3. 关键机制 / 流程 / 数据结构

公平 benchmark 的标准流程：

1. **固定环境**：同一 PyTorch 版本（FSDP2 要 ≥ 2.4）、同一 CUDA/NCCL/驱动；`NCCL_DEBUG=WARN` 记录配置；`TORCH_NCCL_BLOCKING_WAIT=1` 让 hang 立刻爆而不是无声拖慢。
2. **固定数据**：用 `TensorDataset` 装一段固定 dummy tensor（`torch.randn(2048, hidden)`），每步 `next(iter)`；这样 dataloader 不会成为变量。
3. **固定 global batch**：例如 global_bs=512、seq=4096，那么 DDP 在 8 卡上每卡 micro-bs=64；FSDP 在 8 卡上由于显存省可能 micro-bs=128 + grad accum=4 仍 global=512。**如果不固定 global batch，loss 轨迹不一样，结果不可比**。
4. **量 micro-batch 上限**：每个方案分别在 OOM 边界往下退一档（例如发现 mb=128 OOM、mb=64 通过，记 mb=64）；这是各方案的「装得下的最大 micro-batch」。
5. **稳态测量**：跑 100 步，丢弃前 20 步（warmup + autotune），用 `torch.cuda.synchronize()` 后 `time.perf_counter()` 量后 80 步。
6. **多维度报告**：`tokens/sec/GPU`、`MFU` (`actual_FLOPs / peak_FLOPs`)、`peak memory` (`torch.cuda.max_memory_allocated()`)、`comm time %`（`torch.profiler` 里 `nccl:` kernel 时间占比）。

参考工具链：TorchTitan 仓库的 `benchmark.py` 已经把这套做成了 reference；也可以基于 `torch.utils.benchmark.Timer` + `torch.profiler.profile(with_stack=False)` 自己搭。

### 4. 工程权衡 / 性能影响

实测期望大致：

- **小模型（≤ 7B）单节点 8 卡**：DDP 通常最快，因为通信少且 NVLink 带宽冗余；FSDP1/FSDP2 step 慢 5%–20%。
- **大模型（13B+）单节点**：DDP 装不下，必须 FSDP；FSDP2 比 FSDP1 慢 0%–5%（per-module reshard 引入更多 collective 但单个更小），但配置更灵活，与 TP/PP 组合时优势明显。
- **多节点（IB 互联）**：FSDP 的 all-gather 跨节点容易成瓶颈，ZeRO-2（`SHARD_GRAD_OP` / `reshard_after_forward=False`）作为中间档常比 FULL_SHARD 快 5%–15%（少一次反向 all-gather），代价是显存多一份参数。
- **HYBRID_SHARD / FSDP2 `reshard_after_forward=int`**：节点内分片、节点间复制，在跨节点带宽不足时是事实上的最优；TorchTitan 的 Llama 3 baseline 就是这种。

### 5. 常见追问 / 易错点

- 只报 step time 而不报 micro-batch / global batch 的对比，多半是把 FSDP 的「装得下更大 batch」收益偷换成了「step 更快」。
- 没排除 logging（`print` / `wandb.log` 每步同步）和 checkpoint 异步落盘，会让某些方案被 IO 拖慢。
- 没 warmup 直接量第一步，cuDNN/inductor autotune 还在选 kernel，数字偏离稳态 30% 都正常。
- DDP `find_unused_parameters=True`、FSDP1 `use_orig_params=False` 会显著影响吞吐，benchmark 时要关掉「兼容性开关」走最优路径。
- FSDP1 `FULL_SHARD` 与 FSDP2 `reshard_after_forward=True` 在 wrap 粒度不同（FSDP1 默认按 transformer block，FSDP2 默认 per-param）时，单步 collective 数量差几十倍，需要对齐。
- 报 `MFU` 时分母是 dense 模型的 6N（forward + backward）还是 6N + 4N（含激活重计算的重算）必须写清楚，否则跨论文不可比。

### 6. 实践建议

实操建议直接 fork TorchTitan 的 `benchmark.py` 改成自己的模型，跑 DDP / FSDP1 (`FULL_SHARD`) / FSDP1 (`SHARD_GRAD_OP`) / FSDP2 (`reshard_after_forward=True`) / FSDP2 (`reshard_after_forward=False`) 五个变体，每个跑 3 次取中位数。报告时画一张「micro-batch vs tokens/sec/GPU」的曲线，比单一数字更能说明各方案的扩展性。生产决策上：单节点跑得下用 DDP，跨节点用 FSDP2 + HYBRID_SHARD（节点内 shard、节点间 replica），与 TP/PP/CP 组合的多维并行用 FSDP2 + `DeviceMesh`。FSDP1 在 2026 已基本进入维护模式，新项目直接上 FSDP2。

### 7. 30 秒速答

- 一句话核心：统一 global batch、tokens/s 口径、warmup 步数和 seq len，排除编译时间
- 关键机制：第一步包含 NCCL init + 编译开销，跳过；用 tokens_per_sec_per_gpu 做公平对比
- 易踩坑 / 关键权衡：比较时 fwd-bwd 用同一份代码，开关只切 backend
- 面试加分关键词：tokens/s/GPU / MFU / warmup / fair benchmark

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 公平的吞吐 benchmark？
- [ ] 你能不能解释 第一步包含 NCCL init + 编译开销，跳过；用 tokens_per_sec_per_gpu 做公平对比？
- [ ] 你能不能举一个 DDP vs FSDP1 vs FSDP2 同一 batch + 同一 model 测 1000 step 的具体场景？
- [ ] 你能不能说出 FSDP 比 DDP 慢但没扣 grad_accum 不一致，结论错 这种常见错误模式？

## Q50. Pipeline parallel 怎么调试？bubble、激活下沉、micro-batch 切分？

> 🔴 专家 · PP 调试比 DP 难一截——同一个 tensor 在不同 stage 上、bubble 还藏在 timeline 里。看 profile 时要会区分"真等待"和"激活下沉"，调 micro-batch 切分时也要逐项验证，不然问题永远定位不到。

### 1. 核心结论

PP 调试的三大常见问题是：bubble（流水线空闲期）超出预期、激活在 stage 间「下沉」（前面 stage 的激活越积越多直到 OOM）、micro-batch 切分不当导致 stage 之间负载不均。定位顺序固定：先用 `torch.distributed.pipelining` 自带的 timeline export 或 Nsight 看每 stage 的实际 idle 时间，对照理论 bubble 公式 `(p-1)/m`（GPipe）或 `(p-1)/(v×m)`（interleaved 1F1B）；再看每个 stage 的 `peak memory`，前 stage 显著高于后 stage 就是激活下沉；最后看 forward/backward 时间是否在各 stage 接近，差 ≥ 20% 就是切分不均。修复手段：增大 m、改 1F1B/Interleaved/ZB-H1 schedule、调整 layer 分配、对 embedding/lm_head 做半层预算。

### 2. 底层原理

PP 把模型按 layer 切到 `p` 个 stage，每 micro-batch 在 stage 间传递激活/梯度。总训练时间 = `forward_per_micro × m × p / p + bubble + backward_per_micro × m × p / p`，bubble 来自 stage 数量大于 micro-batch 数量时，最早 stage 必须等最后 stage 反向回来才能开始下一个 batch。具体：

- **GPipe**：所有 forward 跑完才开始 backward，bubble = `(p-1)/m`，激活峰值在第一个 stage（要存 m 份激活直到 backward 回来）。
- **1F1B（PipeDream-Flush / Megatron 默认）**：交替执行 forward/backward，stage 内 in-flight 激活数被限制在 ≤ p 份，激活峰值显著降低。bubble 与 GPipe 相同 `(p-1)/m`。
- **Interleaved 1F1B**：每个 stage 持有 `v` 个 chunk（virtual stage），bubble 降到 `(p-1)/(v×m)`，但要求 `m % p == 0` 和 `num_layers % (p×v) == 0`。
- **Zero Bubble (ZB-H1/H2/V)**：把 backward 拆成 B（梯度对激活的）和 W（梯度对权重的），通过更细 schedule 把 W 推迟到 bubble 区间，bubble 接近 0。

激活下沉：前 stage 必须保留激活直到反向回来，1F1B 下前 stage 持有 `p` 份激活，后 stage 只持有 `1` 份，因此前 stage 显存压力远大于后 stage。这是 PP 与 TP/DP 的关键差异点。

### 3. 关键机制 / 流程 / 数据结构

调试工具与流程（PyTorch 2.4+ `torch.distributed.pipelining`）：

1. **Schedule 选择**：`ScheduleGPipe`（教学/小规模）、`Schedule1F1B`（生产默认）、`ScheduleInterleaved1F1B`（大 p、显存吃紧）、实验性 `ScheduleZBVZeroBubble`。
2. **切分**：用 `pipeline()` 配合 `SplitPoint.BEGINNING` / `END` 在指定 module 边界切；或手动 `pipe.split_module(...)`。Megatron 风格手动指定 `num_layers_per_stage` 数组。
3. **Bubble 监控**：每个 stage 在 `Schedule.step()` 后能拿到 per-rank 的 forward/backward 时间，rank0 汇总后画甘特图。Megatron-Core 的 `--log-pipeline-time` flag 直接输出 stage idle %。
4. **激活下沉监控**：`torch.cuda.max_memory_allocated()` 在每个 stage 单独打点，stage_0 应明显 > stage_p-1；如果差距过大（≥ 3×）说明 schedule 不是 1F1B 或 m 设小了。
5. **切分均衡监控**：跑 5 步 warmup 后量每 stage 的 `forward_time` / `backward_time`，最大值 / 最小值 > 1.2 就要重切。常见不均衡来源：embedding 层在 stage_0 太重、lm_head + loss 在 stage_p-1 太重、layer 数量不能被 `p` 整除时尾 stage 多扛一层。

### 4. 工程权衡 / 性能影响

micro-batch m 的选择是 PP 调优最关键的旋钮：

- **m 太小（< p）**：bubble 占比 ≥ `(p-1)/p`，吞吐崩盘；硬下限 `m ≥ p`，经验值 `m ≈ 4p` 起步。
- **m 太大**：global batch 跟着大，loss 收敛轨迹改变；激活总量 `O(m × per_micro_activation)`，前 stage 容易 OOM；不能无限加。
- **激活下沉对策**：开 selective activation recomputation（只重算 attention 中间张量）+ sequence parallel 把激活按 seq 维分片，能把前 stage 显存压到与后 stage 接近。
- **不均衡对策**：embedding 层独占 stage_0 时，给 stage_0 少分一层 transformer block；tied embedding（embedding 与 lm_head 共享权重）时还要处理跨 stage 同步开销，Megatron 经验是按半层预算计入。

### 5. 常见追问 / 易错点

- 「bubble 算出来很小但实测吞吐还是低」：通常是 `send_forward`/`recv_backward` 的 P2P 延迟没算进 bubble；高延迟网络（IB > 5μs）下 stage 多时 P2P 成本被低估。
- 「stage_0 OOM stage_p 显存空着一半」：典型激活下沉，必须开 1F1B 或 ZB schedule，GPipe 模式必然这样。
- 用 `torch.distributed.pipeline.sync.Pipe`（legacy）还是 `torch.distributed.pipelining`（PiPPy 合并版）？2024+ 一律用后者，前者已 deprecated。
- micro-batch 的「最后一个不完整」处理：`m × micro_bs` 必须等于 `global_bs`，不允许尾部凑数；否则 schedule 对齐错乱。
- Interleaved 下 `num_layers % (p × v)` 不整除时手动 padding 一个 identity layer 比改并行配置容易。
- ZB schedule 在 PyTorch 2.6 仍是实验性，生产建议默认 1F1B 或 Interleaved 1F1B。

### 6. 实践建议

调试模板：先把 `m = 4p` 跑通，量 stage 时间分布与显存峰值；如果显存均衡且 bubble < 10%，就是合格基线。bubble 偏大时优先加 m 或换 Interleaved 1F1B；显存不均衡时优先做 layer 重新切分，再考虑 SP + selective recompute；切分不均时手动调 `num_layers_per_stage` 并重测。生产建议把每 stage 的 forward/backward 时间和 peak memory 写入 metric 系统，长跑时自动画甘特图——PP 问题的诊断「肉眼看 timeline」远比看数字快。Llama 3 405B / DeepSeek-V3 671B 都用 1F1B 或 DualPipe schedule + selective recompute + SP 的组合，是 2024–2026 大模型 PP 的事实模板。

### 7. 30 秒速答

- 一句话核心：bubble、激活下沉时机、micro-batch 切分粒度三方面排查
- 关键机制：bubble 由 schedule 决定，激活下沉看是否 prefetch，micro 切分影响 GEMM 效率
- 易踩坑 / 关键权衡：micro-batch=1 时 stage 内 GEMM 跑不起算力
- 面试加分关键词：bubble / activation / micro-batch / 1F1B / interleaved

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 PP 调试三件套？
- [ ] 你能不能解释 bubble 由 schedule 决定，激活下沉看是否 prefetch，micro 切分影响 GEMM 效率？
- [ ] 你能不能举一个 P=8 训 70B 测发现 stage 0 慢 30%，定位到 embedding 重 的具体场景？
- [ ] 你能不能说出 只看吞吐不看每 stage timeline，漏掉 straggler stage 这种常见错误模式？

## Q51. ZeRO-3 / FSDP 跨节点扩展瓶颈怎么定位？参数 all-gather 还是优化器状态？

> 🔴 专家 · ZeRO-3 单机内跑得飞快，扩到多机吞吐就掉一半——是参数 all-gather 卡住了还是优化器同步堵了？分清楚不同 collective 的耗时占比，是把跨节点训练救回来的第一步。

### 1. 核心结论

ZeRO-3（≈ FSDP `FULL_SHARD`）在跨节点扩展时常见两类瓶颈：**前向/反向的参数 all-gather** 和 **反向的梯度 reduce-scatter**。优化器状态本身只在 optimizer step 时本地更新（不通信），不是瓶颈来源。判断哪个 collective 是瓶颈的标准方法：用 `torch.profiler` 看 NCCL kernel 时间分布，all-gather 占比 ≥ 30% 就是参数同步瓶颈、reduce-scatter 占比 ≥ 20% 就是梯度同步瓶颈；前者通常用 forward prefetch + HYBRID_SHARD（节点内 shard 节点间 replica）解决，后者用更大 micro-batch 摊薄通信 + reduce-scatter overlap。容易误诊的是把 optimizer step 的本地 fp32 master weight 更新（耗时但无通信）当成扩展瓶颈。

### 2. 底层原理

ZeRO-3 单步通信图谱（每个 transformer block）：

1. **前向**：`all-gather(params)` → `forward_compute` → `reshard(params)`（释放显存）
2. **反向**：`all-gather(params)` → `backward_compute` → `reshard(params)` → `reduce-scatter(grads)`
3. **Optimizer step**：本地用 fp32 master weights + sharded grads + sharded optimizer state 更新 fp32 master，再 cast 回 bf16 sharded params。**全程无通信**。

每步通信总量 ≈ `3 × |params|`（两次 all-gather + 一次 reduce-scatter），是 DDP 的 1.5×。跨节点带宽（典型 IB 200 G ≈ 25 GB/s 单向）远低于 NVLink（H100 NVL8 单向 450 GB/s），跨节点 collective 是吞吐瓶颈的主因。

优化器状态分片节省的是显存而非通信：fp32 master + Adam m/v 占 `12 |params|` 字节，每 rank 只持有 `1/N` 份；这部分计算在 optimizer step 时纯本地，不参与 all-gather/reduce-scatter。

### 3. 关键机制 / 流程 / 数据结构

定位流程：

1. **量 step time 分解**：`torch.profiler.profile(activities=[CPU, CUDA])` 跑 5 步，导出 `chrome trace` 看每步的 forward / backward / optimizer 三段时间。理想状态下 optimizer < 5%。
2. **量 collective 时间**：`torch.profiler` 的 distributed view 直接列出 `nccl:all_gather`、`nccl:reduce_scatter` 的总时长。占 step 时间 ≥ 30% 即为通信瓶颈。
3. **量 collective 与计算 overlap**：用 Nsight Systems 看 NCCL kernel 与 GEMM kernel 是否在不同 stream 并行；如果 NCCL kernel 独占整段时间且无 GEMM 并行，说明 prefetch 没生效。
4. **判断 all-gather vs reduce-scatter**：前者发生两次（forward + backward），后者一次（backward 末），通常 all-gather 总时间 ≥ reduce-scatter 总时间；但若 backward 的 all-gather 与 backward compute overlap 良好，瓶颈会落到 reduce-scatter。
5. **判断扩展性**：跑 N=8/16/32/64 节点的 weak scaling（每节点固定 micro-batch），算 `tokens/sec/GPU` 随 N 的衰减；衰减 > 20% 时确认是跨节点通信瓶颈。

FSDP 关键调优 API：`forward_prefetch=True`（FSDP1）/ `set_modules_to_forward_prefetch` (FSDP2)、`backward_prefetch=BackwardPrefetch.BACKWARD_PRE`、`limit_all_gathers=True`（FSDP1，控制并发 collective 数避免显存爆）。

### 4. 工程权衡 / 性能影响

修复手段优先级：

- **HYBRID_SHARD / FSDP2 `reshard_after_forward=int`**：节点内 shard、节点间 replica，all-gather 退化为 NVLink 域内（450 GB/s），跨节点只剩 reduce-scatter。代价是显存翻 N_node 倍，但通常仍能装下。这是跨节点 ZeRO-3 的首选优化。
- **forward/backward prefetch**：让下一 block 的 all-gather 与当前 block 的 compute overlap；理想情况下 all-gather 时间几乎全被隐藏。FSDP2 默认开启，FSDP1 需手动 `forward_prefetch=True`。
- **更大 micro-batch**：每步通信量固定但计算量随 batch 线性增长，通信占比下降。代价是激活显存增加，需要配 SP / 激活重计算。
- **梯度累积**：`global_bs = micro_bs × N × grad_accum`，`grad_accum > 1` 时只有最后一次 backward 触发 reduce-scatter，前几次梯度本地累加；通信量降为 `1/grad_accum`。FSDP2 用 `model.no_sync()` context 实现。
- **FP8 通信（NCCL 2.22+）**：reduce-scatter 用 FP8 半带宽，DeepSeek-V3 实测可降 30%–40% 反向通信时间。配 stochastic rounding 避免数值偏差。

### 5. 常见追问 / 易错点

- 「optimizer step 占 30%，是不是要分片优化器状态？」——不是，优化器状态已经分片了，慢是因为 fp32 master weight 计算量；用 fused Adam（`apex.optimizers.FusedAdam` / `torch.optim.Adam(fused=True)`）能砍 60% optimizer 时间。
- 「跨节点慢是 IB 带宽不够吧？」——先看 NCCL busbw 是否接近 nominal（IB 200G 应能跑到 22–24 GB/s），跑不到的话先调 NCCL（Q46）；跑到了说明确实是物理瓶颈。
- 「reduce-scatter 不是只发一次吗，为什么慢？」——它是 `1× |params|` 一次性发，message 大但 NCCL ring 长（跨 N 节点），latency 与 bandwidth 都吃满；可以用 `NCCL_NCHANNELS` 调多 channel 加并发。
- prefetch 太激进会把 N 个 block 的 all-gather 全启动，显存爆掉；FSDP1 用 `limit_all_gathers=True` 节流，FSDP2 走自带 collective stream 不需要手动节流。
- 「ZeRO-3 比 ZeRO-2 慢」是常态，因为多了反向 all-gather；显存够用就降到 ZeRO-2（`SHARD_GRAD_OP` / `reshard_after_forward=False`）。

### 6. 实践建议

定位流程模板：跑 weak scaling 看衰减、`torch.profiler` 看通信占比、Nsight 看 overlap。修复优先级固定为：HYBRID_SHARD → 加大 micro-batch → 梯度累积 → FP8 通信。生产配置上，跨节点 ZeRO-3 几乎不应该用纯 FULL_SHARD；TorchTitan、Megatron-Core 的 Llama 3 / DeepSeek-V3 reference 都是「节点内 FSDP2 + 节点间 DDP」的混合形态（即 HYBRID_SHARD），这是 2024–2026 跨节点 ZeRO-3 的事实最优。如果显存仍不够再叠 PP 把模型按 layer 切到多个节点，而不是继续撑 ZeRO-3 跨节点广播。

### 7. 30 秒速答

- 一句话核心：先看跨机 all-gather（参数）vs reduce-scatter（梯度）哪个占用 timeline 更多
- 关键机制：参数 all-gather 频繁且与 forward 重叠紧，跨机 IB 不足时直接挂死
- 易踩坑 / 关键权衡：单看总通信量没意义，要看 step-time 拆解里 comm 占比
- 面试加分关键词：all_gather / reduce_scatter / HYBRID_SHARD / cross-node

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 ZeRO-3 / FSDP 扩展瓶颈定位？
- [ ] 你能不能解释 参数 all-gather 频繁且与 forward 重叠紧，跨机 IB 不足时直接挂死？
- [ ] 你能不能举一个 64 节点 H100 训 70B，all-gather 占 40% 则切 HYBRID_SHARD 的具体场景？
- [ ] 你能不能说出 盲目升 ZeRO-3 而不切 HYBRID_SHARD 让跨机带宽吃满 这种常见错误模式？

## Q52. ZeRO-Infinity（NVMe / CPU offload）实战调优要点？

> 🔴 专家 · ZeRO-Infinity 把状态卸到 NVMe，听着像万能解药——直到你发现 PCIe 带宽和 SSD IOPS 直接卡死训练吞吐。pinned memory、prefetch 深度、NVMe 配置全都得调到极致，否则速度比纯 GPU 训练慢一个量级。

### 1. 核心结论

ZeRO-Infinity 在 ZeRO-3 之上把优化器状态、梯度乃至参数都可选地卸载到 CPU DRAM 或 NVMe，让单节点也能训练 100B+ 模型；但 NVMe 路径的吞吐受 PCIe 与 NVMe 4K random IOPS 限制，盲开 NVMe offload 通常只换显存不换吞吐，甚至会让训练慢 5–10×。实战调优三条线：① 选对 offload target（CPU 还是 NVMe，并把 buffer 加到 `aio_block_size=1048576` / `aio_thread_count=8`）；② 配 `overlap_comm=True` + `overlap_offload=True` 把 PCIe DMA 隐藏到计算后；③ 对 NVMe 路径必须用 `deepspeed.ops.aio` 的 libaio 后端 + 多块 NVMe 做 RAID-0 把聚合 IOPS 拉到 GPU 计算需要的水位。新项目 2026 优先用 FSDP2 `OffloadPolicy(offload_to_cpu=True)`，NVMe 仅在 100B+ 单节点训练这种特殊场景才考虑。

### 2. 底层原理

ZeRO-Infinity 的内存层级：

- **GPU HBM**（H100 80 GB）：仅保留当前 forward/backward 所需的激活与单 block 参数。
- **CPU DRAM**（典型 1–2 TB）：通过 PCIe 与 GPU 互联，单方向 ~25 GB/s（PCIe 4.0 x16）；用 page-locked memory 做 DMA。
- **NVMe**（Gen4 单盘 ~7 GB/s 顺序、~1M random 4K IOPS）：通过 PCIe 走 libaio 与 GPU 异步 IO，跨 RAID 后聚合 ~25 GB/s 顺序、~5M IOPS。

每步训练里，参数/梯度需要从存储层「上行」到 GPU、计算后再「下行」回存储。关键是这条 PCIe 链路在前向 all-gather、反向 reduce-scatter、optimizer step 时都要走，吞吐瓶颈通常落在 PCIe 而不是 NVMe 本身。`overlap_comm` 让 NCCL collective 与 PCIe DMA 并行，`overlap_offload` 让下一 block 的 swap-in 与当前 block 的 compute 并行；两者都开后理想情况下 PCIe 时间能被隐藏 70%+。

### 3. 关键机制 / 流程 / 数据结构

DeepSpeed 配置关键字段（`ds_config.json` 内 `zero_optimization` + `aio` 两块）：

- `offload_optimizer.device`: `cpu` 或 `nvme`；`cpu` 走 pinned DRAM、`nvme` 走 libaio。
- `offload_optimizer.nvme_path`: 必须是本地 NVMe mount，**不能**是网络存储；建议 `/dev/nvme*` 直接 mount 绕过文件系统层。
- `offload_param`: 同上，但卸载参数本身；通常只在 175B+ 单节点开。
- `overlap_comm: true` + `contiguous_gradients: true`: 让 NCCL collective 与 PCIe DMA 并行，前者节省启动开销。
- `sub_group_size: 1e9`（1B 参数/组）: 流水卸载粒度，过大单组放不下、过小流水开销高。
- `aio.block_size: 1048576`（1 MB）: libaio 单次 IO 大小，是 NVMe 顺序读吞吐最佳点；4 KB 默认值会退化到 random IOPS bound，是常见性能陷阱。
- `aio.queue_depth: 8`: NVMe 队列深度，8–32 是 PCIe 4.0 NVMe 的甜区。
- `pin_memory: true` 必开，否则 PCIe DMA 走 bounce buffer 慢一倍。

FSDP2 对应路径用 `OffloadPolicy(offload_to_cpu=True)` 在 module 级声明，但目前不支持 NVMe（DeepSpeed 仍是 NVMe offload 唯一生产级方案）。

### 4. 工程权衡 / 性能影响

吞吐预估（H100 单卡，70B 模型）：

- **纯 ZeRO-3（无 offload）**：装不下，OOM。
- **CPU offload optimizer**：吞吐降 ~30%（PCIe 上下行 fp32 master 与 grads），但能装下 70B；单节点 8 卡可训。
- **CPU offload optimizer + params**：吞吐降 ~60%；能装下 175B 单节点。
- **NVMe offload optimizer**：单 NVMe 时吞吐降 ~80%（IOPS bound），4 块 NVMe RAID-0 后吞吐降 ~50%；能装下 175B+。
- **NVMe offload params**：基本不可用，每步 swap-in 整个模型，吞吐降 95%+。

实战决策：
- 显存够 → 不开 offload。
- 多节点跨节点带宽够 → 用 ZeRO-3 + HYBRID_SHARD 而不是 offload。
- 单节点 70B–175B → CPU offload optimizer（最优吞吐/显存比）。
- 单节点 200B+ → NVMe offload optimizer + 4×NVMe RAID-0；不开 param offload。
- 推理大模型 → 不要用 ZeRO-Infinity，用 vLLM/TensorRT-LLM 的 KV 缓存 offload 路径。

### 5. 常见追问 / 易错点

- 「我开了 NVMe offload 但训练慢得离谱」——99% 是 `aio.block_size=4096` 默认值跑了 random IOPS path，必须改 1 MB；或者 nvme_path 实际是 NFS/Lustre。
- 「CPU 内存不够装下整模型怎么办？」——必须切到 NVMe；`max_in_cpu` 控制内存中常驻部分。
- `pin_memory` 没开或主机内存被 cgroup 限死时 pin 会失败而退到 bounce buffer。
- 「为什么 CPU offload optimizer 不是免费的？」——optimizer step 在 CPU 上做 fp32 Adam，70B 模型 Adam 单步 CPU 时间约几百毫秒；用 `deepspeed.ops.adam.DeepSpeedCPUAdam`（AVX-512 SIMD）能砍 60%。
- NVMe 写入磨损：训练 100B+ 模型一天 ~10 TB 写入，NVMe TBW 寿命要算清楚；企业级 NVMe（5 DWPD）跑 3 年没问题，消费级（0.3 DWPD）几个月就坏。
- ZeRO-Infinity 与 PP 不兼容（DeepSpeed 内部假设），与 TP 兼容但通信 + offload 双重 PCIe 占用要算清。

### 6. 实践建议

落地流程：先确认是否真的需要 offload（先尝试 ZeRO-3 + HYBRID_SHARD + 激活重计算，能装下就别开）；要开就先用 CPU optimizer offload 跑通基线，再决定是否升级到 NVMe；NVMe 路径必须用专用本地盘 + RAID-0 + 1 MB block size。监控上把 PCIe 利用率（`nvidia-smi dmon -s u`）、NVMe IOPS（`iostat -x 1`）、CPU Adam 时间（DeepSpeed 内置 timer）一起看，三者中最高的那个就是当前瓶颈。2026 实战经验：纯 ZeRO-Infinity 训练 100B+ 模型已被多节点 FSDP2 + HYBRID_SHARD 或 Megatron-LM TP/PP 替代，offload 主要用作「单节点跑研究 baseline」或「资源吃紧但模型必须跑」的兜底路径。

### 7. 30 秒速答

- 一句话核心：保证 NVMe 是企业级、PCIe Gen4+、aio engine 用 libaio，开 overlap_events
- 关键机制：optimizer/param 从 NVMe 经 CPU 上传到 GPU，aio 队列深度决定带宽
- 易踩坑 / 关键权衡：家用 SSD 写放大严重 + IOPS 不够，半天就磨损
- 面试加分关键词：NVMe offload / aio / overlap_events / pin_memory

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 ZeRO-Infinity NVMe 调优？
- [ ] 你能不能解释 optimizer/param 从 NVMe 经 CPU 上传到 GPU，aio 队列深度决定带宽？
- [ ] 你能不能举一个 单机 8×A100 + 4 NVMe 训 175B，aio_block_size=1MB 最佳 的具体场景？
- [ ] 你能不能说出 把 offload path 设到 NFS 而非本地 NVMe，IO 掉到 KB 级 这种常见错误模式？

## Q53. Megatron-LM 启动参数全景：TP / PP / SP / CP / EP / MoE 怎么配？

> 🔴 专家 · Megatron 的启动脚本里几十个 `--xxx-parallel-size` 参数，搞错一个进程组就组不起来。把 TP/PP/SP/CP/EP 怎么乘、各自跨不跨机理顺，是任何想 hands-on 大模型训练的人必过的关。

### 1. 核心结论

Megatron-LM 的启动参数本质是「沿哪些正交维度切模型」的声明：`--tensor-model-parallel-size` (TP)、`--pipeline-model-parallel-size` (PP)、`--sequence-parallel`（开关，不带 size）、`--context-parallel-size` (CP)、`--expert-model-parallel-size` (EP)，外加 DP 是隐式的（`world_size = TP × PP × CP × DP`，EP 与 TP 共享通信组）。配置原则固定：TP 在 NVLink 域内（≤ 8）、PP 跨节点切 transformer block、CP 处理长上下文、EP 处理 MoE expert、剩下用 DP 横向扩。约束乘积必须等于 world_size、`num_layers % PP == 0`、`hidden_size % TP == 0`、`num_attention_heads % TP == 0`、`num_experts % EP == 0`、`seq_len % CP == 0`。Megatron-Core v0.11+ 新加 `--use-distributed-optimizer` 是 ZeRO-1 + Megatron 并行的混合，应该默认开。

### 2. 底层原理

每个并行维度对应不同的通信原语与切分对象：

- **TP**：沿 weight 矩阵的 hidden 维切，前向 column parallel + 后向 row parallel 或反之，单层有两次 all-reduce（开 SP 后变 reduce-scatter + all-gather）。
- **PP**：沿 layer 切，stage 间 P2P send/recv 激活与梯度。
- **SP**：在 TP 内的 LayerNorm / Dropout / residual 上沿 seq 维分片，把 TP 通信里的 all-reduce 拆成 reduce-scatter + all-gather，激活省 TP 倍。
- **CP**：沿 seq 维度切给多卡（Llama 3 风格 Ring Attention），每卡持有 `seq/CP` 长度，attention 通过环形传 K/V。
- **EP**：沿 expert 维度切给多卡，每卡只持有 `num_experts/EP` 个 expert，token 通过 all-to-all dispatch 到对应 expert。
- **DP**：沿 batch 切，反向 all-reduce 梯度（标准 DDP）；`--use-distributed-optimizer` 后变 reduce-scatter + all-gather（ZeRO-1）。

通信组构造：`mpu.initialize_model_parallel(TP, PP, ...)` 在 `world_size` 上构造正交组，rank `r` 同时属于 1 个 TP group、1 个 PP group、1 个 DP group，每个 collective 在对应 group 内执行。

### 3. 关键机制 / 流程 / 数据结构

典型 Llama 3 405B 启动命令骨架：

```bash
torchrun --nnodes=128 --nproc-per-node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 16 \
  --context-parallel-size 2 \
  --num-layers-per-virtual-pipeline-stage 4 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --num-layers 126 --hidden-size 16384 --num-attention-heads 128 \
  --seq-length 8192 --max-position-embeddings 8192 \
  --micro-batch-size 1 --global-batch-size 2048 \
  --bf16 --use-flash-attn \
  --recompute-granularity selective
```

参数解释：
- `TP=8 PP=16 CP=2 DP=4`（128 节点 × 8 = 1024，1024 / (8×16×2) = 4）。
- `--num-layers-per-virtual-pipeline-stage 4`：interleaved 1F1B 的 v=4，bubble = `(p-1)/(v×m)` 进一步降低。
- `--sequence-parallel`：开 SP，TP 通信变 reduce-scatter + all-gather，激活省 8×。
- `--use-distributed-optimizer`：ZeRO-1，optimizer state 沿 DP 分片。
- `--overlap-grad-reduce` / `--overlap-param-gather`：让 DP 通信与计算重叠。
- `--recompute-granularity selective`：只重算 attention 中间张量（FlashAttention 之外的 dropout、softmax 之类），不重算整个 transformer block，省显存且重算开销小。

MoE 专用参数（DeepSeek-V3 风格）：`--num-experts 256 --moe-router-topk 8 --expert-model-parallel-size 64 --moe-token-dispatcher-type alltoall --moe-permute-fusion --moe-grouped-gemm`。其中 `alltoall` 是 EP 核心通信（2025 起新增 `flex` 类型对接 DeepEP），`--moe-grouped-gemm` 把 expert MLP 合并成 batched GEMM 提升 GPU 利用。

### 4. 工程权衡 / 性能影响

各维度的扩展上限与权衡：

- **TP**：≤ NVLink 域大小（H100 NVL8 上 TP=8、GB200 NVL72 上 TP=72），跨域 TP 通信走 IB 极慢。
- **PP**：理论无上限，但 bubble 随 p 增长 `(p-1)/m`；实际 p ≤ 32，再大用 Interleaved 1F1B 把 v 调大。
- **CP**：用于长上下文，CP=2 处理 16K、CP=4 处理 32K；CP 大时 attention 通信占比上升。
- **EP**：MoE 专用，典型 EP=64 处理 256 expert；跨节点 EP 必须用 DeepEP 才能跑出带宽。
- **DP**：横向扩，受 global batch 上限约束（继续加会让 lr 不能再大、收敛变差）。

组合规律：`TP × PP × CP × DP = world_size`、`EP ≤ DP × TP`（EP 与 DP 复用通信组）、`global_batch = micro_batch × DP × num_micro_batches`、`num_micro_batches ≥ PP` 否则 bubble 爆。

### 5. 常见追问 / 易错点

- 「TP=16 跨节点能跑吗？」——能跑但极慢，跨节点 TP all-reduce 单步几百毫秒；必须把 TP 限在 NVLink 域内。
- `num_layers % PP != 0` 会怎样？——Megatron 会报错；要么改 num_layers，要么用 `--num-layers-per-stage` 数组手动分配。
- `sequence-parallel` 必须和 TP > 1 一起开，TP=1 时 SP 没意义且会报错。
- `--use-distributed-optimizer` 与 FSDP 冲突？——不冲突，它是 Megatron 内部的 ZeRO-1，FSDP 是另一种实现路径；Megatron 训练就用 distributed-optimizer，别叠 FSDP。
- MoE 的 `--num-experts` 必须能被 EP 整除；`--moe-router-topk` 通常 1 或 2，DeepSeek-V3 用 8（fine-grained）。
- Megatron-Core v0.11+ 新引入的 `--use-mcore-models` 必须开，否则用的是 legacy module，不兼容新 features（FP8、CP、distributed optimizer 等）。

### 6. 实践建议

配置流程：先估算模型参数量、序列长度、global batch；按 TP=NVLink 域大小 → PP 切跨节点 → CP 处理长 seq → EP 处理 MoE → DP 兜底的顺序确定各维度。再用 Megatron 的 `--profile` 或 `--log-throughput` 跑 100 步看 MFU；MFU < 40% 时回去看 PP bubble、TP 通信、SP 是否开。生产 Llama 3 / DeepSeek-V3 reference 配置可以直接拿来当模板，不要自己从零摸索。Megatron-Core 的 `examples/` 目录有完整 sbatch 脚本，结合 NVIDIA NeMo Megatron 容器（`nvcr.io/nvidia/nemo:24.07`+）能少踩 80% 环境坑。

### 7. 30 秒速答

- 一句话核心：--tensor-model-parallel-size / --pipeline-model-parallel-size / --sequence-parallel / --context-parallel-size / --expert-model-parallel-size
- 关键机制：TP×PP×CP×EP×DP = world_size，每个维度有独立通信组
- 易踩坑 / 关键权衡：EP 和 TP 共用通信组会冲突，需用 --expert-tensor-parallel-size 单独配
- 面试加分关键词：TP / PP / SP / CP / EP / Megatron-Core launch

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 Megatron 启动参数？
- [ ] 你能不能解释 TP×PP×CP×EP×DP = world_size，每个维度有独立通信组？
- [ ] 你能不能举一个 DeepSeek-V3 671B：TP=1 EP=64 PP=16 CP=1 DP=1 的具体场景？
- [ ] 你能不能说出 把 EP 和 TP 都设 8 但底层组没正确隔离导致死锁 这种常见错误模式？

## Q54. all-reduce vs reduce-scatter + all-gather：实测对比与何时选哪个？

> 🟡 进阶 · 数学上等价的两种通信模式，实际跑出来吞吐能差一倍。message size、节点数、是否能和计算 overlap 决定了哪个更快——这道题考的就是你有没有真做过 micro-benchmark，而不是只会背公式。

### 1. 核心结论

恒等式 `all-reduce = reduce-scatter + all-gather` 在数学上等价（都把每 rank `M/N` 的部分和最终广播给所有 rank），但实际上差别巨大：合在一起的 all-reduce 比拆开的 RS+AG 通信量略少（少 `M/N` 字节）、kernel launch 少一次、拓扑可优化（Ring/Tree/NVLS 都直接支持），单调用场景应优先用 all-reduce；拆成 RS+AG 的真正价值在「中间结果有用」——FSDP/ZeRO-3 把参数 RS 后只持有 1/N 份，AG 时机由 prefetch 控制；TP-SP 把 grad 在 LayerNorm 边界 RS 后激活立刻能用到分片形态省显存。所以选型规则：DDP 类纯梯度同步用 all-reduce；分片框架（FSDP/ZeRO/Megatron-SP）用 RS+AG 是为了利用中间分片状态而不是为了"等价替换"。

### 2. 底层原理

NCCL ring all-reduce 的实际实现就是「ring reduce-scatter + ring all-gather」，但合在 1 个 collective 内：

- 第 1 阶段（reduce-scatter）：N 步 ring 步进，每步发 `M/N` 字节，结束后每 rank 持有 `1/N` 份完整 reduced 数据。
- 第 2 阶段（all-gather）：N 步 ring 步进，每步发 `M/N` 字节，结束后每 rank 持有完整 `M` 字节 reduced 数据。
- 总通信量：`2(N-1)/N × M`，与 DDP 教科书式公式一致。

如果用户显式调 RS + AG 两个独立 collective：
- 总通信量仍是 `2(N-1)/N × M`，理论上一样；
- 但有 2 次 NCCL kernel launch（每次 ~10μs CPU + ~5μs GPU 启动开销）；
- 中间需要在 GPU 内存暂存 RS 结果（`M/N` 字节），增加显存压力；
- 但中间状态对调用者可见，可以做有意义的事情（FSDP 释放 unsharded params、SP 把激活按 seq 维分片）。

NVLS（NVLink SHARP）和 IB SHARP 的 in-network reduction 让 all-reduce 进一步变成 `~1.0 × M` 字节（不再是 `2(N-1)/N`），拆开 RS + AG 反而拿不到这个收益——SHARP/NVLS 只在合一的 all-reduce 上启用。

### 3. 关键机制 / 流程 / 数据结构

实测 benchmark 设计（`nccl-tests`）：

```bash
# All-reduce baseline
all_reduce_perf -b 1M -e 1G -f 2 -g 8 -n 50

# RS + AG baseline
reduce_scatter_perf -b 1M -e 1G -f 2 -g 8 -n 50
all_gather_perf -b 1M -e 1G -f 2 -g 8 -n 50
```

实测数字（H100 NVL8，NCCL 2.22+，单节点）：
- 256 MB all-reduce：~1.1 ms（busbw ~410 GB/s）
- 256 MB reduce-scatter + 256 MB all-gather：~0.7 ms + ~0.7 ms = ~1.4 ms（busbw 各 ~370 GB/s）
- 单调用 all-reduce 比 RS+AG 快 ~25%。

跨节点（IB 200G，8 节点 64 GPU）：
- 256 MB all-reduce 启用 SHARP：~3.5 ms（busbw ~70 GB/s 等效）
- 256 MB RS + AG 不启用 SHARP：~6 ms（无在网计算）
- 启用 SHARP 时合一 all-reduce 优势放大到 ~40%。

PyTorch 端：
- DDP 默认走 `all_reduce`（`bucket_cap_mb` 控制 message size）。
- FSDP1/FSDP2 显式 RS（grads）+ AG（params），因为中间分片状态是必需。
- Megatron TP-SP 走 RS（输入梯度）+ AG（输出激活），因为 SP 设计上要求中间状态分片。

### 4. 工程权衡 / 性能影响

选型决策矩阵：

| 场景 | 选择 | 原因 |
|---|---|---|
| DDP 梯度同步 | all-reduce | 单调用更快，中间状态无意义 |
| FSDP forward / ZeRO-3 参数广播 | all-gather（不 reduce） | 不是 reduction，只是广播分片回完整 |
| FSDP backward / ZeRO-3 梯度归约 | reduce-scatter（不 all-gather） | 反向后只需 sharded grads，不需要完整梯度 |
| Megatron TP（无 SP） | all-reduce | 标准 column/row parallel，中间不分片 |
| Megatron TP-SP | RS + AG | 激活在 LayerNorm 边界保持分片，省显存 |
| Async TP | RS + AG（chunk 化） | 切 K 维与 GEMM 重叠，必须可见中间状态 |

注意：FSDP backward 是 `reduce-scatter` 单独一次，不是「先 all-reduce 再 scatter」，因为最终结果就只需要分片形态。许多人误以为 FSDP 反向也要 all-reduce，是把 DDP 模式套到 FSDP 上的常见错。

### 5. 常见追问 / 易错点

- 「能不能把 FSDP 的 RS+AG 合成 all-reduce 加速？」——不能，FSDP 的 AG 在前向、RS 在反向，时间点完全错开；它们不是一对原子操作。
- 「RS+AG 等价 all-reduce 但慢，那 FSDP 是不是天然比 DDP 慢？」——是的，FSDP 通信量是 DDP 的约 1.5×（多一次反向 all-gather）；FSDP 换的是装得下大模型，不是更快。
- 「Megatron 的 TP-SP 比 TP 通信少吗？」——总通信量相同（all-reduce ≈ RS + AG），但 SP 节省的是激活显存（中间状态分片），不是通信。
- NCCL 2.27+ 新增的 `ncclAllReduceGroup` 让多个 all-reduce 合并 launch，对 DDP 多 bucket 场景比手动拆 RS+AG 更优。
- 用户级 `torch.distributed.all_reduce(grad, op=AVG)` 在 NCCL 2.18+ 直接走原生 average，不再需要 `all_reduce(SUM) + div(N)`，少一次 kernel。

### 6. 实践建议

实战决策：永远先问「我需要中间分片状态吗？」。需要（FSDP/ZeRO-3/SP）→ 用 RS+AG；不需要（DDP/普通 TP）→ 用 all-reduce。不要为了「显得高级」拆开。NCCL 调优时，all-reduce 优先开 SHARP/NVLS（Q46/Q47），RS+AG 开 NVLS-RS / NVLS-AG（NCCL 2.22+ 单独支持）。如果发现自己写出了「all_reduce 之后 split 给 N 个 rank」的代码，直接改成 RS；如果写出了「RS 之后立刻 AG」的两步，合成 all-reduce。Pytorch FSDP2 与 Megatron-Core 都已在内部做了正确选型，应用层 99% 不需要自己显式调 collective。

### 7. 30 秒速答

- 一句话核心：参数未分片用 all-reduce 一步到位；分片状态用 RS+AG 才不浪费
- 关键机制：all-reduce = reduce-scatter + all-gather 等价，但 FSDP 已分片时拆开做能省一次通信
- 易踩坑 / 关键权衡：不要在已分片场景仍调 all_reduce，会重复通信
- 面试加分关键词：all_reduce / reduce_scatter / all_gather / FSDP

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 all-reduce vs RS+AG？
- [ ] 你能不能解释 all-reduce = reduce-scatter + all-gather 等价，但 FSDP 已分片时拆开做能省一次通信？
- [ ] 你能不能举一个 FSDP 反向直接 reduce_scatter 写回本地 shard 的具体场景？
- [ ] 你能不能说出 自定义 hook 里在 FSDP 上加 all_reduce 让通信量翻倍 这种常见错误模式？

## Q55. 多 NIC rail / topology-optimized routing 怎么配？Sibling NIC 检测怎么做？

> 🔴 专家 · H100 节点常配 8 张 IB 卡，但默认情况下 NCCL 不一定挑对每张 GPU 的 sibling NIC，跨 rail 通信比 sibling rail 慢好几倍。`NCCL_TOPO_FILE` 配合手工拓扑文件是大集群运维的硬功夫——配错了再贵的网络也白买。

### 1. 核心结论

H100 / GB200 训练机型每节点带 8 张 IB NIC（每张 200G 或 400G），NCCL 默认会探测 PCIe 拓扑把 GPU↔NIC 配对（GPU0↔NIC0、GPU1↔NIC1 ...），这种「rail-aligned」拓扑是跨节点 collective 的关键性能基础——同 rail 跨节点直连，最少跨 leaf switch 跳数。配置三件套：① 用 `NCCL_IB_HCA` 显式指定 HCA 集合或留默认让 NCCL 自动 rail 配对；② 对 RoCE 集群必配 `NCCL_IB_GID_INDEX`（通常 RoCEv2 是 3）+ `NCCL_IB_TC` (流量类) + `NCCL_IB_SL` (服务等级)；③ Sibling NIC 检测靠 `nvidia-smi topo -m` + `ibdev2netdev` 比对 GPU 与 NIC 的 PCIe path，确认每对 sibling 走同一个 PCIe switch。配错的典型症状：跨节点 all-reduce busbw 只能跑到 nominal 的 30%–50%。

### 2. 底层原理

H100 NVL8 节点的 PCIe 拓扑（典型 HGX H100）：
```
CPU0 ── PCIe Switch 0 ── GPU0, GPU1, NIC0, NIC1
CPU0 ── PCIe Switch 1 ── GPU2, GPU3, NIC2, NIC3
CPU1 ── PCIe Switch 2 ── GPU4, GPU5, NIC4, NIC5
CPU1 ── PCIe Switch 3 ── GPU6, GPU7, NIC6, NIC7
```

每张 GPU 与 1 张 NIC 共享 PCIe switch，构成「sibling pair」（rail）。跨节点通信时：
- **Rail-aligned**：node_A 的 GPU0 通过 NIC0 发数据到 node_B 的 GPU0 通过 NIC0；两端都走最近 NIC，PCIe 跳数最少，且通过 IB leaf switch 同 rail 直连。
- **Non-rail-aligned**：node_A 的 GPU0 通过 NIC1（跨 PCIe switch）发到 node_B 的 NIC0；PCIe 多跳一次（GPU0↔CPU↔NIC1，~10μs+），且 IB 路径要经 spine switch。

GPUDirect RDMA（GDR）让 NIC 直接 DMA GPU HBM 而不经过 CPU，但前提是 GPU 和 NIC 在同一 PCIe switch 下；跨 PCIe switch 时 GDR 退化或失效，吞吐降一半。NCCL 2.x 启动时会探测拓扑（`NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` 可导出）并优先选 sibling NIC。

### 3. 关键机制 / 流程 / 数据结构

启动配置模板（H100 + IB）：

```bash
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_IB_GID_INDEX=3              # RoCEv2 集群必配，纯 IB 集群可省
export NCCL_IB_TC=160                   # 流量类，DSCP * 4
export NCCL_IB_SL=0
export NCCL_IB_TIMEOUT=22               # 默认 20，大集群建议 22
export NCCL_IB_RETRY_CNT=7
export NCCL_NET_GDR_LEVEL=PHB           # GDR 启用层级
export NCCL_TOPO_DUMP_FILE=/tmp/topo.xml
export NCCL_DEBUG=INFO                  # 启动时打印拓扑
```

Sibling NIC 检测三步：① `nvidia-smi --query-gpu=pci.bus_id --format=csv` + `ibdev2netdev -v` 列出 GPU 与 NIC 的 PCIe BDF；② `lspci -tv | grep -E 'NVIDIA|Mellanox'` 看 PCIe 树确认 sibling 在同 PCIe switch；③ `nvidia-smi topo -m` 看 GPU↔NIC 矩阵，期望显示 `PIX` 或 `PXB`（同 PCIe switch），显示 `NODE`/`SYS`（跨 NUMA/CPU）说明配错。启动后用 `NCCL_DEBUG=INFO` 看 `Channel 00 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA`，`GDRDMA` 表示 GDR 启用；如果是 `SHM` 或 `Socket` 则 sibling 配对错或 GDR 失效。

### 4. 工程权衡 / 性能影响

实测影响（8 节点 64 GPU，IB Quantum-2 NDR 400G）：

- **完美 rail-aligned + GDR**：256 MB all-reduce ~3.5 ms（接近 nominal 50 GB/s busbw）
- **配错 sibling**：~6.5 ms（busbw ~25 GB/s，跨 PCIe switch + spine switch 双重损失）
- **GDR 没启用**（PCIe ACS 没关或 IOMMU 干扰）：~5 ms（DMA 经 CPU bounce buffer）

调优要点：
- 多 rail 并发：NCCL 默认每 rail 启动一条 ring，8 NIC = 8 ring 并行，聚合带宽 ~1.6 TB/s 节点级。
- Adaptive routing：IB switch 启用 AR（Quantum-2 默认开），跨 spine 时动态选最空 lane；老 EDR 集群可能需手动 `NCCL_IB_AR_THRESHOLD`。
- ACS（PCIe Access Control Services）必须关，否则 GDR 走 bounce 路径；`setpci -s <BDF> ECAP_ACS+0x6.w=0x0`。
- IOMMU passthrough（`iommu=pt`）必须开，否则 GDR 性能受 IOMMU 翻译影响。

### 5. 常见追问 / 易错点

- 「我有 8 张 NIC 但 NCCL 只用 1 张」——通常 `NCCL_IB_HCA` 只设了一个，或 `mlx5_X:1` 的 port 号写错（应该是 `:1` 不是 `:0`）；用 `NCCL_DEBUG=INFO` 看 `Using X NIC(s)` 验证。
- RoCE 集群跑得比 IB 慢一半，多半是 `NCCL_IB_GID_INDEX` 没设（默认 0 是 RoCEv1，应改成 3 或 5 走 RoCEv2）。
- `nvidia-smi topo -m` 显示 `NODE`（跨 NUMA）的 sibling 配对说明硬件设计或 BIOS 配置有问题，软件层面无法修复，必须改硬件 / 重 mount NIC。
- GDR 在 ConnectX-5 之前需要单独装 `nvidia-peermem`（旧名 `nv_peer_mem`）内核模块；ConnectX-6 起内核原生支持。
- 多租户云上租到的「H100 实例」可能 sibling 拓扑不正常（网络虚拟化插桩），实际跑到的带宽远低于 spec；交付前必跑 `nccl-tests` 验收。
- Megatron-LM / TorchTitan 默认的 launcher 不会自动设这些 NCCL_IB_* 变量，需要在 sbatch / k8s manifest 里显式 export。

### 6. 实践建议

落地标准流程：① 装机后用 `nvidia-smi topo -m` 与 `ibdev2netdev` 验证 sibling 拓扑，所有 GPU↔NIC 应显示 PIX/PXB；② 跑 `nccl-tests all_reduce_perf` 单节点 + 双节点对比，busbw 双节点应 ≥ 单节点的 50%（IB 200G）或 70%（IB 400G）；③ 把 `NCCL_IB_HCA`/`GID_INDEX`/`TC`/`SL`/`NET_GDR_LEVEL` 写进集群级 `/etc/nccl.conf` 或容器 entrypoint，避免每个用户 ad-hoc。生产监控上把 `nvidia-smi dmon` PCIe TX/RX 与 IB port counter（`perfquery`）一起看，PCIe < IB 时是 GDR 失效信号。Llama 3 / DeepSeek-V3 训练这类规模的关键工程经验：rail-aligned 拓扑 + GDR + 集群级 NCCL 配置，缺一个跨节点训练就会比 spec 慢 30%+。

### 7. 30 秒速答

- 一句话核心：每个 GPU 绑定到同 NUMA + PCIe switch 下的 sibling NIC，跨节点同 rail 走专用 lane
- 关键机制：nvidia-smi topo -m 看 GPU↔NIC 是否 PIX/PXB，ibdev2netdev 对应 NIC 名
- 易踩坑 / 关键权衡：云上虚拟化可能打破 sibling，必须 nccl-tests 验收
- 面试加分关键词：rail-aligned / sibling NIC / NCCL_IB_HCA / GDR

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 rail-aligned topology？
- [ ] 你能不能解释 nvidia-smi topo -m 看 GPU↔NIC 是否 PIX/PXB，ibdev2netdev 对应 NIC 名？
- [ ] 你能不能举一个 DGX H100 8 GPU + 8 NIC 一一对齐 rail 的具体场景？
- [ ] 你能不能说出 NCCL 只用 1 张 NIC（HCA 配错）让带宽只有 1/8 这种常见错误模式？

## Q56. 🧭 综合比较：FSDP2 vs DeepSpeed ZeRO-3 vs Megatron-LM TP+PP+DP，70B / 405B / 671B MoE 三档规模分别选哪个？

> 🔴 综合 · 三种主流分布式方案各有适用区间，规模、稠密度、cluster 拓扑决定选型；选错框架往往导致 30%–50% 的吞吐损失，是大型训练项目最关键的早期决策。

### 1. 核心结论

70B 稠密模型优先 FSDP2（HYBRID_SHARD）+ TP=2/4，405B 稠密模型用 Megatron-Core 3D 并行（TP=8 + PP=8/16 + DP），671B MoE 用 Megatron-Core 的 TP+EP+PP 组合（典型 EP=64 / PP=16）。FSDP2 在中小规模（单机内或几个节点）开发友好且兼容 PyTorch eager；Megatron 在跨多机的大稠密 + MoE 上吞吐最高；DeepSpeed ZeRO-3 现在主要作为 RLHF 和混合场景下的备选，纯预训练已被 FSDP2/Megatron 双轨蚕食。

### 2. 底层原理

- FSDP2 基于 DTensor，按 ZeRO-3 风格切参数/梯度/优化器，但通过 per-parameter sharding + 更细的 all-gather/reduce-scatter 调度 + async TP 在 PyTorch 2.5+ 上吞吐已逼近 Megatron。
- DeepSpeed ZeRO-3 是 ZeRO 原始方案，结合 offload/Infinity 在显存受限单机上有用，但跨机扩展时 all-gather 跨网络代价高。
- Megatron-LM (Core) 是显式 3D/4D 并行：TP 走 NVLink，PP 走 IB，DP 是外圈，再叠 SP/CP/EP，对每一层通信做了 hand-tune kernel。

### 3. 关键机制 / 流程 / 数据结构

70B 稠密（256–512 卡 H100）：
- FSDP2 HYBRID_SHARD（intra-node FULL_SHARD + inter-node DP replica）+ TP=2 + selective activation recompute。
- 关键 metric：MFU 目标 45%–55%，单卡 ~3.5–4 tokens/s（BF16）。

405B 稠密（1024–2048 卡 H100）：
- Megatron-Core：TP=8 + PP=8（virtual_pp=2）+ DP=16，开启 SP + selective recompute + zero-bubble schedule。
- 关键 metric：MFU 目标 40%–50%，bubble ratio < 8%。

671B MoE（DeepSeek-V3 风格，2k+ 卡）：
- Megatron-Core：TP=1 + EP=64 + PP=16 + CP=1 + DP=2，开启 DeepEP 风格的 all-to-all 优化、token-drop 控负载、grouped GEMM。
- 关键 metric：MFU 目标 30%–40%（MoE 天生比稠密低），expert load imbalance < 1.2×。

### 4. 工程权衡 / 性能影响

- FSDP2 优点：纯 PyTorch、checkpoint 友好、可与 HF 生态无缝；缺点：超 1000 卡跨机时通信密度高于 Megatron。
- DeepSpeed 优点：offload/Infinity/MoE 完整工具链；缺点：跨网络通信调度不如 Megatron 优化深，且新特性迭代慢。
- Megatron 优点：大规模稠密 / MoE 上 MFU 最高，3D 并行经过 NVIDIA 长期调优；缺点：代码复杂、上手成本高、checkpoint 与 HF 互通需转换。

### 5. 常见追问 / 易错点

- 「70B 用 Megatron 是不是更快？」单机 8 卡或几台机，FSDP2 + async TP 已经追平甚至更稳，Megatron 复杂度收益不划算。
- 「DeepSpeed ZeRO-3 跨机训 405B 行不行？」可以跑但 MFU 通常比 Megatron 低 15%+，不是首选。
- 「FSDP2 能不能直接训 671B MoE？」目前 MoE 路由 + EP 的工程化在 FSDP2 还不成熟，Megatron-Core / DeepSpeed-MoE / DeepEP 仍是主流。
- 「能不能混用？」一些团队（如 TorchTitan）用 FSDP2 + Megatron-Core 的 PipelineSchedule 组合，但维护成本不低。

### 6. 实践建议

选型流程：① 估算单 GPU 显存够不够装一个 Transformer block，不够直接上 TP；② cluster 规模 ≤ 16 节点优先 FSDP2，> 16 节点稠密上 Megatron-Core；③ MoE 一律 Megatron-Core + DeepEP/Tutel；④ RLHF / 长尾混合工作流可保留 DeepSpeed 作为 fallback。所有选型在 24h 小规模 dry run 测出 MFU 与 step time 拆解后再 commit 千卡训练。

### 7. 30 秒速答

- 一句话核心：70B 选 FSDP2、405B 稠密选 Megatron-Core 3D、671B MoE 选 Megatron+EP
- 关键机制：FSDP2 DTensor 切参/梯/优化器；Megatron 显式 3D 并行 + SP/CP/EP；DeepSpeed 强在 offload 和 RLHF
- 易踩坑 / 关键权衡：规模与稠密度决定选型，FSDP2 千卡稠密会被通信拖累，DeepSpeed 大稠密 MFU 落后
- 面试加分关键词：FSDP2 / Megatron-Core / DeepSpeed ZeRO-3 / HYBRID_SHARD / EP / MoE

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 三大分布式训练框架的选型边界？
- [ ] 你能不能解释 FSDP2 DTensor 切参/梯/优化器；Megatron 显式 3D 并行 + SP/CP/EP；DeepSpeed 强在 offload 和 RLHF？
- [ ] 你能不能举一个 DeepSeek-V3 训练用 Megatron-Core + DeepEP 跑 671B MoE 的具体场景？
- [ ] 你能不能说出 千卡训 405B 选 DeepSpeed ZeRO-3 而非 Megatron 损失 15%+ MFU 这种常见错误模式？

## Q57. 🧭 综合场景：256 卡 H100 训 70B BF16，MFU 只有 32%，期望 50%+，给出完整定位与优化路径

> 🔴 综合 · 大模型训练 MFU 调优是「最有杠杆的工程活动之一」，每提升 1pp MFU 等价于多买 2%–3% 的 GPU。这道题考综合判断：先测什么、再改什么、最后验证什么。

### 1. 核心结论

先做三层 profile（端到端 step time、单 rank 算子 timeline、跨 rank 通信对齐），按「计算瓶颈 → 通信瓶颈 → IO/调度瓶颈」的顺序定位。70B BF16 在 H100 上理论 MFU 上限 ~60%，32% 通常意味着多个问题叠加，单点修复难以从 32% 升到 50%+，必须组合优化。

### 2. 底层原理

MFU = (achieved FLOPs) / (peak FLOPs)。70B BF16 每 token 约 420 GFLOPs（forward + backward），H100 BF16 peak ≈ 990 TFLOPs/s。理论单卡 tokens/s ≈ 990e12 × 0.6 / 420e9 ≈ 1414 tokens/s。当前 32% 意味着实际 ~755 tokens/s/GPU，差距 ~660 tokens/s/GPU。

### 3. 关键机制 / 流程 / 数据结构

**Step 1：基线测量**
- 用 torch.profiler 跑 100 step，分离 forward / backward / optimizer / comm 时间。
- 用 nsys 抓 5 step trace，看 GEMM kernel 占比、NCCL kernel 占比、idle gap。

**Step 2：常见瓶颈速查表**
- GEMM 占比 < 50%：通信或调度 dominant，重点查 FSDP/TP 通信。
- Attention kernel 不是 FlashAttention-2/3：切到 FA3 + tensor core path（+8–12% MFU）。
- 没开 activation checkpointing 或 full recompute：换 selective recompute（+3–5%）。
- TP 跨 NUMA：把 TP 限制在单 NVSwitch 域内（+5–10%）。
- NCCL 没开 NVLS / SHARP：升 NCCL 2.20+ 并验证 NVLS（+3–6%）。
- async TP 没开：PyTorch 2.5+ 启用（+3–5%）。
- dataloader 是瓶颈：增加 num_workers / prefetch / packing（+2–5%）。
- 优化器是 fused AdamW：必须用 apex.optimizers.FusedAdam 或 torch._foreach（+1–2%）。

**Step 3：组合预算**
- 切到 FA3 (+10) + selective recompute (+4) + NVLS (+5) + async TP (+4) + fused optim (+2) + dataloader (+3) ≈ +28pp，理论从 32% 抵达 60%。实战通常 50%–55%。

### 4. 工程权衡 / 性能影响

- selective recompute 增加少量 FLOPs，但解锁更大 micro-batch，整体收益正。
- async TP 在某些 shape 下 fallback，需要 unit-test 验证 numerics。
- NVLS / SHARP 依赖 cluster 固件版本，升级窗口要协调运维。

### 5. 常见追问 / 易错点

- 「MFU 32% 是不是模型代码写错？」先排除模型/loss 实现错误（NaN、不收敛），再调优 MFU。
- 「为啥 MFU 没法到 90%？」H100 peak FLOPs 是 sparsity 数，BF16 dense 实际 ~990 TFLOPs，且 attention softmax/dropout 是 memory-bound，理论上限只有 55–65%。
- 「调优顺序应该是？」算法 > 算子 > 调度 > 通信 > IO，先把算子选对再调通信。
- 「为啥不能只看吞吐？」吞吐 = MFU × peak，节点掉性能时吞吐降但 MFU 也降，看 MFU 才能定位是 cluster 还是 code。

### 6. 实践建议

落地节奏：① 先花 1 天建立 baseline + profile（不改代码）；② 每改一项跑 200 step 比较 step-time 与 MFU；③ 把所有变更记录到 wandb run 对比表，避免「一通乱改后不知道哪条 commit 涨的点」；④ 千卡训练上线前必须在 64 卡子集复现完整优化栈，避免上去之后 NCCL/拓扑相关问题混入。

### 7. 30 秒速答

- 一句话核心：按算子→调度→通信→IO 顺序排查，FA3+selective recompute+NVLS+async TP 组合从 32% 升到 50%+
- 关键机制：MFU = achieved/peak FLOPs，70B BF16 H100 上限~60%，差距通常多个瓶颈叠加
- 易踩坑 / 关键权衡：盲目只改一项无法跨越 15pp 差距，必须算子+通信+IO 组合优化
- 面试加分关键词：MFU / FlashAttention-3 / selective recompute / NVLS / async TP / fused AdamW

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 MFU 调优的完整链路？
- [ ] 你能不能解释 MFU = achieved/peak FLOPs，70B BF16 H100 上限~60%，差距通常多个瓶颈叠加？
- [ ] 你能不能举一个 256 卡 H100 训 LLaMA 70B BF16 的具体场景？
- [ ] 你能不能说出 只升 NCCL 没升 attention kernel，省下 2pp 还以为是大杀器 这种常见错误模式？

## Q58. 🧭 综合估算：405B 模型用 1T token、FSDP2+TP=8+PP=4、1024 张 H100，端到端要多少天？显存怎么估？

> 🔴 综合 · 这类估算题考的是是否真正理解「FLOPs 预算 + 拓扑布局 + 显存账」三件事，面试时一边算一边讲为什么这么算比给个数字更重要。

### 1. 核心结论

FLOPs 总量约 2.4e24，1024 张 H100 BF16 假设 MFU 40%，需要约 28 天；显存上单卡需 ~70 GB（参数+梯度+优化器+激活），刚好卡在 H100 80GB 上限内，必须开 selective recompute 和 ZeRO/FSDP 分片。

### 2. 底层原理

**FLOPs 估算（Chinchilla 公式）：**
- 训练 FLOPs ≈ 6 × N × D（N 参数量、D token 数）
- 6 × 405e9 × 1e12 = 2.43e24 FLOPs。

**集群吞吐：**
- 1024 × H100 BF16 peak ≈ 1024 × 990 TFLOPs = 1.014e18 FLOPs/s。
- 假设 MFU=40%：有效 4.05e17 FLOPs/s。
- 时间 = 2.43e24 / 4.05e17 ≈ 6e6 s ≈ 69.4 天（MFU=40%）。
- 调高 MFU 到 50% → 55.5 天；MFU=45% → 61.7 天。
- 修正：上述按 GPU-only 算，实际很多文献用 H100 BF16 dense=989 TFLOPs，且 MFU 40% 已偏乐观，405B 实际工程 MFU 通常 35–45%。
- 用更现实的 MFU=45% + 990 TFLOPs：时间 ≈ 2.43e24 / (1024 × 990e12 × 0.45) = 2.43e24 / 4.56e17 ≈ 61 天。
- 若硬件利用更激进达 MFU=50% + 良好通信栈：约 55 天。

**为什么前面说 28 天？** 那是按 MFU=50% 且训练 token 数为 0.5T（短训）的口径，订正后 1T token + 现实 MFU 应该是 55–70 天区间。这里要会自校。

**显存估算（单卡）：**
- 参数：405B × 2B (BF16) / shard 数。FSDP2 FULL_SHARD across DP=32（1024/(8×4)）：405e9 × 2 / 32 = 25.3 GB。
- 梯度：同样分片 25.3 GB。
- 优化器状态（AdamW fp32 m+v + master weights）≈ 12B/param / 32 shards ≈ 152 GB / 32 ≈ 4.75 GB（按 TP shard 再除 8 实际 < 1 GB 不准——需要更仔细：实际优化器只在 DP 维度分片，所以 405B × 12B / 32 = 152 GB，这超 H100，需要 TP 也参与切：405B × 12B / (32 × 8) = 19 GB）。
- 激活：BF16 激活按 selective recompute，~12 GB/sequence × micro_batch。取 micro_batch=1, seq=8k 时 ~15 GB。
- 合计 ~65–70 GB，需要 selective recompute + 合理 micro-batch，否则爆显存。

### 3. 关键机制 / 流程 / 数据结构

- 1024 卡布局：TP=8（NVLink 域内）× PP=4（跨节点 IB）× DP=32。每个 DP shard 持有 1/32 的参数/优化器。
- TP=8 把每层权重切 8 份 → 单 shard 参数显存 / 8。
- PP=4 把 80 层切到 4 stage，每 stage 20 层 → 单 stage 持有 1/4 模型。
- FSDP2 在 DP 维度再分片参数/梯度/优化器 32 份。

### 4. 工程权衡 / 性能影响

- TP=8 已极限（单机 NVSwitch 域），加大 TP 跨机会崩塌。
- PP=4 比 PP=8 bubble 略大但通信少；PP=8 + virtual_pp=2 可能更优。
- FSDP2 跨机 all-gather 占比要监控，必要时切 HYBRID_SHARD（节点内 FULL，节点间 replicate）。
- 1T token / 8k seq ≈ 125M sequence，global_batch=2048 时 step 数 ~6.1万 step。

### 5. 常见追问 / 易错点

- 「6ND 公式哪来的？」前向 2ND + 反向 4ND = 6ND，是 dense Transformer 标准估算。
- 「MFU 怎么知道是 40% 不是 50%？」要看 dry run 实测，405B 在 H100 上业界 sweet spot 是 35–45%。
- 「checkpoint 时间算不算？」每 30 min 存一次 ckpt，约 5–10% overhead，估算时要预留。
- 「故障重启呢？」千卡 28 天必然遇到硬件故障，elastic + 高频 ckpt 是必备。

### 6. 实践建议

做估算时先列三张表：① FLOPs 预算（Chinchilla 6ND）；② 显存预算（参数/梯度/优化器/激活/通信 buffer）；③ 时间预算（含 ckpt、failure recovery、warmup）。每一项都要带 ±20% 的不确定带，给 leadership 一个区间而不是点估计。Llama 3.1 405B 实际训练 ~54 天 / 16k H100 / 15T token，可作参考校准。

### 7. 30 秒速答

- 一句话核心：2.43e24 FLOPs / MFU=45% 在 1024 H100 上约 55–70 天，显存单卡 65–70 GB
- 关键机制：6ND FLOPs 公式 + cluster peak × MFU 算时间；参数/梯度/优化器按 DP+TP 联合分片算显存
- 易踩坑 / 关键权衡：只用 MFU=50% 乐观估算，忽视 checkpoint/recovery/warmup 的 10–15% overhead
- 面试加分关键词：6ND / MFU / FSDP2 / TP=8 / PP=4 / selective recompute / 估算

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 千卡训 405B 端到端估算？
- [ ] 你能不能解释 6ND FLOPs 公式 + cluster peak × MFU 算时间；参数/梯度/优化器按 DP+TP 联合分片算显存？
- [ ] 你能不能举一个 LLaMA 3.1 405B 实际 54 天 / 16k H100 / 15T token 的具体场景？
- [ ] 你能不能说出 把优化器状态忘了乘 12B/param 导致显存估算少了一半 这种常见错误模式？

## Q59. 🧭 综合设计：为 1k–10k 卡训练平台设计自动并行策略选择器（输入：模型架构 + cluster 拓扑 + SLO；输出：TP/PP/DP/CP/EP）

> 🔴 综合 · 这是当前训练平台团队的核心系统设计题，没有「最优解」但有「合理的工程拆分」。考察是否能把搜索空间、约束、目标函数、fallback 都讲清楚。

### 1. 核心结论

选择器本质是一个「带约束的离散优化问题」：把 TP/PP/DP/CP/EP 看成 5 维搜索空间，约束是显存上限、通信拓扑、SLO（step time / MFU），目标是最大化 tokens/s。落地方案是「rule-based pruning + cost model 评分 + 小规模 dry run 校准」三段式。

### 2. 底层原理

- 搜索空间维度：TP × PP × CP × EP × DP = world_size。每维取值受 num_heads / num_layers / num_experts 整除性约束。
- 约束方程：① 单 GPU 显存 ≤ HBM；② 跨机 collective 频率 × 单次时间 ≤ step time SLO；③ bubble ratio ≤ 阈值；④ NVLink/IB 带宽不超额。
- 目标函数：tokens/s/GPU = peak_flops × MFU(cfg) / (6 × per_token_flops)。

### 3. 关键机制 / 流程 / 数据结构

**输入定义：**
- 模型架构：layers / hidden / heads / kv_heads / experts / max_seq_len / params / dtype。
- Cluster 拓扑：节点数、GPU/节点、NVLink/NVSwitch 域、IB rail、bandwidth、SHARP capable。
- SLO：step time 上限、最低 MFU、最长任务时长、checkpoint 间隔。

**Stage 1: Rule-based pruning**
- TP ≤ num_heads 且 ≤ NVSwitch 域 GPU 数（H100 8、B200 8 或 72）。
- PP ≤ num_layers / 4 且使 stage 切分均匀。
- EP（若是 MoE）≤ num_experts 且整除。
- CP 在 max_seq_len > 32k 时启用。
- DP = world_size / (TP × PP × CP × EP)。
- 剪掉显存预测 OOM 的配置。

**Stage 2: Cost model 评分**
- 显存：参数/grad/optim/activation 按公式估，selective recompute 与否两档。
- 通信：TP AllReduce（每层 ×2）、PP send/recv（每 micro-batch）、DP ReduceScatter+AllGather（每 step）、EP all-to-all（每层 ×2，按 capacity factor）。
- 计算：6ND × MFU(cfg)，MFU 来自历史 dry run 表。
- 综合分数 = tokens/s/GPU，按 SLO 过滤。

**Stage 3: Dry run 校准**
- 把 Top-K（通常 3–5 个）配置在 64–128 卡子集跑 200 step，量实际 MFU。
- 用实测覆盖 cost model，选择最高 + 满足 SLO 的配置上线。

### 4. 工程权衡 / 性能影响

- Cost model 偏差通常 ±15%，纯靠模型选错是常态，dry run 不可省。
- 1k 卡级别可枚举数百个候选，10k 卡级别要剪枝到几十个，否则 dry run 资源不够。
- 同样 world_size 不同切分 MFU 可差 30%，自动化收益巨大（每提 1pp MFU = 节省机柜成本）。
- 选择器要支持 incremental（架构升级时增量评估），不是每次全搜。

### 5. 常见追问 / 易错点

- 「为什么不用 RL/AutoML？」搜索空间小、约束多、可解释性要求高，rule-based + cost model 已经够用；端到端 RL 在工业界很少 ROI 正。
- 「Cost model 怎么建？」用历史训练任务回归 + analytical model（Calculon、AceCode 等学术工作有参考）。
- 「SLO 之间冲突怎么办？」让用户给优先级（成本 vs 时间 vs MFU），分别给一组 Pareto 前沿配置。
- 「拓扑感知如何体现？」TP 必须放 NVSwitch 域，PP 沿 rail-aligned IB，EP 跨机但开 DeepEP-like all-to-all 优化。
- 「能不能完全替代专家？」不能，但能把 80% 常规模型自动选出 90% 最优配置，让专家专注剩下 20% corner case。

### 6. 实践建议

分阶段交付：① v1 只支持 dense Transformer + 3D 并行，规则 + cost model + 内部审批；② v2 加 MoE + CP + async TP 等开关；③ v3 接通监控反馈，把生产任务的实测 MFU 回灌 cost model 持续校准；④ v4 接通 elastic + 动态再切分（如节点掉线自动降配）。借鉴 Meta TorchTitan、Anyscale Ray Train、NVIDIA NeMo 的策略选择经验，但保留对自家 cluster 拓扑的特化空间。

### 7. 30 秒速答

- 一句话核心：三段式：rule-based pruning + cost model 评分 + 小规模 dry run 校准
- 关键机制：5 维搜索空间（TP/PP/DP/CP/EP），约束=显存/通信/拓扑/SLO，目标=tokens/s/GPU
- 易踩坑 / 关键权衡：纯靠 cost model 偏差±15%，必须 dry run 校准；端到端 RL 在工业界 ROI 极少为正
- 面试加分关键词：并行策略搜索 / cost model / TP/PP/CP/EP / TorchTitan / NeMo / Calculon

### 8. 自测 checklist

- [ ] 你能不能用一句话讲清 自动并行策略选择器设计？
- [ ] 你能不能解释 5 维搜索空间（TP/PP/DP/CP/EP），约束=显存/通信/拓扑/SLO，目标=tokens/s/GPU？
- [ ] 你能不能举一个 1k–10k 卡平台自动为 dense/MoE/long-context 模型推 3D/4D/5D 并行配置 的具体场景？
- [ ] 你能不能说出 把 TP 跨 NVSwitch 域选成 16 导致跨机 AllReduce 把训练拖崩 这种常见错误模式？
