# 推荐系统推理 infra 卷

## 主题边界
本卷聚焦推荐系统（recommendation system）的在线推理服务化基础设施。涵盖传统 DLRM 范式（Deep Learning Recommendation Model）以及 2024 后兴起的生成式推荐（Generative Recommendation, GR）范式：Meta HSTU、Google TIGER、字节 GR 等。讨论范围包括 embedding lookup 分布式、候选生成 vs 排序两阶段流水、特征工程在推理路径上的工程化、序列推荐 KV cache 形态、推荐 P99 latency 工程、多目标建模、特征服务（Feature Store）、A/B 测试 infra、跨方向工程对照（推荐 vs LLM vs 图像生成）。推荐算法本身、效果调参、业务策略不在本卷展开。

## Q1. 推荐系统推理跟 LLM 推理 / 图像生成推理在工程模式上的本质差异是什么？

> 🟢 基础 · LLM 是序列化生成、图像是迭代去噪、推荐是多目标矩阵乘加 embedding 查表——三种推理工程模型从底层到调度都不同。

### 1. 核心结论
推荐系统推理的工程特点：（1）**embedding lookup 主导**，DLRM 类模型 90% 时间在查 embedding 表、不在算 matmul；（2）**latency 极紧**，信息流场景 P99 < 50ms，比 LLM/图像松一两个量级；（3）**两阶段流水**，候选生成（Retrieval / Recall）+ 排序（Ranking），各自独立的服务；（4）**特征工程在 path 上**，需要实时从 Feature Store 拉用户/物品/上下文特征；（5）**多目标建模**，同时输出 CTR / CVR / 时长 等多个 head；（6）**没有 KV cache 概念**（传统 DLRM）或者**有简化 KV**（生成式推荐 GR）；（7）**容量随 DAU 线性涨**，跟 LLM 按 token 算的成本结构完全不同。

### 2. 底层原理
传统 DLRM 推理模式：input = {sparse 用户/物品 ID 特征} + {dense 数值特征} → embedding lookup → feature interaction (FM / DCN / Cross) → MLP → multi-head output。Embedding 表是模型最大部分（几十 GB-TB 量级），MLP 反而很小（几十 MB）。

生成式推荐（GR）模式：把用户行为序列当作 prompt，LLM 风格的 transformer 模型生成候选 item id 序列。Vocabulary 是 item ID 集合（百万到亿级）。推理类似 LLM 但 sequence 较短（hundreds vs LLM thousands）、vocabulary 极大。

跟 LLM 区别：LLM 推理是 autoregressive、KV cache 主导、memory-bound；推荐 DLRM 是 embedding lookup 主导、cache miss 主导、IO-bound。生成式推荐 GR 处在两者之间，倾向 LLM 模式但 vocabulary 大。

### 3. 关键机制 / 流程 / 数据结构
第一，DLRM 推理流。用户 ID + N 个物品 ID 候选 → 查 embedding（user_emb 1×d + N 个 item_emb N×d + 100+ 个 sparse feature emb）→ 拼接 → MLP forward → 输出 N 个 score → topK 返回。

第二，候选生成阶段。从亿级 item pool 选出几百到几千候选。常用方法：双塔模型（user embedding × item embedding 算 cosine similarity）+ ANN 检索（HNSW / FAISS）。耗时 ~10-30ms。

第三，排序阶段。对候选生成的 N 个 item 跑精排模型（DLRM / DIN / DCN）算精确 score 排序。N 通常 100-1000。耗时 ~10-30ms。

第四，特征实时拉取。每次请求要从 Feature Store（Redis / 自研 KV store）拉用户特征（最近行为、画像）、物品特征（统计、内容向量）、上下文特征（时间、设备）。Feature lookup 占总 latency 30-50%。

### 4. 工程权衡 / 性能影响
延迟 budget 分配。50ms 总预算：（a）feature lookup ~15ms；（b）候选生成 ~10ms；（c）排序 ~15ms；（d）网络 + 序列化 ~10ms。每段都要严格控制。

吞吐特性。推荐系统 QPS 比 LLM 高 1-2 个量级（DAU × 频次）。抖音级别 fleet ~百万 QPS（推荐请求），LLM 服务 ~万 QPS。

成本结构。Embedding 表存储是大头（TB 级，需要分布式 KV store + 内存 cache）。GPU 算力相对小（DLRM 是 IO bound）。跟 LLM（GPU 算力主导）相反。

### 5. 常见追问 / 易错点
第一，DLRM 不是单一模型。"DLRM" 是 Meta 论文的具体架构，但业界泛指"传统 ranking 模型"（DCN / DIN / DLRM / xDeepFM 等）。每家公司有自家变体。

第二，离线训练 vs 在线推理。训练时 batch 几千、用 distributed training；推理时 batch=1 或几十（latency 优先）。Embedding 表在线推理时只读（不更新）。

第三，跟广告系统的关系。推荐 / 广告 / 搜索系统底层架构非常类似（多阶段 retrieval + ranking + serving）。本卷主线讲推荐，工程上跟广告/搜索 80% 相通。

第四，generative recommendation 不是要替代 DLRM。GR 跟 DLRM 是并行发展。GR 在序列建模和长尾推荐上有优势，DLRM 在工程成熟度和大规模特征上有优势。2024-2025 工业上两者共存。

### 6. 实践建议
新业务起步：用开源框架（TorchRec / DeepCTR / EasyRec）跑通 DLRM 基线。

性能优化优先级：（1）embedding lookup 优化（cache + 分布式 KV）；（2）feature service P99；（3）模型 compute（MLP）量化；（4）网络 + 序列化。

跟广告团队 collaborate：技术栈高度相通，可复用 infra。

监控：QPS / P50/P99 latency / cache hit rate / feature service P99 / 各阶段耗时分布。

### 7. 30 秒速答
- DLRM 是 embedding lookup 主导（90% 时间），不是 matmul
- 两阶段流水：候选生成 + 排序，各 ~10-30ms
- P99 latency 50ms 量级，比 LLM 紧一两个量级
- 生成式推荐 GR 是 2024+ 新范式，倾向 LLM 模式

### 8. 自测 checklist
- [ ] 你能不能讲清 DLRM 推理时各阶段时间分布？
- [ ] 你能不能解释候选生成 vs 排序分别用什么模型？
- [ ] 你能不能说出推荐 fleet 跟 LLM fleet 在成本结构上的差异？
- [ ] 你能不能识别 GR 跟 DLRM 适合的不同场景？

## Q2. DLRM 模型架构与推理瓶颈

> 🟡 进阶 · DLRM = sparse features × embedding + dense features → MLP → CTR/CVR——朴素的架构，但工业 fleet 每天跑万亿次推理，瓶颈分布跟 LLM 完全不同。

### 1. 核心结论
DLRM (Deep Learning Recommendation Model, Meta 2019) 是工业推荐系统的代表架构。结构：（1）sparse features（用户 ID / 物品 ID / 类别等）通过 embedding lookup 得到 dense vector；（2）dense features（统计特征）通过 bottom MLP 编码；（3）feature interaction 层（dot product / cross / FM）让 sparse + dense 互相交互；（4）top MLP 输出 CTR / CVR 等 head。推理瓶颈不在 compute（MLP 计算量小），在 embedding lookup（亿级表 + 几十到几百 features × N 候选物品 → 大量随机访存）和 feature 拉取（IO bound）。

### 2. 底层原理
Embedding 表组成：每个 sparse feature 一张 embedding 表，shape [vocab_size, embedding_dim]。User ID 表可能几亿行；item ID 表几亿行；类目表几万行；其它特征几千到几十万行。总参数量几十亿到万亿（embedding 主导）。

Feature interaction：DLRM 原始论文用 dot product（pairwise feature dot）。后续 DCN 用 cross network 做高阶交互。DIN 用 attention 让用户行为序列加权聚合。xDeepFM 用 CIN 做显式高阶。

MLP 部分：bottom MLP 几层 dense（128-512 维），top MLP 几层 dense + sigmoid head。整体参数 < 100MB。

推理时计算分布：embedding lookup 60-70%，MLP 10-15%，feature interaction 10%，其它（softmax / sigmoid 等）5-10%。

### 3. 关键机制 / 流程 / 数据结构
第一，embedding lookup pattern。每次请求 lookup pattern：1 个 user ID + N 个 candidate item IDs（N=100-1000）+ M 个其它 sparse features（M=50-200）。总 lookup 次数 N+M+1，每次返回 dim=16-128 vector。

第二，embedding 存储分级。Hot embedding（活跃用户 / 热门物品）放 GPU HBM 或 Redis；warm 放 SSD-backed KV store；cold 放对象存储。LRU 或 LFU eviction。

第三，sparse feature batch lookup。N 个 item 的 lookup 可 batch：concat ID list → 一次 lookup 调用 → 返回 [N, dim] tensor。Bulk read 比单次 lookup 快 5-10x。

第四，跟训练的差异。训练时 embedding 表在多机 distributed（TorchRec 的 ShardedEmbedding），用 AllToAll 通信。推理时全量 read-only，可以全 replicate 到每个 inference instance（如果显存够）或分布式 lookup service。

### 4. 工程权衡 / 性能影响
显存 / 内存预算。10 亿 user × 64-dim FP16 = 128GB。1 亿 item × 64-dim FP16 = 12.8GB。其它 100 个 features 平均 100 万 × 32-dim = 6.4GB。总 ~150GB。单 GPU 显存（80G）装不下，需要 CPU 内存 cache + GPU 热点 cache 分层。

延迟分布（典型 DLRM）。Feature lookup 远端 RPC ~5-10ms；embedding lookup 本地 hot path ~2-5ms；MLP forward ~1-3ms on CPU 或 ~0.5ms on GPU；feature interaction ~1ms。总 10-20ms 单候选；100 候选 batch 30-50ms。

吞吐特性。CPU 推理：单核 ~几千 QPS；多核 ~几万 QPS / 实例。GPU 推理：受 PCIe 限制（embedding 在 CPU），~万 QPS / 实例。生产 fleet 数千实例打底。

### 5. 常见追问 / 易错点
第一，DLRM 用 GPU 不一定快。Embedding lookup 是 CPU 友好（大随机访存 + 大 cache），GPU 上 lookup overhead 大（PCIe 传输）+ HBM 装不下整表。生产 DLRM 推理常 CPU + GPU hybrid（GPU 跑 MLP，CPU 跑 lookup）。

第二，embedding 表的"维度爆炸"。一个新业务要加 100 个 sparse features，每个 feature 自己的 embedding 表，总参数量翻倍。新业务上线前要 feature engineering 收缩特征。

第三，feature hash trick。Vocabulary 巨大时（如 user ID 全网十亿）用 hash 把 ID 映射到固定大小（如 1M）embedding 桶。冲突造成精度损失但显著省内存。

第四，distributed embedding 推理。如果 embedding 表必须分布式（fleet 共享），每次 lookup 走 RPC。Latency 紧时不可行。常用做法：完整复制（每实例独立持表）+ 增量更新（每天 push 新版）。

### 6. 实践建议
新业务起步：用 TorchRec + DLRM baseline 跑通。Embedding 表先放本地内存，发现内存不够再分层。

性能优化路径：（1）embedding lookup 用 CPU + AVX SIMD 加速；（2）热点 embedding cache 到 L2 / Redis；（3）feature hash 控制 vocabulary 大小；（4）MLP 量化 INT8 / FP16。

成本控制：CPU instance 比 GPU 便宜 5-10x for DLRM（embedding 主导）。GPU 只在 MLP 复杂 / 多 head 时收益。

监控：每阶段 latency / embedding cache hit / feature service P99 / 内存 / QPS。

### 7. 30 秒速答
- DLRM = embedding lookup + MLP + feature interaction
- 瓶颈在 embedding lookup（60-70% 时间）+ feature 拉取（IO bound）
- 显存装不下完整表（150GB+），需要分层 cache
- CPU 推理常比 GPU 便宜（embedding 友好 CPU）

### 8. 自测 checklist
- [ ] 你能不能拆解 DLRM 一次推理的各阶段耗时？
- [ ] 你能不能说出 embedding lookup 跟 MLP 计算的相对开销？
- [ ] 你能不能解释为什么 DLRM 用 CPU 也能跑得不错？
- [ ] 你能不能识别 feature hash trick 的 trade-off？

## Q3. 大规模 Embedding 表的分布式存储与查询

> 🔴 专家 · 几个亿用户 × 64-dim embedding = 上百 GB；几个亿物品 × 几个 features = TB 级——embedding 表怎么存怎么查，是推荐 infra 的核心问题。

### 1. 核心结论
大规模 embedding 表的分布式存储与查询方案：（1）**全量复制**：每个推理实例本地一份完整表，最快但内存浪费；（2）**hash sharding**：按 hash(ID) % N 分到 N 个 shard，每实例只持一个 shard，查询走 RPC；（3）**row-based sharding**：按 ID 范围切片，类似 hash；（4）**hierarchical caching**：本地 hot cache + 远端 KV store + cold storage 分层。生产 fleet 一般 GPU 实例本地全量复制小表（user / item ID），远端共享大表（行为序列 / cross-feature）。代表方案：Meta 自研 distributed embedding lookup、字节 ByteRec、TorchRec ShardedEmbedding。

### 2. 底层原理
全量复制方案：每个推理 instance 启动时 load 整个 embedding 表到内存。新 model 上线时 rolling restart 重新 load。优点：lookup 无 RPC 延迟，本地纳秒级访问。缺点：内存占用大（每实例几百 GB），新表上线慢（load 几分钟）。适合表 < 100GB。

Hash sharding 方案：每个 shard server 持一部分表（如 ID hash mod N == shard_id）。Lookup 时按 ID 算 hash 路由到对应 shard server，走 RPC 读 embedding。优点：单实例内存压力小（仅 1/N），新表 rolling 更新方便。缺点：每 lookup 一次 RPC（亚毫秒级），高 QPS 时网络压力大。

Hierarchical：local LRU cache（最近热点 IDs in L1/L2）+ 本机 RAM（中频）+ 远端 Redis cluster（低频）+ S3/HDFS（冷）。Hit rate 调优是关键。

### 3. 关键机制 / 流程 / 数据结构
第一，embedding lookup batch RPC。多个 ID lookup 攒成一个 RPC（batch lookup），分摊 RTT。同时支持 multi-shard：客户端按 hash 把 IDs 分组，并行发到各 shard，合并结果。

第二，热点 cache 管理。Skewed 分布（少数 ID 占大部分查询）让 cache hit 率高。LFU 比 LRU 更适合（频次而非时间）。Hot ID 列表可定期推送给 inference 实例。

第三，embedding 更新机制。训练好的新 embedding 表 push 到 fleet：（a）全量 push（几小时但简单）；（b）增量 push（diff 部分）；（c）reload signal 触发实例 swap 表。生产里全量 + cross-region replication 是常见。

第四，sparse update 推理时不更新。推理实例 read-only，embedding 训练完离线更新。Online learning（边推理边训练）系统更复杂，需要专门 infra（如 Alibaba PAI EAS）。

### 4. 工程权衡 / 性能影响
存储成本对比。10 亿 user × 64-dim FP16 = 128GB / replica。100 实例全量 = 12.8TB 内存。Sharding 到 100 shard = 128GB / shard，10 实例×10 shard = 1.28TB。复制省 lookup 延迟，sharding 省总内存但增 RPC。

延迟对比。全量复制 lookup ~纳秒（L1/L2 cache）到微秒（RAM）。Sharding RPC ~亚毫秒（kernel + 网络 0.1-0.5ms）。延迟敏感场景全量复制赢。

新业务规模 < 100GB：直接全量复制。100GB-1TB：考虑 sharding 或分层 cache。> 1TB：必须分布式 + heavy caching。

### 5. 常见追问 / 易错点
第一，hash collision。Hash mod 不均匀，某些 shard 负载高（hot key）。Consistent hashing + virtual nodes 缓解。

第二，feature 维度爆炸。Cross feature（如 user_id × item_id）vocabulary 是乘积，几亿 × 几亿 = 不可能存。常用做法：cross 在线计算（不存表）或 hash 到固定桶。

第三，跟 model 训练 sharding 一致性。训练时 ShardedEmbedding 用某种 sharding 策略，推理时要保持一致或重新 reshard。Reshard 计算 expensive，最好训推一致。

第四，long tail problem。Cold ID 很少被查，cache miss 多。Active learning 优化：定期 retraining 把长尾 ID 升级（如果业务有需求）。

### 6. 实践建议
小规模（< 100GB 表）：全量复制每实例，最简单高效。

中规模（100GB-1TB）：分层 cache（GPU HBM hot + RAM warm + Redis cold）。

大规模（> 1TB）：分布式 shard service + 客户端 batch RPC + 热点 cache 在 inference 实例。

监控：cache hit rate（多层）、shard server P99 latency、RPC 吞吐、内存占用。

### 7. 30 秒速答
- 三种方案：全量复制（小表）/ hash sharding（中大）/ 分层 cache（混合）
- Trade-off：复制省 latency 但费内存；sharding 省内存但增 RPC
- 生产典型：本地 hot cache + 共享 KV store + cold storage 三层
- Hot key 用 LFU + consistent hashing 防 skew

### 8. 自测 checklist
- [ ] 你能不能算 10 亿 user × 64-dim 的内存预算？
- [ ] 你能不能讲清 hash sharding 的 hot key 防御？
- [ ] 你能不能解释 cross feature vocabulary 为什么不能直接存？
- [ ] 你能不能识别 embedding 全量更新 vs 增量更新的 trade-off？

## Q4. 推荐系统两阶段流水：候选生成 vs 精排

> 🟡 进阶 · 从亿级物品池选出最好的几十个推给用户，靠的不是单个模型——而是候选生成（快粗）+ 精排（慢细）+ 重排（业务约束）三段式流水。

### 1. 核心结论
推荐系统在线服务通常分三阶段：（1）**候选生成 / Retrieval / Recall**：从亿级 item pool 召回几千到几万候选，强调速度（10ms 内）+ 召回率；（2）**精排 / Ranking**：用复杂模型给候选打精确分（CTR / CVR / 时长），强调精度，N=100-1000；（3）**重排 / Re-ranking**：业务规则、多样性、新颖性、商业约束，最终输出推给用户的几十个 item。三阶段计算量递增（每阶段 item 数减少但模型复杂度增加）。工程上是三个独立服务通过 RPC 串联。

### 2. 底层原理
候选生成阶段。常用方法：
- **双塔模型 (Two-Tower)**：user 塔输出 user embedding，item 塔输出 item embedding。在线时算 user_emb × item_emb 内积，topK。Item embedding 离线全量算好建 ANN 索引。
- **协同过滤 (CF)**：基于历史共现矩阵分解。
- **graph-based**：图神经网络嵌入。
- **生成式 (GR)**：直接生成 candidate id sequence。

精排阶段。复杂模型（DLRM / DIN / DCN / xDeepFM）。Input 是 user features + 候选 item features + cross features，输出多 head（CTR / CVR / 时长 / engagement）。

重排阶段。规则 + 多目标融合 + 多样性约束。规则如"同一作者不超过 N 个"、"广告占比"等。MMR (Maximum Marginal Relevance) 类算法平衡相关性和多样性。

### 3. 关键机制 / 流程 / 数据结构
第一，候选生成的 ANN。Item embedding（10 亿 item × 64-dim）建 ANN 索引（HNSW / IVF + FAISS）。查询时 user emb 跟索引比对，topK 返回。索引构建几小时（离线），在线查询几毫秒。

第二，多路召回。生产里不止一路召回：双塔召回 + 协同过滤召回 + 标签召回 + 热门召回，每路出 1000 候选，合并去重后给精排。多路提高召回多样性。

第三，精排 batch。100-1000 候选 batch 一次 forward。Batch 内每个 item 跟同一 user 配对，user features 一次准备共享。

第四，多目标融合。精排输出 [CTR, CVR, 时长]，重排阶段加权融合：score = α·CTR + β·CVR + γ·时长。权重业务调。

### 4. 工程权衡 / 性能影响
延迟 budget 分配（50ms 总）：feature 拉取 10-15ms / 候选生成 5-10ms / 精排 15-20ms / 重排 + 序列化 5-10ms。

吞吐瓶颈。候选生成阶段查 ANN，单机几千到几万 QPS。精排阶段对 N 个 item 跑 DLRM，N 越大 GPU 跑越合算。

模型 size 阶梯。候选生成模型轻（几 MB-几十 MB 双塔），精排模型重（几百 MB-几 GB DLRM）。重排几乎无模型（规则 + 排序）。

### 5. 常见追问 / 易错点
第一，候选生成跟精排的目标不一致。候选生成优化"召回率"（top-K 里是否包含用户最终点击 item），精排优化"精度"（topK 排序质量）。两者训练数据和 loss 不同。

第二，曝光偏差。精排只看历史曝光过的 item，长尾物品没机会进入数据。需要专门策略（exploration / 强冷启动 boost）。

第三，多目标冲突。CTR 高的物品不一定 CVR 高，可能 CTR 高的是 clickbait。多目标融合权重需要仔细调。

第四，重排的 hard rules。规则太多会盖过精排（如强制广告位会让用户体验差）。需要业务跟 algorithm collaborate。

### 6. 实践建议
新业务起步：双塔召回 + DLRM 精排 + 简单 MMR 重排，跑通端到端。

性能瓶颈定位：先 profile 三阶段时间，哪段占比大优化哪段。生产里精排常是大头。

A/B 测试：每改一阶段（如换召回算法）都跑 A/B 测试看端到端业务指标（点击率 / 时长 / 留存）变化。

监控：三阶段独立 latency 监控 / 召回多样性 / 精排校准度 / 重排规则触发率。

### 7. 30 秒速答
- 三阶段：候选生成（亿→千）/ 精排（千→百）/ 重排（百→几十）
- 候选生成快粗（双塔 + ANN）；精排慢细（DLRM）；重排规则 + 多目标
- 50ms 总预算：feature 10 / 召回 5-10 / 精排 15-20 / 重排 + 网络 10
- 多路召回 + 多目标融合是生产标配

### 8. 自测 checklist
- [ ] 你能不能讲清双塔模型为什么适合候选生成？
- [ ] 你能不能算 1 亿 item × 64-dim HNSW 索引的内存？
- [ ] 你能不能解释候选生成跟精排训练目标的差异？
- [ ] 你能不能识别多路召回的去重与合并机制？

## Q5. 双塔模型 + ANN 检索的候选生成方案

> 🔴 专家 · 双塔模型几乎是工业候选生成的事实标准——user 塔和 item 塔解耦让 item 部分离线建索引，user 部分在线轻量计算，整个流水 5-10ms 出几千候选。

### 1. 核心结论
双塔模型 (Two-Tower) 把 user 和 item 各自映射到同一 embedding 空间，相似度用内积或 cosine。训练时正样本（用户点击的 item）拉近、负样本拉远。推理时 item 塔离线全量算好 embedding，建 ANN 索引（HNSW / IVF / ScaNN）。在线时 user 塔实时算 user_emb，ANN 检索 topK 最近 item。ANN 是工业级近似最近邻搜索算法，million-billion item 库上能做到毫秒级 topK。常用库 FAISS / ScaNN / Milvus / Qdrant。

### 2. 底层原理
双塔训练。Loss = sampled softmax 或 in-batch negative sampling。Sampled softmax 选 K 个负样本算 cross-entropy；in-batch 用同 batch 其它 user-item 对作负样本（效率高）。Item embedding 直接是输出向量（不查表）。

ANN 索引算法：
- **HNSW (Hierarchical Navigable Small World)**：构建多层 graph，每层节点稀疏，查询时从顶层粗找到底层精找。Query O(log N)，构建 O(N log N)。准确率 95%+ at 100x faster than exact。
- **IVF (Inverted File)**：聚类 + 倒排。N item 聚成 K 簇，查询时找最近的 nprobe 个簇内做 exact search。
- **ScaNN (Google)**：anisotropic quantization + tree。Google 自家算法，特别适合 dot product 检索。
- **DiskANN**：HNSW 变体支持 SSD-resident 索引，10 亿规模可单机。

Embedding 维度通常 64-128，太低质量差太高索引慢。

### 3. 关键机制 / 流程 / 数据结构
第一，item embedding 离线生成。每天/小时 train item 塔 → batch inference 全量 item → 存到向量库。新 item 上线先走 cold-start path（基于内容特征算 embedding）再等下次训练。

第二，user embedding 在线生成。每次请求拿 user features → user 塔 forward → user_emb → ANN 查询。User 塔通常轻量（几 MB），CPU 跑足够快。

第三，ANN 索引构建。10 亿 item × 64-dim HNSW 索引大概 300-500GB（每点几百 byte 元数据 + 邻居指针）。单机难装，分布式 ANN 服务（Milvus / 字节自研）。

第四，多路负采样。In-batch negative + 全局采样 + hard negative mining（找混淆的难负样本）。Hard negative 大幅提升模型质量。

### 4. 工程权衡 / 性能影响
ANN 准确率 vs 速度。HNSW 参数 ef_search 控制：ef_search=10 快但召回率 90%；ef_search=100 慢 3x 但召回率 99%。生产里业务调。

构建时间。HNSW 10 亿 item 索引构建几小时（多核 CPU）。每天重建一次。

显存 vs CPU 内存。ANN 索引通常在 CPU 内存（GPU HBM 太贵装不下）。10 亿点 500GB 内存的机器。

替代方案。当 item 库 <1 亿时，brute-force dot product on GPU 反而比 ANN 快（GPU matmul 极快）。Item 库 > 1 亿时 ANN 必须。

### 5. 常见追问 / 易错点
第一，cold start。新 item 没行为数据，item 塔可能输出乱七八糟的 embedding。生产里用 content-based features（标题 / 类目 / 图像 embedding）跑 item 塔，保证 cold start 基础质量。

第二，user embedding cache。同 user 在短时间内多次请求，user_emb 可 cache 几分钟。Cache hit rate 80%+ 大幅减算力。

第三，跟实时行为 disconnect。User 塔输出基于离线训练时的用户特征，user 行为快速变化（如刚买了某品类）时 user_emb 滞后。需要实时特征注入（real-time user behavior sequence）。

第四，distance metric 选择。内积 (dot product) vs cosine vs L2。生产常用内积，但要 normalize（否则 magnitude 主导）。

### 6. 实践建议
新业务起步：FAISS HNSW + 64-dim embedding + sampled softmax loss，跑通 baseline。

性能优化：ANN 参数调（ef_construction, ef_search, M）；user embedding cache；GPU brute-force（小库）。

冷启动专项：content-based item embedding + cold-start boost in 重排。

监控：召回率 (top-K 包含点击 item 的比例) / ANN P99 latency / 索引构建时间 / 索引大小。

### 7. 30 秒速答
- 双塔：user 塔在线算 + item 塔离线建索引
- ANN（HNSW / IVF / ScaNN）million-billion item 毫秒级 topK
- 训练用 in-batch negative + hard negative mining
- 冷启动用 content-based 特征兜底

### 8. 自测 checklist
- [ ] 你能不能讲清双塔为什么解耦后能离线建索引？
- [ ] 你能不能算 HNSW 10 亿点的内存预算？
- [ ] 你能不能解释 in-batch negative 的局限？
- [ ] 你能不能识别 cold start 的工程方案？

## Q6. 生成式推荐 (GR) 范式：从 DLRM 到 HSTU / TIGER

> 🟡 进阶 · DLRM 是矩阵乘加 embedding 查表；GR 是 LLM 风格生成 item id sequence——2024 后大厂都在试验 GR，工程模式跟传统推荐根本不同。

### 1. 核心结论
生成式推荐 (Generative Recommendation, GR) 把用户行为序列当作 prompt，用 Transformer 模型生成下一个或下几个 item id。代表工作：（1）**Meta HSTU (Hierarchical Sequential Transduction Units, 2024)** 把 DLRM 替换成统一 Transformer 范式；（2）**Google TIGER (Transformer Index for GEnerative Recommenders, 2023)** 用 semantic ID 让 transformer 输出可学的 token sequence；（3）**字节 GR** 内部范式（vLLM-GR 推理框架）。GR 优势：序列建模能力强、长尾召回好、统一模型替代多模型组合。劣势：vocabulary 巨大（百万-亿 item id）、推理 latency 更紧、工程成熟度不如 DLRM。

### 2. 底层原理
GR 范式核心思路：行为序列 = NLP 的 prompt，下一 item = NLP 的 next token。模型输出 logits over vocabulary（all items），sampling / topK 得到候选 items。

HSTU 架构：堆叠 HSTU block，每个 block 是 self-attention + pointwise gating + cross-feature interaction。跟标准 transformer 类似但有推荐特有的设计（attention mask 处理序列分段、user/item dense feature 注入）。

TIGER 架构：核心创新是 **semantic ID** —— 用 RQ-VAE (Residual Quantization VAE) 把 item 表示为几个 codebook 索引（如 [125, 387, 92, 41]），然后 transformer 输出 semantic ID sequence。Vocabulary 从亿级变成几千 × 几千的笛卡尔积，更可学。

跟 LLM 区别：vocabulary 是 item id（百万-亿），不是 word token（几万）；sequence 是用户行为（几十到几百），不是文本（几千-几万）；输出是 top-K candidate，不是单一 token stream。

### 3. 关键机制 / 流程 / 数据结构
第一，HSTU 工作流。User behavior sequence [item_1, item_2, ..., item_T] + user/context features → HSTU model → logits over all items → top-K sampling → candidate items。

第二，TIGER semantic ID。Item → RQ-VAE encoder → semantic ID [c1, c2, c3, c4]（4 个 codebook 索引）→ transformer 生成时输出 4 个 token 表示一个 item。Generation 时 beam search 在 (codebook size)^4 空间。

第三，跟 DLRM 协同 vs 替代。早期方案 GR 跟 DLRM 并行（GR 出候选 + DLRM 精排），后期方案统一 GR 端到端（GR 直接输出 final ranking）。Meta 论文 HSTU 是统一替代。

第四，vLLM-GR 风格框架。把 vLLM 推理引擎适配 GR：sequence 较短、vocabulary 巨大、batch 多 user 并发。需要改 attention kernel、KV cache 形态、sampling logits。

### 4. 工程权衡 / 性能影响
模型大小。HSTU 实验规模 0.1B-1.5B 参数。生产规模可能 1-10B。比 LLM 70B 小，但比 DLRM 大 10-100x。

推理 latency。GR 是 autoregressive 生成 K 个 item（K=100-1000）。每 step ~1ms × 100 = 100ms，比 DLRM 的 50ms 慢 2x。优化方向：投机解码（spec decode）/ 减 step / batch。

vocabulary 问题。Output projection [hidden_dim, vocab_size] 巨大（hidden=1024, vocab=1亿 → 100GB matmul）。常用 sampled softmax 或 hierarchical softmax 或 TIGER semantic ID 控制 vocab 大小。

工程对照 LLM。HSTU/GR 推理跟 LLM 工程模式更接近：attention / KV cache / autoregressive decode 都通用。这就是为什么 vLLM-GR 类框架成为可能。

### 5. 常见追问 / 易错点
第一，是不是要替代 DLRM。短期不会。DLRM 在大特征 + 工程成熟度上仍领先，GR 在序列建模 + 长尾上有优势。2024-2025 是 GR 探索期，2026+ 可能更主流。

第二，GR 的训练成本。Vocabulary 巨大 → embedding 表巨大 → 训练显存压力大。Meta HSTU 用 distributed training 1024 GPU 量级。

第三，跟 LLM 类比的局限。User 行为序列没有像自然语言那么强的"语义连贯性"，纯 transformer 在推荐上未必比 DLRM 强多少（除非数据量极大）。

第四，semantic ID 的稳定性。TIGER 的 RQ-VAE codebook 重训会变，semantic ID 跟着变，下游训练数据失效。生产部署稳定性问题。

### 6. 实践建议
GR 探索：用开源 HSTU 实现（Meta 已开源）+ 业务序列数据，跟 DLRM baseline 对比。

如果用 GR：要准备一套不同的 infra（vLLM-style 推理引擎 + 巨大 vocabulary 处理 + KV cache 管理）。

跟 DLRM 长期共存。新业务用 DLRM 起步，规模上来后探索 GR。

监控：序列 length 分布 / vocabulary coverage / topK 召回率 vs DLRM baseline。

### 7. 30 秒速答
- GR = Transformer 范式生成 item id sequence（LLM 风格）
- 代表：HSTU (Meta) / TIGER (Google) / 字节 GR
- 优势：序列建模强 / 长尾召回好 / 统一模型
- 劣势：vocabulary 巨大 / latency 紧 / 工程不成熟

### 8. 自测 checklist
- [ ] 你能不能讲清 HSTU 跟标准 Transformer 的差异？
- [ ] 你能不能解释 TIGER semantic ID 怎么减少 vocabulary？
- [ ] 你能不能说出 GR 推理跟 LLM 推理的相同与不同？
- [ ] 你能不能识别 GR 短期不会替代 DLRM 的原因？

## Q7. vLLM-GR 思路：把 LLM 推理框架适配推荐

> 🔴 专家 · 不重写一套推荐推理引擎，而是改造 vLLM 让它能跑 HSTU——vLLM-GR 是 2024-2025 出现的关键工程实践，背后是 GR 推理跟 LLM 推理的高度同构。

### 1. 核心结论
vLLM-GR 是把 vLLM 推理引擎适配生成式推荐模型 (GR) 的工程实践。核心思路：GR 模型本质是 Transformer + autoregressive generation，跟 LLM 高度同构，可以复用 vLLM 的 PagedAttention / continuous batching / CUDA Graph 等优化。改造点：（1）attention kernel 适配推荐特有 attention mask（行为序列分段）；（2）vocabulary 巨大时的 sampling 优化（sampled softmax / top-K logits）；（3）KV cache 适配较短 sequence（推荐序列 ~hundreds vs LLM thousands）；（4）batch 调度按推荐场景特点（一个 user 多个候选 vs LLM 多 user 各一 sequence）。代表实践：华为 vLLM-GR（Omni-Infer 集成）、字节内部类似框架。

### 2. 底层原理
vLLM 核心优化对应到 GR：
- **PagedAttention**：仍适用，但 sequence 短时 block 浪费可能更明显（GR sequence ~200 vs LLM ~2K）。
- **Continuous batching**：仍适用，user 请求间动态加入退出。
- **CUDA Graph**：捕获 decode loop，对短 sequence 收益更大（cold path 占比小）。
- **KV cache 管理**：复用，但 user 行为序列的 KV 不像 LLM 那样长生命周期。

跟 LLM 不同的地方：
- **vocabulary 处理**：LLM 几万 vocab，输出 logits 一次 matmul OK。GR 几百万-亿 vocab，logits matmul 爆炸。需要 sampled softmax 或 hierarchical 或 TIGER semantic ID。
- **attention mask**：用户行为序列可能有分段（不同 session）、padding（长度不齐）。
- **sampling 策略**：LLM 输出 1 token 用 sampling；GR 输出 top-K item 用 beam search 或 top-K logits 直接选。

### 3. 关键机制 / 流程 / 数据结构
第一，HSTU on vLLM。HSTU model class 注册到 vLLM model registry → vLLM 加载 → 自定义 attention kernel（HSTU 有 pointwise gating 等扩展）→ 自定义 sampling head（top-K logits over vocab）。

第二，short sequence 优化。Sequence 短时 PagedAttention block_size=16 浪费明显。可以调小 block_size（如 8）减少内部碎片。或者用 simple contiguous KV cache 反而效率更高。

第三，vocabulary 优化。输出 [batch, vocab] logits 巨大。常见做法：（a）输出 logits 只保留 top-K（K=1000）；（b）sampled softmax（训练时已用，推理时近似）；（c）TIGER semantic ID 让 vocab 变小。

第四，batch 调度差异。LLM batch：多 user 各一 prompt，prefill + decode 混合。GR batch：单 user 单序列出多个 candidate item（autoregressive K 步）+ 多 user 并发。结构略不同。

### 4. 工程权衡 / 性能影响
适配成本。从 vLLM 改到 vLLM-GR 工作量大概 1-3 月（取决于 GR 模型复杂度）。完全自研推理引擎 6-12 月。复用 vLLM ROI 高。

性能。vLLM-GR 在 GR 模型上的速度跟自研引擎接近（两者都基于 PagedAttention + CUDA Graph）。复用 vLLM 优势主要在工程效率和持续 upgrade。

工程组织。需要熟悉 vLLM 内部 + GR 模型 + 推荐业务知识三件套，团队招聘门槛高。

### 5. 常见追问 / 易错点
第一，vLLM upstream 不会接受 GR。GR 模型属于业务特殊路径，开源 vLLM 主线不会维护 GR 适配。需要自己 fork 或长期维护 patch。

第二，vocabulary 巨大的 logits 计算。即使 sampled softmax，sampled batch 内 logits 仍可能 GB 级 tensor。Memory pressure 大。

第三，跟 ANN 检索的组合。生产里 GR 出 candidate id 后还可能跑 ANN 找相似的（增加多样性），或者直接用 GR 输出。两种方案各有 trade-off。

第四，模型尺寸 vs 业务效果。GR 0.1B vs 1B vs 10B，效果跟 LLM 类似 scaling law 但收益递减更快（推荐数据信号没 NLP 文本那么丰富）。

### 6. 实践建议
新业务尝试 GR：用开源 HSTU + vLLM-GR fork（如华为 Omni-Infer 已集成）跑起来。

工程团队需要 vLLM 内部熟悉的人。建议先做 LLM serving 1 年再上 GR。

跟传统 DLRM 并行试错：GR fleet 是新探索，不要 day 1 全量切。10% 流量 A/B 测试是稳健做法。

监控：GR vs DLRM 业务指标对比（点击率 / 时长 / 留存）+ GR latency + vocabulary 利用率。

### 7. 30 秒速答
- vLLM-GR = 把 vLLM 引擎适配 GR 模型（HSTU 等）
- 复用 PagedAttention / continuous batching / CUDA Graph
- 主要改造：vocabulary 处理（sampled softmax / semantic ID）、attention mask、sampling head
- 比自研引擎 ROI 高，但需要长期 fork 维护

### 8. 自测 checklist
- [ ] 你能不能列出 vLLM-GR 跟标准 vLLM 的 4 个改造点？
- [ ] 你能不能解释 vocabulary 巨大对 logits 计算的影响？
- [ ] 你能不能讲清 short sequence 在 PagedAttention 下的工程问题？
- [ ] 你能不能识别 vLLM-GR 的工程组织挑战？

## Q8. 序列推荐模型（SASRec / BERT4Rec）与 KV cache 形态

> 🟡 进阶 · 序列推荐里用户行为是 sequence，模型是 transformer，跟 LLM 框架几乎重合——但 sequence 短、batch 多、vocabulary 大，推理工程要重新设计。

### 1. 核心结论
序列推荐 (Sequential Recommendation) 用 transformer 建模用户历史行为序列，预测下一个交互 item。代表模型：（1）**SASRec (Self-Attentive Sequential Recommendation)**：causal transformer，预测下一 item；（2）**BERT4Rec**：bidirectional masked LM，预测被 mask 的 item；（3）**S3-Rec / FDSA**：加 self-supervised pre-training。SASRec 跟 LLM 推理高度同构（autoregressive + KV cache），可复用 LLM 推理 infra。BERT4Rec 是 bidirectional 推理一次完成，工程更简单。Sequence 长度 ~100-500 比 LLM 短得多。

### 2. 底层原理
SASRec 架构：item embedding + position embedding → N 层 causal self-attention → 预测下一 item logits。Causal mask 让每个位置只 attend 历史。推理时给 user 历史序列，输出 logits，topK 出 candidates。

KV cache 形态：SASRec 的 KV cache 形态跟 LLM 完全一样（causal attention + 按 token KV 存储）。可直接复用 PagedAttention。

BERT4Rec 架构：bidirectional self-attention，训练时 mask 15% item 让模型预测。推理时给完整 user 序列 + 在末尾加 [MASK] token，输出 [MASK] 位置的 logits 作为下一 item 预测。

跟 LLM 推理差异：sequence 短（100-500 vs LLM 几千），decode 步数少（推荐 top-K 出 K item vs LLM 出几百 token），vocabulary 巨大（百万 item vs LLM 几万 word）。

### 3. 关键机制 / 流程 / 数据结构
第一，SASRec 推理流。User 历史 [item_1, ..., item_T] → embedding → N 层 causal attention → 最后位置的 hidden state h_T → logits = h_T @ item_emb_matrix.T → topK。

第二，KV cache 在序列推荐。给定 user 序列做一次 prefill 把所有位置的 KV 算好。推理时只需要最后一个位置的 K/V（如果只预测下一 item）或 incremental decode（如果预测 sequence）。

第三，BERT4Rec 推理。Bidirectional 一次 forward 完成，没有 autoregressive 也没 KV cache 必要。简单。

第四，跟 DLRM 协同。序列推荐通常作为候选生成的一路（跟双塔召回并行），出 top-K 后给 DLRM 精排。也可端到端用序列模型直接出 ranking。

### 4. 工程权衡 / 性能影响
延迟。SASRec 100-200 长 sequence 单次 prefill + topK ~5-10ms。BERT4Rec 同样 5-10ms。比双塔 ANN 检索（~3-5ms）慢 1-2 倍，比 DLRM 精排（10-20ms）快。

模型大小。SASRec/BERT4Rec 模型本身几十 MB-几百 MB（远小于 LLM）。Item embedding 表跟模型分离（几 GB-几十 GB）。

vocabulary 处理。生产用 sampled softmax 训练。推理时 output projection [hidden, item_vocab] 大矩阵乘，几百 MB-几 GB 量级。可以做 LSH（local sensitive hashing）或 candidate filtering 提前缩小 vocab。

### 5. 常见追问 / 易错点
第一，bidirectional vs causal 选择。BERT4Rec bidirectional 训练时能用全序列信息，效果略好但训练更慢（mask 15% 训练效率低）。生产里 SASRec causal 因为简单 + 跟 LLM 一致而更受欢迎。

第二，序列长度 cutoff。User 历史可能几千 item，模型训练只用最近 100-500。怎么 cutoff（recent vs sampled）影响效果。

第三，跟 KV cache 不匹配。LLM PagedAttention block_size=16 是为长序列设计的。推荐 sequence 100-200 用 block=16 浪费高，可调小 block 或用 contiguous。

第四，跟实时行为的 lag。User 刚发生的行为（如刚点击某 item）要快速反映到模型输入。需要实时序列服务（user behavior stream）。

### 6. 实践建议
新业务起步：SASRec + sampled softmax，复用 LLM 推理 infra（vLLM / SGLang）跑通。

效果调优：序列 cutoff（最近 N 个 + 重要性 sampling）、negative sampling 策略、cold start 处理。

跟双塔召回并行：序列推荐 + 双塔召回多路合并，DLRM 精排。

监控：序列推荐召回率 / topK 多样性 / latency / vocabulary coverage。

### 7. 30 秒速答
- SASRec 是 causal transformer 序列推荐，KV cache 形态跟 LLM 同构
- BERT4Rec bidirectional 一次 forward，更简单
- 复用 LLM 推理 infra（vLLM）可行，但 sequence 短 + vocab 大要调整
- 跟双塔召回并行作为多路 retrieval 是常见生产 setup

### 8. 自测 checklist
- [ ] 你能不能讲清 SASRec 跟 BERT4Rec 的训练目标差异？
- [ ] 你能不能解释推荐 sequence 在 PagedAttention 下的工程问题？
- [ ] 你能不能算 100M item × 64-dim output projection 的显存？
- [ ] 你能不能识别实时行为 lag 的工程方案？

## Q9. 推荐 P99 latency 工程：信息流场景 50ms 怎么达到

> 🔴 专家 · 信息流推荐请求要 50ms 出几十个 item，从 user 打开 app 到滑出第一屏整个链路都被这个 SLO 约束——每 ms 都要算计。

### 1. 核心结论
信息流推荐 P99 50ms 是工业标杆。延迟预算分配：（1）网络 + 序列化 5-10ms（client → 接入层 → 推荐服务）；（2）feature service 10-15ms（拉 user / item / context features）；（3）召回 5-10ms（多路召回合并）；（4）精排 15-20ms（DLRM 100 候选）；（5）重排 + 业务规则 5ms；（6）日志 + 后处理 5ms。每段必须 P99 满足以保证整体 P99。关键优化：feature service 高可用 + cache、模型推理 GPU/CPU 优化、精排 batch、超时 fallback。

### 2. 底层原理
P99 长尾来源 top 5：
1. **Feature service 抖动**：Redis / 自研 KV 抖动，单个 feature 拉取 100ms+。
2. **GC pause**：JVM 大 heap GC 几百 ms。
3. **网络抖动**：跨 AZ 网络偶发 100ms+。
4. **Cold cache miss**：缓存失效后第一次访问触发回源。
5. **资源争抢**：CPU / memory bandwidth 临时争抢。

防御策略：**hedged request**（同时打多个副本取最快返回）、**timeout + fallback**（超时降级到简单策略）、**多副本冗余**、**hot key cache**、**资源隔离**（cgroup / GPU MIG）。

### 3. 关键机制 / 流程 / 数据结构
第一，feature service P99。Feature store 多副本部署 + 客户端 hedged read（同时打 2 副本，取先返回的）。Hot feature 在 inference 实例本地 cache，cache miss fallback 远端。

第二，模型推理 P99。GPU 推理 P99 长尾来自 CUDA Graph cold start、kernel dispatch 抖动。预 warmup + 固定 batch size + CUDA Graph capture 解决。

第三，timeout + fallback。各阶段设 timeout（如召回 8ms / 精排 18ms）。超时时启用 fallback：候选生成超时用热门 item 兜底，精排超时用召回 score 直接排。

第四，请求级 trace。每请求生成 trace ID，跟踪各阶段时间。P99 长尾时 trace 分析定位是哪段慢。

### 4. 工程权衡 / 性能影响
冗余 vs 成本。Hedged request 每请求打 2 次后端 → 后端 QPS 翻倍。延迟降但成本翻倍。生产里只在 critical path 用。

Timeout 设置 trade-off。太紧（如 5ms）经常超时降级 → 效果差；太松（如 50ms）失去意义。生产里 timeout = P99 * 1.2 是经验值。

GPU 推理 vs CPU 推理。GPU latency 更稳（CUDA Graph 后），但 P99 长尾还是有（如多请求争 GPU）。CPU latency P50 比 GPU 高但 P99 更稳。生产里精排 GPU + 简单兜底 CPU 混合。

### 5. 常见追问 / 易错点
第一，P99 跟 P50 差距巨大。P50 20ms 但 P99 100ms 是常见。P99 优化才是重头。

第二，单点改善有限。改一个 hot spot 可能 P99 降 5ms，但其它环节抖动还是会让整体 P99 高。需要全链路优化。

第三，跟 client 体验对齐。Server P99 50ms 不等于用户 P99 50ms。Client 端可能加 100ms 网络。SLO 要拆 server / client 两段。

第四，成本约束。极致 P99（如 P99 30ms）成本是 P99 50ms 的 2-3 倍。业务定 SLO 时要 trade-off 成本。

### 6. 实践建议
新业务起步：定 P99 100ms 起，逐步收紧到 50ms。

最大长尾优化优先：先 profile 找 P99 长尾在哪段，针对性优化（feature service 是 #1 常见根因）。

监控 dashboard：各阶段 P50/P95/P99/P99.9 latency + 超时率 + fallback 触发率。

跟 SRE collaborate：infra 抖动是 P99 大头，需要 SRE 资源（专属机器 / 资源隔离 / 网络优化）。

### 7. 30 秒速答
- P99 50ms = 6 段预算严格分配（feature / 召回 / 精排 / 重排 / 网络 / 后处理）
- P99 长尾根因：feature service / GC / 网络 / cold cache / 资源争抢
- 防御：hedged / timeout + fallback / 冗余 / hot cache / 资源隔离
- 全链路 trace 是 P99 长尾定位的根本工具

### 8. 自测 checklist
- [ ] 你能不能列出 P99 长尾的 top 5 来源？
- [ ] 你能不能讲清 hedged request 的成本 vs 收益？
- [ ] 你能不能设计 timeout + fallback 的具体值？
- [ ] 你能不能识别 P99 50ms vs P99 30ms 的成本差距？

## Q10. 多目标建模与推理融合（CTR / CVR / 时长 等多 head）

> 🟡 进阶 · 一次推荐请求要同时预测点击率、转化率、停留时长、互动率等多个目标——多 head 模型是工业标配，融合方式决定业务效果。

### 1. 核心结论
推荐系统需同时优化多个业务目标：CTR（点击率）/ CVR（转化率）/ 时长（dwell time）/ 完播率（视频）/ 互动率（点赞评论）/ 留存（次日打开）。多目标建模有两类方案：（1）**多 head 共享 backbone**：底层 embedding + interaction 层共享，顶层 N 个 head 各算 N 个目标（MMoE / PLE）；（2）**多任务联合训练**：单 head 但 loss 加权融合。推理时多 head 一次 forward 出 N 个分数。重排阶段加权融合：final_score = Σ α_i × score_i。权重 α_i 业务调（季节 / 业务策略）。

### 2. 底层原理
共享 backbone 架构：features → embedding + bottom MLP → shared representation → 分叉到 N 个 head（每 head 几层 MLP）。每 head 输出对应目标的预测值。

MMoE (Multi-gate Mixture of Experts)：底层是 K 个 expert 网络，每 task head 用一个 gate 选 expert 加权和。让不同 task 用不同的底层 representation 子集。

PLE (Progressive Layered Extraction)：MMoE 改进，更精细地区分共享 expert 跟 task-specific expert。

多任务训练 loss：L = Σ w_i · L_i。Weight w_i 重要：CTR loss 数据多容易 dominate，需要 reweight 让 CVR / 长尾目标也学好。

### 3. 关键机制 / 流程 / 数据结构
第一，推理时多 head 并行。一次 forward 同时输出 N 个 score。Output shape [batch, N_targets]。

第二，重排加权融合。Final score = α·CTR + β·CVR + γ·duration + δ·interaction + ...。权重业务调。

第三，目标之间的 trade-off。CTR 高的 item 可能 dwell time 短（标题党）；CVR 高可能 CTR 不极致。多目标融合权衡这些 trade-off。

第四，目标 calibration。各 head 输出的分数可能不在同一 scale，需要 calibration（如把 CTR 输出从 logits 转概率）才能加权融合。

### 4. 工程权衡 / 性能影响
推理时延几乎不变。N head 共享 backbone，只是 head 层多几个 MLP，对总 latency 影响 <5%。

模型大小增加。每多一个 head 增加几 MB（几层 MLP）。N=5-10 个 head 总大小增几十 MB。

训练复杂度高。多目标 loss 平衡需要精心调参。某些 task 数据少容易学偏。

### 5. 常见追问 / 易错点
第一，权重不是固定的。Production 里 α/β/γ 等会按业务策略动态调（如电商大促时 CVR 权重上升）。生产 fleet 支持 runtime weight 切换。

第二，calibration 重要。两个 head 输出可能 scale 不同（CTR 0-1，duration 0-300s）。需要 normalize 才能加权。

第三，目标冲突。某些目标可能负相关（如多样性 vs 相关性）。多目标可以 mitigate 但不能完全解决。

第四，长期 vs 短期。CTR / 时长是短期指标，留存 / LTV 是长期指标。生产里长期指标更难量化和实时优化。

### 6. 实践建议
新业务起步：3-5 个 head（CTR / CVR / 时长 / 互动）共享 backbone，简单加权融合。

进阶：MMoE / PLE 让 task 表征解耦，提升多目标效果。

权重调优：A/B 测试不同权重，看哪组业务总收益最高。每月或每季度审视权重。

监控：每 head 输出分布 / calibration drift / 各目标线上业务指标。

### 7. 30 秒速答
- 多目标：CTR / CVR / 时长 / 互动 / 留存 等同时预测
- 架构：共享 backbone + N 个 head（MMoE / PLE 更优）
- 推理：一次 forward 出 N 个分数
- 融合：重排加权 final = Σ α·score_i，业务调权

### 8. 自测 checklist
- [ ] 你能不能讲清 MMoE 跟 PLE 的差异？
- [ ] 你能不能解释多目标 loss 权重的训练挑战？
- [ ] 你能不能说出多 head 推理的开销？
- [ ] 你能不能识别目标 calibration drift 的现象？

## Q11. 实时特征 vs 离线特征：Feature Store 设计

> 🔴 专家 · 用户刚买完一双鞋，5 秒后再刷信息流要不要看到鞋广告？这取决于 Feature Store 的实时性设计，是推荐 infra 最复杂的子系统之一。

### 1. 核心结论
推荐 Feature Store 是给推理服务提供 features 的子系统，按时效性分：（1）**离线特征**：T+1 计算（昨天行为的统计），从 Hive/Spark 离线 batch 生成，存到 Feature Store；（2）**近实时特征**：分钟级延迟，从 Kafka 流计算（Flink / 自研流引擎）；（3）**实时特征**：秒级延迟，从 user 行为事件直接更新（如刚点击的 item id）。Feature Store 架构：离线 batch job + 流处理 job + 在线 KV store（Redis / 自研）+ inference client SDK。代表系统：Feast、字节 ByteFS、Meta 内部 FS、Uber Michelangelo。

### 2. 底层原理
离线特征生成：Hive / Spark job 每天跑，算用户/物品的统计特征（最近 7 天点击 / CTR / 类目偏好等），写入 Feature Store 的 KV。

近实时特征：Flink job 消费 Kafka 行为流，sliding window 算 features（如最近 1 小时点击次数），定时 flush 到 KV。Latency 分钟级。

实时特征：行为事件直接同步更新 KV（如 user 刚点 item_X → 立即写"最近点击 item"特征）。Latency 秒级。

推理服务读取：Inference 请求时通过 client SDK 拉取 user/item/context features。一次拉取几十到几百个 features，可 batch RPC。

### 3. 关键机制 / 流程 / 数据结构
第一，feature schema。每个 feature 有 name + dtype + version。Schema 在 metadata service 注册，feature 计算 job 和 inference 服务通过 schema 协调。

第二，feature 存储。KV store 按 (entity_type, entity_id, feature_name) 组织。如 user_123 的 "ctr_7d" feature。可以 columnar 存（一行所有 features）或 row-based 存（每 feature 一行）。columnar 适合 batch read。

第三，feature versioning。Feature 定义改变（如统计窗口从 7 天改 30 天）创建新版本。Inference 服务可以指定 version 保证 consistency。

第4，online-offline consistency。训练时用的 features（offline 算）和推理时用的（online 读）应一致。否则训练-推理 skew 导致效果下降。Feature Store 关键责任是保证这个一致性。

### 4. 工程权衡 / 性能影响
延迟。Feature 拉取 P99 应 <15ms（占总推荐 P99 的 30%）。多 feature batch 拉取 + hot cache + 多副本冗余。

吞吐。推荐 fleet QPS 几十万-几百万，feature service 必须能扛同等 QPS。生产规模 Redis cluster 几千节点。

存储。User features 几亿用户 × 几百 features × 几 byte = 几 TB。Item features 类似规模。Total ~10TB+。

一致性。Online / offline 一致性是大坑。要严格保证 feature 定义、计算逻辑、数据源在 training 和 serving 完全一致。

### 5. 常见追问 / 易错点
第一，feature leakage。Training 时用了 inference 时拿不到的 future feature（如 "今天总点击数" 在 inference 时算不准）。会导致 offline 效果好 online 差。

第二，feature drift。Feature 分布随时间变（如 user 行为习惯季节性变化）。模型如果不及时 retrain 效果下降。

第三，cold start features。新 user / 新 item 没历史 features，需要 default 值或 content-based feature 兜底。

第四，跨业务 feature 复用。多个推荐业务共用 features（如用户基础画像）。Feature Store 应支持 sharing 避免重复计算。

### 6. 实践建议
新业务起步：用开源 Feast 跑通，避免自研。Redis 后端足够。

性能优化：feature batch RPC、client 本地 hot cache、columnar 存储。

数据治理：feature schema registry / lineage tracking / 一致性检查。

监控：feature service P99 latency / freshness（last update time）/ feature drift detection。

### 7. 30 秒速答
- 三类 feature：离线（T+1）/ 近实时（分钟）/ 实时（秒）
- Feature Store = batch job + 流处理 + 在线 KV + client SDK
- 推理 P99 中 feature 拉取占 ~30%（10-15ms）
- 关键挑战：online-offline consistency / freshness / scaling

### 8. 自测 checklist
- [ ] 你能不能讲清三类 feature 的时效差异？
- [ ] 你能不能解释 feature leakage 怎么发生？
- [ ] 你能不能说出 Feature Store 主要组件？
- [ ] 你能不能识别 online-offline skew 的现象？

## Q12. 推荐系统 fleet 调度与容量规划

> 🔴 专家 · 抖音晚高峰 QPS 是平峰 5-10x，怎么 fleet 容量规划既不浪费又不挂——推荐 fleet 调度跟 LLM serving 完全不同的问题域。

### 1. 核心结论
推荐系统 fleet 容量规划核心：（1）按 peak QPS × P99 latency / 70% utilization buffer 算实例数；（2）多 region / 多 AZ 部署冗余；（3）autoscaling 跟 traffic 波动；（4）spot / preemptible 实例混合降成本。挑战跟 LLM serving 不同：（1）QPS 比 LLM 高 1-2 个量级（百万 QPS vs 万 QPS）；（2）单请求成本极低（$0.0001 量级）；（3）latency P99 极紧（50ms）让 over-provisioning 必须；（4）feature service / DB / cache 等强耦合下游需要联动 scaling。

### 2. 底层原理
容量公式：required_instance = peak_qps × p99_latency_sec / target_utilization / per_instance_qps。例：peak 100万 QPS, P99 50ms, target util 70%, per instance 1000 QPS → 100万 × 0.05 / 0.7 / 1000 ≈ 71 instances。

Traffic pattern：信息流推荐 daily 周期（早晚高峰 vs 凌晨低峰），weekly 周期（工作日 vs 周末），seasonal（节假日大促）。

Autoscaling 策略：基于 metric（CPU / latency / QPS）触发，缩容 cool-down 5-10 分钟防抖。Scale up 必须比 scale down 快（防过载）。

混合实例。Reserved 60-70%（基础容量）+ on-demand 20%（buffer）+ spot 10-20%（成本节省）。Spot 突然回收时降级到 reserved。

### 3. 关键机制 / 流程 / 数据结构
第一，多 region 部署。一个 region 容灾。User request 路由到最近 region（GeoDNS）。Feature Store 也跨 region 复制。

第二，fleet 分层。Critical path（推荐请求）独占 fleet。Background job（特征计算 / 模型训练）单独 fleet 避免互相影响。

第三，灰度发布。新模型 / 新策略上线，先 1% canary 流量观察，逐步放量。失败立即回滚。

第四，容量监控。Real-time monitor: 当前 QPS / latency / error rate / instance util。Capacity dashboard: 当前余量 / 预计扛多大峰值。

### 4. 工程权衡 / 性能影响
Over-provisioning 成本。70% util 意味着 30% 浪费，但保 P99 必要。极致优化用 90% util 但要可靠 burst handling（容易掉链）。

Spot 风险。Spot 30-70% 便宜但随时被回收。容易批量回收造成瞬时容量缺口。混合策略 + diversified spot pool 缓解。

跨 region 一致性。模型版本 / feature 计算 / cache 等跨 region 同步。Eventual consistency 是 ok 的（推荐对秒级一致性敏感度不高）。

### 5. 常见追问 / 易错点
第一，flash crowd（瞬时高峰）。如热搜事件突发 traffic 5-10x。Autoscaling 慢，需要 over-provisioning 应对。或者 traffic shaping（限流降级）。

第二，下游依赖故障扩散。Feature Store 挂 → 推荐服务 timeout → 上游接入层 retry → 加剧故障。需要 circuit breaker / bulkhead。

第三，多模型 fleet。一个推荐服务可能跑多个模型（A/B test / 不同业务），fleet 容量按总 traffic 算，但不同模型 GPU 利用率不同。

第四，cost-per-request。算 cost 时分摊 fleet / Feature Store / network / storage 等总成本到每请求。推荐 $0.0001-0.001 / request 量级。

### 6. 实践建议
新业务起步：单 region + reserved instance + 简单 autoscaling。

中规模：多 AZ + 多 region + autoscaling + canary release。

大规模：混合 reserved/spot + 多 region + 全链路监控 + chaos engineering。

监控：QPS / latency / utilization / scaling event / cost。Dashboard 每段 P99。

### 7. 30 秒速答
- 容量公式：peak_qps × latency / util / per_instance
- Traffic pattern：daily 早晚峰 + weekly + 节假日
- 混合实例：reserved 60-70% + on-demand 20% + spot 10-20%
- Flash crowd 用 over-provisioning + traffic shaping 应对

### 8. 自测 checklist
- [ ] 你能不能算 100 万 QPS 推荐 fleet 的 instance 数？
- [ ] 你能不能讲清 spot 实例的风险与防御？
- [ ] 你能不能解释为什么推荐 P99 50ms 让 over-provisioning 必须？
- [ ] 你能不能识别 flash crowd 的应对策略？

## Q13. A/B 测试 infra 与在线评估

> 🟡 进阶 · 推荐系统的所有改动都要 A/B 测试，从模型到 UI 每天上百个 experiment 并发——A/B 测试 infra 是推荐工程的重要组成。

### 1. 核心结论
推荐 A/B 测试 infra 包括：（1）**分流系统**：按 user / device / region 等维度把 traffic 分到不同 experiment bucket；（2）**experiment 管理**：experiment 定义 / 配置 / 生命周期；（3）**指标计算**：实时和离线计算业务指标（CTR / 留存 / GMV）；（4）**统计显著性**：confidence interval / p-value 判断 winner；（5）**多层重叠**：多个 experiment 同时跑，layer 隔离防干扰。代表系统：Google Borg-experiment、Facebook PlanOut、字节 Libra、阿里 TPP。生产 fleet 同时跑几十到几百个 experiment 是常态。

### 2. 底层原理
分流机制：user_id hash + experiment_id → bucket。同一 user 在同 experiment 中始终在同一 bucket（reproducible）。Bucket 数通常 1000-10000，每 experiment 占用其中一部分（如 100 buckets = 10% traffic）。

多层重叠 (Multi-Layer Experiment)：把 experiments 分到不同 layer（如算法 layer / UI layer / 推送 layer），每 layer 内 buckets 隔离不重叠，跨 layer 自由组合。让 100 个 experiment 同时跑不互相干扰。

指标计算：每请求 log experiment_id + outcome。离线 batch job 按 experiment 聚合 metric。实时 dashboard 用流处理近实时算。

统计显著性：t-test / chi-square 等检验。生产里 sequential testing（实时监控边算边判断）比 fixed-horizon 更高效。

### 3. 关键机制 / 流程 / 数据结构
第一，experiment 配置。每个 experiment 配置：name / description / start/end date / buckets / treatment vs control / metrics / 责任人。配置改了立即生效，不用重启服务。

第二，分流 SDK。Inference 服务用 SDK：bucket_id = hash(user_id, experiment_id) % total_buckets；查 experiment config → 决定走哪个 treatment。

第三，metrics 计算 pipeline。每请求 log 到 Kafka → Flink 实时聚合 → 离线 Hive 详细分析 → dashboard 展示。

第四，多层 + 互斥。Layers: 召回 / 精排 / 重排 / UI / 推送 等。每 layer 内 buckets 互斥（一个 user 在 layer A 只在一个 bucket），跨 layer 自由组合。

### 4. 工程权衡 / 性能影响
分流系统性能。每请求做 hash + 查 config，几微秒。Negligible。

experiment 数量限制。多 layer 设计让理论上无限 experiment 并发。生产里几百个 experiment 同时跑是常态。

数据量。每 experiment 几亿 user × 几十 events / day = TB 级日志。Kafka + S3 + Hive 存储 + 处理。

统计能力。指标算出来还要业务人员看懂。Dashboard / alerting / 周报机制配套。

### 5. 常见追问 / 易错点
第一，sample size 不够。新 experiment 流量太少，统计不显著。需要 power analysis 预算需要多少 traffic 多少时间。

第二，simpson's paradox。整体 metric 改善但分群体（如新用户 vs 老用户）相反。分群体看必要。

第三，novelty effect。新功能上线初期用户好奇心驱动 metric 高估。要跑足够长（2 周+）确认效应稳定。

第四，跨 experiment 干扰。多层设计应避免，但 metric 互相关联时仍可能（如召回改善影响精排 metric）。

### 6. 实践建议
新业务起步：用开源 PlanOut 类系统跑通，或自研轻量分流系统。

experiment 流程：propose → review → launch 1% → monitor → ramp to 10% / 50% / 100%。

数据 quality：log完整性 / sample balance / 统计 power 提前检查。

monitor：experiment dashboard 实时显示各 metric / p-value / confidence interval。异常预警。

### 7. 30 秒速答
- 分流：hash(user, experiment) % buckets，多层 layer 隔离防干扰
- 指标：实时流处理 + 离线 batch 双轨
- 统计：t-test / sequential testing 判断 winner
- 生产几十到几百个 experiment 并发是常态

### 8. 自测 checklist
- [ ] 你能不能讲清多层 layer 设计怎么防 experiment 干扰？
- [ ] 你能不能解释 power analysis 的必要性？
- [ ] 你能不能识别 Simpson's paradox 的现象？
- [ ] 你能不能设计一个新 experiment 的完整 lifecycle？

## Q14. 推荐冷启动：新用户 / 新物品 / 新业务

> 🟡 进阶 · 新用户没行为没画像，新物品没曝光没数据——冷启动是推荐 system 的永恒挑战，要在 0 信号下做出合理推荐。

### 1. 核心结论
推荐冷启动分三种：（1）**新用户冷启动**：刚注册 user 没行为；（2）**新物品冷启动**：刚上线 item 没曝光；（3）**新业务冷启动**：新场景没历史数据。常用策略：基于 content / 画像的 fallback model、热门 item bootstrap、explicit exploration、迁移学习、人工运营干预。冷启动是推荐效果的重要 floor，新 user 留存 / 新 item 长尾分布都靠它。

### 2. 底层原理
新 user 冷启动：没有行为序列、historical features 都缺。解决：（1）注册时收集 explicit preference（兴趣选择）；（2）人口统计 features（年龄 / 性别 / 地域）+ content-based 推荐；（3）相似 user lookalike；（4）默认热门推荐 + exploration 探索兴趣。

新物品冷启动：item 没曝光数据，embedding 没训出来。解决：（1）content-based item embedding（标题 / 类目 / 图像）；（2）作者历史 item embedding 平均；（3）explicit boost（运营人工选）；（4）exploration（强制曝光一段时间收集数据）。

新业务冷启动：从 0 build。解决：（1）从相关业务迁移 model + features；（2）人工运营策略 + 简单 rule-based；（3）逐步收集数据 train basic model。

### 3. 关键机制 / 流程 / 数据结构
第一，content-based embedding。Item 内容（标题 / 标签 / 图像）→ pre-trained encoder（BERT / CLIP）→ embedding。新 item 一上线就有 embedding 进入召回池。

第二，exploration strategies。Epsilon-greedy（1-5% 流量随机探索）/ UCB (Upper Confidence Bound) / Thompson sampling。让新 item 有曝光机会。

第三，lookalike for new users。新 user 看注册信息找相似老 user，复用老 user 的 embedding 或推荐结果。

第四，冷启动 funnel。新用户 day 1 / day 7 / day 30 留存。每个阶段不同策略：day 1 偏 broad popular / day 7 collect preference / day 30 personalized。

### 4. 工程权衡 / 性能影响
冷启动跟主推荐 trade-off。Exploration（强制曝光新 item）降低短期 CTR 但提升长期生态健康。需平衡。

content-based 质量依赖 encoder。CLIP 类多模态 encoder 给 item embedding 质量影响很大。生产 fleet 一般用专门 fine-tune 的版本。

工程复杂度。Cold start 增加多条 path（content-based / exploration / lookalike），调试和监控复杂。

### 5. 常见追问 / 易错点
第一，filter bubble。Personalized 推荐让 user 永远只看到熟悉内容，新兴趣发现不了。Exploration 是破解 filter bubble 的关键。

第二，曝光偏差累积。冷启动差 → user 体验差 → 离开 → 数据更少 → 冷启动更差。死循环。Day 1 体验关键。

第三，新业务初期数据量太少难以训出好模型。早期靠运营策略 + 简单规则，模型 train 出来要等几周。

第四，跟个性化的 tension。冷启动 fallback 是 non-personalized，会让早期推荐"通用"，影响体验。需 graceful 过渡。

### 6. 实践建议
新业务起步：构建 content-based encoder + exploration mechanism + 热门兜底。

新用户 day 1 体验：注册时引导 explicit preference + 简单 demographic-based + popular。

新物品上线：content embedding + exploration boost（1 周内 1-2% 流量曝光）。

监控：新 user 7 天留存 / 新 item 7 天 CTR / exploration 流量占比。

### 7. 30 秒速答
- 三种冷启动：新 user / 新 item / 新业务
- 新 user：explicit preference + demographic + lookalike
- 新 item：content embedding + exploration boost
- exploration 是平衡短期 vs 长期的关键机制

### 8. 自测 checklist
- [ ] 你能不能讲清 epsilon-greedy 跟 UCB exploration 的差异？
- [ ] 你能不能解释 content-based embedding 在冷启动的关键作用？
- [ ] 你能不能设计新 user day 1 的推荐策略？
- [ ] 你能不能识别 filter bubble 的工程根因？

## Q15. 推荐推理服务架构：单体 vs 微服务

> 🔴 专家 · 推荐服务从单进程到几十个微服务，每个 hop 增加 latency 但带来弹性——架构选择决定整体系统的 SLO 上限和迭代速度。

### 1. 核心结论
推荐推理服务架构演进：（1）**单体架构**：所有逻辑（feature / 召回 / 精排 / 重排）在一个进程，简单但难 scale 和迭代；（2）**模型即服务**：每个模型独立服务（feature service / retrieval service / ranking service），通过 RPC 串联；（3）**精细化微服务**：每个阶段甚至子阶段独立服务，几十个服务联动。Trade-off：微服务化让团队独立迭代 + 故障隔离，但每 hop 几 ms RPC latency 和复杂度上升。生产里中大规模推荐都是微服务架构，但具体粒度按团队规模和迭代节奏决定。

### 2. 底层原理
单体架构：一个 binary 进程包含 feature 拉取 + 召回 + 精排 + 重排所有逻辑。优点：函数调用无 RPC overhead、debug 简单、deploy 简单。缺点：一次改动需要全量重新 deploy、team 之间 conflict、单点故障。

模型即服务：每个模型独立服务，inference 服务调用它们。优点：模型独立迭代 + 独立 scaling。缺点：每个模型 service 增加几 ms RPC。

精细化微服务：召回 / 精排 / 重排 / feature 都是独立服务，可能各自又拆。优点：极致灵活和 scale。缺点：复杂度爆炸、debug 难、latency 累积。

### 3. 关键机制 / 流程 / 数据结构
第一，RPC 框架选择。gRPC / 自研 thrift 类。RPC overhead 1-3ms（包括序列化 + 网络 + 反序列化）。每多一 hop 累加。

第二，服务发现。Consul / etcd / 自研注册中心。服务实例 register 上线，client 通过服务名找到 endpoint。

第三，service mesh。Istio / Linkerd 等让流量管理（路由 / 熔断 / 限流 / 重试）standardize。推荐生产部分用 mesh 部分自研。

第四，链路追踪。Jaeger / Zipkin / 自研。每请求 trace ID 串联各服务，定位 P99 长尾。

### 4. 工程权衡 / 性能影响
延迟。单体 0 hop，所有调用 in-process。微服务每 hop 1-3ms。50ms 预算下 10 hop 占 10-30ms 是上限。

吞吐 / scale。单体只能整体 scale，资源浪费。微服务各自独立 scale，资源高效但 capacity planning 复杂。

迭代速度。单体 ：team conflict 让 deploy 慢。微服务：各自迭代但需要 API contract 管理。

故障隔离。单体一挂全挂。微服务故障可以局部影响 + degradation。

### 5. 常见追问 / 易错点
第一，过早微服务化。小规模业务过早拆微服务（每个 service QPS 几百），运维成本远超收益。建议 user 量起来后再拆。

第二，cascading failure。一个 service 挂导致依赖它的全挂。Circuit breaker / bulkhead / fallback 必备。

第三，distributed tracing 复杂度。100+ services 互相调用，trace 数据量巨大。需要专门 tracing 团队。

第四，跨服务数据 consistency。各服务各自 cache，可能数据不一致。需要 cache invalidation 协调。

### 6. 实践建议
新业务起步：单体架构跑通 product market fit。

中规模：拆出"模型即服务"（ranking model / retrieval / feature service）三大块。

大规模：精细化微服务，但配套 mesh / tracing / chaos engineering。

技术债意识：每多拆一个 service 增加运维成本，权衡好处和成本。

### 7. 30 秒速答
- 架构演进：单体 → 模型即服务 → 精细化微服务
- 单体简单但难 scale；微服务灵活但复杂
- 每 RPC hop ~1-3ms，50ms 预算下 10 hop 是上限
- 中大规模业务必须微服务，但粒度按团队规模决定

### 8. 自测 checklist
- [ ] 你能不能讲清单体 vs 微服务在 latency 上的差异？
- [ ] 你能不能解释 service mesh 的核心价值？
- [ ] 你能不能识别过早微服务化的风险？
- [ ] 你能不能设计 distributed tracing 关键 metric？

## Q16. 推荐系统跟广告 / 搜索的工程相似与差异

> 🟡 进阶 · 推荐 / 广告 / 搜索三个系统在工程上 80% 重叠——理解它们的差异和共性，让你能在不同业务间快速迁移。

### 1. 核心结论
推荐 / 广告 / 搜索系统在 infra 上高度相似（80%+ 共享：feature store / 多阶段 ranking / A/B 测试 / fleet 调度），但业务约束有 key 差异：（1）**搜索**：query 是 explicit 信号，relevance 是核心 metric；（2）**推荐**：无 query 主动推送，engagement metric（CTR/时长）核心；（3）**广告**：钱货关联，bid + relevance + budget 联合优化，CPM/CPC 是计费单位。三者底层 model 架构很像（DLRM / Transformer），但 ranking 目标不同。生产里很多公司这三个团队 share infra（feature / model platform / experiment infra）。

### 2. 底层原理
搜索特点：query → 召回相关 doc → ranking。Relevance 是首要 metric（BM25 / dense retrieval 都用）。User 主动表达 intent，匹配相对容易。

推荐特点：无 query，根据 user history 推 item。Engagement 是首要 metric。User intent 隐含，需要从行为推断。

广告特点：跟搜索 / 推荐场景类似但有 bid 和 budget。Ranking 公式 = bid × CTR × relevance × budget_smoothing。CPM 计费按曝光，CPC 按点击。

技术栈共享：feature store / DLRM-style ranking model / 双塔召回 / ANN 检索 / A/B 测试 / fleet 调度。

### 3. 关键机制 / 流程 / 数据结构
第一，搜索的 query understanding。Query → 分词 / 实体识别 / 意图分类 / query expansion，比纯推荐多这一段。

第二，推荐的多场景。同一推荐系统支持信息流 / 详情页 / push 等多场景，每场景 user intent / 时延要求不同。

第三，广告的拍卖机制。GSP (Generalized Second Price) 拍卖最常见。Ranking 后按 bid 排，但实际收 second-price。

第四，跟搜索引擎深度结合。淘宝 / 京东等电商既是搜索也是推荐，user 进入有 query 是搜索、无 query 是推荐。同一底层 infra。

### 4. 工程权衡 / 性能影响
延迟约束：搜索 P99 100-200ms 较松（user 等结果耐心高）；推荐信息流 50ms；广告随场景。

ranking 模型复杂度：搜索 ranking 相对简单（relevance + 少量 personalization）；推荐 ranking 复杂（多目标 + 个性化）；广告最复杂（multi-objective + bid + budget）。

冷启动差异：搜索新 query 是常态（多数 query 是 long-tail），系统天然要处理。推荐新 user / 新 item 都是冷启动问题。广告新 ad / 新广告主类似。

商业模式：搜索靠广告变现；推荐靠 engagement → ads / 电商；广告直接卖位置。

### 5. 常见追问 / 易错点
第一，team organization。大公司 search / rec / ads 是独立 BU，但底层 infra 团队（model platform / feature store）共享。

第二，metric calibration。搜索 NDCG / 推荐 CTR / 广告 RPM，metric 不同但 underlying ranking 模型可能同架构。

第三，跨业务知识迁移。从搜索转推荐 ramp-up ~1 月（learn 召回 + 精排区别）。从推荐转广告 ramp-up ~1 月（learn bid + auction）。

第四，infra reuse 实际度。理想 100% reuse 实际 60-70%。Search 有 query understanding，ads 有 auction service，rec 有 sequence model，各自有特殊组件。

### 6. 实践建议
跨业务迁移：先理解 metric 差异（不同 north star）+ 业务约束差异（query / bid / budget）。

infra 复用：feature store / model platform / experiment infra 优先共享。

团队 collaboration：跨 search / rec / ads team 定期 sync，技术 best practices 共享。

招聘：跨业务背景候选人有价值，能 cross-pollinate 经验。

### 7. 30 秒速答
- 三系统 infra 80% 共享，业务约束 20% 差异
- 搜索 query-driven relevance；推荐 history-driven engagement；广告 bid-driven 钱
- 底层 model 架构（DLRM / Transformer）几乎通用
- 大公司组织上独立 BU 但底层 infra team 共享

### 8. 自测 checklist
- [ ] 你能不能讲清三系统的核心 metric 差异？
- [ ] 你能不能解释 GSP 拍卖的工程实现？
- [ ] 你能不能识别 infra reuse 的实际边界？
- [ ] 你能不能设计跨业务迁移的学习路径？

## Q17. 推荐 / LLM / 图像生成三大推理范式对照

> 🧭 综合 · 三种推理范式对应三类不同的工程模型——理解差异让 cross-domain 工程师能在不同领域间灵活切换。

### 1. 核心结论
推荐 / LLM / 图像生成是三种根本不同的推理范式：（1）**推荐 (DLRM/GR)**：embedding lookup + ranking，IO bound，P99 50ms，QPS 百万级；（2）**LLM (autoregressive)**：KV cache + token by token，memory bound（decode），P99 几秒，QPS 万级；（3）**图像生成 (diffusion)**：multi-step iterative denoise，compute bound，P99 几秒到几分钟，QPS 千级。三者的 fleet 架构 / 调度 / 单位经济学完全不同，但底层组件（GPU / KV cache / scheduler）有重叠。理解三种范式让工程师能在不同业务间快速迁移。

### 2. 底层原理
推荐推理特征：
- 计算模式：embedding lookup + MLP，单请求几 ms
- 数据流：features → embedding → interaction → multi-head output
- 瓶颈：embedding lookup（IO/memory）+ feature service（IO）
- 单位经济：$0.0001-0.001/request

LLM 推理特征：
- 计算模式：prefill + autoregressive decode，sequence × layers
- 数据流：tokens → embedding → N layers → next token
- 瓶颈：decode 阶段 HBM bandwidth（KV cache read）
- 单位经济：$0.5-2 / 1M tokens

图像生成推理特征：
- 计算模式：multi-step iterative denoising，N step × Unet/DiT
- 数据流：text + noise latent → N step UNet → VAE decode
- 瓶颈：Unet/DiT compute（compute bound）
- 单位经济：$0.001-0.05 / image

### 3. 关键机制 / 流程 / 数据结构
推荐架构：feature service + retrieval service + ranking service + reranker。
LLM 架构：scheduler + worker（prefill + decode）+ KV cache pool。
图像架构：scheduler + UNet/DiT + VAE，无 KV cache。

调度差异：
- 推荐：static batch（按 latency 收 batch）
- LLM：continuous batching（动态加入退出）
- 图像：static batch（每 step 内固定）

容量规划：
- 推荐：peak QPS × P99 latency
- LLM：tokens/s × peak tokens/s
- 图像：imgs/s × peak imgs/s

### 4. 工程权衡 / 性能影响
跨域迁移成本：LLM 工程师转推荐 ~3 月（学 embedding / multi-stage ranking）；LLM 转图像 ~2-4 周（学 diffusion）；推荐转 LLM ~2 月（学 KV cache / autoregressive）。

infra 复用：feature store / GPU fleet / monitoring / A/B test infra 三者都需要，但具体形态不同。

招聘 implications：跨域候选人有价值。"既懂推荐又懂 LLM"工程师比单一方向稀缺。

### 5. 常见追问 / 易错点
第一，多模态推理处于交叉位置。VLM 兼有 LLM 和 image encoder。Recommendation with multimodal items（如视频推荐）兼有推荐和图像。这些"杂交"业务越来越多。

第二，工具复用边界。vLLM 不能直接跑推荐或图像。diffusers 不能直接跑推荐。Tool 是范式 specific 的。

第三，性能 metric 不可互比。LLM 跟图像不能直接比 "速度"（一个是 tokens/s 一个是 imgs/s）。需要单位经济（$/output）才可比。

第四，团队组织。大公司三个范式可能独立团队，infra 部分共享（model platform / GPU fleet）。

### 6. 实践建议
跨域学习：先专精一个范式 1 年，再扩展到第二个。同时学三个浅。

工具栈：vLLM / SGLang for LLM；diffusers / ComfyUI for image；TorchRec / 自研 for 推荐。

业务对接：明确业务属于哪种范式，technical stack 完全不同。

招聘标准：senior 工程师应至少深入一个范式，且能讲清跟另两个范式的工程差异。

### 7. 30 秒速答
- 推荐：embedding 主导 / IO bound / P99 50ms / QPS 百万
- LLM：autoregressive + KV cache / memory bound / P99 几秒 / QPS 万
- 图像：multi-step diffusion / compute bound / P99 几秒-分钟 / QPS 千
- 工具栈完全不同，infra 部分（fleet / monitoring）可共享

### 8. 自测 checklist
- [ ] 你能不能列出三种范式各自的瓶颈类型？
- [ ] 你能不能讲清调度策略差异（static / continuous / static iterative）？
- [ ] 你能不能算三种范式的单位经济密度？
- [ ] 你能不能识别多模态业务的"杂交"位置？

## Q18. 比较题：DLRM vs HSTU vs TIGER 三大推荐架构

> 🧭 综合 · 同样是"推荐模型"，三种架构在 model / data / infra / scaling 上完全不同——理解差异让架构选型有据可依。

### 1. 核心结论
DLRM / HSTU / TIGER 代表三种推荐架构范式：DLRM 是 sparse features + MLP，工业最成熟；HSTU 是 Transformer 序列推荐 + 多头多目标，Meta 提出的 GR 代表；TIGER 是 Transformer + semantic ID，Google 提出的 generative retrieval 代表。三者在数据需求 / 模型 size / 推理工程 / scaling 性质上有显著差异。生产选型：DLRM 适合成熟业务 + 大特征工程 + 工程团队成熟；HSTU 适合序列信号丰富业务 + 团队能 invest GR；TIGER 适合 retrieval-heavy 场景 + 需要 generative 灵活性。

### 2. 底层原理
DLRM 范式：input = sparse features + dense features → embedding lookup + bottom MLP → feature interaction (FM/DCN) → top MLP → multi-head output。Strength：工业成熟、大 vocabulary 特征丰富、ranking 模型质量高。

HSTU 范式：input = user behavior sequence + features → HSTU blocks (self-attn + pointwise gating) → logits over items → topK。Strength：序列建模强、long-tail 召回好、统一模型替代多模型组合。Weakness：vocabulary 大处理复杂、工程不成熟。

TIGER 范式：item → RQ-VAE encoder → semantic ID (codebook indices) → transformer generates semantic ID sequence → decode to item。Strength：semantic ID 让 vocabulary 大幅减小、可学性强、cold start 友好。Weakness：semantic ID 训练稳定性、跟 ranking 阶段如何配合。

### 3. 关键机制 / 流程 / 数据结构
DLRM 数据流：features → embedding tables (TB 级) → MLP (MB 级) → multi-head。模型部署需要大内存机器或分布式 embedding。

HSTU 数据流：user 行为序列 → embedding (item vocab) → HSTU layers → output logits over all items。推理类似 LLM。

TIGER 数据流：user query → transformer → 生成 [c1, c2, c3, c4] semantic ID → 查 RQ-VAE codebook 还原 item。推理是 autoregressive 但 sequence 很短（4-8 step）。

跟现有 infra 复用：DLRM 复用 TorchRec / DeepCTR；HSTU 复用 vLLM-GR；TIGER 复用 vLLM / 自研。

### 4. 工程权衡 / 性能影响
模型 size。DLRM 总参数主要在 embedding（TB 级），MLP 小（MB 级）。HSTU/TIGER 主要在 transformer（GB 级）+ item embedding（GB 级）。

延迟。DLRM 推理 P99 ~30-50ms（embedding lookup + MLP）。HSTU/TIGER 因 autoregressive decode 可能慢 1-3x（取决于生成 K item）。

scaling law。DLRM 没有清晰 scaling law（特征工程驱动）。HSTU/TIGER 类似 LLM scaling law（model + data 增益相对可预测）。

工程成熟度。DLRM 工业 10 年沉淀。HSTU 2024 兴起。TIGER 2023 提出。

### 5. 常见追问 / 易错点
第一，DLRM 不会消失。多数业务（中等规模、特征丰富、效果稳定）DLRM 仍是最优选择。GR 是补充不是替代。

第二，HSTU 跟 TIGER 不直接对比。HSTU 是端到端 ranking 模型，TIGER 是 generative retrieval 模型。不在同一阶段。

第三，semantic ID 的 codebook drift。RQ-VAE 重训 codebook 变，TIGER 输出 ID 漂移，下游数据失效。工程稳定性是 active research。

第四，跟 cold start。HSTU/TIGER 跟 DLRM 一样面临冷启动，没消除这个问题。

### 6. 实践建议
保守业务：DLRM 起步 + 双塔召回 + 多目标。技术债低。

激进业务：HSTU 或 TIGER 探索性 pipeline 跟 DLRM 并行，A/B 测试看长期收益。

学习路径：DLRM 是必修课。HSTU/TIGER 选一深入，跟踪学术进展。

选型矩阵：业务成熟度 / 数据规模 / 团队能力 / latency 约束 → 选型。

### 7. 30 秒速答
- DLRM：sparse features + MLP，工业成熟，TB embedding
- HSTU：Transformer 序列 + 多 head，Meta GR 代表
- TIGER：Transformer + semantic ID，Google generative retrieval
- 选型：保守 DLRM；激进 GR；不互相替代

### 8. 自测 checklist
- [ ] 你能不能讲清三者在模型 size 上的差异？
- [ ] 你能不能解释 TIGER semantic ID 怎么减 vocabulary？
- [ ] 你能不能说出 HSTU 跟 TIGER 是否在同一阶段？
- [ ] 你能不能为新业务选型给出 trade-off 分析？

## Q19. 场景题：从 0 设计一个新闻推荐系统的推理服务

> 🧭 综合 · 给你 1000 万 DAU、信息流场景、P99 50ms 约束 —— 怎么设计推理服务从 fleet 大小到模型选型到 feature store 的完整方案。

### 1. 核心结论
1000 万 DAU 信息流新闻推荐系统设计：（1）**Fleet 容量**：peak QPS ~10 万（每 user 10 req/min × 0.01 同时在线），需 ~200 服务实例；（2）**架构**：feature service + 多路召回（双塔 + 协同过滤 + 标签 + 热门）+ DLRM 精排（100 候选）+ 重排（业务规则 + 多目标融合）；（3）**Feature Store**：Redis cluster ~10 实例，存 user/item/context features，P99 <15ms；（4）**模型**：双塔 user/item 各 64-dim + DLRM 精排 ~500MB；（5）**冷启动**：content-based item embedding + popular fallback + exploration 5%；（6）**A/B 测试**：多层 layer，100+ experiments 并发。

### 2. 底层原理
容量计算。1000 万 DAU × 平均会话长度 30 分钟 / 天 × 平均刷 100 条 / 会话 = 10 亿 req / 天。Peak QPS（晚高峰 5x 平均）= 10亿/86400 × 5 ≈ 6 万。Buffer 70% util → fleet 100 instances（每 instance 1000 QPS）。

延迟预算（50ms 总）。Feature service 15ms / 召回 10ms / 精排 15ms / 重排 5ms / 网络 + 序列化 5ms。每段 P99 严格控制。

模型选择。新闻推荐序列信号弱（user 兴趣分散），DLRM + 双塔召回足够。HSTU 可作长期探索方向。

存储预算。1 亿 user × 几十 features + 1000 万 news × 几十 features ≈ 1-2 TB feature data。Redis cluster sharded across ~50 nodes。

### 3. 关键机制 / 流程 / 数据结构
第一，整体架构。Client → API gateway → recommendation service → (feature service / retrieval service / ranking service / rerank service) → response。

第二，模型工作流。Retrieval service: 双塔（5000）+ 协同过滤（1000）+ 热门（500）+ 标签（500）= 多路合并去重 ~5000 → 截取 1000 给 ranking。Ranking service: DLRM forward 1000 候选 → 精排 score → top 200。Rerank: 多目标融合 + 业务规则 → top 30。

第三，Feature Store 设计。User features（demographics / 兴趣画像 / 行为序列）+ Item features（新闻标题 embedding / 类目 / 时效性 / 热度）+ Context features（时间 / 设备 / 网络）。Redis cluster 多副本读 + 写。

第四，A/B 测试 infra。多 layer experiment（算法 / UI / push）+ 中心化 experiment service。每 user 路由到对应 buckets。

### 4. 工程权衡 / 性能影响
fleet 配置。100 instances × 32 CPU 64GB（混合 CPU 跑 DLRM 推理 + 缓存）or 50 instances × 8x A10 GPU。CPU 方案性价比高 for DLRM。

新闻特性。新闻时效性强（小时级过期）。Item embedding 需要小时级更新。Feature update pipeline 高频。

挑战。冷启动新闻多（每天新发万级）。Content-based embedding + 强制 exploration（每 list 1-2 条新新闻）。

成本估算。100 服务器 × $500/月 + Redis cluster × $5000/月 + GPU fleet（如果用）+ 网络 + storage ≈ $80-200k / month。Per request ~$0.0001。

### 5. 常见追问 / 易错点
第一，新闻时效性 vs 用户兴趣稳定。User 长期兴趣画像变化慢，新闻时效短。需要兴趣画像（slow）+ 实时点击（fast）混合 features。

第二，多模态新闻。新闻含标题 + 图片 + 视频。Multi-modal embedding 提升推荐质量但增加 infra 复杂度。

第三，反作弊。Click farm / bot 行为污染数据 + 推荐结果。需要 anti-fraud 系统。

第四，编辑干预。新闻业务需要编辑手动 boost 重要新闻或 demote 低质新闻。Re-ranking 阶段加 editorial layer。

### 6. 实践建议
phase 1（0-3 月）：MVP fleet ~20 instances + 简单 DLRM + Redis FS + 基础 A/B test。

phase 2（3-6 月）：scale fleet + 多路召回 + 多目标精排 + 冷启动优化。

phase 3（6-12 月）：精细化微服务 + 多层 A/B + 多模态 + HSTU 探索。

监控：QPS / P99 latency / cache hit / 各 service health / 业务 metric（CTR / 时长 / 留存）。

### 7. 30 秒速答
- 1000 万 DAU → peak 6 万 QPS → fleet ~100 instances
- 架构：FS + 多路召回（双塔+CF+热门）+ DLRM 精排 + 重排
- Feature Store: Redis cluster sharded ~50 nodes, 1-2TB
- 新闻特性：时效性强，冷启动多，需要 explicit exploration

### 8. 自测 checklist
- [ ] 你能不能算 1000 万 DAU 的 peak QPS？
- [ ] 你能不能拆分 50ms 延迟预算到各阶段？
- [ ] 你能不能设计新闻冷启动策略？
- [ ] 你能不能预算这个服务的月成本？

## Q20. 估算题：推荐 fleet 单 request 成本核算

> 🧭 综合 · LLM 算 tokens/$, 图像算 imgs/$, 推荐算 requests/$ ——单 request 成本 $0.0001-0.001 量级，跟 LLM/图像差几个数量级。

### 1. 核心结论
推荐 fleet 单 request 成本核算公式：(fleet GPU/CPU 成本 + feature service 成本 + storage + network) / 总 requests。典型值：成熟推荐系统单 request $0.0001-0.001。占比：fleet compute 50-60%；feature service 20-30%；storage + network 10-20%；其它 5-10%。1000 万 DAU 推荐服务月 $80-200k cost / ~30 亿 requests = $0.00003-0.00007 / request。比 LLM 单 request 便宜 1000-10000x（因为单请求耗时短 + 模型小）。

### 2. 底层原理
成本组成：
- Fleet compute: 实例费 + GPU 时间。CPU instance $0.1-0.5/h / GPU instance $1-3/h。
- Feature service: Redis / 自研 KV cluster 实例费 + 跨 AZ 网络。
- Storage: embedding tables / feature data / logs，S3 / 自建。
- Network: cross-region / cross-AZ 网络出向 ($0.01-0.1/GB)。
- Operations: monitoring / on-call / oncall infra。

单 request 时间。DLRM 推荐请求 ~10-30ms compute time。CPU instance $0.2/h × 0.02s / 3600 = $0.000001 / request。加上 feature / storage / network 约 $0.00005-0.0005。

跟 LLM 比。LLM 70B serving 单 token ~ms 计算 + KV cache 等，$0.5-2 / 1M tokens ≈ $0.005-0.02 / 1000 token response ≈ $0.005-0.02 / request。

跟图像比。SDXL 单图 $0.001-0.05 / image。

### 3. 关键机制 / 流程 / 数据结构
第一，监控成本 dashboard。Per-day cost / per-request cost / 各 component breakdown。Anomaly detect cost spike。

第二，cost attribution。按业务 line / 推荐 channel / experiment 分摊 cost。每业务知道自己 cost contribution。

第三，cost optimization。Spot/reserved 混合 / 推荐模型量化 INT8 / cache 命中率提升 / 长期不用 feature 下线。

第四，commercial 模型。广告业务直接看 RPM (revenue per mille)；推荐业务靠下游变现（电商 / 广告 / 订阅）。Cost / request 是 baseline，需要 revenue / request 才完整。

### 4. 工程权衡 / 性能影响
精度 vs 成本。More features / heavier model = better metric but higher cost。每多 1ms latency / 1MB feature 都要算 ROI。

cost cliff。某些优化点跨过去（如 fleet 实例 N → N+1）成本阶梯式增加。Aware of cliff 防意外超预算。

新业务 ramp up cost。新业务初期 fleet 小但 baseline cost（如 Redis cluster minimum size）高。Unit cost 暂时高。

### 5. 常见追问 / 易错点
第一，long-tail vs 头部业务。头部业务高 QPS 摊薄 baseline cost。Long-tail 业务 per-request cost 高（baseline 没摊开）。

第二，dev / staging fleet 也算钱。生产 only 60-70% 总 cost，dev / staging / experiment 也是 cost。Total cost 比直觉高。

第三，data egress fee。Cross-region 数据传输 $0.02-0.09/GB。Global fleet 网络成本不容忽视。

第四，跨业务 reuse 摊薄。Feature service / model platform 等共享 infra 跨业务摊销 cost，但内部 chargeback 模型要 fair。

### 6. 实践建议
新业务起步：粗略算 cost / request，business plan 用。

精细化：每月 cost breakdown by service / business line。Cost spike alert。

优化：fleet rightsizing / spot 利用 / cache 命中率 / 长期不用 feature 下线。

cross-team collaboration: cost optimization 跟 SRE / finance 联合规划。

### 7. 30 秒速答
- 单 request $0.0001-0.001 量级
- 成本组成：fleet compute 50-60% + feature service 20-30% + storage / network 10-20%
- 比 LLM 便宜 1000-10000x（单请求耗时短 + 模型小）
- 监控 + cost attribution + spot 混合 是 cost 优化三件套

### 8. 自测 checklist
- [ ] 你能不能算 1000 万 DAU 推荐服务的月 cost？
- [ ] 你能不能拆分 cost / request 到各 component？
- [ ] 你能不能对比推荐 / LLM / 图像的 unit cost 差异？
- [ ] 你能不能设计 cost monitoring dashboard 的关键 metric？
