# 战略与工程决策

## 主题边界

本卷聚焦把 ML 系统能力转化为工程产出的"战略层"问题：云与硬件采购、容量与峰值规划、技术债与路线图决策、ML 平台团队与工程文化、单位经济学。每题以 6 节模板组织：核心结论 / 底层原理 / 关键机制 / 工程权衡 / 常见追问 / 实践建议。

## Q1. 公有云 vs 自建 GPU 集群：在 2025–2026 的算力市场怎么选？

> 🔴 专家 · CFO 拍着桌子问"自建是不是更便宜"，工程负责人这时候回答错了，团队就要被 18 个月的电力申请和 IB 调试拖死。算清盈亏点 `~75% 利用率` 与 24 个月需求曲线，是这个决策的硬门槛。

### 1. 核心结论
做最终决策的不是"哪种便宜"，而是 **峰值规模、确定性周期、毛利目标** 三者的组合。一般经验：年消耗 < 5000 GPU-月、增长曲线不明朗、现金流紧 → 留在公有云；年消耗 > 20000 GPU-月、需求曲线已经明朗 18 个月以上、并且能拿到电力与机房 → 自建或长租。介于中间则用"公有云 reserved + 第三方 neocloud（CoreWeave/Lambda）spot"的混合形态。

### 2. 底层原理
GPU 算力的成本结构由三块构成：硬件折旧（按 3–4 年线性折旧 H100/H200，B200 通常按 4–5 年）、电力与机房（每 kW 年化 $1k–$2k 含 PUE）、运维与软件人力。公有云在这三块都加了 30–60% 毛利与弹性溢价。自建省下的是溢价，但要承担**利用率风险**：自建 H100 节点的盈亏点通常在 65%–75% 利用率，低于这个数自建更贵。

### 3. 关键机制
- **公有云按需价**：H100 80GB SXM 在 AWS p5/p5e、GCP A3、Azure ND H100 v5 上 2026 年现价 $4–$6/GPU-hour 区间；H200 略高 5–10%；B200 上线初期 $7–$10/GPU-hour。
- **公有云 1Y/3Y reserved / Savings Plan**：相对 on-demand 折扣 30%–55%，但要承担前付与跨代风险。
- **第三方 neocloud（CoreWeave、Lambda、Crusoe、Nebius）**：H100 报价 $2.0–$3.0/GPU-hour，长租可压到 $1.8–$2.5；牺牲一部分多区域、合规、托管服务深度。
- **自建**：H100 整机 8 卡 SXM5 节点 2026 年现货 $250k–$320k，加 InfiniBand HDR/NDR 网络与液冷，每卡全成本 $1.4–$1.8/hour（按 3 年折旧 + 75% 利用率）。
- **盈亏点反推**：自建年化全成本 ≈ 硬件 / 3 + 电力 / 年 + 运维人力 / 年。以 1024 卡集群算，3 年 TCO 约 $40M–$55M；要至少 75% 利用率 + 24 个月稳定需求才能跑赢 reserved 云。

### 4. 工程权衡
公有云的真正价值不是单价，而是**第二天就能扩到 1024 卡**与多区域容灾。自建的真正成本不是采购价，而是 18–36 个月的爬坡期：电力申请、机柜布线、IB 调试、NCCL/topology 调优、运维值班体系。介于两者之间的 neocloud 报价漂亮，但要写进合同的有 **MTBF 承诺、故障替换 SLA、单卡掉线计费规则、跨 rail 网络收敛**——很多便宜报价不包含这些保证。

### 5. 常见追问 / 易错点
- "我能不能现在 spot 训完一个模型？"——可以，但要预算 NCCL hang/checkpoint resume 的工程成本，且只适合 < 200 卡级别。
- 误把"云上 reserved 折扣后单价 = 自建单价"当作平价线，忽略了自建还要叠加 18 个月人力与失败重试成本。
- **二手 H100**：2026 年灰市流通的 H100 SXM 模组单价 $20k–$25k，但缺保修、HBM 老化未知、BIOS/IPMI 来源不可信，企业级生产**不建议**使用。

### 6. 实践建议
画一张 **24 个月需求曲线** + **3 条价格曲线（云按需 / 云 reserved / 自建）**，看交叉点。如果交叉点在 12 个月内出现且需求曲线置信度 > 70%，启动自建评估。否则先用云 reserved 锁住基线 40%–60% 容量，剩余用 spot/neocloud 跑实验。把"我们能多快从 0 拉到 1024 卡训练"作为决策的第二维度——很多团队只算 $/hour 就拍板，忽略了启动延迟。每季度复盘真实利用率与故障率，必要时调整 reserved / on-demand / spot 配比。

### 7. 30 秒速答
- 不是哪种便宜，而是峰值规模、确定性周期、毛利目标三者决定。
- 自建盈亏点 ~75% 利用率 + 24 个月明确需求；否则混合云。
- 别只看 $/hour——18 个月爬坡期与电力门槛才是真成本。
- 加分关键词：盈亏点、TCO、24 个月需求曲线、reserved + spot 混合

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清自建与公有云的决策核心？
- [ ] 你能不能解释 75% 利用率盈亏点是怎么推出来的？
- [ ] 你能不能举一个混合云组合的具体场景（reserved + neocloud + spot）？
- [ ] 你能不能说出 3 个低估的自建隐性成本？

## Q2. AWS / GCP / Azure 三大云在 ML 训练侧的差异点是什么？

> 🔴 专家 · 别只看 H100 单卡报价表——三家在 `EFA / TCPX / IB` 网络栈、Capacity Block 机制、托管训练栈成熟度上的差异，才是 1024 卡 all-reduce 跑得动跑不动的真实分水岭。

### 1. 核心结论
三家在单 GPU 价格上差距 < 10%，**真正的差异在 GPU pod 网络拓扑、Capacity Block 机制、托管训练栈的成熟度**。AWS 强在 EFA 与 SageMaker HyperPod 的 8000+ 卡训练规模；GCP 强在 TPU 与 A3 Mega/Ultra 的 NVLink switch 拓扑；Azure 强在 ND H100/H200 v5 的 Mellanox InfiniBand 与企业合规。

### 2. 底层原理
大模型训练的瓶颈早就不是单卡 FLOPs，而是 **集合通信带宽 × 节点数 × MTBF**。三家在底层网卡层使用不同方案：
- AWS：自研 EFA（Elastic Fabric Adapter），用户态 SRD 协议，p5/p5e 节点 8×400 Gb/s。
- GCP：自研 TPU 用 ICI；A3 Mega 8×200 Gb/s GPUDirect TCPX，A3 Ultra 切到 RoCE + NVL。
- Azure：直接用 NVIDIA Quantum-2 InfiniBand NDR 400 Gb/s，与本地 HPC 栈一致。

### 3. 关键机制
- **AWS Capacity Blocks for ML**：可预订 1–14 天连续 H100/H200/B200 容量，提前 8 周锁定，按 $/GPU-hour 一次性付清。适合"我下个月要训 100B 模型 7 天"这类需求。
- **GCP DWS（Dynamic Workload Scheduler）Calendar/Flex**：Calendar 模式锁定固定起止时间；Flex Start 则承诺 7 天内启动并连续运行。
- **Azure ND H200 v5**：8×H200 + 8×400 Gb/s NDR，与微软自家训练（Maia/MI300）解耦，纯 NVIDIA 路线。
- **存储与训练协同**：AWS FSx for Lustre / GCP Parallelstore / Azure Managed Lustre 是各家 HPC 训练侧建议存储；选 vendor 时同时确认存储吞吐 SLA。

### 4. 工程权衡
选 AWS：你的栈深度依赖 SageMaker / S3 / EFA，团队接受用户态 SRD 调试。选 GCP：你愿意把数据放 GCS、且 TPU 是 plan B，A3 Mega 在 NCCL 层调优后 8000 卡可达 50%+ MFU。选 Azure：你已经是 Microsoft 企业客户、需要严格合规与 HPC 团队熟悉的 IB 栈。

跨云迁移成本被低估的部分：网络拓扑差异导致 NCCL 参数（`NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、`NCCL_NET_PLUGIN`）每家都要重新调；存储侧 S3/GCS/ABFS 的吞吐曲线、限速规则、multipart 大小最佳值都不一样。

### 5. 常见追问 / 易错点
- "三家 H100 价格差不多，所以可以互换"——错。互换的是单卡，不互换的是 1024 卡级集合通信稳定性。
- 忽视 **跨可用区训练**：单 AZ 容量不够时被迫跨 AZ，p99 RTT 从 5 μs 跳到 1 ms，all-reduce 直接拉胯。要么不跨，要么提前预订 Capacity Block。
- 把 SageMaker / Vertex / Azure ML 的"托管训练"当成主路径——大模型团队最终都退回到原始 EC2/GCE/VMSS + 自管 SLURM/K8s，托管层调度粒度太粗。

### 6. 实践建议
做一张矩阵：行是"我现有的栈（数据、IAM、合规、人才）"，列是三家云。如果某行明显倾斜，从这家起步；不要一上来就 multi-cloud。准备一个 **"32 卡 NCCL benchmark + checkpoint S3/GCS/ABFS 写吞吐"** 的标准化 1 天测试，每家云入场前必跑，跑完才签合同。Capacity Block 类的预订要提前 6–8 周开口，避免被业务突袭。

### 7. 30 秒速答
- 单 GPU 价差 < 10%，真正差异在 pod 网络栈、Capacity Block 与托管栈。
- AWS EFA、GCP TCPX / NVL、Azure IB NDR——NCCL 参数每家都不同。
- 别迷信"托管训练"，大模型团队最终都退回原始 EC2/GCE/VMSS。
- 加分关键词：EFA、TCPX、Capacity Block、HyperPod、bisection 带宽

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清三家云的核心差异在哪一层？
- [ ] 你能不能解释 EFA / TCPX / IB 三种网络栈的差别？
- [ ] 你能不能举一个 Capacity Block 的使用场景？
- [ ] 你能不能说出跨云迁移被低估的 3 个成本项？

## Q3. CoreWeave / Lambda / Crusoe / Nebius 这类 neocloud 怎么评估？

> 🔴 专家 · 看到 H100 报价比 AWS 便宜一半就想全切过去，结果合同里没写单卡替换 SLA 与网络收敛比，整 region 故障一次直接打穿全年 SLO。便宜的代价必须事前看清。

### 1. 核心结论
neocloud 的核心价值是 **比三大云便宜 30%–50% 的 GPU-hour 单价**，代价是合规深度浅、托管服务弱、单 region 容灾差。把它放在 **"训练专用 / 实验专用"**位置，不要把它当推理生产首选；选 vendor 时把"故障 GPU 替换 SLA"和"NCCL/IB 拓扑文档"作为硬门槛。

### 2. 底层原理
neocloud 通常是 GPU-only 公司，**没有自研网卡 / 存储 / 数据库栈**，靠 NVIDIA 参考设计 + Mellanox/Spectrum + 第三方对象存储拼起来。这让它能把毛利压到 15%–25%（vs 三大云的 35%–55%），但也意味着任何不在 NVIDIA 参考路径上的能力（合规、加密、复杂 IAM、跨区灾备）都要打折扣。

### 3. 关键机制
- **CoreWeave**：H100/H200/B200 全代覆盖，节点级 InfiniBand NDR，Kubernetes-native。2026 年 H100 长租 $1.8–$2.3/GPU-hour，B200 上线初期 $5–$7。
- **Lambda**：1-Click Clusters 与 Reserved Cloud；H100 短租 $2.5–$3.0、Reserved 1Y $2.0 左右；偏中小客户。
- **Crusoe**：能源优势（自建数据中心 + 余热回收），H200 长租报价进入 $1.8–$2.2 区间。
- **Nebius / Voltage Park / Together AI**：差异化在地理位置（欧洲 / 北美中部）、合规、或叠加 inference API。
- **入场必要文档**：节点拓扑图（含 leaf↔spine 收敛比）、IB / RoCE 网卡型号与固件版本、对象存储后端与吞吐 SLA、计费粒度（按秒还是按分）。
- **常见溢价项**：管理 K8s（vs 自管）、托管 SLURM、托管模型仓库；这些往往加 10%–25% 单价。

### 4. 工程权衡
省下的钱要花在两件事上：（1）**自建运维体系**，因为对方不会帮你调 SLURM、不会帮你 debug NCCL；（2）**多 vendor 容灾**，单家 neocloud 整 region 故障案例每年都有，关键训练要 checkpoint 到 S3/GCS 这种独立后端。**合同条款重点**：单卡掉线后多久替换（业界惯例 < 2 小时）、单节点 ECC 报错累计多少触发免费替换、IB 链路 down 是否计费、是否承诺 800 Gb/s 全 bisection。

### 5. 常见追问 / 易错点
- "CoreWeave 比 AWS 便宜一半，全切过去"——忽视了你的数据已经在 S3，跨云传 PB 级数据 egress 费用 + 时间窗口可能吃掉 6 个月节省。
- 误以为"reserved 1Y" 就是稳定容量——很多 neocloud 合同允许 vendor 在容量紧张时**优先服务大客户**，小客户即使付了 reserved 也可能被降级到 best-effort。
- 二手或翻新 GPU：少数 neocloud 报价异常低（< $1.5/H100-hour），通常是混入了二手卡，要在合同里写明"全新 + NVIDIA 原厂保修"。

### 6. 实践建议
入场标准化流程：（1）签 1 周 32 卡 PoC，跑标准 NCCL all-reduce + 1 个真实训练 step；（2）拿到拓扑文档（节点 → leaf → spine 收敛比）；（3）合同里加单卡替换 SLA 与免费降级条款；（4）至少留 20% 容量在三大云做"逃生通道"。永远别把生产推理 SLO 完全押在单家 neocloud 上。

### 7. 30 秒速答
- neocloud 便宜 30%–50%，代价是合规浅、托管弱、单 region 容灾差。
- 关键合同条款：单卡替换 SLA、收敛比、降级条款、容量保障。
- 定位"训练 / 实验"，不要押推理生产 SLO；留 20% 三大云逃生通道。
- 加分关键词：CoreWeave、Lambda、Crusoe、收敛比、neocloud 风险

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 neocloud 的核心定位？
- [ ] 你能不能解释为什么"reserved 1Y" 在 neocloud 不等于稳定容量？
- [ ] 你能不能举一个 neocloud + 三大云的混合架构？
- [ ] 你能不能说出 5 条入场必看的合同条款？

## Q4. Spot / Reserved / On-demand / Capacity Block 四种采购模式怎么组合？

> 🔴 专家 · 全押 3Y RI 求最大折扣，第 18 个月 B200 上量手里 H100 RI 当场贬值。四种模式本质是把"容量风险"在客户与云方之间分配，组合错了要么多付钱要么没卡用。

### 1. 核心结论
**基线 reserved + 弹性 on-demand + 实验 spot + 集中突发 Capacity Block**。reserved 比例覆盖你 12 个月 P50 需求；on-demand 留 10%–20% 当作短期 buffer；spot 用于无 SLA 的实验、ablation、batch inference；Capacity Block 用于"我必须连续训练 7–14 天"这种对齐到日历的大任务。

### 2. 底层原理
四种模式就是**云厂商把容量风险在客户与自己之间分配**：
- on-demand：客户付溢价，云方承担容量风险。
- reserved（1Y/3Y）：客户承诺消费，云方折扣 30%–55%。
- spot：客户承担被回收风险，云方折扣 50%–90%。
- Capacity Block：客户预付固定时间窗口，锁定容量，云方按窗口收费。

### 3. 关键机制
- **AWS Spot 中断**：2 分钟通知；GCP Preemptible/Spot 30 秒通知；Azure Spot 30 秒通知。训练侧需要做 SIGTERM 捕获 → 异步 checkpoint → exit 0。
- **AWS Capacity Block for ML**：1–14 天，支持 H100 p5、H200 p5e、B200 p6（按区域）；提前最多 8 周预订；典型溢价 30%–60% vs reserved。
- **AWS Savings Plans (Compute SP)**：比 RI 灵活，按 $/hour 承诺、跨实例族适用。
- **GCP Committed Use Discount (CUD)**：1Y/3Y，资源型与消费型两种。
- **Azure Reserved VM Instances + Savings Plan**：可换实例族但限定 region。

### 4. 工程权衡
spot 适合**短任务（< 1 小时）+ 有 checkpoint** 场景；大模型 1024 卡训练上 spot = 找死，因为单卡被回收会让整 job 失败。Capacity Block 在 H100/H200 紧张窗口几乎是唯一办法，但要忍受**严格的窗口起止时间**——不到时间不能开始、过了时间立即关机，意味着你的预热与收尾都要压在窗口内。reserved 的最大风险是**跨代贬值**：买了 3Y H100 RI，第 18 个月 B200 普及，剩余 18 个月 RI 在二级市场只值 60%。

### 5. 常见追问 / 易错点
- 把所有产能买成 3Y RI 求最大折扣——忽视了 18 个月一代的硬件迭代，等于赌 3 年内不会出更划算的卡。
- spot 训大模型时 NCCL hang——不是 spot 本身的问题，是没做 SIGTERM 处理 + 没用 elastic launcher（torchelastic / TorchTitan elastic）。
- 误把 Capacity Block 当成"提前 1 天就能订"——实际是按 region 容量提前 6–8 周。

### 6. 实践建议
做一张**容量曲线**：12 个月里每月需要的 GPU-月数。底下一层（P30）用 3Y RI 锁住；中间一层（P30–P70）用 1Y RI / Savings Plan；上面一层（P70–P95）用 on-demand；峰值（> P95）用 Capacity Block 提前预订。spot 完全独立，专门给"非关键路径 + 可中断" 的批处理用。每季度复盘 RI 利用率，掉到 < 70% 启动卖出/转移流程。

### 7. 30 秒速答
- 基线 reserved + 弹性 on-demand + 实验 spot + 突发 Capacity Block。
- 四种模式本质是容量风险在客户与云方之间分配。
- 全押 3Y RI = 赌 3 年没新硬件；spot 上大模型 = 找 NCCL hang。
- 加分关键词：Capacity Block、Savings Plan、CUD、容量风险分配

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清四种采购模式的核心差异？
- [ ] 你能不能解释 spot 中断对训练任务的影响与缓解？
- [ ] 你能不能举一个 P30 / P70 / P95 容量分层的具体配比？
- [ ] 你能不能说出 3Y RI 的"跨代贬值"风险？

## Q5. 长期 GPU 租赁谈判（1Y / 2Y / 3Y）的关键条款是什么？

> 🔴 专家 · 谈判桌上只盯 `$/GPU-hour` 是新手做派——单卡替换 SLA、网络收敛比、跨代 swap、提前终止条款，每一条没写进合同，后面 12 个月都是被动挨打。

### 1. 核心结论
长租谈判**别只盯 $/GPU-hour**。关键条款 5 条：单卡替换 SLA、网络收敛比承诺、计划停机窗口与计费、跨代升级权（H100 → H200 / B200 mid-term swap）、提前终止条款。其中**跨代升级权**是 2025–2026 最值得争取的——H100 寿命已经从 3 年压到 2 年。

### 2. 底层原理
长租本质是客户**用流动性换确定性**。云方/neocloud 拿到 12–36 个月稳定收入，客户拿到固定单价。这个交易的价格波动主要来自三个变量：（1）下一代 GPU 何时上量（影响 H100 残值）；（2）能源价格（影响数据中心运营成本）；（3）行业总需求（影响二手市场流动性）。

### 3. 关键机制
- **单卡替换 SLA**：业界惯例"单卡 ECC uncorrectable 错误 / xid 79 / NVLink down"触发 4–24 小时内替换；写明"不计入合同 uptime"。
- **网络收敛比**：要求 vendor 提供 leaf↔spine bisection bandwidth 数据，写明"任意 2 节点间 ≥ X Gbps"。
- **计划停机**：每季度允许 4–8 小时计划停机，超出部分按 1.5–2× 退款。
- **跨代 swap**：常见 mid-term 条款"在合同期 50% 时点以 RFP 价格换购下一代 GPU，已付费用按比例转移"。
- **提前终止**：early termination fee 通常是剩余合同金额的 50%–80%；可以谈到 30%–50% 如果你预付了较高比例。
- **价格调整条款**：电力涨跌、关税变化、新一代上市时的 price floor / ceiling 都要写明，避免合同中段被单方面调价。
- **审计与报告**：要求 vendor 每月提供 utilization、incident、replacement 数据；这些是 mid-term 谈判的弹药。

### 4. 工程权衡
预付比例与单价折扣呈反比：100% 前付能再砍 5%–10%，但锁死现金流。租期越长单价越低，但跨代风险越高——2026 年签 3Y H100 长租是高风险动作，2Y 是合理上限。**网络承诺**尤其要写死：很多便宜报价是"oversubscribed fabric"（4:1 收敛比），训练性能比 1:1 全对分慢 30%+。

### 5. 常见追问 / 易错点
- "vendor 答应了，但合同没写"——后续替换 / 维护 / 网络都要写进 SOW，否则法务上无法追责。
- 忽视**电力涨价 pass-through 条款**：部分 neocloud 合同允许 vendor 把电价上涨 50% 以上转嫁给客户。
- 没有 audit 权：要求季度性提供利用率报告、故障历史、网络拓扑变更记录。

### 6. 实践建议
谈判前做三件事：（1）拿三家以上报价做 BATNA；（2）找有 2024–2025 长租实战经验的法务/采购顾问，他们知道哪些是"业界已经写进去过"的条款；（3）准备一份你的 SLO 反推清单（你能容忍多少 GPU-hour/年的故障停机），把它翻译成合同里的具体小时数。每月跟踪利用率与故障率，到了 mid-term 触发 swap 选项的时间点要主动开口。

### 7. 30 秒速答
- 别只盯 $/GPU-hour，5 条关键条款决定 12 个月被不被动挨打。
- 单卡替换 SLA、收敛比、计划停机、跨代 swap、提前终止。
- 2026 年 H100 长租 2Y 是上限；跨代 swap 权是最值得争取的。
- 加分关键词：跨代 swap、收敛比、early termination、电价 pass-through

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清长租谈判的本质？
- [ ] 你能不能解释跨代 swap 条款的具体写法？
- [ ] 你能不能举一个 1:1 vs 4:1 收敛比对训练的影响？
- [ ] 你能不能说出 3 条"vendor 答应但合同没写"的常见坑？

## Q6. 二手 H100 / 灰市 GPU 的工程风险评估

> 🔴 专家 · 灰市 H100 比全新便宜 25%–35%，听上去香——但 HBM 老化未知、VBIOS 可能被刷过、保修不可转让，生产环境用一次就知道为什么大厂宁可贵也要原厂卡。

### 1. 核心结论
二手 H100 在 2026 年灰市价格 $20k–$25k（vs 全新 $30k–$35k SXM 模组），价差 25%–35%。**不建议用于生产训练或推理**：风险点是 HBM 老化未知、BIOS/IPMI 不可信、保修无法转让、NVLink 拓扑可能被改装。可以容忍的场景：内部研究、ablation 实验、纯推理且能接受 30% 卡掉线率。

### 2. 底层原理
H100 的失效模式集中在 **HBM3 ECC uncorrectable error 与 NVLink CRC 错误**。HBM 在持续高温（液冷不足或风冷）下使用 18 个月以上，UCE 概率上升明显。二手卡来源大多是早期采购方（2023 H1–H2 拿货）淘汰下来的，已经过了高强度训练 12–24 个月。NVIDIA 原厂保修不可转让，意味着你后续故障只能找经销商或自负盈亏。

### 3. 关键机制
入手前必跑的 burn-in：
- **HBM stress**：DCGM `dcgmi diag -r 4` 跑 30+ 分钟，监控 HBM 温度与 ECC 计数。
- **NVLink loop**：`nvidia-smi nvlink -e 0`、`nccl-tests` all-reduce 跑 4 小时观察 CRC 错误。
- **BIOS / VBIOS 校验**：`nvidia-smi -q | grep VBIOS` 对照 NVIDIA 发布版本，灰市卡常见被刷成 mining 固件或开锁版。
- **IPMI / BMC fingerprint**：检查 BMC 固件签名，避免拿到带后门的二手服务器。
- **温度循环**：50 °C↔ 80 °C 反复跑 24 小时观察是否触发 thermal throttling 或 ECC 突增。
- **批次抽检**：同一批二手卡里随机抽 10% 做完整 burn-in，外推整批合格率。

### 4. 工程权衡
省 30% 钱的代价是**每年 5%–15% 卡损耗率**（vs 全新卡 1%–3%）。如果你是租给第三方推理服务，这意味着 SLO 难以保证；如果是自己做训练，意味着 1024 卡训练 24 小时 hang 概率从 5% 升到 20%+。**真正合理的二手场景**：批处理推理（无 latency SLO）、内部实验集群（接受 down 时手动重启）、教育/科研用途。

### 5. 常见追问 / 易错点
- "二手卡也有保修，经销商 90 天替换"——大多数灰市经销商保修不覆盖 HBM ECC 与 NVLink，写仔细看。
- **海关与合规风险**：经过制裁国家流通的二手卡可能被海关扣押或后续维护被禁。
- 误把"价格差 25%" 等于"等效 25% 节省"——加上 burn-in 时间、人力、未来故障替换成本，实际节省可能只有 5%–10%。

### 6. 实践建议
如果非要走二手路线：（1）只买**单一来源**（一次清仓的整批卡），可以批量做 burn-in 与统计；（2）保留 10%–15% 预算做替换池；（3）在合同里要求经销商提供原始 NVIDIA 序列号与购买凭证以追溯保修历史；（4）独立测试一周再上线。生产训练或外部推理服务，**不要用二手卡**。

### 7. 30 秒速答
- 二手 H100 便宜 25%–35%，但 HBM 老化未知、保修不可转让。
- 失效集中在 HBM3 UCE 与 NVLink CRC，年损耗 5%–15%。
- 只适合内部研究 / 批处理推理；生产训练与外部推理不要碰。
- 加分关键词：DCGM burn-in、xid 79、VBIOS 校验、批次抽检

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清二手 H100 的核心风险？
- [ ] 你能不能解释 HBM ECC 与 NVLink CRC 怎么测出来？
- [ ] 你能不能举一个二手卡合理使用的场景？
- [ ] 你能不能说出 burn-in 的 5 项必跑检查？

## Q7. AWS / GCP / Azure 跨区域容量分布的战略意义？

> 🔴 专家 · 三大云的 H100 容量集中在少数 region，新代际首发的 us-east-1 / us-central1 与新加坡某 region 之间能差出 6–12 个月的 GA 时间。选 region 这件事的权重，跟选 vendor 一样大。

### 1. 核心结论
2024–2026 年三大云的 H100/H200/B200 容量在 **少数核心 region 集中**：AWS us-east-1 / us-west-2 / eu-north-1，GCP us-central1 / asia-southeast1 / europe-west4，Azure East US 2 / South Central US / West Europe。其他 region 容量稀缺、配额低、新代际迟到 6–12 个月。**战略选择 region 与战略选择 vendor 一样重要**。

### 2. 底学原理
GPU 部署受电力、冷却、光纤布局约束，云厂商优先在已有大型数据中心扩容。新 region（尤其是 LATAM / 中东 / 东南亚二线）需要 18–36 个月才有 GPU 容量。因此：可用 region 数 << 可用 region 总数；H100 quota 上限可能远低于公开声明。

### 3. 关键机制
- **配额申请流程**：AWS Service Quotas、GCP Quota requests、Azure Quota requests，H100 单账户初始上限通常 0–8 卡，需要 case-by-case 申请上调。
- **Capacity Block 区域**：AWS Capacity Blocks 仅在少数 region 提供。
- **数据驻留 / 主权云**：GCP Sovereign Controls、Azure Confidential Cloud、AWS GovCloud，这些 region GPU 容量更稀缺。
- **新代际首发 region**：H200 首发往往在 us-east-1 / us-central1，B200 也类似；其他 region 落后 6–12 个月。
- **跨 region 网络**：同一 cloud 内 inter-region 1ms–80ms RTT；egress $0.02–$0.08/GB；跨 cloud 加上 internet 段更慢更贵。
- **AZ 容量分布不均**：单 region 内 AZ 间 H100 容量差距可达 5–10×，预订时要指定具体 AZ 而非"region anywhere"。

### 4. 工程权衡
把容量集中在 1–2 个 region 简化网络与配额管理，但承担**整 region 故障**风险（2024–2025 已发生数起）。多 region 容灾的代价：（1）数据复制带宽费用，PB 级跨 region $0.02–$0.08/GB；（2）latency 影响 RAG/推理 routing；（3）双份 reserved capacity。**权衡点**：训练侧可以集中（一次性大任务），推理侧应分散（用户面向）。

### 5. 常见追问 / 易错点
- "我在 us-east-1 申请 256 H100 quota，云方说能给"——能给不等于现在能给，可能要等 4–8 周；正式启动训练前要做 dry-run 确认。
- 忽视**网络出口费用**：用户在欧洲、训练在美东、推理在东南亚——egress 费用一年可以多花 $1M。
- 主权云 / 政府云 region 的 GPU 容量极少，且新代际可能永远不到——签订合同前必须确认。

### 6. 实践建议
画一张**用户分布 × 训练 region × 推理 region** 矩阵。训练向新代际首发 region 集中，推理就近用户分布。每个核心 region 至少做 32 卡 PoC 验证拓扑与配额。新 region 上线前 3 个月开始申请 quota，留足时间。把数据驻留要求与 region 选择放在同一张决策表里，别让法务事后否决。

### 7. 30 秒速答
- H100/H200/B200 容量集中在少数核心 region，新代际首发更稀缺。
- 选 region 与选 vendor 权重一样大；AZ 容量差距可达 5–10×。
- 训练向新代际首发 region 集中，推理就近用户分布。
- 加分关键词：us-east-1、us-central1、East US 2、quota、AZ 容量

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 region 选择的核心约束？
- [ ] 你能不能解释为什么"申请 quota 通过"不等于"现在能给"？
- [ ] 你能不能举一个用户 / 训练 / 推理三地分布的容灾设计？
- [ ] 你能不能说出主权云 region 的 GPU 容量陷阱？

## Q8. ML 平台从单云到 multi-cloud 的演进时机？

> 🔴 专家 · "我们要 multi-cloud 容灾"听起来政治正确，实际多数团队过早上 multi-cloud 等于自掏腰包给账单加倍。容灾、容量、合规这 3 个信号没出现，就别折腾。

### 1. 核心结论
**别太早 multi-cloud**。早期（< 1000 卡级）单云足够，跨云成本 > 收益。触发 multi-cloud 的 3 个信号：（1）单 vendor 容量不够你下一阶段需求；（2）出过整 region 故障且业务被打断 > 4 小时；（3）合规要求多区域多 vendor。其他场景 multi-cloud 的 ROI 通常为负。

### 2. 底层原理
跨云的隐性成本：（1）**egress 费用**，PB 级数据迁移 $20k–$80k；（2）**双倍工程团队**，每家云的 IAM / 网络 / 存储 / 监控都要单独搭；（3）**抽象层负担**，写 K8s + Terraform 试图统一，最后被云专属能力（EFA、GCS、Azure NetApp）反噬；（4）**采购杠杆下降**，分散到多家后单家 reserved 谈判力变弱。

### 3. 关键机制
渐进路径（典型 24 个月演进）：
- **阶段 1（< 500 卡）**：单云，用 reserved + on-demand；
- **阶段 2（500–2000 卡）**：单云 + 1 家 neocloud 做实验/spot；
- **阶段 3（2000–8000 卡）**：双云架构，主云训练 + 备云推理或反之；
- **阶段 4（> 8000 卡）**：完整 multi-cloud + 自建混合。
- **抽象层选项**：SkyPilot / Volcano / Run:ai / Anyscale 各自定位；选型时确认它们对 IB / EFA / TCPX 的支持深度。
- **数据复制策略**：核心数据集每家云保留只读副本；checkpoint 写主云，异步复制到备云。

### 4. 工程权衡
multi-cloud 真正的工程负担在**抽象层选择**：用 K8s / Volcano / SkyPilot 统一调度，要承担抽象不完整带来的 bug；用云专属栈，要承担多套人力。SkyPilot 这类工具在 2025–2026 成熟度提升，可作为统一作业层；但底层网络、存储、加密仍要分别处理。**checkpoint 与数据集双写**也是大头，PB 级数据集双写每月数十万美元。

### 5. 常见追问 / 易错点
- "multi-cloud 等于高可用"——错。高可用要看你能否在主云挂掉 1 小时内切到备云。如果切换需要人工干预 4 小时，multi-cloud 只是给账单加倍。
- 误把 **K8s portability** 等同 **GPU 训练 portability**。K8s 能跑，但 EFA / TCPX / IB 配置每家不同，NCCL 性能差异巨大。
- 忽视**法务 / 财务 multi-cloud 的复杂度**：发票、对账、税务、reserved 闲置都要在多家之间平衡。

### 6. 实践建议
先把 **"多 region 容灾"** 当成 multi-cloud 的过渡形态——同一家云的 us-east-1 + us-west-2 已经能解决 80% 的容灾需求，且没有跨云成本。真正要 multi-cloud 时，先做**单一 workload 跨云**（比如离线 batch inference），积累 6 个月经验，再扩到训练侧。建立**跨云 cost dashboard**，每月看真实跨云成本与节省，ROI 跌入负值就退回去。

### 7. 30 秒速答
- 别太早 multi-cloud；信号没出现，账单先翻倍。
- 3 个触发：容量不够、整 region 故障、合规要求。
- 多 region 容灾先做，能覆盖 80% 高可用需求且无跨云成本。
- 加分关键词：SkyPilot、egress、双倍工程团队、抽象层负担

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 multi-cloud 的合理触发条件？
- [ ] 你能不能解释 K8s portability ≠ GPU 训练 portability？
- [ ] 你能不能举一个 4 阶段 multi-cloud 渐进路径？
- [ ] 你能不能说出 multi-cloud 4 个隐性成本？

## Q9. 训练数据中心选址 / 自建机房的工程门槛

> 🔴 专家 · 自建机房真正卡人的不是钱，是电力申请 12–24 个月 + 液冷工程师招不到 + IB fabric 验收 4–8 周。算盘打得再响，"无电可申请"四个字一出来，所有计划归零。

### 1. 核心结论
自建训练数据中心的真正门槛不是钱，是**电力 + 冷却 + 网络运维团队**。门槛量级：5 MW 起步、18–36 个月建设周期、需要 HPC 网络与液冷工程师团队 10+ 人。如果以下三条任一不成立，**不要自建**：（1）24 个月需求 > 5000 GPU-月；（2）能拿到 5+ MW 电力；（3）有 HPC 老兵能 hire。

### 2. 底层原理
H100/H200 SXM5 整机 8 卡功耗 10–12 kW，B200 节点 14–16 kW。1024 卡 H100 集群 ≈ 1.4 MW IT 负载，叠加 PUE 1.2–1.5 总电力 ≈ 2 MW。冷却从 1 kW/机柜 风冷模式跳到 30 kW+/机柜液冷或后门换热，土建与运维难度阶跃式上升。**网络**：1024 卡级集群需要 spine-leaf InfiniBand NDR 或 800GbE RoCE，光纤数千根，调试周期 1–3 个月。

### 3. 关键机制
- **电力**：从申请到通电 12–24 个月，2025–2026 北美热门区域已经"无电可申请"。
- **冷却选型**：直触液冷（DLC，rear-door heat exchanger 或 cold plate）支持 30–80 kW/机柜；浸没式（immersion）支持 100+ kW/机柜，但维护成本高。
- **PUE 目标**：现代 GPU 数据中心 PUE 1.15–1.30；老旧机房 1.5+ 直接拉高电费 30%。
- **网络**：NDR IB 800Gb/s 或 RoCE v2，需要 RDMA-aware 交换机与运维。
- **IT 团队**：HPC 系统工程师、SLURM/K8s 运维、网络工程师、设施工程师，每个角色 2+ 人备份。
- **典型 capex 占比**：GPU 占整个建设成本 65%–75%、网络 5%–10%、机柜与冷却 8%–12%、电力 / UPS 5%–8%、土建 2%–5%。
- **opex 结构**：电力占 50%–65%、运维人力 15%–25%、备件与维保 10%–15%、网络与对外连接 5%–10%。

### 4. 工程权衡
**完全自建** vs **共建/colocation**：colocation（你出 GPU + 网络，别人出机房 + 电）大幅降低门槛，但每年电力费用率高 20%–30%。共建（与超算中心 / 大学 / 私募合作）适合 1000–5000 卡规模，避免自负土建。**完全自建只对 10000+ 卡且需求确定的客户值得**。

### 5. 常见追问 / 易错点
- "电力签了就能通"——签合约 ≠ 通电；电力公司可能延期 12 个月以上。
- 忽视**节能审查 / 碳排放报告**：欧盟、加州、新加坡等地法规收紧，PUE > 1.3 可能不予审批。
- 网络调试低估：NDR IB 1024 卡 fabric 验收一般要 4–8 周，期间 GPU 闲置烧钱。

### 6. 实践建议
如果"必须自建"：第 0 个月开始申请电力（关键路径），第 6 个月签 colocation 或买地，第 12 个月开始装机，第 18 个月做 NCCL 大规模验证，第 24 个月正式上线。建议 80% 团队还是用云 + neocloud，**自建只承载 P30 以下基线训练**。把"何时退出自建"作为初始决策的一部分——不要做无限期投入。

### 7. 30 秒速答
- 真正卡人的是电力 + 冷却 + HPC 运维团队，不是钱。
- 1024 卡 H100 ≈ 1.4 MW IT 负载、18–36 个月建设周期。
- 北美热门区已"无电可申请"；colocation 是更现实的中间态。
- 加分关键词：PUE、DLC 液冷、IB fabric 验收、capex/opex 拆分

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清自建的核心门槛？
- [ ] 你能不能解释 1024 卡集群的电力换算？
- [ ] 你能不能举一个 colocation 替代完全自建的场景？
- [ ] 你能不能说出建设期的 4 个关键里程碑？

## Q10. 训练硬件选型：H100 / H200 / B200 / MI300 / TPU v5p 决策矩阵

> 🔴 专家 · 销售 PPT 都是 `4× FP4 加速`，落到生产 LLM 训练端到端 1.8–2.5× 已经是好结果。每个 SKU 的显存、互连、生态成熟度差异很大，错配一次就是几百万烧掉。

### 1. 核心结论
2026 年默认选 **H200**（成熟、生态完整、长租价格合理）；新建集群可以混 **B200**（FP4/FP8 训练吞吐 2–3× H100，但软件栈仍在打磨）；推理侧 **MI300X** 在大显存场景（192 GB HBM3）有 cost 优势但生态弱；Google 客户继续用 **TPU v5p / Trillium**；H100 转向 spot/二手市场，2026 H2 起逐步替换。

### 2. 底层原理
不同 SKU 的本质差异在 **HBM 容量、FP8/FP4 算力、互连带宽**：
- H100 SXM5：80 GB HBM3、FP8 ~1979 TFLOPS、NVLink 900 GB/s、上市 2022。
- H200 SXM5：141 GB HBM3e、FP8 ~1979 TFLOPS、NVLink 900 GB/s、上市 2024。
- B200 SXM6：192 GB HBM3e、FP4 ~9 PFLOPS、NVLink 5 1800 GB/s、上市 2024 H2 / 2025。
- MI300X：192 GB HBM3、FP8 ~2614 TFLOPS、Infinity Fabric 896 GB/s、上市 2024。
- TPU v5p：95 GB HBM、bf16 ~459 TFLOPS、ICI 4800 Gb/s、Google 内部 + GCP。

### 3. 关键机制
- **生态成熟度**：CUDA + PyTorch + NCCL（H/B 系列）> ROCm + PyTorch（MI300） > XLA + JAX/PyTorch-XLA（TPU）。
- **训练 vs 推理偏向**：H200 / B200 训练推理通用；MI300X 推理友好（大显存 + 单卡推理 70B 不分片）；TPU v5p 训练强、推理生态弱。
- **价格曲线**：H100 长租 2026 跌到 $1.5–$2.0/hour；H200 $1.8–$2.5；B200 上线初期 $5–$7，2026 H2 跌到 $3–$5；MI300X $1.5–$2.5；TPU v5p 仅在 GCP 包月。

### 4. 工程权衡
**B200 的陷阱**：FP4 训练数值稳定性、NCCL 5 与 NVLink 5 的早期 bug、Megatron/TorchTitan 对 B200 路径还在收敛。早期 adopter 要预留 3–6 个月调优期。**MI300X 的陷阱**：ROCm 6.x 在 PyTorch 主线已经稳定，但**vLLM / SGLang / FlashAttention** 等关键栈跟随略慢；运维复杂度高于 NVIDIA。**TPU 的陷阱**：JAX/XLA 学习曲线陡，且只能在 GCP，被 vendor-lock。

### 5. 常见追问 / 易错点
- "B200 比 H100 快 4×，全切 B200"——4× 是 FP4 marketing 数据，实际 LLM 训练 FP8/BF16 端到端 1.8–2.5× 已经很好。
- 误把 MI300X 192 GB 当成无脑替代 H100：通信带宽 / NCCL 替代品 RCCL 的成熟度、长尾算子覆盖都要单独验证。
- TPU 在 attention / MoE 实现上与 GPU 路径不一致，模型从 GPU 迁移到 TPU 经常需要重写一部分。

### 6. 实践建议
**默认路径**：新订单 H200 主力 + 少量 B200 (20%) 先行验证；H100 走 spot 与二手市场承接非关键负载；MI300X 用于"大显存推理"场景（70B 单卡）；TPU 仅在 GCP 重客户里保留作为 plan B。每代际切换前留 3 个月 PoC + 6 个月并行运行；不做"硬切换"。把硬件选型与团队技能矩阵放一起评估——再好的卡，团队不会调也不会有 MFU。

### 7. 30 秒速答
- 2026 默认 H200 主力 + 20% B200 验证；H100 走 spot。
- 差异在 HBM 容量、FP8/FP4 算力、互连带宽、生态成熟度。
- "FP4 4× 加速"是 marketing；端到端 1.8–2.5× 已是好结果。
- 加分关键词：HBM3e、NVLink 5、ROCm、TPU v5p、生态成熟度

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 2026 的硬件主力配比？
- [ ] 你能不能解释 H100 / H200 / B200 / MI300 / TPU 的关键差异？
- [ ] 你能不能举一个 MI300X 适合的场景？
- [ ] 你能不能说出 B200 早期上线的 3 个陷阱？

## Q11. tokens / $ 作为推理产品的北极星指标怎么定义与跟踪？

> 🔴 专家 · 只看 GPU `$/hour` 不算分母，团队会一直觉得自己很省，结果毛利每月都在掉。把训练摊销、出口、安全、重试全摊到 token 上，才看得见真实账。

### 1. 核心结论
**tokens / $ = (有效输出 token 数) / (摊销后基础设施 + 模型 + 运维成本)**，分母要覆盖训练摊销、GPU 租赁、网络出口、KV 缓存内存、安全审核、长尾失败重试。2026 年开源 70B 推理在 H100 集群典型水平 5k–20k output tokens / $；闭源 frontier 模型在 200–2000 区间。这条曲线决定推理产品定价与毛利。

### 2. 底层原理
推理成本不只是 GPU 时长。完整账本：（1）GPU 租赁/折旧（占 60%–75%）；（2）网络与对象存储（5%–15%）；（3）模型权重摊销（训练成本 ÷ 模型生命周期 token 总量，5%–15%）；（4）观测、日志、安全（5%–10%）；（5）失败重试与冗余（5%）。**只看 GPU $/hour 而不算分母**会让你低估真实成本 30%–50%。

### 3. 关键机制
- **吞吐侧**：vLLM / SGLang / TensorRT-LLM 在 H100 上 70B 模型 INT8/FP8 输出 ~3k–6k tokens/s/卡（batch saturated）。
- **核算公式**：`tokens / $ = throughput_per_gpu × utilization × 3600 / hourly_cost`。
  例：6k tok/s × 0.7 × 3600 / $2.5 = 6048k tok/$，即 ~6M tok/$。
- **训练摊销**：70B 模型训练 $5M、生命周期推理 5T tokens，摊到每 token $1e-6。
- **跟踪粒度**：按模型 SKU × 区域 × 客户群组三维统计。
- **input vs output 拆分**：input prefill 通常 $/token 是 output 的 1/5–1/10；对外定价要分开。
- **峰谷成本曲线**：低谷期单 token 成本是高峰期的 2–3×（batch 不饱和），需要按时段计入平均。

### 4. 工程权衡
**追求 tokens / $ 极致 vs 追求 SLO**：批量场景可以把 batch size 拉到 256+，单 token 成本降到 1/3，但 TTFT 上升到 2–5 秒；交互式场景必须 batch 32–64，单 token 成本翻倍。**模型选择 vs tokens / $**：70B 比 405B tokens / $ 高 5–10×，但任务质量门槛通过不去就是另一回事。**FP8/FP4 vs FP16**：FP8 提升 1.5–2× 吞吐但偶发数值漂移要监控。

### 5. 常见追问 / 易错点
- "tokens / $ 越高越好"——错。如果质量降到客户流失，tokens / $ 高也无意义。要与"客户接受率"双指标看。
- 忽视 **input vs output token 不对称**：input 一次性吃 prefill，output 是 decode 循环，成本结构差 5–10×。计算时区分。
- 把 spot $/hour 直接代入——spot 中断重试摊到分母后 effective $/hour 高 20%–40%。

### 6. 实践建议
搭一个 **tokens/$** 仪表盘，分 SKU × 区域 × batch 区间。每周复盘 P50 / P90 / P99 三点。找 TOP3 的"高 spend 低 yield" 客群打磨：要么调 batching 策略，要么换模型 SKU，要么涨价。把 tokens/$ 写进季度 OKR，研究、训练、推理三方共担。每次模型更换、量化、kernel 升级前后都做 A/B，落到这个指标上。

### 7. 30 秒速答
- tokens/$ = 有效输出 token 数 / 摊销后全成本，是推理北极星。
- 分母必须含训练摊销、出口、安全、重试，不只是 GPU $/hour。
- 极致 tokens/$ vs 客户接受率要双指标看，单边优化必踩坑。
- 加分关键词：input/output 不对称、prefix cache、batch 饱和、spot 实效成本

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 tokens/$ 的公式？
- [ ] 你能不能解释为什么 input 与 output 要分开核算？
- [ ] 你能不能举一个 H100 上 70B 推理的 tokens/$ 估算？
- [ ] 你能不能说出 5 项必须计入分母的成本？

## Q12. MFU 与 HFU 的合理目标值与瓶颈分解

> 🔴 专家 · 报 `MFU 50%` 之前先想清楚自己是几卡几配置——1024 卡 H100 BF16 50% 是顶尖，64 卡还停在 50% 是没好好调。把 MFU 当 KPI 也得有瓶颈分解，否则就是数字游戏。

### 1. 核心结论
**MFU（Model FLOPs Utilization）** 是模型理论 FLOPs / 硬件理论 FLOPs。2026 年大规模训练经验：H100 BF16 训练 LLM 期望 MFU 35%–55%，B200 FP8 训练 30%–45%（早期）。**HFU（Hardware FLOPs Utilization）** 把 recomputation 也算进分子，比 MFU 高 10%–20%。MFU < 30% 说明有明显的工程问题，> 60% 在大规模集群罕见且通常是"取巧"。

### 2. 底层原理
MFU 的损耗来源：（1）通信（all-reduce / all-gather / pipeline bubble）；（2）数据加载抖动；（3）optimizer step / kernel launch 间隙；（4）activation recompute（不计入 MFU 但消耗时间）；（5）热点 kernel 未优化。每一项都能从 profiling 看出贡献，目标是把"非 GPU compute 时间"压到 < 25%。

### 3. 关键机制
- **MFU 计算**：`MFU = 6 × N_params × tokens_per_step / (peak_FLOPs × step_time × num_gpus)`（Transformer 训练）。
- **典型瓶颈**：
  - 通信 25%–40%：`NCCL_ALGO`、async TP、ZeRO 分片。
  - 数据加载 5%–15%：DataLoader workers、本地 NVMe 缓存。
  - kernel 间隙 5%–10%：cudagraph、torch.compile。
  - bubble 5%–15%：流水线并行（pipeline parallel）切分不均、micro-batch 数。
- **profiling 工具**：Nsight Systems、PyTorch profiler、HTA、Megatron `--profile`。

### 4. 工程权衡
**短期提 MFU vs 长期可维护性**：手写 fused kernel 把 MFU 从 45% 提到 52%，但下次升级 PyTorch / Hopper → Blackwell 又要重写。优先用 torch.compile / Triton / FlashAttention 这种"上游持续维护"的路径。**FP8 训练的 MFU 陷阱**：FP8 把 peak FLOPs 翻倍，MFU 数值反而下降（分母翻倍）；判断时要看绝对吞吐而非相对 MFU。

### 5. 常见追问 / 易错点
- "MFU 50% 已经业界顶尖，不动了"——不一定。看绝对数：1024 卡 H100 MFU 50% 是顶尖，64 卡 H100 MFU 50% 还有 20%+ 提升空间。
- 把 HFU 当成 MFU 报告——recomputation 砍 activation 显存但消耗 FLOPs，不能算"模型有效计算"。
- 忽视 **跨 step 抖动**：均值 MFU 50% 但 P90 step time 比 P50 慢 30%，意味着 straggler。

### 6. 实践建议
建立 **每作业 MFU + 瓶颈分解仪表盘**。新模型上线前必须做 4 阶段调优：（1）单节点 MFU 极限测试；（2）8/16/32 节点 scale-out；（3）加 checkpoint / FT 重启路径；（4）大规模 + 长时间稳定性。每次模型架构变化（MoE 引入、序列长度翻倍、TP/PP 拓扑变更）都要重新调优。**MFU 跌 5% 以上就该立 bug**，不要把它当作"自然衰减"。

### 7. 30 秒速答
- H100 BF16 LLM 训练 MFU 35%–55%，B200 FP8 早期 30%–45%。
- 损耗来源：通信 25%–40%、数据加载 5%–15%、kernel 间隙、bubble。
- 看绝对吞吐不是相对 MFU——FP8 把分母翻倍会假性降低 MFU。
- 加分关键词：HFU、bubble、torch.compile、Nsight、瓶颈分解

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 MFU 与 HFU 的差别？
- [ ] 你能不能解释 MFU 的计算公式？
- [ ] 你能不能举一个 1024 卡 vs 64 卡的合理目标差异？
- [ ] 你能不能说出 4 个主要 MFU 损耗来源？

## Q13. 训练预算反推：从模型规格到 GPU-月数

> 🔴 专家 · "训一个 70B 大概多少钱？"这个问题没有 `6 × N × T / (peak × MFU)` 这把尺子，立项就是拍脑袋。算成功一次还要乘 ablation × failed runs × 1.3，否则下半年现金流必爆。

### 1. 核心结论
**训练预算 ≈ 6 × N_params × N_tokens / (peak_FLOPs × MFU)**，单位是 GPU-秒。换算成 GPU-月再叠加 1.3–1.5× 工程系数（首训失败重试、ablation、调优）。例：70B × 15T tokens × H100 BF16 × MFU 45% ≈ 5M GPU-小时 ≈ 7000 GPU-月。

### 2. 底层原理
Transformer 前向 ~2N FLOPs/token，反向 ~4N FLOPs/token，合计 6N。这是 Kaplan/Hoffmann 之后的业界共识。再叠加：activation recomputation（×1.0–1.33，看是否全部 recompute）、optimizer step（< 5% 通常忽略）、loss scaling/精度调整、failed runs 系数。

### 3. 关键机制
- **公式**：`GPU_seconds = (6 × N × T) / (peak_FLOPs × MFU)`
  - N = 参数量
  - T = 训练 token 数
  - peak_FLOPs：H100 BF16 ~989 TFLOPS，FP8 ~1979 TFLOPS；B200 FP8 ~4.5 PFLOPS
  - MFU：BF16 0.40–0.50，FP8 0.30–0.40
- **Chinchilla scaling**：T ≈ 20 × N（最优 token/参数比），实际生产模型常用 30–80 × N（过训以提升下游）。
- **工程系数**：×1.3 ~ ×1.5 应对 ablation、failed runs、scale-up 调试。

### 4. 工程权衡
**精度选型对预算影响**：FP8 训练理论吞吐翻倍，MFU 30%–40% → 实际 1.5–1.8× 端到端加速 → 预算节省 35%–45%。但 FP8 数值稳定性需要监控、loss scaling、master weights，调试期可能吃掉部分节省。**模型形状（layer 数 vs hidden）**：相同参数量下，hidden 大、layer 少，通信开销低、TP 友好；hidden 小、layer 多，PP 友好。预算估算时要带形状。

### 5. 常见追问 / 易错点
- 算预算只算"成功一次的训练"——忽略 ablation（×3–5）、failed runs（×1.2–1.5）、最终 scale 验证（×1.1）。生产预算应是首训公式的 4–6×。
- 忘记 **eval 与 RLHF**：post-training（SFT + RLHF + DPO）通常占 5%–15% 训练预算。
- MoE 模型公式不同：active params × tokens ≠ total params × tokens，要分别算。

### 6. 实践建议
建立**预算计算器**（Python 脚本或 spreadsheet），输入 N、T、精度、硬件、MFU 假设，输出 GPU-月与 $。模型立项时第一件事就是跑这个计算器，结果交评审。预留 30% buffer 给"我们不知道我们不知道的"。每次实际训练完毕，把 actual vs estimated 写进复盘——3–5 次后估算误差能压到 ±15%。

### 7. 30 秒速答
- 训练预算 = 6 × N × T / (peak × MFU)，单位 GPU-秒。
- 工程系数 ×1.3–1.5；ablation × 失败重试，生产是首训的 4–6×。
- MoE 用 active params；FP8 端到端 1.5–1.8×；不要只算成功一次。
- 加分关键词：Chinchilla、6N、active params、工程系数、ablation

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清训练预算公式？
- [ ] 你能不能解释 6N 是怎么来的？
- [ ] 你能不能举一个 70B × 15T tokens 的 GPU-月估算？
- [ ] 你能不能说出预算反推的 4 个工程系数？

## Q14. 推理峰谷扩缩容：触发门槛与最小副本数

> 🟡 进阶 · 把 GPU 推理当 web 服务做 `scale to zero`，第一波突发流量直接 5 分钟冷启动把 SLA 打穿。模型权重 140 GB 加载 + graph capture 不是 pod 启动 5 秒能搞定的事。

### 1. 核心结论
**最小副本数 = 你能容忍的 cold-start 延迟 / 单副本启动时间**。GPU 推理副本启动 60–300 秒，远高于 CPU 容器（5–15 秒），所以**永远要保留 baseline replicas 应对 P95**，不能像 web 服务那样"零基线 + scale to zero"。扩容触发门槛：GPU utilization > 70% 持续 60 秒，或 queue depth > 阈值。

### 2. 底层原理
GPU 推理冷启动慢的根本原因：（1）模型权重加载（70B FP16 = 140 GB，从 S3 拉到本地 NVMe + 加载到 GPU 显存 60–180 秒）；（2）CUDA context / kernel JIT 编译 30–60 秒；（3）vLLM/TRT-LLM 的 graph capture 5–30 秒；（4）K8s / autoscaler 调度 + image pull 30–120 秒。**总冷启动 90–390 秒**，远超大多数 SLO。

### 3. 关键机制
- **副本数公式**：`min_replicas = ceil(P95_QPS × P95_latency / per_replica_capacity)`
- **HPA / KEDA 触发指标**：GPU util、TGI/vLLM `requests_running`、`requests_waiting`、TTFT P95。
- **预热策略**：weights 缓存到 hostPath / NVMe；image 提前 pull；模型 graph 提前 capture；保留"warm pool"（idle 但不 release）。
- **缩容**：触发阈值要远低于扩容（GPU util < 30% 持续 5 分钟），避免抖动。
- **scale step**：每分钟最多扩 + 30% 容量，避免 thundering herd 把模型仓库 / 网络打爆。
- **就绪检查**：readiness probe 必须等 graph capture 与首批 token 通过后才返回 200，避免流量切到未热的副本。

### 4. 工程权衡
**baseline 越大，冷启动风险越低，但闲置成本越高**。SaaS 推理服务典型配比：baseline = P50 流量 × 1.2，max = P99 × 1.5，scale 速度 = 每分钟扩容 ≤ 30%（避免 thundering herd 把存储 / network 打爆）。**多 SKU 共享 GPU**：MIG / multi-instance 让一张 H100 跑 2–7 个轻量模型，提升利用率，但隔离弱、调度复杂。

### 5. 常见追问 / 易错点
- "GPU 也能 scale to zero"——理论可以，实际除非客户接受 5 分钟冷启动否则不行。
- 用 Pod CPU/Memory util 触发 GPU 推理扩容——signal 错位，应该用 GPU util / queue / TTFT。
- 忽视 **model loading thundering herd**：同时启 50 个副本去 S3 拉权重，把 S3 / 网络打爆；要错峰或预热缓存。

### 6. 实践建议
画 24 小时流量曲线 + 副本数曲线，看二者是否对齐（leading 指标 5–10 分钟）。每个模型 SKU 单独评估冷启动时间，写入 runbook。在区域级别保留 **"warm pool"**（已 ready 的备用副本），数量 = P99 - 当前副本数 + 安全 buffer。每月做一次"突发演练"（QPS 翻 3 倍），观察扩容 SLA 是否达标。

### 7. 30 秒速答
- GPU 推理冷启动 90–390 秒，不能像 web 服务那样 scale to zero。
- 最小副本数 = 容忍 cold-start 延迟 / 单副本启动时间。
- 触发用 GPU util / queue / TTFT，不要用 CPU/Memory。
- 加分关键词：warm pool、graph capture、KEDA、thundering herd

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 GPU 推理为什么不能 scale to zero？
- [ ] 你能不能解释冷启动 90–390 秒的来源？
- [ ] 你能不能举一个 baseline / max / step 的副本配比？
- [ ] 你能不能说出 model loading thundering herd 的缓解方法？

## Q15. 容量规划：训练与推理资源的耦合与隔离

> 🔴 专家 · "训练晚上空着，给推理用就行了"——一句话听上去合理，实际训练 NCCL all-reduce 跟推理 P99 互相伤害，混布到同节点是 SLO 灾难现场。默认隔离，按需共享。

### 1. 核心结论
**默认隔离，按需耦合**。训练侧的 SLO 是"几天内完成"、推理侧是"P99 ms 级"，两者放同一池子会互相伤害。耦合的合理场景：低优先级 batch inference / fine-tuning 用训练池 spare capacity；推理 burst 高峰临时借训练 spot capacity（接受 SIGTERM）。

### 2. 底层原理
训练任务是 **gang-scheduled、长时间、高带宽通信**；推理任务是 **single-pod 或 small-replica、短时、low-latency-sensitive**。混布的根本矛盾：训练 NCCL all-reduce 占满 NIC 时，同节点推理 latency 抖动 10–100 ms；推理 burst 抢占训练 GPU 时，训练 NCCL hang。**MIG**（H100 7-way）和 **MPS** 能做轻量隔离，但通信侧仍然冲突。

### 3. 关键机制
- **集群划分**：训练分区（topology-aware、IB 高带宽、长任务）、推理分区（多 region、SLO-aware、自动扩缩容）、共享 burst 池（spot/preempt）。
- **优先级调度**：SLURM `Priority` / K8s `PriorityClass` + preemption；推理 high priority、训练 medium、batch low。
- **共享存储**：模型权重在 shared object store，训练完直接 promote 到推理。
- **MIG**：H100 切 7×10GB instance；适合小模型推理与开发。
- **MPS（Multi-Process Service）**：CUDA 进程级 GPU 共享，比 MIG 灵活但隔离弱，适合开发环境。
- **NUMA / 网卡亲和**：训练侧严格绑定 GPU↔NUMA↔NIC，推理侧可放松；混布要保留训练侧的拓扑硬约束。

### 4. 工程权衡
**完全隔离**（双集群）：成本高 5%–15%（无法跨用），但 SLO 可保证；**完全混布**：成本低，SLO 难。**实战中位选择**：90% 隔离 + 10% 共享 burst 池。共享池里跑：fine-tuning（短任务、无 SLO）、batch inference、ablation。共享池**不跑**：客户面向推理、关键训练 run。

### 5. 常见追问 / 易错点
- "训练完晚上 GPU 闲，给推理用"——推理 burst 不在你控制，训练日历也变化，靠人工对齐很快失败；要么用调度器自动化，要么放弃。
- MIG 跨多个 client 时**显存碎片**：1 张 H100 切 7 份后再合并需要 reset GPU，影响在线服务。
- 把 LLM 70B 推理塞进 MIG 7 份——单份显存不够，方案错位。

### 6. 实践建议
做容量规划时画 4 维：（训练 vs 推理）×（专用 vs 共享）。专用容量按 P50 配，共享 burst 池按 P95-P50 差额配。建立**容量市场**（内部）：训练团队 borrow 推理闲时容量要付影子账、反之亦然，让浪费 visible。每季度复盘"我们多大比例的容量在共享池里"，理想 5%–15%，过低浪费、过高 SLO 风险。

### 7. 30 秒速答
- 默认隔离，按需耦合；训练 NCCL 和推理 P99 互相伤害。
- 训练 gang-scheduled 长时高带宽，推理 small-replica 低延迟。
- 实战 90% 隔离 + 10% 共享 burst 池跑 fine-tune / batch inference。
- 加分关键词：MIG、MPS、PriorityClass、topology-aware

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清训练 / 推理为什么默认隔离？
- [ ] 你能不能解释 NCCL 与推理 P99 的互相伤害机制？
- [ ] 你能不能举一个共享 burst 池适合跑的任务？
- [ ] 你能不能说出 MIG / MPS 各自的使用边界？

## Q16. SLO 反推容量：从 P99 latency 到副本与显存预算

> 🔴 专家 · 销售签了 `P99 TTFT < 800ms`，工程没有反推一遍副本数 + KV 预算就接单，上线第一周就要紧急加机。SLO 拍出来之后立刻翻成 `replicas = QPS × TTFT / (1-ρ)` 是基本功。

### 1. 核心结论
**给定 SLO（P99 TTFT < X ms、TPOT < Y ms），反推所需 GPU 数与显存配置**。流程：（1）测单副本最大并发；（2）算 P99 队列延迟 → 副本数；（3）算单副本 KV 缓存显存上限 → batch size 上限；（4）叠加 20%–30% headroom。

### 2. 底层原理
Latency = serving_time + queue_time。serving_time 由模型 + 硬件决定（不可压缩）；queue_time 由副本数 + 并发分布决定（可调）。M/M/c 排队论：当 ρ = λ/(c×μ) > 0.7 时 P99 显著抬升。**目标 ρ ≤ 0.6** 给 P99 留余量。

### 3. 关键机制
- **TTFT 拆分**：调度排队 + prefill 计算 + KV 写入。70B FP8 在 H100 prefill 1k tokens ≈ 80–200 ms。
- **TPOT**：每个 decode token 时间，70B 在 H100 batch 32 ≈ 30–50 ms/token。
- **副本数**：`replicas = QPS_p99 × TTFT_target / (1 - ρ)`。
- **显存预算**：weights + KV 缓存 + activations。70B FP8 = 70GB；KV per token per layer = 2 × hidden_dim × bytes；80 layers, hidden 8192, FP16 KV ≈ 2.6 MB/token；2k context × 32 batch ≈ 170 GB（超 H100，需要分片或缩 batch）。
- **prefix cache**：高 prefix hit rate（> 50%）时 effective TTFT 降到 1/3–1/5；要把 hit rate 写进容量模型。
- **multi-tenant batching**：请求合批后单 GPU 吞吐 5–10× 单请求，但要付出 P99 抖动；调度器要平衡 fairness 与 throughput。

### 4. 工程权衡
**降低 P99 的 3 条路**：加副本（直接但贵）、减 batch 抖动（要求调度器更聪明）、压模型（量化 / 蒸馏 / cascade）。前两条是 ops，第三条是 model。**优先做 ops**——投入 1 周减抖动比训一个新模型快 100×。**KV 缓存 offload to CPU/disk** 是 2025 之后的新选项，能把 effective context 拉到 128k+ 但 P99 抖动要单独评估。

### 5. 常见追问 / 易错点
- 用平均 latency 反推副本数——P99 反推才有意义，否则永远超 SLO。
- 忽视 **input length 长尾**：1% 请求 input 8k+ token，prefill 时间 8×，挤占其他请求。要么 admission control，要么单独 SKU。
- 没有 chaos testing：实际生产 P99 受 GC / kernel preempt / NCCL 抖动影响，纸面公式与实测差 20%–50%。

### 6. 实践建议
建一个**容量计算器**：输入 SLO + QPS 分布 + 模型规格，输出建议副本数 + 单副本 batch + KV 缓存配置。每季度用真实流量回灌测试，校准参数。生产环境同时跑 **shadow traffic**（影子流量），观察新配置在真实分布下的 P99。把 SLO 反推作为模型上线 gating 条件——不达标不发布。

### 7. 30 秒速答
- 副本数 = QPS_p99 × TTFT_target / (1 - ρ)，目标 ρ ≤ 0.6。
- TTFT = 排队 + prefill + KV 写；用 P99 不用平均反推。
- 降 P99 三条路：加副本、减抖动、压模型；优先做 ops。
- 加分关键词：M/M/c、prefix cache、admission control、shadow traffic

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 SLO 反推副本数的公式？
- [ ] 你能不能解释 ρ 为什么要压到 0.6 以下？
- [ ] 你能不能举一个 70B FP8 H100 batch 32 的显存预算？
- [ ] 你能不能说出 input 长尾对 P99 的破坏？

## Q17. PyTorch 主版本升级（2.4 → 2.5 → 2.6 → 2.7+）的时机与回滚策略

> 🟡 进阶 · 看到 release note 就直接 inplace 升级，第二天 `torch.compile` 行为变了 / NCCL 绑定改了 / numerics 漂了，一连串 incident 找不到锚。升级 PyTorch 从来不是 patch level 的事。

### 1. 核心结论
**主版本一年滞后 1 个 minor**。新版本 release 后留 3–6 个月观察 issue 与上游生态（FlashAttention / vLLM / DeepSpeed / Megatron）跟随情况，再上量产。**永远保留前一个 minor 作回滚版本**，CI 跑双版本至少 2 个月。强制升级触发：（1）新硬件需要新版本（B200 要求 2.5+）；（2）严重 CVE；（3）下游栈不再支持旧版本。

### 2. 底层原理
PyTorch 主版本 minor 间通常包含：（1）torch.compile 行为变化；（2）FSDP / FSDP2 API 调整；（3）NCCL / cuDNN 版本绑定升级；（4）算子默认精度或 layout 变化。这些都可能在你不可见的地方改变 numerics 或性能。"看似 patch level 升级"实际等价于半重写一部分模型代码。

### 3. 关键机制
- **CI 矩阵**：主线 + 候选版本双跑 2 个月，跨 train + infer + checkpoint roundtrip。
- **Numerics 回归**：固定种子训练 100 step，对比 loss 曲线、grad norm、actual logits；差异 > 1e-4 触发 review。
- **性能回归**：基准 MFU、step time、memory peak；下降 > 5% 阻断升级。
- **回滚预案**：保留旧 wheel 镜像、旧 CUDA 镜像；线上配置可以一键切换。
- **依赖联动**：PyTorch upgrade 通常要同步 cuDNN / NCCL / FlashAttention；列出 transitive deps 一并升级避免 ABI 冲突。
- **canary 比例**：5% → 25% → 50% → 100%，每阶段至少 24 小时观察期。

### 4. 工程权衡
**频繁升级**（每个 minor）：吃到性能改进与新特性，但每次 1–4 周工程投入。**滞后升级**（年度）：稳定但错过 FP8 / FlashAttention v3 / FSDP2 这种关键能力。**实际节奏**：训练栈（Megatron / TorchTitan）跟随激进，推理栈（vLLM / SGLang）跟随更激进，平台底层（K8s operator / monitoring）跟随保守。

### 5. 常见追问 / 易错点
- "patch 升级（2.5.0 → 2.5.1）安全"——一般是，但 PyTorch 的 patch 偶尔包含 numeric fix，仍要回归测试。
- 忽视 **CUDA / cuDNN 绑定**：升 PyTorch 经常隐式升 CUDA，影响所有自定义 kernel。
- 在生产推理服务直接 inplace 升级——必须 canary，先 5%、再 50%、再 100%。

### 6. 实践建议
画**版本依赖图**：PyTorch / CUDA / cuDNN / NCCL / FlashAttention / Triton / vLLM / Transformers，明确每个版本与其他版本兼容矩阵。建立 **2 个并行的"黄金镜像"**：current production + next candidate；后者跑 dev/staging 至少 4 周。每次升级写 RFC，列变更项 / 风险 / 回滚步骤。把 "最近一次升级到 GA 时间" 作为团队健康度指标，> 12 个月就该警惕技术债。

### 7. 30 秒速答
- 主版本一年滞后 1 个 minor；强制升级触发：新硬件、CVE、下游栈断供。
- minor 升级隐含 NCCL/cuDNN/算子精度变化，不是 patch 等价。
- 保留前一个 minor 作回滚；CI 双版本至少 2 个月。
- 加分关键词：FSDP2、torch.compile、numerics regression、canary

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 PyTorch 升级的节奏？
- [ ] 你能不能解释为什么 minor 升级要做 numerics regression？
- [ ] 你能不能举一个 5/25/50/100 canary 流程？
- [ ] 你能不能说出 3 个强制升级触发条件？

## Q18. FSDP1 → FSDP2 迁移的决策与时间点

> 🔴 专家 · FSDP2 基于 `DTensor` per-parameter shard、和 `torch.compile` 配合更好，新项目直接用没毛病；存量稳定的 FSDP1 急着迁就是给自己挖坑——但 12–18 月内不规划，技术债就锁死。

### 1. 核心结论
FSDP2（2024 上线，2025 稳定）相对 FSDP1 提供更细粒度的 sharding（per-parameter）、更好的 compile 兼容、更清晰的 hook 模型。**2026 年新项目默认 FSDP2**；存量 FSDP1 如果稳定且达到 MFU 目标，**不必急着迁移**——但要在 12–18 个月内规划，因为 FSDP1 长期会进入维护模式。

### 2. 底层原理
FSDP1 基于 `FlatParameter` 把整个 module 的参数 flatten 后 shard，简单但对 mixed-precision、hooks、自定义 init 不够友好。FSDP2 基于 `DTensor`，每个 parameter 独立 shard，与 PT2 graph capture / torch.compile 配合更好，且支持更复杂的 partial replication 与 2D mesh。

### 3. 关键机制
- **API 变化**：`FSDP(...)` → `fully_shard(module, ...)`；no more `_orig_mod` indirection。
- **Mixed precision**：FSDP2 通过 `MixedPrecisionPolicy` 显式控制 param/reduce/buffer 三种 dtype。
- **State dict**：FSDP2 与 `torch.distributed.checkpoint` 集成更紧；权重 save/load 用 DCP API。
- **Compile 兼容**：FSDP2 + torch.compile 在 2025 H2 后 production-ready。
- **2D / 3D parallelism**：FSDP2 与 TP / PP 用 `DeviceMesh` 表达，比 FSDP1 + Megatron 自管 process group 更清晰。
- **CPU offload**：FSDP2 的 `CPUOffloadPolicy` 选项更细，可只 offload optimizer state 不 offload params。

### 4. 工程权衡
**迁移成本**：API 重写、checkpoint format 不互通、CI 全量重跑。中型项目 2–6 周，大型项目（Megatron 风格自定义）3–6 个月。**收益**：MFU 提升 5%–15%（compile + per-param）、可与 TP/PP/EP 组合更灵活。**风险**：早期 FSDP2 在某些 MoE / 长序列场景仍有 corner case；不要做先锋。

### 5. 常见追问 / 易错点
- 把 FSDP1 checkpoint 直接 load 到 FSDP2——格式不互通，要走 DCP 或全量 load 后重新 shard。
- 迁移时同时升 PyTorch 主版本——一次只改一件事，否则定位回归极困难。
- 误以为 FSDP2 自动比 FSDP1 快——纸面快，实际要重新调 `forward_prefetch` / `reshard_after_forward` / `bucketing`。

### 6. 实践建议
分 4 阶段：（1）小模型 PoC（< 1B）验证 API；（2）单节点 70B 验证 numerics + perf；（3）多节点 scale-out；（4）完整 checkpoint roundtrip + 重启。每阶段写 RFC + benchmark 报告。**不要一次性迁移所有训练任务**，优先迁新项目，存量等成熟。预留 6 周 buffer，重大迁移历来超期。

### 7. 30 秒速答
- 2026 新项目默认 FSDP2，存量 FSDP1 稳定就不必急迁。
- FSDP2 基于 DTensor per-param shard，与 torch.compile 配合好。
- 一次只改一件事；checkpoint 格式不互通，走 DCP 转换。
- 加分关键词：DTensor、fully_shard、DCP、DeviceMesh、CPUOffloadPolicy

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 FSDP2 vs FSDP1 的本质差异？
- [ ] 你能不能解释 FSDP2 与 torch.compile 的兼容优势？
- [ ] 你能不能举一个 4 阶段迁移路径？
- [ ] 你能不能说出 FSDP1 checkpoint 不能直接 load 到 FSDP2 的原因？

## Q19. 模型版本冻结：什么时候停止追新模型

> 🔴 专家 · 每周追新模型听上去很 cool，客户的 prompt、评测基线、安全规则全在你脚下漂移。模型在生产里是行为契约，不是版本号——冻结 6–12 个月是商业责任不是技术保守。

### 1. 核心结论
**生产服务的模型版本要冻结 6–12 个月**，期间只接受安全 patch 与极小的 tuning。频繁换模型版本带来：（1）客户 prompt regression；（2）评测基线漂移；（3）下游应用兼容性问题；（4）成本不可控。冻结期内做 A/B 验证下一代，到期一次性切换。

### 2. 底层原理
模型在产品中起到 **"行为契约"** 的角色——客户写的 prompt、应用集成的 schema、安全审核规则、评测集，全部是基于某个具体版本调出来的。换版本即使主观感觉"更好"，也会让 5%–20% 的 use case 出现 regression。这是一个**生态性技术债**，不是单个模型问题。

### 3. 关键机制
- **冻结策略**：每个模型 SKU 标 `frozen-since` 时间戳；线上仅接受 critical fix 与 safety patch。
- **下一代评估**：影子流量并跑 4–8 周，对比客户级指标（接受率、重写率、退订率），不只是 MMLU/HumanEval。
- **切换流程**：宣布 EOL → 90 天 deprecation 期 → 新旧并跑 → 切换 → 30 天观察期。
- **回滚开关**：切换后保留旧模型 30–90 天热备。
- **版本元信息**：API response 里返回 `model_version` / `served_by`，便于客户端排查回归。
- **safety patch 通道**：与新版本独立，patch 只允许动 safety 层（拒答规则 / 内容审核），不动 base model。

### 4. 工程权衡
**追新 vs 冻结**：追新拿到能力红利，冻结拿到稳定性。SaaS 推理产品偏冻结（客户依赖契约），内部应用可激进（自家可控）。**多 SKU 策略**：同时维护 stable / preview 两条线，客户可选择。preview 线允许快速换版，stable 线严格冻结。

### 5. 常见追问 / 易错点
- "新模型 benchmark 高，立即切换"——benchmark 高 ≠ 客户 use case 表现好，要分布式评估。
- 忽视 prompt drift：旧模型与新模型对相同 prompt 行为不同，所有客户 prompt 都要重审。
- 不留 deprecation 期，直接切——客户合同义务 / 法律风险。

### 6. 实践建议
把模型当成 **API 契约管理**，遵循 SemVer 思维：major（行为变化，需 deprecation）、minor（能力增强，向后兼容）、patch（safety / bug fix）。建立 **"model usage dashboard"**：每个 SKU 当前 QPS、客户分布、依赖关系。冻结期写进 SLA。每年规划模型路线图，提前 6 个月通知客户即将 EOL 的 SKU。

### 7. 30 秒速答
- 生产模型版本冻结 6–12 个月，期间只接受 safety patch。
- 模型在产品中是行为契约，换版本必引起 5%–20% regression。
- 双 SKU 策略 stable + preview；EOL 走 90 天 deprecation。
- 加分关键词：行为契约、prompt drift、deprecation 期、SemVer 思维

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清模型冻结的核心理由？
- [ ] 你能不能解释 benchmark 高 ≠ 客户表现好？
- [ ] 你能不能举一个 deprecation 完整流程？
- [ ] 你能不能说出"行为契约"在产品里的具体体现？

## Q20. 自研框架 vs 用开源（PyTorch / Megatron / TorchTitan / DeepSpeed）的边界

> 🔴 专家 · 一冲动想自研训练框架，5 年人力 + 招聘困难 + 错过上游 FlashAttention v3 / FSDP2 红利全部到账。99% 的团队都不是 Meta，老老实实用开源加 plugin 才是正解。

### 1. 核心结论
**默认 100% 用开源**。自研只在两种情况合理：（1）你的硬件 / 拓扑 / scale 上游不支持；（2）你有明确的 5+ 年差异化需求并能投入 20+ 人年维护。其他情况自研都是技术债与人才浪费。2026 年开源训练栈（PyTorch + TorchTitan / Megatron-LM / NeMo）已经能撑 1 万卡级训练。

### 2. 底层原理
自研框架的真实成本：（1）核心开发 5–20 人年；（2）持续维护 3–8 人年/年；（3）跨硬件代际 porting（H100 → B200 → 下一代）；（4）招聘困难（候选人不熟悉你的栈）；（5）失去上游优化红利（FlashAttention / FSDP2 / async TP）。**只有当上游缺失能力对你的产品差异化关键**时才划算。

### 3. 关键机制
- **拥抱开源的边界**：用 PyTorch 主线 + TorchTitan 或 Megatron-LM 做训练，vLLM / TensorRT-LLM 做推理。在它们之上做自定义 kernel、调度、数据加载。
- **自研合理项**：自定义 fused kernel（Triton 写）、调度器（基于 K8s/SLURM 加层）、checkpoint 格式（在 DCP 之上扩展）。
- **不要自研**：autograd 引擎、分布式通信库、attention 实现。
- **判断标准**：上游有 90% 你需要的能力？用并 contribute 剩下的 10%。
- **plugin / extension first**：能用 hook / callback / custom op 解决的，不要 fork 主框架。
- **upstream contribution policy**：自研出来的通用能力定期回流上游，避免长期 fork divergence。

### 4. 工程权衡
**自研框架的人才陷阱**：核心 2–3 个人离职后整个栈无人维护；新人上手 6–12 个月。**用开源的脆弱**：上游 breaking change 强迫跟随；某些能力延迟到下个 release。**实战中位选择**：用开源做 80% 主路径 + 自研 20% 差异化，且差异化部分用 plugin 形式而不是 fork。

### 5. 常见追问 / 易错点
- "我们规模大，开源不够用"——2024–2026 上游训练栈已经支持 10000+ 卡（如 Llama 3 用 Meta 内部 fork 的 PyTorch / FSDP，但核心仍是 PyTorch）。99% 团队的"规模"达不到那个边界。
- fork 上游做大改——3–6 个月后 merge 不回去，永远停在某版本。
- 自研推理引擎与 vLLM / SGLang 竞速——除非你有团队 30+ 人专做，否则跟不上。

### 6. 实践建议
做"build vs buy"评估时，列**3 张表**：（1）上游能力清单 vs 我们需求清单；（2）自研 5 年总成本（人力 + 机会成本）；（3）依赖上游的风险（vendor lock、breaking change）。除非自研有清晰 ROI，**默认 buy**。每个季度复盘自研模块，问"上游是否已经有了？"——如果是，放弃自研、迁回上游。

### 7. 30 秒速答
- 默认 100% 用开源；自研只在硬件不支持或 5+ 年差异化合理。
- 自研真实成本 = 核心 5–20 人年 + 维护 3–8 人年/年 + 失去上游红利。
- 用 plugin/extension 解决，不要 fork 主框架。
- 加分关键词：TorchTitan、Megatron-LM、vLLM、plugin first

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 build vs buy 的判断标准？
- [ ] 你能不能解释自研框架的"人才陷阱"？
- [ ] 你能不能举一个合理的自研边界（kernel / 调度 / checkpoint）？
- [ ] 你能不能说出 fork 上游做大改的长期后果？

## Q21. 训练 / 推理栈的演进路线图：12–24 个月规划怎么写

> 🔴 专家 · 路线图只画 12 个月，硬件采购周期偏偏 18 个月，B200 一延期整个计划崩盘。三层 cadence（硬件 / 栈 / 产品）不对齐，路线图就是 wishlist。

### 1. 核心结论
路线图分**3 层**：硬件层（GPU 代际、网络、存储）、栈层（PyTorch / NCCL / vLLM 等版本）、产品层（模型 SKU、客户能力、SLO）。这三层有 **不同的 cadence**：硬件 18–24 个月一代，栈 6–12 个月一波，产品 3–6 个月一迭代。路线图要把三层对齐到同一时间轴并标依赖。

### 2. 底层原理
路线图失败的常见模式：（1）只列产品 feature，没考虑底层栈支持；（2）硬件计划与软件升级时间错位（B200 上来时 PyTorch 还没准备好）；（3）没有 "kill switch"——某条路径如果延期，依赖项需要回滚。**好路线图**回答三个问题：现在做什么、做完什么算成功、失败回到什么备选。

### 3. 关键机制
- **硬件路线**：当前代占比 → 下一代 PoC 时间 → 全面切换时间 → 上一代退役时间。
- **栈路线**：PyTorch / NCCL / vLLM / Megatron 每个组件的 current / next / target 版本。
- **产品路线**：模型 SKU 列表 + 每个 SKU 的 GA / EOL / 下一代发布。
- **依赖图**：用 GraphViz 或表格画哪个能力依赖哪个底层升级。
- **里程碑模板**：每条 milestone 含 owner、deliverable、success metric、risk、rollback。
- **预算关联**：每条 milestone 标注消耗的 GPU-月与人月，避免 wishlist。

### 4. 工程权衡
**激进路线**（追新硬件 + 新栈）：吃到 1.5–2× 性能 / cost 红利，承担 3–6 个月不稳定。**保守路线**（一年一升级）：稳定但错过窗口，被竞品超越。**最佳实践**：硬件保守（晚 6 个月入场，避开早期 silicon bug），栈激进（紧跟 PyTorch / vLLM 主线），产品中位（季度迭代）。

### 5. 常见追问 / 易错点
- 路线图只到 12 个月——硬件采购周期 18 个月，必须画到 24 个月。
- 没有 "if not, then" 分支——B200 延期 6 个月时计划崩盘。
- 把研究 ambition 写进工程路线图——"我们要训 1T 模型"如果没对应预算与硬件，是 wishlist 不是 roadmap。

### 6. 实践建议
每季度更新一次 12 月滚动路线图，每年更新一次 24 月长程版本。每条 milestone 写**3 件事**：deliverable、success metric、rollback。建立**月度路线图 review**：跨研究 / 训练 / 推理 / 平台四方对齐。把 vendor commit（云容量、长租 GPU）与路线图绑定——路线图变 → 容量重谈。每年至少一次"红队评审"：假设 50% milestone 延期，业务还能活吗？

### 7. 30 秒速答
- 路线图分 3 层：硬件 18–24 月、栈 6–12 月、产品 3–6 月。
- 每条 milestone 写 deliverable / success metric / rollback。
- 实战节奏：硬件保守、栈激进、产品中位。
- 加分关键词：cadence 对齐、kill switch、24 月长程、依赖图

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清三层 cadence 的差异？
- [ ] 你能不能解释路线图为什么必须到 24 月？
- [ ] 你能不能举一个 milestone 的完整写法？
- [ ] 你能不能说出"研究 ambition vs 工程 roadmap"的区别？

## Q22. ML 平台团队的组织结构与人数配比

> 🔴 专家 · 平台团队 5 人扛不住、20 人变官僚，行业经验值是总 ML 头数的 5%–10%。招进来全是 ML 研究员就更尴尬——平台需要的是 distributed systems / SRE 背景。

### 1. 核心结论
**ML 平台团队 ≈ 5–10% 总 ML 头数**（含研究 + 工程）。一个 50 人 ML 组织通常配 3–5 人平台团队；500 人组织 30–50 人。**核心 4 个子团队**：训练基础设施、推理基础设施、数据基础设施、平台 SRE。研究团队 / 应用团队是平台团队的客户，不是同级。

### 2. 底层原理
平台团队存在的目的是**抽象掉"重复 + 系统级"工作**，让研究与应用聚焦模型与产品。如果平台团队过小，研究团队各自造轮子（重复浪费）；过大则成为瓶颈与官僚。**5%–10%** 是行业经验值，对应"每个研究员 / 应用工程师有 1 个平台工程师服务 10–20 人"。

### 3. 关键机制
- **训练 infra 团队（30%–40%）**：调度器、训练框架封装、checkpoint 系统、训练 observability。
- **推理 infra 团队（25%–35%）**：推理引擎封装、服务化、autoscaling、SLO 工具链。
- **数据 infra 团队（15%–25%）**：数据集存储、ETL、向量库、特征存储。
- **平台 SRE（10%–20%）**：on-call、cluster ops、容量规划、cost FinOps。
- **跨职能**：技术负责人、产品经理、research engineer 接口人。
- **背景结构**：60% 系统/分布式背景 + 30% ML 背景 + 10% 产品/业务背景。
- **report 关系**：平台团队向 CTO 或独立 VP 汇报，避免被研究 / 产品单独绑架。

### 4. 工程权衡
**集中型平台团队**：抽象层好、效率高、但可能与研究脱节。**嵌入型 platform engineers**（每个研究小组配 1 人）：贴近需求、但工具链碎片化。**实战折衷**：核心平台集中 + 研究小组配 1–2 个 embed engineer，每季度轮换。**避免**：研究员自己写训练框架（短期快、长期维护噩梦）。

### 5. 常见追问 / 易错点
- "我们 10 个人就先把平台搭起来"——10 人能搭起来但难持续；6 个月后单点风险爆雷。最小可持续团队 5 人。
- 平台团队招了一堆 ML 研究员——错位，平台需要的是 distributed systems / SRE 背景的工程师。
- 把 platform 与 research 放进同一个 OKR——目标错位，研究追前沿、平台追稳定。

### 6. 实践建议
组织 50 人以上 ML 团队时设独立平台部门，向 CTO / VP Eng 汇报（不是向研究 VP）。**3 大职责写进 charter**：通用工具、统一抽象、SLO 责任。每季度做 internal CSAT 调查（研究员 / 应用工程师对平台团队评分），低于 70 分启动整改。**禁止平台团队成为 ticket queue**——要主动产品化，不是被动响应需求。

### 7. 30 秒速答
- 平台团队 ≈ 5%–10% 总 ML 头数；50 人组织配 3–5 人。
- 4 子团队：训练 infra、推理 infra、数据 infra、平台 SRE。
- 60% 系统/分布式 + 30% ML + 10% 产品；向 CTO 汇报。
- 加分关键词：embed engineer、internal CSAT、charter

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清平台团队的核心职责？
- [ ] 你能不能解释为什么 5%–10% 是合理比例？
- [ ] 你能不能举一个 4 子团队的人数配比？
- [ ] 你能不能说出招 ML 研究员去做平台的错位？

## Q23. on-call / 值班轮换：ML 平台与传统 SRE 的差异

> 🔴 专家 · 把 NCCL hang 的 alert 丢给传统 SRE 团队，他们看一眼日志直接 escalate 给 ML 组，等于双倍延迟。ML 值班需要 ML+sys 双背景，runbook 还得长得不一样。

### 1. 核心结论
ML 平台 on-call 与传统 SRE 有 3 大差异：（1）训练任务 hang 比 service down 复杂（NCCL / topology / numerics）；（2）值班需要 ML + sys 双背景，纯 SRE 处理不了；（3）alert SNR 低，调优期长。**轮换最佳实践**：2–3 人 primary + secondary，1 周一轮，研究/工程混编。

### 2. 底层原理
传统 SRE alert 对应"服务 down / 慢"，根因常在 5–6 个共性领域（DB、缓存、网络、磁盘、容量、配置）。ML 平台 alert 多了：训练 NCCL hang、checkpoint 损坏、推理 P99 抖动、模型 numeric drift、GPU ECC 故障、autoscaler 抖动。这些根因需要懂 ML stack + sys，且 alert 误报率高。

### 3. 关键机制
- **alert 分级**：
  - P1：客户面向推理 down、训练大规模 hang > 30 min；
  - P2：训练 step time 退化、单节点 GPU 故障；
  - P3：集群利用率异常、模型 numerics 漂移。
- **响应时间**：P1 5 min ack / 30 min mitigate；P2 30 min ack / 2 h mitigate；P3 当日工作时段处理。
- **runbook**：每个 alert 链接 runbook，含定位步骤 + 常见根因 + 升级路径。
- **post-mortem**：P1 + P2 都写复盘，P3 季度汇总。

### 4. 工程权衡
**专职 on-call** vs **轮班 on-call**：专职单点风险高、burn-out；轮班分摊但每个人 ramp-up 慢。**实战**：轮班 + 强 runbook + escalation 路径。**24×7 vs 工作时段**：客户面向推理 24×7；纯训练值班可以"工作日工作时段 + 凌晨重大事故 escalate"。

### 5. 常见追问 / 易错点
- 把 ML 平台 alert 全送 SRE 团队——他们看不懂 NCCL log，会快速 escalate 给 ML 团队，等于双倍延迟。
- runbook 写得过于教条——ML 故障常见 unique，runbook 应包含"调查步骤"而非"固定脚本"。
- 没有 alert quiet 机制——单节点 GPU ECC 一晚上发 50 个 alert，疲劳。

### 6. 实践建议
建立 **3 类 dashboard**：训练健康（MFU、step time、checkpoint）、推理健康（QPS、P99、错误率）、基础设施健康（GPU util、网络、存储）。每周 review alert 噪声，删除或调阈值；目标 P1 < 1 个/月、P2 < 5 个/月。**onboarding 流程**：新人 4 周影子值班，独立轮班前必须独立处理过至少 3 个 P2。把"on-call burden" 量化到周报，超过阈值触发流程优化。

### 7. 30 秒速答
- ML on-call 比传统 SRE 多 NCCL hang / numerics / GPU ECC 等根因。
- 值班需要 ML + sys 双背景，2–3 人 primary/secondary 一周一轮。
- alert SNR 低，每周清理噪声；P1 < 1 / 月、P2 < 5 / 月为目标。
- 加分关键词：runbook、post-mortem、shadow on-call、escalation 路径

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 ML 值班与传统 SRE 的关键差异？
- [ ] 你能不能解释 P1 / P2 / P3 alert 的响应时间标准？
- [ ] 你能不能举一个 NCCL hang 的 runbook 思路？
- [ ] 你能不能说出 alert quiet 机制的必要性？

## Q24. ML 代码评审：与传统软件评审的差异点

> 🟡 进阶 · ML 代码 review 比业务代码多 3 个维度：numerics、性能、可复现。一个 silently incorrect 的 attention 改动跑通 1 周才暴露，那时候已经训了一半模型，谁来背锅。

### 1. 核心结论
ML 代码评审比传统代码评审多 3 个维度：（1）**numerics**（精度、数值稳定）；（2）**性能**（kernel 性能、显存使用）；（3）**可复现性**（seed、版本、环境）。**最低门槛**：每个 PR 至少 1 个 reviewer 来自 ML 系统侧；涉及训练 loop 改动需 2 reviewer；涉及自定义 kernel 需 1 个 CUDA 专家。

### 2. 底层原理
ML 代码的特殊性：（1）正确但慢的代码 = 业务 break（训练超期、推理超 SLO）；（2）silently incorrect 代码（小数值 bug）可能跑通 1 周后才暴露；（3）非确定性（CUDA 浮点、数据加载顺序）让 review 时难以推断行为。这些都需要专门的 review checklist。

### 3. 关键机制
- **review 三段论**：
  - 正确性：对比 reference impl、单元测试、numerics 容差。
  - 性能：profile 数据、MFU 对比、显存对比。
  - 可维护性：依赖、版本、可复现脚本。
- **强制项**：
  - 改训练 loop：附 1 个 small-scale 训练对比图。
  - 改自定义 kernel：附 nsys / ncu profile + 与 reference 数值对比。
  - 改推理引擎：附 P50/P99 latency + throughput 对比。
- **CI gate**：所有改动通过基础 numerics regression（固定种子 100 step loss 一致）。

### 4. 工程权衡
**严格 review** vs **快速迭代**：研究侧迭代快、容忍 hack；生产侧严格、容忍慢。**双轨制**：research / experimental 分支宽松（自评 + 1 reviewer），main / production 分支严格（2 reviewer + CI + perf check）。**review 速度 SLA**：24h 内首轮 review，48h 内 closed loop。

### 5. 常见追问 / 易错点
- "ML 改动太复杂，reviewer 看不懂"——拆分 PR、附测试、附 benchmark。reviewer 不该花 1 小时理解一个 PR。
- 不跑 CI 直接 merge——尤其改 attention / norm / loss，必须跑 numerics regression。
- review 只看 diff，不看上下文——ML 代码常依赖 data 分布与硬件特性，要在 PR description 里注明。

### 6. 实践建议
建立**两份 checklist**：研究 PR checklist（5 条）、生产 PR checklist（10 条）。在 GitHub / GitLab 模板里放进去，提交者自检后再请 reviewer。**Reviewer 池子分类**：sys / kernel / numerics / data 各 1–2 人；自动 routing。每季度做 review 质量评审：抽 10 个已 merged PR 重审，看是否漏过 bug。把 "review turnaround time" 作为团队健康指标。

### 7. 30 秒速答
- ML review 多 3 维：numerics、性能、可复现性。
- 训练 loop 改动需 2 reviewer；自定义 kernel 需 CUDA 专家。
- 双轨制：research 宽松 + production 严格 + CI numerics regression。
- 加分关键词：fixed seed、perf benchmark、reviewer 池子、PR template

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 ML 代码评审的 3 个额外维度？
- [ ] 你能不能解释 silently incorrect 的危险性？
- [ ] 你能不能举一个 attention 改动的强制 review 项？
- [ ] 你能不能说出"研究 / 生产"双轨制的合理标准？

## Q25. 研究团队与工程团队的接口规范

> 🔴 专家 · 研究和工程在源码层耦合，一个 train.py 共享，最后双方都嫌对方拖后腿。把接口定在"产物层"——checkpoint + config + eval 报告，问题瞬间清晰一半。

### 1. 核心结论
**接口边界不在代码层，而在"产物层"**：研究团队交付"训练好的 checkpoint + 评测报告 + reproducible config"；工程团队负责"上 infra、上推理、上产品"。把双方耦合在源码层（共享 train.py）会持续摩擦——研究要快迭代，工程要稳定。

### 2. 底层原理
研究与工程是两种**不同优化目标**的工作模式：研究优化"找到一个能 work 的方案"，工程优化"让方案在生产稳定运行"。两者节奏、风险偏好、代码标准都不同。强制用同一套流程必然出问题：研究嫌工程慢，工程嫌研究脏。**接口规范**就是定义双方的"交付契约"。

### 3. 关键机制
- **研究交付物**：
  - checkpoint（safetensors + metadata）；
  - 训练 config（YAML / JSON，含 git SHA、data hash）；
  - 评测报告（标准 benchmark + custom eval）；
  - 已知限制（known issues、failure modes）。
- **工程交付物**：
  - 推理服务（meets SLO）；
  - 监控 / 告警；
  - 上线计划与回滚预案。
- **协作仪式**：
  - 周会：研究 demo + 工程 readiness review；
  - "promote-to-production" gate：研究侧 sign-off + 工程侧 sign-off。

### 4. 工程权衡
**完全分离**（研究丢 checkpoint 给工程）：清晰但易扯皮，工程拿到 checkpoint 才发现性能不达标。**深度耦合**（研究在生产代码里改）：快但 fragile。**实战折衷**：研究在 monorepo 的 `research/` 分支里自由迭代；工程在 `production/` 分支固化；定期 promote 经过 gate 的代码。

### 5. 常见追问 / 易错点
- "研究员自己上线"——多数情况下灾难，研究员不熟悉 SLO / observability / on-call。
- "工程团队不懂模型"——也是问题，需要工程侧懂 numerics / 评测，否则只会 dummy serving。
- 评测口径不统一——研究侧用 eval set A，工程侧用客户分布 B，结论南辕北辙。

### 6. 实践建议
写一份**双向接口文档**：研究→工程（交付物 + 评测口径 + checkpoint 格式）、工程→研究（infra 能力清单 + SLO 边界 + 故障模式）。每个 quarter review 一次，看协作摩擦点。建立 **"research engineer"** 角色，跨双方语言，担任翻译；这个角色 1–2 人能极大提升流程效率。把 "promote-to-production" 时间作为联合 KPI（共担）。

### 7. 30 秒速答
- 接口边界不在代码层，而在"产物层"：checkpoint + config + eval。
- 研究优化"能 work"，工程优化"稳定运行"，目标本就不同。
- 设 research engineer 角色翻译双方语言；评测口径必须统一。
- 加分关键词：promote-to-production gate、reproducible config、known issues

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清研究 / 工程的接口定义？
- [ ] 你能不能解释代码层耦合的常见摩擦？
- [ ] 你能不能举一个 promote-to-production 的 gate 清单？
- [ ] 你能不能说出 research engineer 这个角色的价值？

## Q26. ML 系统能力的工程文化建设：长期 vs 短期的平衡

> 🔴 专家 · 所有人都被产品 deadline 绑死，platform / tooling / 文档没人投入，6 个月后整支团队都在重做别人 6 个月前做过的事。70/20/10 的容量配比是把"复利"写进 OKR。

### 1. 核心结论
**短期看交付，长期看 platform 复利**。ML 团队常见反模式：所有人都被产品 deadline 绑架，没有时间投入 platform / tooling / 文档，6 个月后所有人都在重复别人 6 个月前做过的事。**健康配比**：70% 当期交付 / 20% platform 投资 / 10% 探索。

### 2. 底层原理
ML 工程的复利效应：好的 dataloader / profiler / checkpoint 系统能让所有未来训练任务节省 10%–30% 时间；好的内部 doc 能让新人 ramp-up 从 3 个月压到 1 个月。这种投入**1 次成本，永久收益**。但 platform 投资难以量化短期 ROI，容易被 shipping pressure 挤掉。

### 3. 关键机制
- **20% time 制度**：每个工程师每周固定 1 天做 platform / tooling / 文档。
- **季度 hack week**：1 周脱产，专攻技术债。
- **"trace-and-fix" 文化**：每次 incident 后必产出至少 1 个 platform improvement。
- **Internal Tech Talk**：每两周一次，分享系统侧学习。
- **Onboarding 文档 owner**：明确角色，每季度更新。
- **internal benchmark suite**：标准化的训练 / 推理基准，新人第一周跑一遍熟悉环境。
- **runbook 库**：每个核心组件配 runbook，季度复审。

### 4. 工程权衡
**Platform 投入过多** → 产品交付滞后，业务抱怨；**过少** → 技术债累积，6 个月后开发速度断崖下降。**平衡的信号**：（1）新人 ramp-up 时间稳定；（2）单 feature 交付时间稳定或下降；（3）on-call burden 不增长。任一指标恶化，调高 platform 投入比例。

### 5. 常见追问 / 易错点
- "我们没时间做 platform"——本质是没有把 platform 写进 OKR。把它写进去，时间就有了。
- 把"重构"当 platform 投入——重构不一定是 platform，platform 是给所有人用的工具与抽象。
- 一次 hack week 解决不了的问题，靠周期性 hack week 解决——慢性技术债需要持续投入，不是一次性冲刺。

### 6. 实践建议
团队每季度划 20% 容量给 platform / tooling / 文档，写进 OKR。**3 个保护机制**：（1）senior 工程师轮值 platform owner，1 季度；（2）每个 incident 必产出 1 个 platform PR；（3）季度 demo day 展示 platform 投入与收益。把"工程师离职率"与"新人 ramp-up 时间"作为文化健康指标——这两个数恶化时，先看是不是 platform 投入不够。

### 7. 30 秒速答
- 70 / 20 / 10：当期交付 / platform 投资 / 探索。
- platform 投入 1 次成本永久收益，但难量化短期 ROI。
- 把 platform 写进 OKR，否则被 shipping pressure 挤掉。
- 加分关键词：复利效应、hack week、trace-and-fix、ramp-up 时间

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 70/20/10 的逻辑？
- [ ] 你能不能解释 platform 投入与产品交付的平衡信号？
- [ ] 你能不能举一个 hack week 解决的典型技术债？
- [ ] 你能不能说出 platform 投入不足的 3 个症状？

## Q27. 推理 $/token 的精细成本分解：从原始 $/hour 到客户报价

> 🔴 专家 · 客户报价不是 GPU `$/hour` 直接除，要把训练摊销、出口、安全审核、冗余、毛利层层叠加。少算一层就是亏一层；spot 单价拿来当成本，effective 还要 +20%–40%。

### 1. 核心结论
**推理 $/output_token = 摊销 GPU $/hour ÷ (吞吐 tok/s × 利用率 × 3600)**。但这是**裸成本**，给客户报价还要叠加：（1）训练摊销；（2）网络出口；（3）观测与安全；（4）冗余与失败重试；（5）目标毛利。从裸成本到客户价，2026 年开源 70B 推理典型 markup 1.8–3.5×。

### 2. 底层原理
推理服务的成本结构是**多层金字塔**，最底是 GPU 算力，往上每层都有规模化分母与不可压缩固定成本。把所有层都摊到 token 上才能得到真实 $/token。常见错误是只看 GPU 层，忽略了上层 5%–30% 的隐性成本。

### 3. 关键机制
- **Layer 1：GPU compute**：H100 长租 $2/hour，70B FP8 vLLM batch 32 输出 ~5k tok/s/卡，利用率 0.7 → $/M tok ≈ 2 / (5000 × 0.7 × 3.6) = $0.16/M tok。
- **Layer 2：训练摊销**：70B 训练 $5M，预期生命周期 5T tokens → $1/M tok 摊销。但若 model 复用 5 个产品，摊到每产品 $0.2/M tok。
- **Layer 3：infra overhead**：网络出口（$0.05/GB × 0.5 KB/token avg = $0.025/M tok）、监控 / 日志 / 安全审核（5%–10% GPU 成本）、failover 冗余（10%–20% GPU 成本）。
- **Layer 4：毛利**：内部用 1.0×，企业 SaaS 1.8–2.5×，公有 API 2.5–4×。
- **input vs output**：input prefill 成本 ~1/5–1/10 of output，typical pricing 反映这点。
- **prefix cache discount**：高 hit 率客户的 effective input 成本进一步降到 1/3–1/5，部分厂商对外公布 "cached input" tier。

### 4. 工程权衡
**降本 4 条路**：（1）量化 FP8 → FP4（吞吐 ×1.5–2.0、质量监控成本上升）；（2）batch 拉大（throughput 上升、TTFT 上升）；（3）硬件代际切换（H100 → B200 单 token 成本降 30%–50%）；（4）混合模型 cascade（小模型先答，70B 兜底）。**注意**：单一指标极致优化容易牺牲其他维度——cost 降 50% 但 P99 翻倍，客户可能流失。

### 5. 常见追问 / 易错点
- "我用 spot $/hour 算成本"——spot 中断重试摊到分母后 effective cost 高 20%–40%。
- 忽视 KV 缓存显存对吞吐的影响——长 context 时 effective batch 缩水，$/token 上升。
- 训练摊销假设过于乐观——5T token 寿命如果实际只用了 1T 就被替换，摊销系数 5×。

### 6. 实践建议
搭建**5 层成本分解 dashboard**，按 SKU × region × customer-tier 切。每月 review 各层占比，找异常。**定价**：报客户的价格 = 裸成本 × markup × 安全 buffer（10%–20%）。每个新模型上线前算清生命周期预期 token 量，倒推训练摊销门槛——如果摊销 > 单 token 成本 50%，这个模型的商业可行性存疑。

### 7. 30 秒速答
- 5 层金字塔：GPU compute + 训练摊销 + infra overhead + 冗余 + 毛利。
- 开源 70B 推理裸成本 ~$0.16/M tok，客户报价 markup 1.8–3.5×。
- input vs output 不对称 5–10×；spot effective cost 高 20%–40%。
- 加分关键词：cached input tier、cascade、量化路径、FP8/FP4

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 5 层成本结构？
- [ ] 你能不能解释 input / output token 成本不对称的原因？
- [ ] 你能不能举一个 70B vLLM 的 $/M tok 估算？
- [ ] 你能不能说出降本 4 条路与各自陷阱？

## Q28. 训练 $/model：单次训练成本核算与摊销

> 🔴 专家 · "70B 训一遍 $1M"是 PR 数据，工程预算要乘 ablation × 5 + failed runs × 1.3 + post-training + eval。这个系数不写进立项表，CFO 月底就要找你聊聊。

### 1. 核心结论
**训练 $/model = GPU-hour × $/GPU-hour × 工程系数（1.3–1.5×）+ 数据成本 + post-training 成本**。2026 年典型 frontier 训练成本：7B $50k–$200k；70B $1M–$5M；405B $10M–$50M；MoE 1T+ $30M–$150M。这些数字是**单次成功训练**的"裸价"，加 ablation / failed runs 实际花费 3–5×。

### 2. 底层原理
训练成本是 sunk cost，必须在预期收益（产品营收 × 客户数 × 模型生命周期）之前**对齐**。如果摊销不下来，训练再大的模型也是消耗品。**业界经验**：训练成本 < 5% 预期 12 个月营收 = 健康；> 20% = 高风险。

### 3. 关键机制
- **裸训练成本**：预算反推（Q13）× 实际 GPU 单价。
- **数据成本**：预训练数据 license $1M–$50M（视来源）；标注 / RLHF 数据 $0.5M–$10M。
- **Post-training**：SFT + DPO + RLHF 通常 5%–20% 预训练成本。
- **Eval 成本**：full benchmark suite + safety eval 每个 candidate $50k–$500k。
- **失败重试系数**：典型 1.3–1.5× 单次预算。
- **机会成本**：训练期间 GPU 不能做别的，要算入实际成本。

### 4. 工程权衡
**模型规模 vs 训练成本**：参数量翻倍训练成本 4× 左右（参数 ×2 + tokens ×2 = 4× FLOPs）。**精度 vs 成本**：FP8 训练比 BF16 省 35%–50% 直接成本，调试期吃掉一些节省。**continual training vs 重训**：continual 比 from-scratch 省 70%–90%，但每代际仍需要 from-scratch 验证。**MoE vs dense**：相同 active params，MoE 训练成本只略高（10%–30%）但推理质量显著提升。

### 5. 常见追问 / 易错点
- 算训练成本只算"成功一次"——必须包含 ablation × 5 + failed runs × 1.3。
- 忽视**数据成本**：高质量 RLHF 数据成本 / token 比预训练数据高 100×。
- 没把 eval 算进去——一次 frontier model 的 eval suite 全跑一遍可能 $200k+ GPU 成本。

### 6. 实践建议
立项时写**训练 $/model 计划表**：列 ablation × 失败重试 × post-training × eval 全部成本，签 CFO。每月跟踪 actual vs planned，超 20% 触发 review。**模型 EOL 时回看**：实际生命周期内 token 量 vs 计划，校准未来摊销系数。把"训练 ROI"作为模型负责人的 KPI 之一，让团队主动控制成本。

### 7. 30 秒速答
- 训练 $/model = GPU-hour × $ × 工程系数 + 数据 + post-training。
- 单次成功是"裸价"，生产实际 3–5× 包含 ablation / 失败。
- 训练成本 < 5% 12 月预期营收为健康，> 20% 高风险。
- 加分关键词：sunk cost、生命周期摊销、RLHF 数据、机会成本

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清训练 $/model 的全成本公式？
- [ ] 你能不能解释为什么"成功一次"不能直接当预算？
- [ ] 你能不能举一个 70B 项目的预算分项？
- [ ] 你能不能说出训练 ROI 的健康阈值？

## Q29. 客户级毛利：从 spend / customer 到 retention 决策

> 🔴 专家 · 推理产品收入分布是长尾，top 10% 客户烧 80% 算力——其中很多是负毛利。靠 ARR 看健康会自我感觉良好，靠 `margin/customer` 才看到谁在亏钱给谁陪跑。

### 1. 核心结论
**客户级毛利 = 客户付费 - 客户实际消耗成本**。SaaS 推理产品的客户毛利分布通常是 **长尾**：top 10% 客户消耗 60%–80% 算力，但毛利可能很低甚至为负。识别"高 spend 低 yield" 客群并做调价 / 限流 / 流失，是 ML 商业化关键工程问题。

### 2. 底层原理
推理产品定价多为 $/token 或 $/request。客户实际成本受 batch effective、prefix cache hit rate、context length 分布影响极大——某些客户用法（长 context、低 batch、cache 不命中）单 token 实际成本可能是 5–10× 平均。**flat pricing + skewed usage = 部分客户负毛利**。

### 3. 关键机制
- **客户级 cost 跟踪**：每个 request 打 customer_id 标签，聚合到 customer 维度。
- **拆解维度**：input vs output token、prefix cache hit、并发度、context 长度分布。
- **margin 公式**：`margin = revenue - (Σ tokens × cost_per_token_actual)`
- **决策门槛**：margin < 10% 客户进入"高风险"列表；margin < 0% 立即触发流程。
- **abuse detection**：异常长 context、单 prompt 高频复读、空闲流量回放等模式自动告警。
- **客户透明度**：在控制台展示每客户的 token 消耗、缓存命中率、平均 context 长度，让客户能自助优化。

### 4. 工程权衡
**分层定价**：tier-1 客户固定折扣（高 margin）；tier-2 按用量；tier-3 按 token 但有 fair-use 限制。**对 abuser 的处理**：（1）软限流（QPS / tokens / context length）；（2）涨价；（3）合同到期不续签。**注意**：top spend 客户也是营收支柱，处理粗暴损失大；要分阶段沟通。

### 5. 常见追问 / 易错点
- 用 ARR 当 health 指标，忽视 margin——某客户 ARR $10M 但实际亏 $5M，越跑越亏。
- 没给客户用量透明度——客户不知道自己用法贵，无法配合优化。
- 一刀切限流——伤客户体验，应优先沟通调整 prompt / context。

### 6. 实践建议
搭 **client-level cost dashboard**：每客户 / 每天 / 每模型的 revenue、cost、margin、usage pattern。每月生成 top-20 客户 margin 报告给销售 / 客户成功团队。**fair-use 限制**默认开启（QPS、tokens/min、max context length），客户有需要可申请提升但触发 review。把客户级毛利写进季度 OKR，与营收同重要。

### 7. 30 秒速答
- top 10% 客户烧 80% 算力，其中很多是负毛利。
- 用 margin/customer 而不是 ARR 看健康；flat pricing + skewed usage 必有负毛利。
- 分层定价 + fair-use 限流 + abuse detection 三件套。
- 加分关键词：长尾分布、prefix cache hit、context length 分布、abuse detection

### 8. 自测 checklist
- [ ] 你能不能用一句话讲清客户级毛利的核心问题？
- [ ] 你能不能解释为什么 ARR 不是好健康指标？
- [ ] 你能不能举一个客户用法导致 5–10× 实际成本的场景？
- [ ] 你能不能说出 3 种 abuser 的处理方式与权衡？

## Q30. 并发模型与 fleet 经济学：多模型 fleet 的容量与定价

> 🔴 专家 · 一直加 SKU 不退役，每个模型独立 baseline + 独立 cache + 独立运维，单个 SKU 利用率掉到 30% 以下整体毛利就被稀释。fleet 不是越多越好，3–7 个 SKU 是甜点区。

### 1. 核心结论
**fleet 经济学**关注"多模型共池"下的整体毛利，不是单模型。核心权衡：模型数 ×（容量 + 运维 + 摊销）vs 客户覆盖度。**最优 fleet 规模**通常 3–7 个 SKU：1–2 个 frontier、2–3 个 mid-tier、1–2 个轻量。**过多 SKU**导致每个 SKU 利用率不足，整体毛利被稀释。

### 2. 底层原理
每多一个模型 SKU 带来：（1）独立训练成本；（2）独立推理容量（单模型最低 baseline）；（3）独立运维（监控、安全、版本）；（4）prefix cache 不能跨 SKU 共享。如果某 SKU 流量低，单位成本飙升。**实战经验**：< 5% fleet 流量的 SKU 通常负毛利。

### 3. 关键机制
- **fleet 设计**：覆盖客户需求曲线（cost vs quality），但避免重叠。
- **utilization tracking**：每 SKU 的 GPU 利用率、平均 batch、QPS 分布。
- **kill list**：连续 3 个月利用率 < 30% 的 SKU 进入 EOL 流程。
- **共享容量**：相同硬件代际、相似 model 大小，可以混部到同一 K8s namespace 共享 buffer。
- **routing**：客户请求自动路由到最优 SKU（cost / latency / quality 联合优化）。
- **fleet 健康仪表盘**：每 SKU 的当期 margin、客户数、流量趋势、容量爬坡，weekly review。
- **degradation 路径**：frontier SKU 容量紧张时，自动降级到 mid-tier 并通知客户，避免 hard failure。

### 4. 工程权衡
**少而精** vs **多而全**：少而精利用率高、毛利稳；多而全覆盖广、客户多但毛利稀。**实战折衷**：核心 fleet 5 SKU + 长尾 1–2 个 specialty（code、multimodal）。**定价策略**：让 mid-tier SKU 是 "甜点"——客户大多数被引导到这里，frontier 仅给真需要的；轻量 SKU 用于价格敏感客户。**MoE 让 fleet 简化**：一个 MoE 模型可以替代 dense 70B + 405B 两个 SKU，但要解决 routing 与 cache 问题。

### 5. 常见追问 / 易错点
- 一直加 SKU 不退役——fleet 越来越臃肿，每个 SKU 利用率下降。
- routing 算法只看 latency，不看 cost——某些 cost 翻倍但 latency 持平的请求被路到 frontier，亏钱。
- 不公布 SKU EOL 时间表——客户接到通知后没时间迁移。

### 6. 实践建议
每季度 fleet review：每 SKU 的 utilization、margin、客户分布、退役候选。建立 **routing 优化器**：基于客户标签、prompt 特征、SLO 要求，自动选择 SKU；A/B 测试不同 routing 策略对毛利的影响。SKU EOL 流程标准化：60 天预告 + 30 天并跑 + 切换 + 30 天观察。把 "fleet 整体毛利" 写进 ML 商业 OKR，让产品 / 工程 / 销售三方共担。

### 7. 30 秒速答
- 多模型 fleet 经济学核心：routing 决策 + 容量切片 + SKU 退役节奏
- routing 不能只看 latency 要看 (cost, margin, SLO) 三元组
- SKU EOL 流程：60 天预告 + 30 天并跑 + 切换 + 30 天观察
- fleet margin 应纳入 OKR，避免 SKU 蔓延

### 8. 自测 checklist
- [ ] 你能不能讲清 fleet routing 与 single-model routing 的根本差异？
- [ ] 你能不能解释为什么 cost-aware routing 比 latency-only 更重要？
- [ ] 你能不能描述一个完整的 SKU EOL 流程？
- [ ] 你能不能举一个 fleet 臃肿导致毛利下滑的具体场景？

## Q31. 比较题：自建 vs 大公有云 vs 专门 GPU 云 vs 国内云的 2026 选型

> 🧭 综合 · 这是 2026 年每个 AI infra 决策者必须回答的问题——四类供应商在性价比、capacity、合规、产品成熟度上各有侧重。

### 1. 核心结论
自建（OEM + colo）单位算力最便宜但前期投入高、运维重；大公有云（AWS/Azure/GCP）产品全栈最成熟、计费灵活但 GPU 价格高且容量紧张；专门 GPU 云（CoreWeave/Lambda/Crusoe）性价比好、capacity 弹性大但产品周边（数据库/可观测/IAM）弱；国内云（阿里/火山/腾讯）合规与本地化优势明显，新代次硬件抵达较慢。70B 量级训练首选专门 GPU 云 + 自有可观测；推理首选大公有云 + multi-region。

### 2. 底层原理
单位 GPU TCO = 折旧 + 电力 + 网络 + 运维人力。自建 H100 三年 TCO ~$1.2/小时；专门 GPU 云 $2/小时；大公有云 $2.5–4/小时；国内云相近。差距来自利用率、采购规模、毛利空间。capacity 视角：B200 在 2026 上半年仍紧俏，大公有云优先供大客户、专门 GPU 云有更多 spot 池。

### 3. 关键机制 / 流程 / 数据结构
关键决策维度：训练 vs 推理（训练需要 NVL72 + 高带宽 IB，推理重价格与多 region）、规模（千卡以下首选云、5k+ 自建更划算）、合规（数据驻留、行业规范）、SLO（专门云 SLA 可能弱于大公有云）、生态（数据/存储/可观测/IAM）。混合架构常见：训练在专门 GPU 云、生产推理在大公有云、批处理走 spot。

### 4. 工程权衡 / 性能影响
自建优点：单位成本低、控制力强、定制网络拓扑；缺点：周期长、硬件升级慢、需要专业团队。大公有云：上手快、产品全；GPU 单价高、capacity 受限。专门 GPU 云：性价比最佳、capacity 弹性；产品周边弱、SLA 偏低。国内云：合规优势；新代次延迟、跨境数据流不便。

### 5. 常见追问 / 易错点
为什么不能"全 AWS"？GPU 价格 + capacity 制约，70B 训练在 AWS 全栈做经济性不优。为什么不能"全自建"？小团队运维不动 1k 卡集群，节点故障率与采购周期都是挑战。能否多供应商 mix？可以，但要解决 cross-cloud 数据搬运成本与 IAM 复杂度。

### 6. 实践建议
30 人以下团队：专门 GPU 云训练 + 大公有云推理。50 人以上 + 5k 卡：开始混合 + 部分自建。1 万卡 +：考虑全自建。多供应商必须有统一的 cost attribution 与 utilization dashboard，不然就是钱花了不知道去哪。

### 7. 30 秒速答
- 训练优先专门 GPU 云（CoreWeave/Lambda），推理优先大公有云（AWS/Azure/GCP）
- 自建仅在 5k+ 卡 + 长期需求时算得过来
- 国内业务必须本地云保合规，新代次硬件接受 6-12 月延迟
- 混合是常态：训练 + 推理 + spot 分别选最优

### 8. 自测 checklist
- [ ] 你能不能讲清自建 vs 公有云的 TCO 拐点？
- [ ] 你能不能解释为什么训练和推理常选不同供应商？
- [ ] 你能不能列出至少 3 个跨云架构的运维难题？
- [ ] 你能不能为一个虚拟团队画出供应商选型矩阵？

## Q32. 场景题：新成立 AI infra 团队首年路线图（10 人 / 3000 万预算 / 70B 训练 + 推理 SLO）

> 🧭 综合 · 当 leader 这一年的工作不是写代码，是把 10 人 3 千万切成具体季度可交付的能力。

### 1. 核心结论
首年路线四阶段：Q1 容量与基础设施（采购 capacity block / 跑通 SLURM / Ray / 可观测栈）；Q2 训练能力（FSDP2 + 70B 跑通 + checkpoint / fault tolerance）；Q3 推理 SLO（vLLM serving + 多 region + 灰度 / 监控）；Q4 后训练 + 优化（SFT/DPO/GRPO pipeline + cost 治理）。预算分配：硬件租用 60%、人力 30%、数据 + 评测 10%。人员配置：训练 3、推理 3、平台 2、SRE/oncall 2。

### 2. 底层原理
70B 训练 capacity 估算：256× H100 三个月集中训练 + 1024× H100 一周冲刺 ≈ $1500 万。推理生产 fleet：50–100× H200 长期租用 ≈ $500–800 万。剩 $700–1000 万给人力 + 数据 + 评测 + 不可预见。SLO 目标：训练 MFU > 45%，推理 TTFT p99 < 500ms / TPOT < 50ms，可用性 > 99.5%。

### 3. 关键机制 / 流程 / 数据结构
Q1 必交付：硬件就位 + DCGM/Prometheus dashboard + 单卡 baseline + 1k 卡 NCCL stable bench。Q2：FSDP2 70B 训练能稳跑 1 万 step + ckpt 1 小时内可恢复。Q3：vLLM 多 region 灰度上线、SLO 看板、oncall 流程。Q4：完整后训练 pipeline + tokens/$ 治理 + 自动化 cost report。

### 4. 工程权衡 / 性能影响
全自营 vs 用现成框架：首年应该 80% 用 vLLM / TorchTitan / Ray 等开源，20% 自研用于差异化。先求"能跑"再求"跑快"。预算优先级：硬件 > 人 > 工具 > 评测。不要砸钱给买不到的卡（capacity block 提前 6 个月签）。

### 5. 常见追问 / 易错点
预算只给 5 千万行不行？大幅缩规模：65× H200 推理 + 64× H100 训练 + 跳 70B 改 13B。完全靠 spot 行不行？训练大批量 spot 不可靠，必须 RI/Capacity Block + spot 混合。10 人够吗？紧但可行；oncall 必须轮值，平台/SRE 各招 1 个 senior 是关键。

### 6. 实践建议
首年三大风险：1) capacity 抢不到 → Q1 必须签长约；2) 模型质量做不出 → 留 buffer 让团队迭代；3) SLO 上不去 → Q3 之前必须有压测。OKR 季度按"能力 + 数字"双轨：能力指"FSDP2 70B 跑通"，数字指"MFU 45%"。

### 7. 30 秒速答
- 四阶段：Q1 基础设施、Q2 训练、Q3 推理 SLO、Q4 后训练 + 治理
- 预算 60/30/10 分硬件/人/数据，10 人按训/推/平/SRE 3/3/2/2 切
- 首年 80% 用现成开源，20% 自研给差异化
- 三大风险：capacity、模型质量、SLO，三个都需季度看板

### 8. 自测 checklist
- [ ] 你能不能把首年路线分解成 16 个季度可验收 KR？
- [ ] 你能不能解释为什么 Q1 必须签 capacity 长约？
- [ ] 你能不能列出 10 人团队的 oncall 排班最小可行方案？
- [ ] 你能不能讲清"能力 + 数字"双轨 OKR 的具体写法？

## Q33. 估算题：ChatGPT-like 产品 1000 万 DAU 的算力账单

> 🧭 综合 · 让人在白板上把 token 量级折算成 GPU 数与年度账单，这是 AI 业务最常见的费米估算。

### 1. 核心结论
1000 万 DAU × 10 prompt/天 × (400 prompt + 500 output) tokens = ~9 万亿 tokens/月。70B FP8 推理 tokens/$ ~10–50 万 token/$（中等 batch H100）。月成本 ~$1800 万–9000 万；年化 ~$2 亿–10 亿。GPU 需求：稳态 ~400–1000 张 H100/H200，峰值 2x。如果用 7B 模型成本可降 5–10x。

### 2. 底层原理
DAU × 频次 × tokens/请求 = 月 tokens。/  tokens/$。乘除完。关键变量：模型大小（成本平方级影响）、quantization（FP8 比 FP16 省 40%）、batch 利用率（稀疏流量推高单 token 成本）、cache hit rate（agent / RAG 可省 30–80%）。GPU 估算：年 token 量 / (单卡 tokens/s × 卡 utilization × 时间秒数)。

### 3. 关键机制 / 流程 / 数据结构
分解层级：业务（DAU / 频次 / 请求大小）→ 模型（参数量 / quantization / context）→ 引擎（vLLM v1 / chunked prefill / KV 量化）→ 硬件（H100/H200/B200, 单价）→ 单位经济（tokens/$, $/DAU/月）。每层放 1-2 个数代入估算。

### 4. 工程权衡 / 性能影响
影响最大的变量：1) 模型大小（70B vs 7B = 10x cost）、2) cache hit rate（agent 用例）、3) GPU 代次（B200 比 H100 单 token 便宜 30–50%）、4) batch 利用率（峰谷流量决定 fleet 大小）。

### 5. 常见追问 / 易错点
峰谷流量怎么处理？peak/avg 通常 2–3x，需要 spot + autoscaling 处理峰值。海外多 region 成本怎么算？每 region 都要 fleet，全球部署 cost 1.5–2x 单 region。免费用户怎么压缩？分级降级（流量大用小模型 / 降 quantization / 缓存压缩）。

### 6. 实践建议
做估算时给上下限：保守上限（无 cache、全 70B、低 batch）+ 激进下限（30% cache、70B + 7B 混合、高 batch）。差距通常 5–10x。商业模型据此定价。Profile + 灰度 + cost dashboard 是把估算变现实的三件套。

### 7. 30 秒速答
- 公式：DAU × 频次 × tokens/请求 / (tokens/$)
- 1000 万 DAU × 70B FP8 ≈ $2 亿–10 亿/年（区间宽）
- 主要变量：模型大小、cache hit rate、GPU 代次、batch 利用率
- 估算要给上下限：保守 vs 激进差 5–10x

### 8. 自测 checklist
- [ ] 你能不能在白板上 5 分钟做出这个估算？
- [ ] 你能不能解释 70B vs 7B 成本差距背后的物理原因？
- [ ] 你能不能列出至少 5 个影响 tokens/$ 的变量？
- [ ] 你能不能描述峰谷流量对 fleet 大小的影响？

## Q34. 设计题：LLM 平台成本治理体系

> 🧭 综合 · cost attribution + budget + quota + tokens/$ 监控 + SKU 优化，缺一不可的成本治理五件套。

### 1. 核心结论
五层架构：1) **遥测**（每请求记录 model/version/input/output/cache/region/team）；2) **attribution**（按 team / product / customer / cohort 聚合）；3) **budget**（季度 / 月度上限，超限报警 + 自动降级）；4) **quota**（per-team rate limit + token bucket）；5) **SKU 优化**（按 cost-per-token 自动 routing + SKU EOL）。覆盖训练（per-job cost）+ 推理（per-token cost）两条线，统一进 cost dashboard。

### 2. 底层原理
单位 cost = (GPU 时间 × 单价 + 网络 + 存储) / 单位（token、step、job）。attribution 难在多租户共享 fleet 时如何切分时间。常见算法：按 tokens 加权（推理）、按 GPU-hours 加权（训练）。budget 系统是 finance 与 infra 的契约，必须可解释、可申诉、可调整。

### 3. 关键机制 / 流程 / 数据结构
数据流：请求 → gateway 打标 → 推理引擎执行 → 日志 → ETL → cost data warehouse → dashboard / alert / report。每条请求至少 20 个字段（model_id、tenant_id、tokens_in、tokens_out、cache_hit、ttft、tpot、region、cluster_id、sku、duration_ms、cost_usd、http_status、user_agent、time、request_id、trace_id、product_id、business_unit、retention_class）。

### 4. 工程权衡 / 性能影响
精度 vs 开销：每请求记录 20 字段 ~200 字节，1 亿请求/天 ~20GB/天，可控。实时 vs 批：实时 dashboard 看 SKU/region 级，批处理（每天）做 team/product 细粒度。Budget alert：先告警再硬降级，给团队 24h 缓冲。

### 5. 常见追问 / 易错点
共享 prefix cache 怎么算 attribution？通常给"启发者"（首次 prefill）记主要成本，复用者记少量摊销。Free tier 怎么记？计入 marketing budget。SKU EOL 怎么和 budget 联动？提前 60 天通知，受影响 team 转用新 SKU + 老 SKU 报价上调激励迁移。

### 6. 实践建议
落地路线：Q1 遥测铺到所有引擎、Q2 attribution + dashboard、Q3 budget + quota、Q4 routing optimizer + SKU EOL 流程。一个常被忽略的点：cost data 必须可被业务团队自助查询（self-service SQL / dashboard）；不然 finance 永远要找 infra 临时拉数。

### 7. 30 秒速答
- 五层：遥测、attribution、budget、quota、SKU 优化
- 推理按 token 加权、训练按 GPU-hours 加权做 attribution
- 每请求 20 字段 / 200 字节，1 亿请求/天 ~20GB（可控）
- 落地节奏：遥测 → attribution → budget → routing optimizer

### 8. 自测 checklist
- [ ] 你能不能列出请求级 cost log 的最小字段集？
- [ ] 你能不能解释 prefix cache attribution 的两种算法？
- [ ] 你能不能描述 budget 软告警 + 硬降级的实现细节？
- [ ] 你能不能为新 SKU 上线 / 老 SKU 退役设计完整流程？
