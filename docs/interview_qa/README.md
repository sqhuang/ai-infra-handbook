# AI infra 大百科 · 总索引

一本面向 AI infra 工程师的"问答式百科"。围绕 PyTorch 内核、编译器、CUDA/Triton、分布式训练、显存与通信、推理与服务化、训练 / 推理平台、系统基础、战略决策、事故复盘以及 2026 后训练 / 新一代推理范式十一条主线，整理出可以反复回看的工程心法。覆盖到 2025–2026 年生态的主流实践，但不替代论文与官方文档，用作系统化梳理与面试前压舱石更合适。

## 特性一览

- 466+ 题、11 卷，每题统一按 8 节模板（核心结论、底层原理、工程实践、易错点、面试高分点、延伸阅读、30 秒速答、自测 checklist）展开
- 每题开头有白话引子，并标注难度（🟢 基础 / 🟡 进阶 / 🔴 专家 / 🧭 综合）
- 关键空间概念（拓扑、流水线、显存布局等）配 ASCII 图，便于在终端 / Markdown 里随手参考
- 配套[术语表](glossary.md) 收录 ~110 个高频缩写与中文术语对照
- 独立[关键数字速查表](numbers-quickref.md) 收录硬件 / 软件 / 业务高频数字，估算与决策时用得到

## 卷册列表

| 卷册 | 文件 | 题号 | 题量 | 说明 |
| --- | --- | --- | ---: | --- |
| 卷 01 PyTorch 内核 | `01-pytorch-internals.md` | Q1–Q24 | 24 | 自动微分、动态图、分发、`torch.compile` 与 PT2 调试。 |
| 卷 02 编译器与 IR | `02-compiler-and-ir.md` | Q1–Q20 | 20 | XLA HLO、MLIR、TVM、TorchInductor、ONNX Runtime、TensorRT 等编译器与 IR 主题。 |
| 卷 03 CUDA/Triton 与自定义算子 | `03-cuda-triton-custom-ops.md` | Q1–Q49 | 49 | CUDA 编程模型、Triton DSL、SM/L2/HBM 与多卡拓扑。 |
| 卷 04 分布式训练 | `04-distributed-training.md` | Q1–Q59 | 59 | DDP/FSDP2、张量并行、流水线并行、ZeRO、NCCL 与 Megatron-LM 调优。 |
| 卷 05 显存、通信与 Profiling | `05-memory-communication-profiling.md` | Q1–Q69 | 69 | 显存分配、激活重计算、NVLink/IB、存储栈与 Nsight/HTA profiling。 |
| 卷 06 推理量化与服务化 | `06-inference-quantization-serving.md` | Q1–Q65 | 65 | vLLM/SGLang/TensorRT-LLM、量化、PagedAttention、SLO、PD 分离、RadixAttention、Mooncake、LMCache、Multi-LoRA。 |
| 卷 07 训练与推理平台 | `07-training-inference-platform.md` | Q1–Q84 | 84 | Ray/Kueue、SLURM 全景、节点健康、可观测性与端侧部署。 |
| 卷 08 系统基础 | `08-systems-foundations.md` | Q1–Q39 | 39 | OS、网络、文件系统、虚拟化等通用基础。 |
| 卷 09 战略与工程决策 | `09-strategy-engineering-decisions.md` | Q1–Q34 | 34 | 云与硬件采购、容量规划、技术债、团队组织与单位经济学。 |
| 卷 10 事故复盘与故障教训 | `10-postmortems-and-lessons.md` | Q1–Q29 | 29 | OOM、通信、存储、推理服务与平台编排的典型化复盘。 |
| 卷 11 后训练与新一代推理范式 | `11-posttraining-new-inference.md` | Q1–Q28 | 28 | RL 后训练、MoE/EP、长上下文 CP、解耦推理、投机解码、Blackwell FP4、MLA、Mamba、1M context、RLVR、EAGLE-3、KV 量化 2026。 |
| 卷 12 AIGC 生成式 infra | `12-aigc-generation-infra.md` | Q1–Q25 | 25 | SDXL/Flux/SD3、DiT、diffusers、采样器、量化、ControlNet/LoRA/IPAdapter、ComfyUI、视频生成、业务 SLO。 |
| 卷 13 推荐系统推理 | `13-recommendation-inference.md` | Q1–Q20 | 20 | DLRM、Embedding 分布式、候选生成+精排、HSTU/TIGER 生成式推荐、vLLM-GR、P99 延迟、Feature Store、A/B 测试。 |
| 卷 14 国产芯片与异构推理 | `14-domestic-chips-heterogeneous.md` | Q1–Q20 | 20 | 昇腾 910B/910C、Ascend C、CANN、vLLM-Ascend、寒武纪 MLU、AMD MI300X、Apple Silicon、跨平台量化、信创迁移。 |

全套语料 565 题，分布于 14 卷，每卷题号独立从 Q1 起编。定位某题时建议带上卷号（例如"卷 04 Q12"）。每卷末尾 4 道 🧭 综合题包含比较 / 场景 / 估算 / 设计，跟卷内基础题互补。

## 按背景的推荐入口

- **后端 / SRE 转 ML**：08 → 07 → 04 → 06 → 05 → 03 → 01。先用熟悉的系统语言铺路，再向训练 / 推理与底层内核纵深。
- **ML 研究员转工程**：01 → 03 → 04 → 11 → 05 → 06。从最熟悉的 PyTorch 入门，逐步扩展到自定义算子、并行、后训练与推理工程。
- **ML 平台 PM / 架构师**：09 → 07 → 06 → 11 → 04 → 10。先看战略决策与平台全景，再补关键技术细节，最后用复盘卷收口。
- **面试前突击**：09 + 10 通读，再扫各卷的 🔴 专家题与 🧭 综合题。短时间拉满"系统观 + 故障感"两条压舱主线。
- **2026 时效冲刺**：11 通读 + 06 重读。RL 后训练、MoE/EP、解耦推理、Blackwell FP4 都是当年新增热点。
- **AIGC 图像 / 视频方向**：12 通读 + 06 Q1-30 + 03 attention 章节。SDXL/Flux/diffusers/ComfyUI 全栈梳理。

## 主题学习路径（跨卷串联）

每条路径是"读懂某个工程主线"需要的题集合，跨卷顺序按学习友好度排：

- **长上下文（128k → 1M+）**：05 Q3（激活重计算）→ 03 Q30（FlashAttention 系列）→ 04 Q10/Q12（TP/CP/SP）→ 11 Q8（CP & Ring）→ 11 Q15（长上下文训练）→ 11 Q9（KV 量化）。
- **MoE 训练与推理**：04 Q25（DeepEP）→ 11 Q6（EP）→ 11 Q7（load balancing & collapse）→ 06 系列（推理引擎对 MoE 的支持）。
- **后训练全景**：11 Q1（RL infra 形态）→ 11 Q2（GRPO vs PPO）→ 11 Q3（RLVR）→ 11 Q4（权重同步）→ 11 Q5（long-CoT rollout）→ 11 Q21（PPO/GRPO/DPO 对比）。
- **解耦与新调度**：06 Q3（vLLM/SGLang/TRT-LLM）→ 06 Q5/Q6（continuous batching & chunked prefill）→ 11 Q10（disaggregated）→ 11 Q18（调度器演进）→ 11 Q19（KV offload）。
- **投机解码 & 解码加速**：06 Q11（speculative decoding 概念）→ 11 Q11（Medusa/EAGLE/Lookahead）→ 11 Q22（场景题：H200 + 70B 部署）。
- **Blackwell FP4 实战**：05 Q42/Q43（HBM 代际）→ 11 Q12（B200 vs H100/H200）→ 11 Q13（NVL72 vs NVL8）→ 11 Q14（FP4 训练数值）。
- **Agent infra**：06 Q13（prefix caching）→ 11 Q17（Agent vs stateless）→ 11 Q19（跨节点 KV cache）。
- **可观测性 & 故障**：05 Q40（MFU/HFU）→ 07 Q40（Xid/ECC/DCGM）→ 10 全卷（事故复盘）→ 09 Q15（tokens/$ 单位经济学）。

## 反向索引：按方向反查题集

按"工作方向 / 岗位画像"反向找到对应主题，弥补"按主题分卷"的盲点：

- **训练 perf 工程师**：03（CUDA/Triton）+ 04 Q10–25（并行）+ 05（显存通信）+ 11 Q14（FP4）+ 11 Q15（长上下文）。
- **推理 / serving 工程师**：06 全卷 + 11 Q10/Q11/Q18 + 06 Q20（SLO）+ numbers-quickref（吞吐参考）。
- **平台 / 调度工程师**：07 全卷 + 09 Q5/Q22（容量与多租户）+ 10 Q5/Q15（编排事故）。
- **RL post-training 工程师**：11 Q1–Q5 + 11 Q21 + 04 Q25（DeepEP）+ 03 Q30（FA-3 用于 rollout）。
- **算子 / kernel 工程师**：03 全卷 + 01 Q5–Q10（dispatch）+ 02（编译器栈）+ 11 Q14（FP4 kernel）。
- **数据 / 评测工程师**：07 Q60+（数据栈）+ 09 Q20（评测）+ 11 Q15（NeedleInHaystack）。
- **端侧 / 本地推理工程师**：06 Q45（端侧）+ 02 Q14（ExecuTorch）+ glossary（GGUF/MLX/Marlin）。
- **架构师 / 技术决策者**：09 全卷 + 10 全卷 + 11 综合题 + numbers-quickref。

## 术语与数字速查

读到陌生缩写时，先翻 [`glossary.md`](glossary.md)（在线版：`docs/site/glossary.html`）；每条都给出第一次出现的卷与题号，方便顺藤回到正文深读。

需要做估算（每秒多少 token、几张卡能装多少 KV、训一次要多少美元）时，看 [`numbers-quickref.md`](numbers-quickref.md)，一份硬件 / 软件 / 业务高频数字的速查页。

## 在线阅读

完整语料以编辑设计风的多页站点呈现，入口：

```
docs/site/index.html
```

直接在浏览器中打开（`file://` 协议即可），或本地起一个简易服务器：

```
python3 scripts/preview_site.py
# → http://127.0.0.1:8765
```

部署到 GitHub Pages：仓库 Settings → Pages，Source 选 `master` 分支
`/docs/site` 子目录即可，无需 CI（产物已入库）。

源 Markdown 改动后重新生成站点：

```
python3 scripts/build_interview_qa_site.py
```
