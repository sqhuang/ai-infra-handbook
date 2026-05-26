# AIGC 生成式 infra 卷

## 主题边界
本卷聚焦 AIGC（AI-Generated Content）生成式模型的推理与服务化基础设施，主线覆盖图像生成（Stable Diffusion 系列 / Flux / SDXL / DiT）和视频生成（Sora 风格 / Latent Video）两大方向。讨论范围包括模型架构、推理流水、采样器、量化加速、ControlNet / LoRA / IPAdapter 等推理时插件、ComfyUI 等编排引擎、服务化调度、AIGC 业务 SLO，以及跟 LLM 文本推理在工程上的根本差异。模型训练、模型设计、艺术效果调参等不在本卷展开。

## Q1. Diffusion 模型推理流水的整体结构是什么？跟训练流水有什么差别？

> 🟢 基础 · 一张图能生成出来要走 text encoder → diffusion loop → VAE decoder 三大段，每段都是独立的子模型，工程上的瓶颈分布跟 LLM 完全不一样。

### 1. 核心结论
Diffusion 模型推理流水分三大段：（1）text encoder 把 prompt 编码为 condition embedding；（2）diffusion loop 在 latent 空间多步去噪（典型 20-50 步），每步跑一次 UNet 或 DiT 主体；（3）VAE decoder 把最终 latent 解码为像素图像。三段是顺序串联的，diffusion loop 的多步迭代是主要计算瓶颈，通常占端到端时间 80-95%。训练时是单步前向 + 反向（学的是单步噪声预测），推理时是 N 步循环（每步都用学好的模型预测当前噪声），两者计算模式根本不同。

### 2. 底层原理
Diffusion 的核心数学：训练时给图像加噪 N 步直到纯高斯噪声，模型学"从加噪图像反推原始噪声"。推理时反过来，从纯噪声开始，用模型预测的噪声逐步去噪。每一步的更新公式由 scheduler（DDIM / DPM-Solver / Euler 等）决定，本质都是某种近似常微分方程或随机微分方程求解器。

Latent diffusion（Stable Diffusion 系列）的关键创新是把扩散过程从像素空间挪到 VAE 压缩后的 latent 空间（典型 8× 空间压缩 + 4 通道，512×512 图 → 64×64×4 latent），让扩散主体只在低维 latent 上跑，VAE 仅在首尾各跑一次。这是 SD 能上消费级 GPU 的根本原因。

text encoder（CLIP / T5-XXL）输出的是 cross-attention 用的 condition embedding，每个 token 一个 embedding 向量，diffusion loop 里每步通过 cross-attention 注入 prompt 信息。

### 3. 关键机制 / 流程 / 数据结构
第一，text 编码阶段。Prompt 走 tokenizer → text encoder → embedding tensor（shape: [batch, seq_len, hidden_dim]）。SDXL 用 CLIP + OpenCLIP 双编码器拼接，Flux 用 CLIP + T5-XXL 双编码器。这一段通常占总时间 1-3%，可缓存（同一 prompt 重复生成时跳过）。

第二，diffusion 主循环。初始 latent x_T 来自标准高斯分布 randn(shape=[batch, 4, H/8, W/8])。每步执行：noise_pred = unet(x_t, t, text_emb)，然后 x_{t-1} = scheduler.step(noise_pred, x_t, t)。N 步后得到 x_0。每步 unet/dit 调用是计算大头，UNet 的 self-attention + cross-attention 是热点 op。

第三，VAE 解码。x_0 (latent) → vae_decoder → image (像素)。SD 的 VAE 解码 1024×1024 图大概几十毫秒，DiT 类模型有时用更大 VAE（FLUX 用 16 通道 latent），解码耗时增加但质量更好。

第四，跟训练对比。训练时随机采样 t、加噪、单步反向；推理是按调度好的 t 序列从 T 到 0 串行去噪。训练 batch 大，推理 batch 通常小（端到端 latency 优先）。

### 4. 工程权衡 / 性能影响
Diffusion 推理的根本 cost = N（步数）× 单步 UNet/DiT cost。SDXL 默认 30-50 步，DiT 模型可能 28-50 步。**加速核心思路有两条**：减少 N（DPM-Solver 让 20 步达到 50 步效果、LCM/Hyper-SD 让 4-8 步生成、SDXL-Turbo 让 1 步生成）；减少单步 cost（DeepCache 缓存中间特征、SVDQuant 把 UNet 量化到 FP4）。

跟 LLM 推理对比的差异：LLM 是 autoregressive（每 step 生成 1 token，KV cache 是关键优化），diffusion 是 multi-step iterative（每 step 改 latent，没有 KV cache 概念但有 "consistency between steps" 概念）；LLM batch 内 sequence 长度不齐，diffusion batch 内分辨率必须一致；LLM 输出是 streaming 的，diffusion 通常要全部 step 跑完才出图（除非 progressive preview）。

### 5. 常见追问 / 易错点
第一，guidance scale（CFG）让计算量翻倍。Classifier-Free Guidance 实际跑两次 UNet（一次有 text condition，一次 unconditional），用 (cond - uncond) × scale + uncond 合成最终 noise prediction。所以同样 N 步实际是 2N 次 UNet。CFG distillation / Hyper-SD 等技术能把 2N 压回 N。

第二，UNet vs DiT 选择影响很多事。SDXL 是 UNet 架构（卷积 + attention 混合）；Flux / SD3 / Sora 都是 DiT（纯 Transformer）。DiT 适合 scaling law、infer kernel 更纯（都是 attention + FFN），但显存占用通常更大。

第三，VAE 解码的工程坑。大分辨率（如 2048×2048）解码会爆显存，常用分块解码（tiled VAE）。VAE 量化对图像质量影响显著，业界很少做 VAE 量化，主要量化 UNet/DiT。

### 6. 实践建议
端到端 profile 时分清三段时间占比（text encode / diffusion loop / VAE decode），diffusion loop 占大头时优化方向是减 step 或减单步；text encode 占大头时考虑缓存；VAE 占大头时考虑 TAESD（Tiny AutoEncoder）。

部署时优先级：选合适采样器（DPM-Solver++ / Euler）→ 量化 UNet 到 FP8/FP4 → 加 DeepCache → 极致场景上 LCM/Turbo。

跟算法团队对齐时强调"step 数和质量是 trade-off"，工程优化能压 step 数但需要算法配合训 LCM-LoRA 等。

### 7. 30 秒速答
- 三段流水：text encoder → N 步 diffusion loop → VAE decoder
- diffusion loop 占 80-95% 时间，是优化主战场
- 加速两条路：减步数（采样器 / LCM / Turbo）+ 减单步 cost（量化 / DeepCache）
- CFG 让单步实际跑两次 UNet，端到端是 2N 次

### 8. 自测 checklist
- [ ] 你能不能讲清 latent diffusion 跟 pixel-space diffusion 的差别？
- [ ] 你能不能解释 CFG 为什么让计算量翻倍？
- [ ] 你能不能拆解 SDXL 1024×1024 一张图的端到端时间分布？
- [ ] 你能不能说出三种减少 diffusion step 数的方法？

## Q2. SDXL / Flux / SD3 三大主流图像生成模型架构对比

> 🟡 进阶 · 一个是 UNet 双 text encoder，一个是 DiT + T5-XXL，一个是 MMDiT 多模态融合——架构选择决定整个 serving 栈的形态。

### 1. 核心结论
SDXL（2023）是 UNet 架构 + CLIP + OpenCLIP 双 text encoder + 4 通道 VAE，参数 2.6B（base）+ 6.6B（refiner，可选），是当前 fine-tune / LoRA 生态最丰富的基座。Flux（2024，Black Forest Labs）是 12B DiT + CLIP + T5-XXL 双 encoder + 16 通道 VAE，质量大幅超越 SDXL，是 2025 工业生成首选。SD3（2024，Stability AI）是 MMDiT（Multi-Modal DiT）+ 三 encoder（CLIP×2 + T5）+ 16 通道 VAE，参数从 800M 到 8B 多版本。

### 2. 底层原理
SDXL UNet：编码器-中间-解码器结构，混合 ResBlock 卷积层 + spatial transformer block（含 self-attn + cross-attn），三个分辨率层级（64×64 / 32×32 / 16×16）。Cross-attention 注入 CLIP + OpenCLIP 拼接的 text embedding。

Flux DiT：纯 Transformer 架构，把 latent patch 化（patch_size=2）成 sequence，加 positional embedding（RoPE 或可学的 2D pos），通过 19 个 double-stream block + 38 个 single-stream block 处理。double-stream 让 text token 和 image token 各走一路、再 cross 融合；single-stream 把两路拼一起统一处理。

SD3 MMDiT：跟 Flux 思路类似但更早提出，强调 text 和 image 是同等 modality，joint attention 允许 text token 和 image token 互相 attend。三 encoder 输出拼成长序列。

VAE 通道数变化：SD 1.5/SDXL 是 4 通道 VAE，压缩比高但细节损失明显；Flux/SD3 用 16 通道 VAE，细节保留好，特别是文字和小细节。

### 3. 关键机制 / 流程 / 数据结构
第一，参数与显存对比。SDXL UNet ~2.6B BF16 ≈ 5.2GB；Flux dev 12B BF16 ≈ 24GB（单卡 RTX 4090 24GB 极限）；SD3 8B BF16 ≈ 16GB。Text encoders 也不小：CLIP-L ~123M、OpenCLIP-G ~694M、T5-XXL ~4.8B。Flux 的 T5-XXL 单独占 ~10GB BF16。

第二，推理时显存峰值。除了模型权重，还要 activation（按分辨率、batch、step 数）。SDXL 1024×1024 batch=1 推理峰值 ~8-10GB；Flux 1024×1024 batch=1 量化后 ~16-20GB；SD3 8B 介于两者之间。

第三，速度对比（H100 BF16 / FP8）。SDXL 1024×1024 30 步 ~2-4s；Flux 1024×1024 28 步 ~6-15s；SD3-8B 类似 Flux 一个数量级。

第四，生态成熟度。SDXL 是当前 LoRA / ControlNet / IPAdapter 等推理时插件生态最完整的；Flux 生态正在快速建立（2024 下半年起）；SD3 因许可证问题生态较慢。

### 4. 工程权衡 / 性能影响
选 SDXL 适合：极高吞吐 / 成熟生态 / 低端显卡 / 大量 LoRA 已有；选 Flux 适合：质量优先 / 复杂 prompt / 文字渲染 / 接受较慢生成；选 SD3 适合：开源研究 / 多模态融合实验。

生产部署里 Flux 显存压力大，必须量化（FP8 至少，FP4 更优）+ text encoder offload。SDXL 显存压力小但质量不如 Flux，看业务对质量是否敏感。

跟 LLM 部署不同，AIGC 显卡选型上：text encoder offload 让显存需求降很多，但推理时还得载回来；可以接受较慢生成（消费级用户能等 5-30s）。

### 5. 常见追问 / 易错点
第一，refiner 还在用吗？SDXL refiner（额外 6.6B 模型，专门做最后几步去噪）业界基本弃用，因为加了一倍显存和时间但质量提升有限。多数生产部署只用 base。

第二，Flux dev vs Flux schnell。dev 是 28 步生成的完整模型，schnell 是 4 步蒸馏版（速度 7x），质量略降但速度更快，工业部署常用 schnell。

第三，T5-XXL 必须吗？Flux 的 T5-XXL 占 10GB BF16，量化到 FP8 减半，但很多人尝试 drop T5 只用 CLIP，发现复杂 prompt（如包含 "a dog with text saying HELLO" 这种）效果显著降。生产里 T5 通常保留 + 量化。

第四，分辨率不是任意选。SDXL 训练时是 1024×1024，可生成 768/832/1152/1216 等附近比例，太偏离训练分辨率（如 256×256 或 2048×2048）效果差。Flux/SD3 类似但更鲁棒。

### 6. 实践建议
新业务选型：质量优先 + 显存够 → Flux dev FP8；速度优先 + 低成本 → Flux schnell；最大化复用现有 LoRA 资产 → SDXL。

容量规划：Flux 单卡 H100/A100 80G 跑 batch=4 1024 推理需要 FP8；A6000 48G 必须 FP4 或者 batch=1；24G 卡只能 schnell + 量化。

text encoder offload：T5-XXL 不用时挪到 CPU，需要时载回，省 10GB GPU 显存代价 ~1s 加载时间。生产里 T5 输出常做 prompt-level cache。

### 7. 30 秒速答
- SDXL: UNet 2.6B + 双 CLIP，生态最熟，4 通道 VAE
- Flux: DiT 12B + CLIP + T5-XXL，质量最强，16 通道 VAE
- SD3: MMDiT + 三 encoder，介于两者
- 选型：质量 → Flux；速度 → SDXL 或 Flux schnell；LoRA 生态 → SDXL

### 8. 自测 checklist
- [ ] 你能不能讲清 UNet 跟 DiT 在 attention 层数 / FLOPs 分布上的差异？
- [ ] 你能不能算 Flux dev 在 24GB 卡上能不能跑（什么量化下）？
- [ ] 你能不能解释为什么 SDXL refiner 业界基本弃用了？
- [ ] 你能不能说出 Flux 的 T5-XXL 跟 CLIP 各负责什么？

## Q3. DiT (Diffusion Transformer) 跟传统 UNet 的本质差异

> 🟡 进阶 · 从卷积 + attention 混合，变成纯 attention + FFN——DiT 让 diffusion 模型走上了 LLM 一样的 scaling law 之路。

### 1. 核心结论
DiT 把 diffusion 模型主体从 UNet（CNN + attention 混合）改成纯 Transformer。优势：跟 LLM 同构的 scaling law、infer kernel 更纯（只有 attention + FFN）、生态可复用 LLM 优化（FlashAttention 等）；劣势：参数效率比 UNet 略低（同样质量需要更多参数），显存占用更大。当前 SOTA 模型（Flux / SD3 / Sora / 国产豆包视觉等）都走 DiT 路线。

### 2. 底层原理
UNet 设计思想：编码器逐层下采样压缩 + 解码器逐层上采样恢复，加 skip connection 保留细节。每一层是 ResBlock（卷积 + 残差）+ 可选 attention block。卷积的归纳偏置让 UNet 对图像结构敏感，参数效率高。

DiT 设计思想：把 latent 切成 patch（如 2×2 的 latent patch），flatten 成 sequence，加位置编码，过 N 个 Transformer block。每个 block 是 self-attention + FFN（跟标准 Transformer 一样）。条件信息（timestep + class label / text embedding）通过 AdaLN（Adaptive LayerNorm）注入，每层的 scale/shift 参数由 condition 决定。

为什么走 DiT：（1）attention 全局视野比 CNN 局部感受野更适合长程语义；（2）Transformer 已被证明在 LLM 上有 clean scaling law（模型越大效果越好），DiT 也展现类似性质；（3）工程上更统一（生态可复用 FlashAttention / TP 等）。

### 3. 关键机制 / 流程 / 数据结构
第一，patch 化。Latent shape [B, C, H, W]（如 [1, 4, 128, 128] 对应 1024 图）→ unfold 成 patches [B, C, H/p, p, W/p, p] → reshape [B, (H/p)(W/p), C·p·p]。patch_size=2 让 sequence 长度 = (H/p × W/p) = 4096 for 1024 图，再过 linear 变 hidden_dim。

第二，AdaLN-Zero。DiT 的 condition 注入用 AdaLN：先算 condition embedding → MLP → 输出 6 个值（scale1, shift1, gate1, scale2, shift2, gate2）；前两组用在 self-attention 前的 LayerNorm，后两组用在 FFN 前的 LayerNorm；gate 控制 residual 强度。AdaLN-Zero 初始化让 gate=0 (输出 0)，训练稳定。

第三，DiT block 结构。x → AdaLN(x) → self-attention → ×gate1 → residual → AdaLN → FFN → ×gate2 → residual。比标准 Transformer 多了 AdaLN 注入条件，其它一样。

第四，跟 LLM Transformer 区别。DiT 是 bidirectional self-attention（没有 causal mask），attention 全局（每个 patch attend 所有 patch），sequence 长度取决于分辨率（不像 LLM 是 token 数）。

### 4. 工程权衡 / 性能影响
DiT 跟 UNet 的关键工程对比。FLOPs：同参数量 DiT 跟 UNet 接近，但 DiT 是 O(N²·d) attention（N 是 patch 数），UNet 是 O(N·d²) 卷积，分辨率越大 DiT attention 越主导。Memory：DiT activation 是 [B, N, d] dense tensor，N=4096 d=3072 时 batch=1 activation ~50MB/层，N 层下来几 GB；UNet 是多层不同分辨率的 4D tensor，总量类似但分布不同。

可优化空间：DiT 的 attention 直接用 FlashAttention（成熟），UNet 需要自定义 fused conv+attn kernel；DiT 量化更直接（标准 Transformer 量化路径），UNet 各层结构不齐导致量化复杂。

scaling：DiT 在 sub-1B 上跟 UNet 接近，1B+ 开始 DiT 优势明显，>10B 几乎只能用 DiT。

### 5. 常见追问 / 易错点
第一，DiT 的 attention 复杂度。N×N attention，1024 图 patch=2 时 N=4096，单层 attention 是 4096² × d = 16M × d，d=3072 时 ~50G FLOPs（单层！）。多层加起来比 UNet 重得多。Flash Attention 是必备。

第二，patch_size 选择。patch=2 让 sequence 长但细节多；patch=4 让 sequence 减半但细节少。Flux/SD3 用 patch=2。降低 patch 是减少 attention cost 的直接方法但牺牲质量。

第三，AdaLN 参数量。每层 6 个 scale/shift/gate，每个 size=hidden_dim，N 层下来 6Nd 参数。DiT-XL 这部分参数能占 30-40% 总参数。

第四，DiT 显存峰值不在 attention 上。Activation 全长 sequence × hidden_dim 是主要显存，attention 算 softmax 时短暂峰值高但用 FA-2 后摊薄。

### 6. 实践建议
新 model 训练默认 DiT；只在小模型（< 1B）+ 强 inductive bias 需求时考虑 UNet。

部署优化优先级：FlashAttention-3（Hopper+） → FP8 量化 → DeepCache（DiT 也适用）→ patch caching（同 prompt 第一层 patch 复用）。

profile DiT 看哪一层热点：通常 N 越长的层（前 1-2 层）attention 越主导，后面层有时是 FFN 主导。

### 7. 30 秒速答
- DiT 把扩散主体换成纯 Transformer，跟 LLM 同构
- AdaLN 注入 condition（timestep + text），不用 cross-attention
- 优势 scaling law / kernel 纯 / 生态复用 LLM 优化
- 劣势：参数效率略低、attention O(N²) 显存大

### 8. 自测 checklist
- [ ] 你能不能讲清 DiT 的 condition 怎么注入（AdaLN）？
- [ ] 你能不能算 DiT-XL 在 1024 分辨率单层 attention 的 FLOPs？
- [ ] 你能不能说出 DiT 跟 UNet 在量化友好度上的差异？
- [ ] 你能不能解释为什么 SOTA 模型都走 DiT？

## Q4. diffusers 库的核心抽象（Pipeline / Scheduler / ModelMixin）

> 🟡 进阶 · Pipeline 把"端到端用户体验"封装，Scheduler 把"采样算法"模块化，ModelMixin 是"模型基类"——这三层抽象决定你用 diffusers 是 5 行代码还是改源码改不动。

### 1. 核心结论
diffusers 是 HuggingFace 维护的事实标准 diffusion 库，三层核心抽象：Pipeline（端到端封装，用户接口）、Scheduler（采样算法实现，可换）、ModelMixin（模型基类，定义 save/load/from_pretrained 等）。生产部署里 Pipeline 通常作为起点但需要拆开自定义；Scheduler 是优化时最常改的（换不同采样器试效果）；底层 model（UNet2DConditionModel / Transformer2DModel 等）通常不动，量化时除外。

### 2. 底层原理
Pipeline 模式：StableDiffusionXLPipeline.from_pretrained(...) 一行加载所有组件（text encoders + tokenizers + scheduler + unet + vae），然后 pipeline(prompt=...) 调用 __call__ 跑端到端。__call__ 内部包含：encode_prompt → prepare_latents → scheduler.set_timesteps → for t in timesteps: unet → scheduler.step → decode_latents → postprocess。

Scheduler 接口：scheduler.set_timesteps(num_inference_steps) 决定 t 序列；scheduler.step(model_output, t, sample) 实现单步更新公式。不同采样器是不同 step 实现：DDIM 用确定性 ODE 步骤，DPM-Solver 用多步 solver，Euler 是标准 ODE 求解，LCM 是单步 consistency model。

ModelMixin：所有模型继承（UNet2DConditionModel、AutoencoderKL、Transformer2DModel 等），提供 from_pretrained / save_pretrained / device 管理。

### 3. 关键机制 / 流程 / 数据结构
第一，Pipeline 拆解。生产部署常需要把 pipeline.__call__ 拆开，原因：（a）text encoder 输出可 cache，不必每次重算；（b）需要插入自定义 ControlNet / IPAdapter / LoRA；（c）需要批量优化（如 batch 个 prompt 同时跑）；（d）需要进度回调 / 中间结果输出。

第二，Scheduler 切换。同一 pipeline 可换 scheduler：pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)。换 scheduler 几乎零代价，是优化时第一步尝试。

第三，model 组件量化。pipeline.unet = quantize(pipeline.unet)（用 bitsandbytes / optimum-quanto / GGUF），其它组件保持原样。量化后 Pipeline 接口不变。

第四，组件 offload。pipeline.enable_model_cpu_offload() 让组件按需载入 GPU（VAE 和 text encoder 用完挪回 CPU），节省 ~10GB 显存。代价是每次切换 ~100ms。

### 4. 工程权衡 / 性能影响
Pipeline 的便利 vs 灵活性：直接用 Pipeline 5 行代码出图，但难加自定义优化；拆开重写 60 行代码但可控。生产部署里通常做"Pipeline 子类化"，重写 __call__ 加入业务逻辑。

Scheduler 是性能优化的杠杆点：从 DDIM 50 步换 DPM-Solver++ 20 步效果接近、速度 2.5x；换 LCM Scheduler 4 步、速度 12x（但要配 LCM-LoRA）。

部署生产 vs 研究：研究用 Pipeline + 默认参数；生产把 Pipeline 拆成 modular components（text_encoder_service / unet_service / vae_service），每段独立部署和扩展。

### 5. 常见追问 / 易错点
第一，model.eval() 跟 torch.no_grad()。diffusers Pipeline 内部已经包了 no_grad（推理用），但显式 with torch.no_grad() 仍是好习惯。model.eval() 关 dropout/batchnorm，推理时必须。

第二，分辨率必须是 8 的倍数。VAE 8x 下采样，输入分辨率必须 / 8。Pipeline 会自动 round，但 batch 内不同 sample 必须同一分辨率。

第三，generator 复现。pipeline(generator=torch.Generator().manual_seed(42)) 控制随机性。多次同 seed 应该出同样图，但跨硬件（不同 GPU 型号）由于浮点非 deterministic 可能略差。

第四，prompt embedding 的形状陷阱。SDXL 的 prompt_embeds shape [B, 77, 2048]（两个 CLIP 拼接），pooled_prompt_embeds shape [B, 1280]（OpenCLIP pooled），手动构造时容易错。

### 6. 实践建议
新业务起步：直接用官方 Pipeline 跑通；准备扩展时拆出 components。

生产优化路径：（1）换 Scheduler 跑 20 步；（2）量化 UNet 到 FP8/FP4；（3）text encoder 输出 cache + offload；（4）VAE 用 TAESD 替代；（5）极致场景上 LCM/Hyper-SD 蒸馏。

ComfyUI 内部其实是基于 diffusers 风格的 component 解耦实现 workflow 编排，理解 diffusers 是理解 ComfyUI 的基础。

### 7. 30 秒速答
- Pipeline 是端到端封装（5 行代码出图）
- Scheduler 是采样算法（DDIM / DPM-Solver / Euler / LCM）可换
- ModelMixin 是模型基类（save/load/device 管理）
- 生产优化常拆 Pipeline + 换 Scheduler + 量化 model

### 8. 自测 checklist
- [ ] 你能不能讲清 Pipeline.__call__ 内部主要做哪几件事？
- [ ] 你能不能在不重训的情况下把 DDIM 换成 DPM-Solver？
- [ ] 你能不能说出 enable_model_cpu_offload 省多少显存代价多少时间？
- [ ] 你能不能拆 Pipeline 实现自定义中间步骤回调？

## Q5. DDIM / DPM-Solver / Euler / LCM 采样器对比与工程选择

> 🔴 专家 · 同一个模型，DDIM 50 步出图、DPM-Solver++ 20 步出图、LCM 4 步出图——采样器选对，速度立刻 5-10×，这是 diffusion 推理最有杠杆的优化点。

### 1. 核心结论
四类主流采样器：DDIM（确定性 ODE，50 步基线，质量好但慢）；DPM-Solver / DPM-Solver++（多步 ODE solver，20 步达到 DDIM 50 步质量）；Euler / Euler Ancestral（标准 ODE 求解 / 加随机，简单稳定）；LCM / TCD（consistency model 蒸馏，1-8 步出图但需配 LCM-LoRA 或专门训练的模型）。生产里推荐 DPM-Solver++ 作为默认（速度质量平衡最好），LCM 适合实时场景（如游戏、互动）。

### 2. 底层原理
Diffusion 数学上是从高斯噪声逐步去噪到数据分布的随机过程。可写成 SDE（随机微分方程）或 ODE（常微分方程）形式。采样器本质是 ODE/SDE 求解器：DDIM 用一阶 ODE 步骤；DPM-Solver 用二阶/三阶多步 solver（参考过去几步的输出），更准；Euler 是标准前向 Euler；Euler Ancestral 加噪声让结果有 variance（多次生成有差异）。

LCM（Latent Consistency Model）思想不同：训练时让模型直接预测"从任意 t 到 0 的去噪结果"，推理时 1-4 步直接跳到 t=0。需要专门蒸馏训练（用 LCM-LoRA 或全模型重训）。

### 3. 关键机制 / 流程 / 数据结构
第一，采样器 step 公式。DDIM: x_{t-1} = α_{t-1} · x̂_0 + σ_{t-1} · ε_pred。DPM-Solver++ 二阶: x_{t-1} 用上两步 ε_pred 做多项式外推。Euler: x_{t-1} = x_t - dt · (dx/dt)。LCM: x_0 = f(x_t, t)，直接预测 endpoint。

第二，step 数 vs 质量。SDXL 上典型质量足够的步数：DDIM 50；DPM-Solver++ 20-25；Euler 25-30；LCM 4-8。Flux 上：DDIM 50；DPM-Solver++ 28；LCM 4。step 减少导致细节损失，但 DPM-Solver++ 的二阶项让 20 步质量接近 DDIM 50 步。

第三，确定性 vs 随机。DDIM / DPM-Solver / Euler 是确定性的（同 seed 同 prompt 出同样图）。Euler Ancestral / DPM++ SDE 是随机的（每步加噪声）。LCM 通常确定性。生产 reproducibility 选确定性。

第四，跟模型训练耦合。SDXL 训练用 ε-prediction（预测噪声），采样器用 ε 公式；Flux 训练用 rectified flow（预测 velocity），采样器必须用 RF 公式。混用会出错。

### 4. 工程权衡 / 性能影响
SDXL 1024×1024 在 H100 上的 wall time：DDIM 50 步 ~3s；DPM-Solver++ 20 步 ~1.3s；Euler 30 步 ~2s；LCM 4 步 ~0.3s（配 LCM-LoRA）。质量主观评估：DDIM ≈ DPM-Solver++ > Euler > LCM 4 步（仍可用但细节略丢）。

工业部署选型：默认 DPM-Solver++（2.5x DDIM 速度，质量持平）；实时互动选 LCM（4-8 步）但提前训好 LCM-LoRA；研究/最高质量选 DDIM 50 步。

CFG 配合：DPM-Solver++ 用 CFG=4-7；LCM 不需要 CFG（蒸馏时已 absorbing），CFG=1 即可。

### 5. 常见追问 / 易错点
第一，scheduler 跟模型不匹配。SDXL（ε-pred）用 Flux 的 rectified flow scheduler 会出错。务必用 model.scheduler 默认或 from_config 转换。

第二，step 数低于训练 sigma 范围。每个 scheduler 内部有 sigma 时间序列，4 步可能跳过太多区间。低步数采样需要 sigma 重新调度（trailing 模式 vs leading 模式）。

第三，CFG distillation。SDXL Turbo / Hyper-SD 把 CFG 蒸馏进单 forward，省去 unconditional pass，速度翻倍。但 CFG 强度变得固定，不能 runtime 调。

第四，noise schedule。zero-SNR / v-prediction 等高级技术让低步数生成更稳定，但需要模型训练时支持。

### 6. 实践建议
新部署起步：DPM-Solver++ 2M Karras 25 步，CFG 6.0，几乎所有 SDXL 场景的最佳默认。

需要更快：训 LCM-LoRA → 切 LCM Scheduler → 4-8 步。LCM-LoRA 训练成本几小时单卡 A100。

需要最高质量：DDIM 50 步 + CFG distillation 模型。

A/B 测试时控制变量：固定 seed + 固定 prompt + 只换 scheduler，对比质量与速度。

### 7. 30 秒速答
- DDIM 50 步（基线）/ DPM-Solver++ 20 步（推荐）/ Euler 25-30 步（稳定）/ LCM 4-8 步（实时）
- DPM-Solver++ 是默认最佳：2.5x DDIM 速度质量持平
- LCM 需配 LCM-LoRA 重训，但 4 步出图
- scheduler 必须跟模型 prediction 类型（ε / v / flow）匹配

### 8. 自测 checklist
- [ ] 你能不能解释为什么 DPM-Solver++ 20 步能达到 DDIM 50 步质量？
- [ ] 你能不能讲清 LCM 跟传统采样器的根本区别？
- [ ] 你能不能说出 SDXL Turbo / Hyper-SD 跟 LCM 的关系？
- [ ] 你能不能识别 scheduler 跟模型不匹配的错误现象？

## Q6. DeepCache 加速原理与适用场景

> 🔴 专家 · UNet 后几层的特征在相邻 step 间几乎不变——把它们缓存复用，每步只跑前几层，diffusion 推理免费提速 2-3×。

### 1. 核心结论
DeepCache 是 2023 提出的免训练 diffusion 加速技术，核心观察：UNet 中间和深层特征在相邻 step 之间变化很小，可以跨 step 缓存复用；只有浅层（输入附近）需要每步重算。典型 SDXL 上配置 cache_interval=3（每 3 步重新算一次深层）能 2-3x 加速，质量损失轻微。DiT 类模型也有类似技术（如 PAB / Pyramid Attention Broadcast）。

### 2. 底层原理
理论依据：diffusion 后期 step 主要在细化高频细节，模型的语义特征（中深层）已基本稳定。所以这些层的 activation 跨步几乎一致。

UNet 结构上：encoder 从浅到深下采样、decoder 从深到浅上采样、skip connection 把同分辨率 encoder activation 接到 decoder。DeepCache 把 encoder 的中深层 + middle block 缓存，每 N 步只重算浅层 encoder + decoder 上采样路径。

具体：第 0 步跑全 UNet 保存中深层 activation；第 1 到 N-1 步只跑 shallow encoder（最浅 1-2 层）+ decoder（用缓存的 middle/deep encoder activation 通过 skip 注入）；第 N 步重新跑全 UNet 更新 cache；循环。

### 3. 关键机制 / 流程 / 数据结构
第一，cache 数据结构。cache 字典存 encoder 各 block 的 output activation（4D tensor）。Memory 约几百 MB（1024 分辨率 SDXL）。

第二，cache_interval 选择。SDXL 上推荐 3-5。interval=2 加速比 ~1.5x，质量几乎无损；interval=3 加速 2-3x，质量略降；interval=5 加速 4x，质量明显降。

第三，跟 ControlNet 兼容性。ControlNet 输出也是逐 step 算，跟 DeepCache 协同时要小心：ControlNet 通常每步都算（因为 control signal 注入需要新鲜），不能 cache。所以 DeepCache + ControlNet 时只 cache UNet 部分。

第四，DiT 上的对应技术。FORA / PAB（Pyramid Attention Broadcast）等是 DiT 上的同思路：缓存 attention output 或 FFN output 跨 step 复用。Flux DiT 上 PAB 能 2-3x 加速。

### 4. 工程权衡 / 性能影响
DeepCache 在 SDXL 上的真实 benchmark（H100 1024 BF16）：原始 DPM-Solver++ 20 步 ~1.3s；DeepCache interval=3 ~0.5s；接近 3x。质量评估在 FID / CLIP-Score 上差异 <1%。

跟其它加速技术叠加：DeepCache + FP8 量化 = ~4-5x；DeepCache + LCM = 不兼容（LCM 步数太少 cache 没意义）。

适用范围：standard diffusion 步数 20-50 步效果最佳；步数 <10 收益少；步数 >50 收益更大。

### 5. 常见追问 / 易错点
第一，为什么不能全 cache。最浅层 encoder 跟当前 noise level 高度相关，每步必须重算。Middle/deep encoder 主要 encode 语义，相对稳定。

第二，质量下降不均匀。简单 prompt + 单主体场景 DeepCache 效果好；复杂多主体 + 细节文字场景质量下降明显。生产前要业务 A/B 测试。

第三，跟 Multi-LoRA 切换的交互。换 LoRA 后 cache 失效，需要重置。生产里 LoRA 切换跟 DeepCache 协同要 careful。

第四，DiT 上的差异。DiT 没有 encoder-decoder 结构，PAB 是 cache attention output across blocks，机制不同但思路类似。

### 6. 实践建议
新业务默认开 DeepCache interval=3：免费 2-3x 加速，质量大体可接受。

业务质量要求高：先做 A/B 测试，比较开 DeepCache 跟不开的图像质量（人工或 CLIP-Score 评估），选择 interval。

跟量化 + LCM 配合：interval=3 + FP8 量化 ~5x；LCM 路径不开 DeepCache。

监控：实时 monitor 质量指标（CLIP-Score / 用户满意度），避免 cache 导致质量塌陷。

### 7. 30 秒速答
- 核心思想：UNet 中深层特征跨 step 几乎不变 → 缓存复用
- 实现：每 N 步重算全 UNet，中间步只算浅层 + decoder
- SDXL 上 interval=3 → 2-3x 加速，质量损失轻微
- DiT 对应技术 PAB / FORA，思路类似

### 8. 自测 checklist
- [ ] 你能不能解释为什么深层 activation 跨 step 变化小？
- [ ] 你能不能讲清 DeepCache 跟 ControlNet 怎么协同？
- [ ] 你能不能说出 cache_interval 跟加速比 / 质量的关系？
- [ ] 你能不能识别 DeepCache 在哪些 prompt 类型下质量损失明显？

## Q7. Flash Diffusion / LCM-LoRA / Hyper-SD 等少步采样技术

> 🔴 专家 · 把 30 步压成 4 步、再压到 1 步——这背后是 consistency model + adversarial distillation 等多种蒸馏路线的工程化。

### 1. 核心结论
少步采样技术通过蒸馏让 diffusion 模型在极少步数（1-8 步）下出图。三大主流方案：LCM/LCM-LoRA（Latent Consistency Model，4-8 步）；SDXL Turbo / Adversarial Diffusion Distillation（1-4 步）；Hyper-SD（progressive consistency distillation，1-8 步可调）。Flash Diffusion 是把多种技术整合成训练框架。生产场景：实时互动 / 移动端 / 高吞吐 batch 用少步；最高质量仍用 20-50 步标准采样。

### 2. 底层原理
Consistency Model 核心思想：训练一个模型 f(x_t, t)，让它对任意 t 都能直接预测 x_0（不需要多步迭代）。约束条件是"consistency"——从相邻 t 出发预测的 x_0 应该一致。训练时用 distillation loss 让 student 模型逼近 teacher 的多步采样结果。

LCM（Latent Consistency Model）：把 consistency 思想用在 latent diffusion 上，配合 v-prediction + Skipping-Step 训练目标。LCM-LoRA 是用 LoRA 形式蒸馏（不重训全模型），训练 ~10 小时单 A100。

SDXL Turbo (Adversarial Diffusion Distillation, ADD)：在 LCM 基础上加 adversarial loss（GAN-style discriminator），让 1 步生成质量大幅提升。

Hyper-SD：progressive distillation 思路，先训 32 步 → 16 步 → 8 步 → 4 步 → 1 步 student，每代用上代当 teacher。质量比单步 LCM 更好。

### 3. 关键机制 / 流程 / 数据结构
第一，LCM-LoRA 工作流。加载 base SDXL → 加载 LCM-LoRA → 切 LCMScheduler → 调用 pipeline(num_inference_steps=4, guidance_scale=0)（注意 CFG=0）。出图 4 步完成。

第二，SDXL Turbo 工作流。加载 sd-turbo（专门蒸馏的 checkpoint）→ pipeline(num_inference_steps=1-4, guidance_scale=0)。

第三，Hyper-SD 工作流。加载 base + Hyper-SD LoRA → 切对应 scheduler → 选 1/2/4/8 步配置之一。Hyper-SD 提供多个 LoRA 文件对应不同步数。

第四，CFG=0 不需要原因。蒸馏时已经把 CFG 信号吸收进模型，runtime 不需要额外的 unconditional pass。所以 1 步生成实际是 1 次 UNet 调用而非 2 次。

### 4. 工程权衡 / 性能影响
速度对比（SDXL 1024 H100）。标准 DPM-Solver 25 步 ~1.5s；LCM 4 步 ~0.3s；Turbo 1 步 ~0.15s。结合 FP8 量化进一步翻倍。

质量对比（主观 + CLIP-Score）。标准 25 步 100%；LCM 4 步 ~88-92%；Turbo 1 步 ~75-85%。Hyper-SD 4 步 ~92-95%（比 LCM 好）。

生态情况：LCM-LoRA 资源最多（每个 base model 都有人蒸馏）；Turbo 是 Stability AI 官方；Hyper-SD 字节出品，质量最好但生态稍新。

业务选型：实时互动（如 AI 画画工具）选 Turbo 1 步；标准 AIGC 服务选 LCM 4-8 步或 Hyper-SD；最高质量仍走标准采样。

### 5. 常见追问 / 易错点
第一，CFG=0 vs CFG=7 区别。CFG=0 时 LCM 出图是单次 UNet 调用；如果错误地 CFG > 1，pipeline 会跑 unconditional + conditional 两次 UNet，蒸馏前提被破坏，质量下降。

第二，跟 LoRA 叠加。LCM-LoRA + style LoRA 可以叠（weight 各 0.5-1.0），但效果不稳定。Hyper-SD LoRA + style LoRA 也类似。生产里常需要 A/B 测试。

第三，分辨率敏感。蒸馏模型在训练分辨率（1024）效果好，偏离（如 512 或 2048）质量降明显。

第四，蒸馏丢失能力。某些复杂 prompt（如包含文字、复杂构图）少步生成质量塌陷比标准采样严重得多。文字渲染特别脆弱。

### 6. 实践建议
新业务尝试少步：先 Hyper-SD 4 步 + 业务 A/B 测试（vs 标准 20 步）。

实时场景必备：SDXL Turbo 1 步 或 Hyper-SD 1 步。

跟 ControlNet 协同：少步 + ControlNet 是可以的，但 ControlNet 强度参数 (controlnet_conditioning_scale) 通常要调低（1.0 → 0.6-0.8）。

监控质量回归：上线少步后，跟踪 CLIP-Score / 用户反馈 / 客诉率，发现质量塌陷及时切回标准采样。

### 7. 30 秒速答
- LCM / Turbo / Hyper-SD 三大少步方案，1-8 步出图
- 核心思想：consistency model + adversarial distillation
- CFG=0 是少步生成关键（蒸馏已吸收 CFG 信号）
- 实时互动选 Turbo；标准 AIGC 选 LCM/Hyper-SD；最高质量仍 DPM-Solver 25 步

### 8. 自测 checklist
- [ ] 你能不能讲清 consistency model 跟标准 diffusion 训练目标的差异？
- [ ] 你能不能解释为什么 Turbo CFG 设 0？
- [ ] 你能不能说出 Hyper-SD 比 LCM 好在哪？
- [ ] 你能不能识别 LCM-LoRA + style-LoRA 叠加的常见问题？

## Q8. SVDQuant / SmoothQuant / Q-Diffusion 等 FP4/INT8 量化方案

> 🔴 专家 · LLM 量化生态成熟，AIGC 量化才刚起步——SVDQuant 把 FP4 量化用在 Unet/DiT 上，让 Flux 跑进消费级显卡，是 2024 末最值得关注的 AIGC infra 突破。

### 1. 核心结论
AIGC 模型量化主要做两件事：weight 量化（INT8/FP8/INT4/FP4）减显存；activation 量化（INT8/FP8）减计算。三类主流方案：（1）SmoothQuant for Diffusion 把 LLM 的 SmoothQuant 思路（激活 outlier 迁移）用在 Unet；（2）Q-Diffusion 是 GPTQ-style 校准量化在 Unet；（3）SVDQuant（MIT 2024）针对 DiT 用 low-rank SVD 分解处理 outlier，FP4 量化 Flux 仅损失 <2% 质量。生产部署 SDXL 默认 FP8 weight + FP8 activation；Flux 因显存压力大，FP4 量化更受关注。

### 2. 底层原理
Unet/DiT 量化的特殊挑战：（1）attention activation 有 outlier（跟 LLM 一样）；（2）AdaLN 的 scale/shift 范围动态；（3）latent 的数值范围比 LLM token 范围更不均匀；（4）generation 是 iterative 的，量化误差跨 step 累积。

SmoothQuant for Diffusion：找出 Unet 中激活有大 outlier 的层（通常 attention proj），把 scale s 迁移到权重：Y = (X/s)(sW)。s 是 offline 算的 per-channel 常数，fuse 到前一层 LayerNorm。Runtime 零额外开销。

Q-Diffusion：GPTQ 思路，layer-wise 校准 Unet weight，用 calibration set 跑 forward 收集 Hessian，列贪心量化 + 误差补偿。

SVDQuant：观察 DiT activation outlier 是低秩结构 → 用 SVD 分解 weight 为 low-rank（FP16 保留）+ residual（FP4 量化）。Low-rank 部分占 weight <5% 但吸收主要 outlier，residual FP4 量化误差小。Flux FP4 量化后显存从 24GB 降到 6GB，跑在 RTX 3090 24GB 上能 batch=4。

### 3. 关键机制 / 流程 / 数据结构
第一，量化粒度。Weight 通常 per-channel 或 per-group（128 元素一组 scale）。Activation per-tensor（粗）或 per-token（细，attention 用）。AIGC 上 per-group 128 是 sweet spot。

第二，KV cache 量化在 AIGC。Diffusion 没有 KV cache 概念，所以"KV 量化"不适用。但 cross-attention 的 K/V（来自 text embedding，定长）可以 cache + 量化，每 step 复用。

第三，calibration set。AIGC 量化的 calibration set 是 (prompt, latent, timestep) 三元组，跑几百次 Unet forward 收集 activation 分布。Flux 量化 calibration 用 LAION-style prompts 256-1024 条。

第四，SVDQuant 实现细节。SVD 分解 W = U·Σ·V^T，取 top-r 奇异值作 low-rank（r 通常 32），W ≈ U_r Σ_r V_r^T + R，R 是 residual 量化到 FP4。Inference 时 Y = U_r(Σ_r(V_r^T x)) + dequant(R_q) x。两次 matmul 但 r=32 时 low-rank matmul 几乎免费。

### 4. 工程权衡 / 性能影响
显存对比（Flux dev）。BF16: 24GB；FP8: 12GB；FP4 (SVDQuant): 6GB；INT4 朴素: 6GB 但质量塌陷。

速度对比（H100）。BF16 baseline；FP8 ~1.3x；FP4 ~1.5-1.7x（取决于 kernel）。质量 FP8 ≈ BF16；FP4 SVDQuant 略损 <2%；FP4 朴素损失明显。

工业部署：SDXL 默认 FP8 weight + activation；Flux dev FP8（H100 80G 单卡）或 FP4 SVDQuant（消费级 GPU）；SDXL 移动端 INT8 + 朴素量化（质量妥协换部署）。

### 5. 常见追问 / 易错点
第一，VAE 量化收益小。VAE 只在首尾各跑一次，量化省的时间相对 Unet 跑 N 次微不足道。生产几乎不量化 VAE。

第二，text encoder 量化。CLIP 几百 MB 量化收益小；Flux 的 T5-XXL 10GB 量化到 FP8 减半很有意义，但通常 dynamic per-tensor 量化即可（cache 输出后量化不影响 runtime）。

第三，跟 LoRA 兼容。量化模型 + LoRA 时，LoRA weight 通常保持 FP16，dynamic fuse 到量化 weight 上。LoRA + 量化是当前 active research 方向。

第四，质量评估方法。AIGC 量化质量评估比 LLM 难（图像没有 perplexity）。常用 CLIP-Score（生成图跟 prompt 一致性）+ FID（跟参考分布距离）+ 人工评估。

### 6. 实践建议
SDXL 生产部署：FP8 weight + FP8 activation，bitsandbytes 或 TensorRT-LLM 路径，质量 ≈ BF16。

Flux 生产部署：FP8（H100 80G）或 FP4 SVDQuant（消费级），优先 SVDQuant 因为 FP4 显存和速度双赢。

calibration：用业务真实 prompt 子集 1024 条做 calibration，比通用 LAION 效果好。

跟少步采样 + DeepCache 叠加：FP8 + LCM 4 步 + DeepCache interval=2，端到端 5-10x 加速。

### 7. 30 秒速答
- AIGC 量化主战场是 Unet/DiT，VAE 和 CLIP 量化收益小
- SmoothQuant: 激活 outlier 迁移；Q-Diffusion: GPTQ-style 校准；SVDQuant: SVD 分解低秩+ FP4
- SDXL FP8 主流；Flux FP4 SVDQuant 让消费级显卡能跑
- 质量评估用 CLIP-Score + FID + 人工，不能套 LLM perplexity

### 8. 自测 checklist
- [ ] 你能不能讲清 SVDQuant 的 SVD 分解思想？
- [ ] 你能不能解释 Diffusion calibration set 跟 LLM calibration set 的差异？
- [ ] 你能不能说出 Flux FP4 量化能装到多大显卡？
- [ ] 你能不能识别量化导致的质量塌陷模式（如文字渲染）？

## Q9. ControlNet 推理时集成与多 ControlNet 组合

> 🟡 进阶 · ControlNet 是给 diffusion 加结构控制的事实标准——线稿、姿势、深度图、边缘都能转图。工程上要解决"多 ControlNet 怎么协同"和"ControlNet 显存翻倍"两大问题。

### 1. 核心结论
ControlNet 在 base Unet 旁边复制一个"控制分支"（跟 base Unet 同架构但权重独立），接收 conditioning image（边缘 / 深度 / 姿势等）并把 control feature 注入 base Unet 的各层。推理时 base Unet 和 ControlNet 都要跑（显存近翻倍 + 计算近翻倍）。多 ControlNet（如同时姿势 + 深度）通过 control feature 加权和注入。生产里常用 ControlNet Union（一个模型支持多种 conditioning）减少显存压力。

### 2. 底层原理
ControlNet 架构：复制 base Unet 的 encoder + middle block（decoder 路径不复制），权重独立初始化。Conditioning image（如 canny 边缘图）经过 small encoder（几层 conv）得到 control hint，跟 base Unet 的浅层 activation 加在一起，然后过 control encoder。Control encoder 输出每层一份 control feature，通过 zero-conv（初始化为 0 的卷积层）注入 base Unet decoder 的对应位置。

Zero-conv 设计让 ControlNet 训练时初始不影响 base Unet 输出（zero-conv 输出 0），训练逐步学到非零的控制信号。

DiT 上的对应：ControlNeXt（DiT 版 ControlNet）或者直接 concat condition embedding 到 DiT input。Flux ControlNet 是 simplified 版本，只复制部分 DiT block。

### 3. 关键机制 / 流程 / 数据结构
第一，控制类型。Canny edge / Lineart / Depth (MiDaS) / OpenPose / Normal / Segmentation / Inpainting mask 等。每个类型对应一个 ControlNet checkpoint（SDXL 上每个 ~2.6GB）。

第二，推理时数据流。Conditioning image 由 preprocessor（如 canny detector）从输入图生成 → 跟 prompt 一起送 Pipeline → 每个 Unet step 同时跑 base Unet 和 ControlNet → control feature 加到 base decoder 上。

第三，多 ControlNet 组合。MultiControlNetPipeline 支持多个 ControlNet 同时使用：每个 ControlNet 独立跑，它们的 control feature 加权和注入 base Unet。权重通过 controlnet_conditioning_scale=[1.0, 0.5] 设定。

第四，ControlNet Union。一个统一 ControlNet 支持多种 conditioning（用一个 hint type embedding 区分），显存只占一份。Flux ControlNet Union 是事实标准。

### 4. 工程权衡 / 性能影响
显存对比。SDXL 单 ControlNet 加载后显存 +5GB；两个 ControlNet +10GB。ControlNet Union 只 +5GB 但支持 8+ 种 conditioning。

速度对比。单 ControlNet 让每 Unet step ~1.7x 时间（不到 2x 因为 ControlNet 只有 encoder 路径）；两个 ControlNet ~2.4x；ControlNet Union 跟单个相同。

跟少步采样兼容。LCM / Turbo + ControlNet 可以但效果略下降（控制强度需调低）。Hyper-SD + ControlNet 是常见组合。

跟量化兼容。ControlNet 可以独立量化（FP8 主流）。量化后显存压力大幅缓解。

### 5. 常见追问 / 易错点
第一，conditioning image preprocessor。Canny 用 OpenCV cv2.Canny；Depth 用 MiDaS（独立模型，~100MB）；OpenPose 用 mmpose 或 controlnet_aux 库。Preprocessor 时间通常几十毫秒，不是瓶颈。

第二，跟 inpainting 混淆。Inpainting 是 ControlNet 的一种应用（mask + reference image），但 inpainting 也可以用 inpainting-specific Unet（如 sd-inpaint），两者不同。

第三，control scale 调节。controlnet_conditioning_scale=1.0 是强控制（严格按 conditioning 来），scale=0.5 是弱控制（更自由）。生产里多数场景 0.7-0.9。

第四，多 ControlNet 冲突。姿势 + 深度组合通常协同好；姿势 + canny 边缘可能冲突（姿势希望粗结构，canny 希望精细边缘），需调权重。

### 6. 实践建议
新业务起步：SDXL + ControlNet Union（如果有）或单个 ControlNet（姿势最常用）。

Flux：用 Flux ControlNet Pro / Union（社区维护）。

多 ControlNet：先 1 个跑通 → 加第 2 个调权重 → 不超过 3 个。

显存紧张：用 ControlNet Union + FP8 量化，能在 24GB 卡上跑 Flux + 2 种控制。

### 7. 30 秒速答
- ControlNet = base Unet 旁的"控制分支"，inject control feature 到 base decoder
- 单 ControlNet +5GB 显存 + 1.7x 时间；Union 支持多 conditioning 共享一份显存
- 控制类型：canny / depth / pose / normal / seg 等，preprocessor 算
- 多 ControlNet 加权组合，scale 调节控制强度

### 8. 自测 checklist
- [ ] 你能不能讲清 ControlNet 的 zero-conv 作用？
- [ ] 你能不能解释为什么 ControlNet 显存只增 ~1.7x 而不是 2x？
- [ ] 你能不能说出 ControlNet Union 跟单个 ControlNet 的差异？
- [ ] 你能不能识别多 ControlNet 互相冲突的场景？

## Q10. LoRA 动态加载与多 LoRA 切换的工程实践

> 🔴 专家 · 一个 SDXL 服务通常要支持几百个 LoRA（不同 style / character / object），怎么切换不卡、怎么组合不冲突、怎么跟量化共存——这是 AIGC fleet 工程最复杂的部分之一。

### 1. 核心结论
LoRA 是给 base 模型加 low-rank adaptation 的轻量 fine-tune（每个 LoRA 几 MB-几十 MB），AIGC 生态里几乎每个业务都有几十到几百个 LoRA。生产挑战：（1）LoRA 切换延迟（每次 fuse 进 base 几十毫秒）；（2）多 LoRA 组合（多个 style 叠加）；（3）跟量化模型协同（LoRA FP16 fuse 到 INT8 base）；（4）显存预算（多 LoRA cache 在 GPU 上）。常用方案是 LoRA hot-swap + LRU cache + 离线 merge 热点 LoRA 进 base。

### 2. 底层原理
LoRA 数学：W_lora = W_base + α · (B · A)，其中 A 是 [r, in_dim]，B 是 [out_dim, r]，r 是 rank（通常 4-64）。alpha 是 strength scale（runtime 可调）。

应用方式两种：（1）Fuse: 把 α·B·A 加进 W_base，runtime 用 W_lora 推理（推理快但 fuse 需要时间）；（2）Compute-on-fly: runtime 每次 forward 算 W_base x + α B A x（不 fuse，但每次推理多一对 matmul）。

切换：fuse 模式下换 LoRA 要先 unfuse（W_base - 旧 α·B·A），再 fuse 新 LoRA。Unfuse + fuse 通常几十毫秒。

### 3. 关键机制 / 流程 / 数据结构
第一，LoRA 加载流程。LoRA 文件（safetensors）→ load 到 GPU buffer → 跟 base Unet 各层 weight 匹配（按 layer name）→ fuse 或保存 compute-on-fly mode。

第二，多 LoRA 组合。每个 LoRA 独立 α scale，weighted sum 后 fuse：W = W_base + Σ_i α_i · B_i A_i。生产里支持 2-4 个 LoRA 叠加是常态。

第三，跟量化 base 协同。量化 base 是 INT8/FP4，LoRA 是 FP16。Fuse 时要 dequant base → 加 LoRA → 重新 quant。Cost 较高。Compute-on-fly 模式更友好：base INT8 matmul + LoRA FP16 matmul 加起来。

第四，LoRA cache。GPU 上 cache 热点 LoRA（不卸载），冷 LoRA 在 CPU 内存。GPU cache 大小看显存：保留 50-200 个 LoRA 在 GPU 是常见配置。

### 4. 工程权衡 / 性能影响
切换延迟。Fuse mode: 每次切换 30-100ms（dequant + fuse + quant）；Compute-on-fly: 切换近 0 但每次推理 +5-15% latency。生产里热点 LoRA 用 fuse + cache，长尾 LoRA 用 compute-on-fly。

显存。每 LoRA fuse 后不增显存（merge 进 base）；compute-on-fly 时每 LoRA 几 MB-几十 MB。100 个 LoRA cache 几 GB 显存。

吞吐。LoRA 不影响 base Unet 计算，主要影响调度（切换时不能 batch）。生产里把请求按 LoRA 分组，同 LoRA 批量处理。

Punica / S-LoRA 类技术：用 batched LoRA kernel 让不同 sample 用不同 LoRA 同 batch 推理，无切换开销。vLLM 集成了类似机制。AIGC diffusion 的 PunicaKernels 版本是 active research。

### 5. 常见追问 / 易错点
第一，alpha 跟 scale 区别。LoRA 内部 alpha 是训练时的 normalization 常数（通常 alpha = rank 或 2×rank）；scale 是 runtime 用户控制（0.5-1.5），最终 strength = scale × alpha / rank。

第二，rank 跟显存 / 质量。rank=4 LoRA 几 MB，质量基础；rank=32 几十 MB，质量好；rank=64+ 接近全 fine-tune 但 LoRA 优势减少。生产里 rank=16-32 是 sweet spot。

第三，跟 ControlNet 协同。两者完全独立，可以同时用。注意权重叠加导致结果偏移。

第四，多 LoRA 冲突。同类型（如两个 style LoRA）叠加常出现风格混乱；不同类型（如 style + character）通常协同好。

### 6. 实践建议
LoRA fleet 设计：热点 100 个 LoRA fuse + GPU cache；长尾 LoRA compute-on-fly + CPU pool。

切换策略：sticky routing（同 LoRA 路由同实例），减少切换；request batch by LoRA。

质量保证：上线前 LoRA × prompt 矩阵 A/B 测试，识别冲突 LoRA。

成本控制：定期统计 LoRA 使用频率，长期不用的 LoRA 从 GPU cache 移除。

### 7. 30 秒速答
- LoRA = W_base + α · B · A，rank 4-64，每个 LoRA 几 MB-几十 MB
- Fuse mode 快但切换需要 dequant/fuse/quant；compute-on-fly 慢但切换零代价
- 多 LoRA weighted sum 叠加，生产支持 2-4 个组合
- Punica/S-LoRA 让不同 sample 用不同 LoRA 同 batch 跑

### 8. 自测 checklist
- [ ] 你能不能解释 LoRA 跟量化 base 模型的协同方案？
- [ ] 你能不能讲清 fuse vs compute-on-fly 各自适用场景？
- [ ] 你能不能说出生产里热点 LoRA 怎么 cache？
- [ ] 你能不能识别多 LoRA 叠加导致质量塌陷的场景？

## Q11. IPAdapter 多条件融合机制与推理开销

> 🟡 进阶 · 给 SDXL 加一张参考图就能生成同风格作品——IPAdapter 比 LoRA 更轻量、比 ControlNet 更柔和，是 AIGC 生产工具箱必备。

### 1. 核心结论
IPAdapter（Image Prompt Adapter）让 diffusion 模型接受图像作为额外 prompt（除文本外）。架构上是给 cross-attention 加一个新的 image-conditioned KV 分支，跟 text-conditioned KV 并行；推理时 image 经过 CLIP image encoder 编码后注入。比 LoRA 轻（几十 MB）、比 ControlNet 柔（不强制结构相似），适合风格迁移 / 参考图生成 / face-consistency 等场景。生产里 IPAdapter + LoRA + ControlNet 三组合是 SDXL fleet 常态。

### 2. 底层原理
IPAdapter 核心：训练一个 image encoder（基于 CLIP vision）+ 一组 image-cross-attention projection layers。Image 经 CLIP encode 得 image embedding → projection → image-conditioned K/V，跟 text K/V 一起进 cross-attention。Cross-attention 公式变成：Attention(Q_image_feature, [K_text, K_image], [V_text, V_image])。

Strength 控制：scale 参数（0-1）调节 image 影响强度。0 完全忽略 image（同纯文本 prompt），1 强烈受 image 影响。

FaceID 变种：IPAdapter-FaceID 用人脸专门 encoder（InsightFace ArcFace 等）替代 CLIP vision encoder，针对面部一致性优化。

### 3. 关键机制 / 流程 / 数据结构
第一，加载与初始化。IPAdapter checkpoint（几十 MB） load → image encoder weights 加载到 GPU → projection layers 注入 cross-attention 模块。

第二，推理流程。Reference image → CLIP vision encode（一次性 ~50ms）→ image embedding → projection → image K/V tensors（每层一份）→ cache。每 step 用 cached image K/V 跟当前 text K/V 一起进 cross-attention。

第三，多 reference image。FaceID Plus 支持多张参考图 averaging；IPAdapter-Plus 用 patch-level image features 更细粒度。

第四，跟 ControlNet 协同。IPAdapter 控制风格 + ControlNet 控制结构（姿势/深度）= 强组合。比如 reference image 提供风格，pose ControlNet 提供构图。生产里非常常见。

### 4. 工程权衡 / 性能影响
推理开销。每 Unet step 多一份 cross-attention（K/V 维度翻倍），约 +15-25% 时间。Image encode 一次性 ~50ms。

显存。IPAdapter checkpoint 几十 MB；image embedding cache 几 MB。整体显存增量小。

跟少步采样。LCM / Turbo + IPAdapter 可以，质量略损。

跟量化协同。IPAdapter 的 projection layers 可量化（FP8 主流），image encoder 通常保 FP16。

### 5. 常见追问 / 易错点
第一，IPAdapter vs ControlNet 选择。结构控制（姿势 / 深度 / 边缘）选 ControlNet；风格 / 参考 / 主体一致性选 IPAdapter；两者可叠加。

第二，scale 调节。0.7-0.9 是 sweet spot。scale=1.0 可能过强导致 prompt 被忽略；scale<0.3 影响微弱。

第三，FaceID 局限。FaceID 适合保持同一人脸特征，但姿势 / 表情 / 配饰仍需 ControlNet / prompt 控制。

第四，跟 LoRA 协同。Style LoRA + IPAdapter 风格冲突常见。优先 IPAdapter 提供风格 + LoRA 用于其它特征。

### 6. 实践建议
新业务起步：IPAdapter Plus（SDXL）或 FLUX-IP-Adapter（Flux），scale=0.8 默认。

参考图来源：用户上传 / 业务图库 / 之前生成结果。Image embedding cache 跨请求复用（同参考图）。

跟 ControlNet 组合：pose ControlNet + IPAdapter，scale 各 0.8 / 0.8，是人物生成主流配置。

监控：跟踪 IPAdapter 触发率 / 用户满意度，识别 scale 设置不当的场景。

### 7. 30 秒速答
- IPAdapter 让 diffusion 接受 image prompt（除 text 外）
- 实现：image cross-attention KV 分支，runtime 加权融合
- 适合风格迁移 / FaceID / 主体一致性，比 ControlNet 柔和
- +15-25% 单 step 时间，比 LoRA 重比 ControlNet 轻

### 8. 自测 checklist
- [ ] 你能不能讲清 IPAdapter 的 cross-attention 怎么变化？
- [ ] 你能不能解释 IPAdapter vs ControlNet 各自适合什么？
- [ ] 你能不能说出 IPAdapter image embedding 怎么 cache？
- [ ] 你能不能识别 IPAdapter scale 设置不当的现象？

## Q12. ComfyUI workflow 引擎与节点编排原理

> 🟡 进阶 · ComfyUI 不只是个 GUI，是个 dataflow 执行引擎——理解它的拓扑解析 / 缓存 / 节点 lazy execution，是把 ComfyUI 工作流上 production 的关键。

### 1. 核心结论
ComfyUI 是基于 Python 后端 + Web 前端的 diffusion 模型 workflow 编排工具。核心是 dataflow graph 执行引擎：用户在 UI 拖拽节点连接 → 后端解析成 DAG → 拓扑排序执行 → 每节点输出 cache 复用。设计上对 AIGC 工作流（多模型组合、多步处理、条件分支）非常友好，但作为 production serving 直接部署有局限（单进程 / 并发弱）。生产里通常用 ComfyUI 设计 workflow，然后用 ComfyUI API 或导出成 diffusers 代码部署。

### 2. 底层原理
节点系统：每个节点是 Python 类，定义 INPUT_TYPES / OUTPUT / FUNCTION。例如 KSampler 节点输入 model/positive/negative/latent/seed/steps 等，输出 latent；CLIPTextEncode 节点输入 text + clip，输出 embedding。

Graph 执行：用户拖拽形成的连接被序列化为 JSON（"prompt"），后端 PromptExecutor 解析 → 检查依赖 → 拓扑排序 → 按序执行节点。每节点输出按 (node_id, output_idx) cache，下次同 input 直接复用（不重算）。

Hash-based cache：节点 input hash 不变就复用 output。这让 iterative 编辑（只改一个节点）非常快（其它节点 cache 命中）。

### 3. 关键机制 / 流程 / 数据结构
第一，节点类型。Loaders（CheckpointLoader / LoRALoader / ControlNetLoader）；Conditioning（CLIPTextEncode / ConditioningCombine）；Sampler（KSampler / KSamplerAdvanced）；Latent（EmptyLatent / VAE Encode/Decode）；Image（LoadImage / SaveImage）；Math / Utility。

第二，自定义节点。Python class + ComfyUI 注册接口 → 放 custom_nodes 目录 → 重启自动加载。生态有几千个第三方节点（如 ComfyUI-Manager 列表）。

第三，API 调用。/prompt POST workflow JSON + client_id → 后端 queue 执行 → /history 取结果。生产部署可绕过 GUI 直接 API。

第四，导出代码。社区工具 ComfyUI-to-Python 可把 workflow 导出成等价 diffusers Python 代码，便于服务化部署。

### 4. 工程权衡 / 性能影响
ComfyUI 优势：可视化复杂 workflow（IPAdapter + ControlNet + LoRA + 多 step）；社区生态丰富（节点 / workflow 模板）；快速迭代调试。

劣势：单进程执行（不能跨进程并发同 workflow）；模型加载在主进程（不能多模型隔离）；GPU 利用率不如自研推理引擎。

production 部署模式：（1）ComfyUI API 直接 serve（小流量、原型）；（2）多 ComfyUI 实例 + 前置 router（中流量）；（3）ComfyUI 设计 + 导出到 diffusers/自研引擎部署（大流量）。

### 5. 常见追问 / 易错点
第一，跟 diffusers 关系。底层都基于 diffusers 组件，ComfyUI 是上层编排。能在 diffusers 做的 ComfyUI 都能做，反之未必。

第二，自定义节点稳定性。第三方节点质量参差不齐，更新 ComfyUI 可能挂掉某些节点。生产环境锁版本。

第三，并发限制。ComfyUI 默认单 queue 单 worker，多请求排队执行。要并发要起多个 ComfyUI 实例。

第四，VRAM 管理。ComfyUI 默认 keep-loaded 所有 model（节省切换时间），多模型时显存占满。可以配置 --normalvram / --lowvram / --novram 模式。

### 6. 实践建议
ComfyUI 适合：原型设计 / 创作者工作流 / 内部工具 / 小流量 serving。

不适合：高并发生产 serving / 严格 SLO 服务 / 多租户隔离。

混合模式：用 ComfyUI 设计 workflow → 测试效果 → 导出 diffusers 代码 → 部署到自研推理服务。

生产里常见架构：creator 用 ComfyUI workflow → API 提交到自研后端 → 后端 parser 转 diffusers 调用 → vLLM/SGLang-style serving 引擎执行。

### 7. 30 秒速答
- ComfyUI = 基于 DAG 的 dataflow workflow 引擎
- 节点系统 + JSON 序列化 workflow + hash cache
- 适合原型 / 创作 / 内部工具，不适合大流量生产 serving
- 生产模式：ComfyUI 设计 + 导出 diffusers 代码 + 自研 serving

### 8. 自测 checklist
- [ ] 你能不能讲清 ComfyUI 节点 cache 的 hash 机制？
- [ ] 你能不能解释 ComfyUI 单进程并发的限制？
- [ ] 你能不能说出从 ComfyUI 到生产 serving 的常见迁移路径？
- [ ] 你能不能识别 ComfyUI 自定义节点的稳定性风险？

## Q13. VAE encoder/decoder 加速（TAESD / tiny VAE / 分块解码）

> 🟡 进阶 · VAE 不是 diffusion 推理的主战场，但 1024 图解码 100ms、4K 图解码爆显存——这两件事让 VAE 优化在大分辨率场景变得重要。

### 1. 核心结论
VAE 在 diffusion 推理只跑一次（latent → image 解码，或 image → latent 编码 if img2img），不像 Unet 跑 N 步那么频繁。但 VAE 在大分辨率（>1024）下显存压力大 + 时间不可忽略。三类加速：（1）TAESD（Tiny AutoEncoder for SD）用极小 VAE 替代，几 MB 模型、解码 10ms 但质量略损；（2）Tiled VAE 分块解码避免显存爆炸；（3）VAE 量化（INT8）减显存。生产里 SDXL 1024 默认全 VAE；2048+ 用 tiled；预览用 TAESD。

### 2. 底层原理
SDXL VAE：~50M 参数，BF16 ~100MB。解码 1024 latent（64×64×4）→ 1024 image（1024×1024×3），中间 activation 峰值显存 ~2-4GB（卷积 + ResBlock 多层）。

显存峰值原因：VAE decoder 是多层卷积上采样，每层 activation 是 4D tensor（B×C×H×W），H/W 随上采样翻倍。最后几层 H/W 接近输出分辨率，activation 巨大。

TAESD 设计：把 VAE 压成几层简单卷积，参数 1M 量级。训练时让 TAESD 输出逼近原 VAE 输出。质量略损但解码 10x 快、显存几乎为 0。

Tiled VAE：把 latent 切成 overlap tiles（如 4 块），每块独立解码 → 重叠区域 blend → 拼接。显存峰值降到 1/N（N=tile 数）。代价是边界 artifact（blend 处理）+ 慢 1.5-2x（重叠计算）。

### 3. 关键机制 / 流程 / 数据结构
第一，标准 VAE 解码。`image = vae.decode(latent / 0.18215)`，0.18215 是 SDXL latent scale（FLUX 是 0.3611）。

第二，Tiled VAE 实现。手动切 tile + decode + blend，或用 diffusers 的 vae.enable_tiling() 自动处理。tile_size 默认 512 latent（=4096 pixel），可调小到 128（256 latent）。

第三，TAESD 集成。用 TAESD checkpoint 替代 vae 组件：pipe.vae = TAESD.from_pretrained("madebyollin/taesdxl"). 预览场景常用。

第四，INT8 量化。VAE 量化收益小但能减一半显存，diffusers Pipeline 没原生支持，需自定义 quantize wrapper。

### 4. 工程权衡 / 性能影响
SDXL 1024 解码时间（H100 BF16）：标准 VAE ~50ms；Tiled VAE 4 tile ~80ms；TAESD ~5ms。

SDXL 2048×2048 解码（latent 256×256）：标准 VAE 显存峰值 ~16GB（可能 OOM）；Tiled VAE 4 tile ~4GB；TAESD ~1GB。

质量对比：标准 VAE 100%；Tiled VAE 99%（边界微 artifact）；TAESD 85-90%（细节略丢，预览级）。

生产策略：1024 用标准；2048+ 用 tiled；preview 路径用 TAESD 给用户先看后等高质量。

### 5. 常见追问 / 易错点
第一，VAE 缺失或错误。SDXL VAE 跟 FLUX VAE 不通用（latent shape / scale 都不同）。换模型时要换对应 VAE。

第二，latent scale。0.18215 (SD/SDXL)、0.3611 (FLUX)、不同 model 不同。Pipeline 内部已处理，手动调时容易错。

第三，img2img 的 encode。img2img 也要 VAE encode（image → latent），同样有显存压力，可 tiled encode。

第四，TAESD 的训练数据。TAESD 在 LAION 上训练，特定 domain（如 anime / portrait）效果可能塌陷。生产业务可能要 fine-tune TAESD。

### 6. 实践建议
1024 及以下：标准 VAE，no special。

2048+ 或显存紧张：vae.enable_tiling()，tile_size=512 默认。

实时 preview：TAESD（10ms 出预览图）+ 后台跑标准 VAE 出最终图。

显存极度紧张：VAE offload 到 CPU（除了解码瞬间），代价是每次 ~200ms 加载。

### 7. 30 秒速答
- VAE 解码 1024 图 ~50ms，2048+ 容易爆显存
- TAESD: 几 MB tiny VAE，10ms 解码，质量略损（预览用）
- Tiled VAE: 分块解码，显存降到 1/N，慢 1.5-2x
- 生产策略：1024 标准 / 2048+ tiled / preview 用 TAESD

### 8. 自测 checklist
- [ ] 你能不能讲清 VAE 解码显存峰值来自哪里？
- [ ] 你能不能解释 Tiled VAE 的 overlap blend 必要性？
- [ ] 你能不能说出 TAESD 的适用场景和局限？
- [ ] 你能不能识别 VAE / latent scale 不匹配的现象？

## Q14. Text encoder 优化（CLIP + T5-XXL 的双 encoder 推理）

> 🟡 进阶 · Flux 的 T5-XXL 占 10GB BF16 显存，比 Unet 主体还重——这让 text encoder 优化从可忽略变成 AIGC 推理必须处理的话题。

### 1. 核心结论
SDXL 用 CLIP-L + OpenCLIP-G（合计 ~820M），text encode 耗时几十毫秒、显存 ~2GB。Flux 用 CLIP-L + T5-XXL（合计 ~5B，T5-XXL 占 4.8B），text encode 耗时 100-300ms、显存 ~10GB（T5 BF16）。SD3 类似 Flux。Text encoder 优化主线：（1）量化（T5-XXL FP8/INT8 减半显存）；（2）prompt-level cache（同 prompt 输出不变）；（3）CPU offload（用完挪出 GPU）；（4）shared text service（多个 diffusion 实例共享一个 text encoder 服务）。

### 2. 底层原理
CLIP：图文对训练的双向 encoder，输出 token-level + pooled embedding。SD 用 token-level embedding（cross-attention）+ pooled（global condition）。

T5-XXL：encoder-only Transformer（4.8B 参数），输出每 token embedding。Flux 用它处理长 prompt（最多 512 tokens），细粒度语义比 CLIP 强（特别是文字 / 复杂构图）。

DiT 注入 mechanism：T5 embedding + CLIP pooled 拼接后通过 adapter layers 注入 DiT 的 AdaLN 或 cross-attention。

### 3. 关键机制 / 流程 / 数据结构
第一，标准推理流。prompt → tokenize → text_encoder forward → embedding tensor → cache（如果实现了）→ 送 diffusion loop。每生成一张图算一次。

第二，prompt cache。同 prompt 输出 deterministic，可 cache。Cache key=(prompt, encoder_version)，value=(embedding, pooled)。Cache size 几 MB-几十 MB / 条。Cache hit 时 text encode 跳过。

第三，CPU offload。pipeline.enable_model_cpu_offload() 让 text encoder 用完挪到 CPU。下次推理重新载 GPU（~100ms / GB）。Flux T5-XXL offload 后省 ~10GB GPU 显存。

第四，shared text service。单独部署 text encoder 服务（专用 GPU 或 CPU），diffusion 推理实例通过 RPC 调用。多个 diffusion 实例共享一个 text service。

### 4. 工程权衡 / 性能影响
显存对比（Flux dev）。Naive: T5 + CLIP + Unet = 10 + 0.5 + 24 = 34GB（A100 80G 够，4090 不够）。CPU offload: 10 + 0.5 + 24 with swap = 24GB peak（4090 紧但够）。Shared service: T5 在另一台 → diffusion 端 0.5 + 24 = 24GB。

速度对比。Naive: text encode 200ms + diffusion 6s。Cache hit: 0ms + 6s。Offload: 200ms encode + 100ms load swap + 6s ≈ 6.3s。Shared service: 50ms RPC + 6s。

Quantization。T5-XXL FP8: 5GB 显存（减半），速度略损但精度损失 <1%。INT8 calibrated: 3GB 但调优成本高。

### 5. 常见追问 / 易错点
第一，prompt cache 的边界。Negative prompt 也要 cache（CFG 需要两次 text encode）。Cache key 包含 negative prompt。

第二，长 prompt 处理。SDXL CLIP 77 token 上限，长 prompt 被截断（早期）或分块 encode 后 average / weighted sum（高级）。Flux T5 512 token 上限。

第三，多 batch 时。Batch 不同 prompt 时不能单 cache lookup，但可以 batch encode（一次性 forward 多 prompt）。

第四，T5-XXL FP16 数值稳定性。T5 在 FP16 有溢出问题（attention 输出大），需要 cast 到 BF16 或部分 FP32（attention 内）。HuggingFace transformers 已处理。

### 6. 实践建议
SDXL：text encoder 不优化也 OK（占比小）。Prompt cache 是 nice-to-have。

Flux：必须量化 + cache + offload 组合。Production 推荐 T5 FP8 + prompt cache + 不 offload（latency 优先）。

Fleet 规模：>10 个 diffusion 实例考虑 shared text service。

prompt 模板化：业务里很多 prompt 是 template（"a {style} photo of {subject}"），template + slot cache 命中率高。

### 7. 30 秒速答
- SDXL: CLIP+OpenCLIP 820M，开销小；Flux: T5-XXL 4.8B，10GB 显存
- 优化：量化（T5 FP8）/ prompt cache / CPU offload / shared text service
- Flux 生产推荐：T5 FP8 + prompt cache，省显存 + 加速
- Fleet 规模时考虑 shared text service 减少多实例开销

### 8. 自测 checklist
- [ ] 你能不能讲清 Flux 跟 SDXL 在 text encoder 显存上的差异？
- [ ] 你能不能解释 prompt cache 的 key 设计？
- [ ] 你能不能说出 T5-XXL FP8 量化的精度损失范围？
- [ ] 你能不能识别长 prompt 在 CLIP/T5 上的处理差异？

## Q15. SDXL 服务化吞吐 vs 时延权衡（batch size / step 数 / resolution）

> 🔴 专家 · 同一个 SDXL，单图 1.3s 出 30 图 / 秒，要么是 batch 8 同时跑、要么是 batch 1 跑 8 次——选哪条路决定显存预算和 SLO 形态。

### 1. 核心结论
SDXL 服务化的核心 trade-off：增大 batch 提吞吐但增 latency 和显存；减 step 数减 latency 但损质量；降分辨率减时间但损用户体验。生产里通常分场景配置：高吞吐 batch 离线生成用 batch 4-8 + 25 step；交互在线服务用 batch 1-2 + 25 step 或 LCM 4 step；极致互动用 Turbo 1 step。SLO 维度上要分 TTFI（time-to-first-image）和 throughput（imgs/s）双轴。

### 2. 底层原理
SDXL Unet 是 compute-bound（不像 LLM decode 是 memory-bound）。所以 batch 增加 batch size 时单 step 时间近似线性增加（不像 LLM decode 摊薄到接近免费）。

但 batch 复用 weight load → 单图均摊时间略降（10-30%）。Batch 4 的 throughput ≈ batch 1 的 3.5x，不是 4x。

显存随 batch 线性涨：activation 是 [B, C, H, W] 4D tensor，B 翻倍 activation 翻倍。SDXL 1024 batch 1 ~10GB；batch 4 ~16GB；batch 8 ~24GB。Flux 类似倍数。

### 3. 关键机制 / 流程 / 数据结构
第一，throughput-oriented 部署。batch_size = 4-8, step = 25-30 (DPM-Solver++), 1024 分辨率。H100 batch 4 SDXL 1024 25 step ~4-5 imgs/min/gpu。

第二，latency-oriented 部署。batch_size = 1-2, step = 25 或 LCM 4-8, 1024。Latency 1-2s。

第三，real-time 部署。batch_size = 1, step = 1-4 (Turbo / LCM), 768-1024。Latency <300ms。

第四，混合 fleet。一部分实例 throughput 模式跑 batch 离线；一部分 latency 模式跑实时；路由按请求类型分发。

### 4. 工程权衡 / 性能影响
吞吐 vs 时延数据（H100 SDXL 1024 BF16）。batch=1 step=25: ~1.5s/img, 0.67 imgs/s。batch=4 step=25: ~5s/batch, 0.8 imgs/s/gpu (高 20%)。batch=8 step=25: ~9s/batch, 0.89 imgs/s/gpu (高 33%)。

分辨率影响。512x512 step=25: ~0.5s; 1024 step=25: ~1.5s; 1536 step=25: ~3.5s; 2048 step=25: ~7s (FLOPs 二次方涨)。

step 影响。1024 25 step: ~1.5s; 50 step: ~3s; LCM 4 step: ~0.3s; Turbo 1 step: ~0.15s。

### 5. 常见追问 / 易错点
第一，batch 内必须分辨率一致。SDXL batch 不支持不同分辨率混合（tensor shape 要齐）。生产路由按分辨率分组。

第二，Continuous batching 不适用。LLM 的 continuous batching 利用 decode 阶段 token-level 动态调度，diffusion 是 step-level 调度，每 step 内固定 batch。不能像 LLM 那样动态加入退出。

第三，"warmup" 重要。第一次 batch=N 推理会触发 cuDNN/cuBLAS autotune（选 kernel），首次几秒慢。生产部署要 warmup 所有要跑的 batch size。

第四，多用户并发不等于 batch。10 个用户同时请求不一定 batch=10 跑（除非攒齐），可能 batch=4 + batch=4 + batch=2 多次串行。SLA 紧张时 batch 1 直接出图反而最快。

### 6. 实践建议
新业务起步：先 batch=1 / step=25 / DPM-Solver++ / 1024，跑通 latency 基线（H100 上 ~1.5s）。

throughput 路径：batch 4-8 + DPM-Solver 25 + FP8 量化 + DeepCache。每 GPU ~5-8 imgs/min。

latency 路径：batch 1 + LCM 4 + FP8。Latency 300-500ms。

real-time 路径：batch 1 + Turbo 1 + FP8。Latency 150-250ms。

监控：分别 monitor TTFI p50/p99、throughput imgs/s、GPU util %、显存峰值。

### 7. 30 秒速答
- SDXL 是 compute-bound：batch 翻倍 ≠ 时间翻倍但 throughput 涨 20-30%
- 显存随 batch 线性涨（batch 8 1024 SDXL ~24GB）
- 吞吐：batch 4-8 + 25 step；时延：batch 1-2 + LCM 4-8 step
- 部署 SLO 分 TTFI + throughput 双轴

### 8. 自测 checklist
- [ ] 你能不能拆解 SDXL 1024 batch 4 25 step 端到端时间？
- [ ] 你能不能解释为什么 batch 翻倍 throughput 不翻倍？
- [ ] 你能不能说出 LCM vs Turbo 在 latency 场景的差异？
- [ ] 你能不能识别 batch 内分辨率混合的限制？

## Q16. 多并发图像生成 batch 调度与显存预算

> 🔴 专家 · vLLM 用 PagedAttention 管 KV cache，AIGC 用什么管多并发？答案：固定 max_batch_size + 显存预算分配 + 分辨率分组队列，跟 LLM 完全不同的调度哲学。

### 1. 核心结论
AIGC fleet 调度跟 LLM serving 根本不同：（1）没有 KV cache 概念（diffusion 是 multi-step iterative，不缓存中间状态）；（2）batch 内必须 shape 齐（分辨率 / step / scheduler 一致），不能像 LLM 那样动态加入退出；（3）显存预算是静态的（max_batch_size × per-sample memory）；（4）调度核心是按 (分辨率, model_version, LoRA, ControlNet) 分组凑 batch。生产架构：前置路由器分组 → 实例内按 max_batch 凑 batch → 满足 batch 或超时则发车。

### 2. 底层原理
显存预算计算：`peak_mem = weight + activation × batch + workspace`。SDXL 1024 BF16: weight ~5GB, activation per-sample ~2GB, workspace ~1GB。RTX 4090 24GB 上 batch 8 = 5 + 16 + 1 = 22GB（紧）。

启动期预测 max_batch_size：跑 profiling forward 测峰值 → 反推安全 max_batch。生产保 10% buffer 防 OOM。

调度策略：固定 max_batch（如 4），有请求来时塞进 batch slot；slot 全满立即发车；slot 半满超时（如 50ms）也发车。

按分辨率分组：1024 / 768 / 1536 三组队列，各自凑 batch。跨分辨率 batch 不可行（shape 不齐）。

按 LoRA 分组：同 LoRA 才能 batch（不同 LoRA fuse 后 weight 不同）。Punica-style 技术（待 AIGC 成熟）能跨 LoRA batch。

### 3. 关键机制 / 流程 / 数据结构
第一，请求路由层。Request → router → 按 (分辨率, model, LoRA) 分到对应 instance / queue → 队列里凑 batch。

第二，batch 凑齐策略。三种：（a）严格 batch=N，凑齐才发车（最大吞吐）；（b）超时发车（max_wait_ms），不满也发（保 latency）；（c）混合：first request 触发计时，超时或满 batch 发车。

第三，OOM 防御。每实例启动跑 max_batch_size 的 forward 验证不 OOM；运行时显存监控，接近上限拒新请求；预留 buffer 防 fragmentation。

第四，跟 vLLM 对比。vLLM 滚动 batch（每 token step 动态加入退出）；AIGC 静态 batch（每 N step 内固定 batch）。AIGC 调度更接近传统 GPU batch inference。

### 4. 工程权衡 / 性能影响
batch=1 实例 vs batch=4 实例。batch=1: latency 1.5s, throughput 0.67 imgs/s/gpu。batch=4: latency 4-5s (按用户视角), throughput 0.8 imgs/s/gpu。吞吐高 20%，latency 高 3x。

batch 凑齐超时影响。max_wait=100ms 时大部分请求只等 50-100ms 凑 batch；max_wait=500ms 时几乎都凑齐 batch；trade-off 是 latency tail。

分辨率分组的成本。少分辨率档（如只 1024）实例利用率高；多分辨率档（如 512/768/1024/1536）每档实例都需保持温热，整体利用率下降。

LoRA 调度复杂度。100 个 LoRA × 10 个分辨率 = 1000 个分组，不可能每组一个实例。实际做法：sticky routing + LoRA cache + 长尾 LoRA on-demand 加载。

### 5. 常见追问 / 易错点
第一，没有 streaming 输出。LLM 可以 stream token，diffusion 必须 N step 跑完才有结果。Progressive preview（每 K step 输出预览图）是变通方案。

第二，cold start 慢。首次跑某 batch_size 触发 autotune ~10s 慢。生产 warmup 所有 batch_size。

第三，CUDA Graph 在 AIGC。Unet/DiT 跑同 batch shape 时可 capture CUDA Graph，省 ~5-10% 时间。但跨 batch size 要多个 graph。

第四，GPU 选型 fit。SDXL batch 4 适合 A100 40G；Flux dev 适合 H100 80G（带量化）；SDXL Turbo 实时适合 4090 24G 但 batch 小。

### 6. 实践建议
小规模业务（<10 req/s）：单 instance batch=1 + LCM/Turbo，简单可靠。

中规模（10-100 req/s）：3-10 个 instance，按分辨率分组，batch=2-4 + DPM-Solver。

大规模（>100 req/s）：fleet >50 instances，分辨率 + LoRA 分组 + 全局 router；hot LoRA shard + cold LoRA on-demand。

监控：TTFI p50/p99 / throughput / GPU util / 显存 / LoRA cache hit rate / batch fill rate / OOM count。

### 7. 30 秒速答
- AIGC batch 静态（每 step 内固定），不像 LLM continuous batching
- 调度按分辨率 + model + LoRA 分组凑 batch
- 显存预算静态：max_batch × per-sample mem
- 大规模 fleet 用 hot/cold LoRA + sticky routing 减切换

### 8. 自测 checklist
- [ ] 你能不能讲清 AIGC 调度跟 vLLM continuous batching 的本质差异？
- [ ] 你能不能计算 SDXL 1024 在 H100 80G 上最大 batch_size？
- [ ] 你能不能设计多分辨率 fleet 的路由策略？
- [ ] 你能不能识别 OOM 防御的关键点？

## Q17. Sora 风格视频生成 infra 总览

> 🟡 进阶 · 视频生成 = 图像生成 × 帧数 × temporal attention，单次生成 10 秒 1080p 视频要算上千张图——这是 AIGC 推理的下一个量级挑战。

### 1. 核心结论
Sora 风格视频生成基于 Video DiT（spatiotemporal Transformer），输入文本（+ 可选首帧 / 参考视频）输出 latent video，再经 VAE 解码成像素视频。核心架构：（1）3D VAE 把视频压缩到 latent video（如 spatial 8× + temporal 4× 压缩）；（2）Video DiT 在 latent 上做扩散；（3）spatial attention + temporal attention 处理空间和时间维度。当前主流模型：OpenAI Sora（闭源，估计 10B+）、Runway Gen-3、Pika、Luma Dream Machine、字节 PixelDance、智谱 CogVideoX、HunyuanVideo（开源）。生成 5-10 秒 720p 视频典型耗时 30-180s on H100。

### 2. 底层原理
3D VAE：扩展 2D VAE 加 temporal 维。Spatial 8× 压缩 + temporal 4× 压缩 → 10s@24fps 1080p (240 frames) → latent 30 frames × (latent_h × latent_w × latent_c)。

Video DiT 架构：把 latent video 切 3D patch（如 1×2×2，t×h×w），flatten 成 sequence。Spatial attention 跨同帧 patch；temporal attention 跨同位置不同帧。两种 attention 交替（spatiotemporal block）或合一（full 3D attention）。

参数规模：CogVideoX-5B / HunyuanVideo-13B 是开源 SOTA；Sora 估计 10-30B。比图像 DiT 更大、序列更长。

Diffusion loop：跟图像类似，N 步去噪 latent video。Step 数 30-50 常见。

### 3. 关键机制 / 流程 / 数据结构
第一，输入条件。Text prompt → CLIP / T5 / 自家 encoder → embedding。可选首帧图（image-to-video）→ VAE encode → 作为 condition。

第二，attention 复杂度。Full 3D attention 复杂度 O((T×H×W)²)，10s 720p 视频 latent 序列长几十 K token，attention 爆炸。实际用 spatiotemporal 分离 attention 降到 O((H×W)² + T²) 量级。

第三，输出后处理。Latent video → 3D VAE decode → frame sequence → encode 成 mp4 / webm。VAE decode 视频比图像耗时几十倍（多帧 × 3D 卷积）。

第四，主流模型对比。CogVideoX-5B 开源，720p 6s ~60s on H100；HunyuanVideo-13B 开源，720p 5s ~90s；Runway Gen-3 闭源 API，质量第一梯队；Sora 待发布。

### 4. 工程权衡 / 性能影响
显存挑战。Video DiT 12B + 视频 latent activation。生成 720p 5s 需要 H100 80G 单卡 FP8 量化才行；BF16 要 H200 或 A100 80G。

时间挑战。30s 720p 视频生成单 H100 几分钟，跟实时 24fps 播放不在一个量级。Real-time 视频生成是 long-term goal。

吞吐挑战。单卡每分钟出几个视频 → fleet 规模指数级大。AIGC 视频成本 $1-10/视频是常态。

跟图像对比。图像 ~1s/img ~$0.001/img；视频 ~60s/5s 视频 ~$0.5-2/视频。三个量级差异。

### 5. 常见追问 / 易错点
第一，时序一致性。Temporal attention 让前后帧自然过渡。但 motion 复杂 / 长视频时一致性仍是难题（人物形变 / 背景漂移）。

第二，I2V（image-to-video）vs T2V。I2V 用首帧约束更稳定；T2V 完全靠 prompt 容易飘。生产里 I2V 更常用。

第三，长视频生成。当前模型最长 10-20s；长视频用 auto-regressive 拼接（每 5s 一段，后段以前段最后帧为条件）。一致性损失。

第四，4K / 高分辨率。当前主流 720p / 1080p；4K 视频生成成本爆炸，几乎没业务化。

### 6. 实践建议
新业务起步：HunyuanVideo / CogVideoX 开源模型试水，I2V 模式起步。

显存预算：H100 80G + FP8 量化是 minimum。Multi-GPU TP 拆分 Video DiT 是 active research。

时长 / 分辨率选择：5s 720p 是 sweet spot。延长或提分辨率成本指数涨。

业务定位：当前视频生成更适合短视频素材生成、广告动画、特效辅助，而非纯实时交互。

### 7. 30 秒速答
- Video DiT = 图像 DiT + temporal attention，处理 latent video
- 3D VAE 时空压缩（如 8× spatial + 4× temporal）
- 显存 / 时间挑战：5-10s 720p 在 H100 上 30-180s
- 主流模型：CogVideoX / HunyuanVideo 开源，Runway / Sora 闭源

### 8. 自测 checklist
- [ ] 你能不能讲清 spatiotemporal attention 跟 full 3D attention 的差异？
- [ ] 你能不能算 720p 10s 视频 latent 序列长度？
- [ ] 你能不能解释 I2V 跟 T2V 在工程上的差异？
- [ ] 你能不能识别长视频一致性问题的来源？

## Q18. 视频生成的 latent video 表示与压缩

> 🔴 专家 · 视频生成不在像素空间做，而是在 spatial 8× × temporal 4× 压缩的 latent 空间——这个 3D VAE 是视频生成 infra 的最关键组件之一。

### 1. 核心结论
Latent video 表示是视频生成 infra 的基石：用 3D VAE 把视频压缩到 latent 空间（spatial 8× + temporal 4× 是主流配置），让 Video DiT 在压缩后的 latent 上做扩散。一个 1080p 24fps 10s 视频原始 ~5GB（uncompressed），latent 表示 ~30MB（压缩比 ~150×）。3D VAE 的设计直接影响视频质量、生成时长上限、推理效率。

### 2. 底层原理
3D VAE 架构：encoder 是 3D 卷积下采样（spatial + temporal 同时压）；decoder 是 3D 反卷积上采样。Spatial 8× 通常 3 层下采样（每层 2×）；temporal 4× 通常 2 层（每层 2×）。

Latent shape：input video [B, C=3, T=240, H=1080, W=1920] → latent [B, C'=16, T=60, H=135, W=240]。压缩比 (3×240×1080×1920) / (16×60×135×240) = 大约 150×。

跟 2D VAE 区别：（1）参数量 ~2-3x（3D 卷积参数多）；（2）训练数据需要视频（成本高）；（3）time consistency 由 temporal 卷积保证（前后帧 latent 平滑变化）。

主流配置：HunyuanVideo 用 spatial 8× + temporal 4× + 16 latent channels；CogVideoX 类似；Sora 估计 spatial 16× + temporal 4×（更激进压缩）。

### 3. 关键机制 / 流程 / 数据结构
第一，编码流程。Input video [B,3,T,H,W] → 3D conv 下采样 → latent [B,16,T/4,H/8,W/8]。编码 10s 1080p video 在 H100 ~3-5s（VAE 跑全卷积）。

第二，解码流程。Latent → 3D 反卷积上采样 → output frames。解码 latent video 比 encode 慢（上采样卷积更重），10s 1080p 解码 ~10-20s on H100。

第三，分块解码（Tiled VAE for video）。视频 VAE decode 显存峰值巨大（5GB+），生产里几乎一定要 tiled decode：把 spatial 和 temporal 都切 tile + overlap blend。Spatial tile 512×512、temporal tile 8 frame 是常见。

第四，跟 image VAE 关系。某些模型（如 CogVideoX 早期版本）用 image VAE 逐帧解码 + temporal smoothing 后处理，质量略差但工程简单。SOTA 模型都用专门 3D VAE。

### 4. 工程权衡 / 性能影响
存储 / 传输。Latent video 比像素 video 小 ~150×。生产 fleet 内部传 latent，最后一步才 decode 出 pixel video。

VAE 推理时间分布。10s 720p 视频端到端：text encode 0.3s；diffusion 60s；VAE decode 15s。VAE decode 占 ~20%，是优化重点之一。

显存。3D VAE 比 2D VAE 重得多。HunyuanVideo VAE BF16 ~1.5GB；activation 峰值 5-10GB（取决于分辨率 / 帧数）。

跟 image VAE 复用。某些 video model 把 image VAE + temporal adaptor 拼接，部分复用 image VAE 训练成果。质量妥协换工程简单。

### 5. 常见追问 / 易错点
第一，temporal upsampling 边界。VAE decode 时 temporal 维边界（首帧 / 末帧）的反卷积容易出 artifact。常见做法：edge padding 或 reflection padding。

第二，分块拼接 artifact。Spatial tiled decode 边界 blend，video 上的 blend 比图像更敏感（人眼对边界跳变敏感）。Overlap 区域要大（256 pixel 量级）。

第三，跟 codec 关系。生产输出是 mp4 / webm，VAE decode 出 raw frame → ffmpeg encode mp4。codec encode 也耗时（几秒）。

第四，量化困难。3D VAE 量化质量损失明显（video 主观质量敏感），生产几乎不量化 VAE。

### 6. 实践建议
新业务起步：用开源模型自带 VAE（HunyuanVideo / CogVideoX），不要自训。

分辨率 / 时长选择：720p / 5-10s 是 VAE 处理 sweet spot。1080p+ / 20s+ VAE 成为瓶颈。

显存紧张：必须 tiled decode + 显存监控。VAE OOM 是 video 生成 fleet 最常见 incident。

成本控制：latent caching 复用（同一 prompt + seed 的 latent 中间结果），跨用户共享。

### 7. 30 秒速答
- 3D VAE 是 video 生成压缩基石：spatial 8× × temporal 4× 是主流
- Latent video ~150× 小于 pixel video，存储传输都靠它
- VAE decode 占视频生成 ~20% 时间，必须 tiled decode 防 OOM
- 量化 3D VAE 损失明显，生产几乎不量化

### 8. 自测 checklist
- [ ] 你能不能算 1080p 10s 视频 latent 大小？
- [ ] 你能不能讲清 spatial + temporal tiled decode 的实现？
- [ ] 你能不能识别 VAE temporal 边界 artifact？
- [ ] 你能不能说出 video VAE 跟 image VAE 的关键区别？

## Q19. Video DiT 的 spatial-temporal attention 优化

> 🔴 专家 · 视频生成里 attention 是大头——序列长度 10K+，跑 full attention 就是显存爆炸。spatial-temporal 分离 + 滑窗 + sparsity 是必备工程。

### 1. 核心结论
Video DiT 的 attention 是工程最大挑战：sequence 长度 = T × H × W / patch_size³，10s 720p 视频 latent (60×90×160) patch=2 时 ~10K-50K token。Full attention O(N²) 在 10K 序列就需要 100M+ activation 内存。三大优化方向：（1）spatial-temporal 分离 attention（O(N²/T) + O(N²/HW)）；（2）滑窗 attention（local temporal window）；（3）attention sparsity（PAB / 跨步 cache）。SOTA 模型 HunyuanVideo / CogVideoX 都用前两种组合。

### 2. 底层原理
Full 3D attention：所有 patch 互 attend，复杂度 O((T·H·W)²)。10K seq 一层 attention 矩阵 100M floats = 200MB BF16，多层多 head 直接爆显存。

Spatial-temporal 分离：每层先做 spatial attention（每帧内 H·W 个 token 互 attend，复杂度 O(T·(HW)²)），再做 temporal attention（每位置 T 个时间 token 互 attend，复杂度 O(HW·T²)）。总复杂度从 O((THW)²) 降到 O(THW·(HW+T))，大幅降低。

滑窗 attention：每帧 token 只 attend 邻近 K 帧的对应位置（如 K=8）。复杂度从 T² 降到 T·K。质量略损但显存大降。

PAB（Pyramid Attention Broadcast）：跨 diffusion step 缓存某些 attention 输出复用。类似 image DiT 的 DeepCache。

### 3. 关键机制 / 流程 / 数据结构
第一，spatial attention layer。Reshape latent [B, T, H, W, C] → [B·T, H·W, C]，跑 standard self-attention，复杂度 O((HW)²)。

第二，temporal attention layer。Reshape → [B·H·W, T, C]，跑 self-attention on T 维，复杂度 O(T²)。

第三，cross-attention with text。每 spatial-temporal block 后加 cross-attention with text embedding。跟图像 DiT 类似。

第四，FlashAttention 加速。所有 attention 都用 FlashAttention-2/3，是必备优化。Hopper 上 FA-3 让 attention 比朴素 implementation 快 2-3x。

### 4. 工程权衡 / 性能影响
Spatial vs temporal 分离收益。10s 720p (T=60, HW=21600) full attention complexity ~1.7B；分离后 spatial 21600² × 60 + temporal 60² × 21600 ≈ 28B + 78M = 大幅降低。

滑窗收益。K=8 vs T=60 时 temporal 复杂度降 7.5×。质量损失 1-3%（CLIP-Score）。

PAB 收益。Cache interval=3 时 attention 计算 ~1/3，端到端 1.5-2x 加速。

跟 image DiT 对比。video DiT 一层比 image DiT 单层重 5-10×。

### 5. 常见追问 / 易错点
第一，pos embedding。视频 DiT 用 3D RoPE（spatial 2D + temporal 1D），或单独 learned。RoPE 在 video 上推广是 active research。

第二，长视频外推。模型训练在 10s 视频上，inference 时跑 30s 一般跑不出（位置编码外推差）。生产用 auto-regressive 拼接（每 5s 一段）。

第三，跟 image attention 区别。Image attention 一般 spatial only，pos embedding 2D；video 加 temporal 维需要 3D pos。

第四，flash attention version。FA-2 支持 batch-level mask；FA-3 在 Hopper+ 用 TMA + WGMMA 更快。Video DiT 默认 FA-2 或 FA-3。

### 6. 实践建议
模型选型：用开源 HunyuanVideo / CogVideoX 自带 attention，不自己重写。

优化优先级：FA-3（Hopper）→ spatial-temporal 分离（model 设计）→ PAB（推理 cache）→ 滑窗（质量妥协）。

displacement profile：单层 attention 跑 nsys profile，看 attention vs FFN 时间分布。Video DiT 通常 attention 50-70%。

显存预算：activation 计算 [B, T·H·W, hidden] × layers × 2.5x（forward + backward states）。Inference 只有 forward 但仍要预留 buffer。

### 7. 30 秒速答
- Video DiT 序列长 10K-50K，full attention O(N²) 爆炸
- Spatial-temporal 分离：复杂度从 (THW)² 降到 THW·(HW+T)
- 滑窗：temporal local K 帧（K=8）减 7.5× 复杂度
- PAB / DeepCache：跨 step cache attention 输出

### 8. 自测 checklist
- [ ] 你能不能算 720p 10s 视频 DiT 的 sequence 长度？
- [ ] 你能不能讲清 spatial-temporal 分离的复杂度收益？
- [ ] 你能不能解释 PAB 跟 image DeepCache 的关系？
- [ ] 你能不能识别长视频外推失败的原因？

## Q20. 视频生成 fleet 的存储 / 传输 / CDN 挑战

> 🔴 专家 · 图像生成出 1MB jpg，视频生成出 50MB mp4，全球 fleet 每天产几百万视频——存储 / 传输 / CDN 不是图像 fleet 的简单 scale up，是新的工程问题。

### 1. 核心结论
视频生成 fleet 的存储 / 传输挑战远超图像：（1）单视频体积 50-500MB（vs 图像 1-5MB），存储成本爆炸；（2）生成耗时长（30-180s），中间状态需要持久化（防 instance crash）；（3）用户下载或 CDN 分发对带宽要求高；（4）多分辨率 / 多码率 transcoding 增加 fleet 复杂度。生产架构：分布式存储（S3 / 自研 blob store）+ CDN（CloudFront / 阿里云 CDN）+ 异步生成 + 用户通知。

### 2. 底层原理
存储分层：hot（最近生成、用户活跃）→ S3 standard / Redis；warm（生成完待下载）→ S3 IA；cold（已下载 / 过期）→ S3 Glacier。各层成本差 10-100×。

传输瓶颈：videos 大文件下载 → CDN 是必备。Origin 直传带宽不够、延迟高。

转码 pipeline：Raw video（VAE decode 出的）→ ffmpeg → H.264/H.265 mp4 多 bitrate → S3 → CDN。转码本身耗时几秒到几十秒。

异步任务模型：用户 submit 请求 → 立即返回 task_id → 后台 fleet 生成 → 完成后通知用户（webhook / poll）。跟图像同步 RPC 不同。

### 3. 关键机制 / 流程 / 数据结构
第一，请求 / 响应模型。POST /generate-video {prompt, options} → 202 Accepted + task_id。GET /tasks/{task_id} → status (queued / running / done / failed) + result_url（if done）。

第二，中间状态持久化。Generation 中途 instance 挂掉 → 通过 latent video checkpoint（每 K diffusion step 保存 latent）恢复。Checkpoint 大小几十 MB。

第三，转码 fleet。生成 fleet 跟转码 fleet 分离。Generation GPU heavy；transcoding CPU/GPU heavy。各自扩展。

第四，CDN 集成。Generated video → upload S3 → trigger CDN invalidate / preload → 用户 URL 直接走 CDN。CDN 边缘缓存让全球用户低延迟下载。

### 4. 工程权衡 / 性能影响
存储成本。10 秒 720p mp4 ~5MB；1000 万视频/月 = 50TB/月。S3 standard $0.023/GB/月 → $1150/月 storage。Cold tier 降到 $0.004/GB → $200/月。Lifecycle policy 自动迁移。

带宽成本。S3 出向 $0.09/GB；CDN $0.05/GB（按 region）。1000 万次下载 5MB 视频 = 50TB → $2500-4500。CDN 比直接 S3 便宜一半且更快。

生成成本。单视频 $0.5-2（fleet GPU 时间 + 存储 + 带宽）。商业模式按视频计费 $0.5-5。

转码 fleet 规模。GPU 转码（NVENC）比 CPU 快 5-10×。生产 fleet 通常 1 transcoder：5 generators 比例。

### 5. 常见追问 / 易错点
第一，存储泄漏。失败 / 过期视频不清理 → 存储无限增长。Lifecycle policy + 监控 storage usage 必备。

第二，CDN 预热。新生成 video 第一次访问触发 CDN pull from origin，慢。可主动 preload / push。

第三，多 bitrate adaptive streaming。专业场景输出 HLS / DASH 多码率 manifest，让用户网速差自动降码率。简单场景输出单 mp4。

第四，水印 / DRM。AIGC 生成视频常加 watermark（防止盗用 + 标识 AIGC）；高端业务加 DRM。需要在 transcoding 阶段处理。

### 6. 实践建议
新业务起步：S3 + CloudFront（或国内云对应）。简单 mp4 输出。Lifecycle 30 天删除未下载视频。

中规模：分离 generation / transcoding fleet。Generated raw → 转码队列 → 多 bitrate mp4 → CDN。

成本优化：cold tier 自动迁移；CDN 配置长 TTL；用户配额限制生成数量。

监控：generation 队列长度 / transcoding 队列长度 / CDN hit rate / storage 增长 / 带宽 cost。

### 7. 30 秒速答
- 视频 fleet 存储 / 传输是新挑战（单视频 50-500MB）
- 异步生成 + task_id polling，跟图像同步 RPC 不同
- CDN 必备（成本 + 延迟）
- 分离 generation fleet / transcoding fleet，比例 1:5 起

### 8. 自测 checklist
- [ ] 你能不能算 1000 万视频/月 fleet 的存储 + 带宽成本？
- [ ] 你能不能讲清 generation 中途 crash 的恢复机制？
- [ ] 你能不能解释为什么转码 fleet 跟 generation fleet 要分离？
- [ ] 你能不能识别 CDN 预热的最佳时机？

## Q21. AIGC 业务 SLO 设计（生成质量 + 时延双维度）

> 🟡 进阶 · LLM 服务 SLO 看 TTFT/TPOT，AIGC 服务多一维：生成质量。怎么在 SLO 里同时刻画"快"和"好"，是 AIGC product engineering 的特殊问题。

### 1. 核心结论
AIGC 业务 SLO 比 LLM 多一维：除了 latency（TTFI、total time），还要刻画生成质量（CLIP-Score、用户满意度、客诉率）。生产 SLO 典型：TTFI p99 < 5s（图像）/ 60s（视频）；生成成功率 > 99%；用户满意度 > 80%（点赞 / 接受率）；客诉率 < 0.5%。质量维度难以自动监控（图像质量主观），常用 CLIP-Score + user feedback 双轨。

### 2. 底层原理
SLO 维度。Latency：TTFI（time-to-first-image，对应 LLM TTFT）、total time（图像 / 视频生成完成时间）。Quality：CLIP-Score（生成图跟 prompt 匹配度，自动）、aesthetic score（主观美感，模型评）、user rating（满意度）。Availability：成功率、错误率、NSFW 拒绝率。Throughput：fleet imgs/s 或 videos/min。

业务 SLO vs SLA。SLO 是内部目标（如 99% 请求 <5s），SLA 是对外承诺（如 95% 请求 <10s 否则退款）。AIGC SaaS 业务通常 SLO 比 SLA 紧 2-3x。

### 3. 关键机制 / 流程 / 数据结构
第一，latency 监控。Per-request 记录：submit_time、queue_start、queue_end、generation_start、generation_end、upload_done、user_get。各段差是各 stage latency。

第二，CLIP-Score 自动评估。Generated image + 原 prompt → CLIP encode 双方 → cosine similarity。> 0.3 一般认为质量合格。生产里实时计算 CLIP-Score 作为质量监控。

第三，用户反馈采集。生成图 + thumbs up/down button，或 1-5 star rating。这些是真实质量信号但有偏（用户只给好评 / 差评，中间评价少）。

第四，NSFW / 安全过滤。生成图过 NSFW classifier → 不合规直接 reject。Reject 率是 SLO 一部分（过高说明 prompt 引导不当 / 模型问题）。

### 4. 工程权衡 / 性能影响
Latency SLO 紧 vs 松。p99 < 3s（紧）：成本高，fleet 必须超配 + 用 LCM/Turbo。p99 < 10s（松）：成本低，标准 DPM-Solver 25 步够。

Quality SLO 选择。CLIP-Score > 0.3 是"prompt 匹配"基线；> 0.35 是高质量。Aesthetic score 看模型，>5/10 算合格。

成本 vs SLO 权衡。质量提高（30 步 vs 20 步）成本 +50%；速度提高（LCM 4 步 vs 25 步）成本 -50% 但质量略降。业务定位决定哪个优先。

### 5. 常见追问 / 易错点
第一，质量监控的"sampling"。不能每张图都人工审，要抽样（如 1% 流量 + 用户主动反馈）。Sampling 偏差要 careful。

第二，A/B 测试质量。新版本上线前 A/B 测试，跟踪 CLIP-Score / 用户满意度 / 完成率三轴。

第三，cold start 在 AIGC。第一次跑新 model / 新 batch_size 触发 autotune ~10s 慢。SLO 计算时排除 cold start request 或专门处理。

第四，failed request 计入 SLO 吗。失败请求（OOM / timeout / NSFW reject）不计入 latency SLO（无意义）但计入 availability SLO。

### 6. 实践建议
新业务起步：定 TTFI p99 < 10s + 成功率 > 99% + CLIP-Score > 0.3 三轴 SLO。

监控 dashboard：实时显示三轴 SLO 完成率、违规请求列表、各 stage latency 分布。

灰度 / A/B 测试：新 model / 新优化上线，5% 流量灰度，对比 SLO 不退化才全量。

incident 流程：SLO 违规触发告警 → on-call 排查 → 决定回滚 or hotfix。

### 7. 30 秒速答
- AIGC SLO 双维度：latency（TTFI / total time）+ quality（CLIP-Score / 用户反馈）
- Latency 跟 LLM 类似，quality 是 AIGC 特殊（图像主观）
- 监控方式：实时 CLIP-Score + user rating + NSFW reject rate
- 成本 vs SLO 权衡决定模型 / step / batch 配置

### 8. 自测 checklist
- [ ] 你能不能列出 AIGC 业务的 4 个核心 SLO？
- [ ] 你能不能讲清 CLIP-Score 怎么实时算？
- [ ] 你能不能解释 cold start 对 SLO 监控的影响？
- [ ] 你能不能设计一个 AIGC A/B 测试的成功标准？

## Q22. 图像生成 vs LLM 文本推理工程差异

> 🧭 综合 · 同样是"模型推理 serving"，AIGC 跟 LLM 工程模型根本不同——理解差异是 cross-domain 工程师必修。

### 1. 核心结论
图像生成 vs LLM 推理在工程上有 7 大根本差异：（1）执行模式：multi-step iterative vs autoregressive；（2）状态管理：无 KV cache vs 有 KV cache 主导显存；（3）batch 模式：静态 batch vs continuous batching；（4）输出模式：N 步全跑完才出图 vs streaming token；（5）瓶颈类型：compute-bound vs memory-bound（decode 阶段）；（6）量化敏感：图像主观质量 vs 文本 perplexity；（7）容错：单图失败可重试 vs LLM 失败影响 session。理解差异让跨方向工程师快速 ramp up。

### 2. 底层原理
执行模式。Diffusion：N 步 iterative denoise，每步全模型 forward，N 步独立。LLM：autoregressive token-by-token，每步生成 1 token、读全部 KV，依赖前序状态。

状态管理。Diffusion：每步 latent 是当前状态，update 后丢前一步（不存）。LLM：KV cache 持续累积，token N 时已存 N-1 个 KV。

显存主导。Diffusion：weight + activation（按 batch×分辨率），no KV cache。LLM：weight + KV cache（按 batch×seq_len），activation 短暂。

batch 模式。Diffusion：static，每 step 内固定 batch，shape 必须齐。LLM continuous：每 step 新 sample 可加入、完成可退出。

### 3. 关键机制 / 流程 / 数据结构
第一，输出形态。Diffusion：N 步串行，最后才出完整图。可 progressive preview（每 K 步出预览）。LLM：streaming token，每 token 立刻输出。

第二，瓶颈类型。Diffusion compute-bound：Unet/DiT 算力大，batch 增加单步时间线性涨。LLM decode memory-bound：从 HBM 读 KV 主导，batch 增加单 token 时间几乎不变（"摊薄"）。

第三，量化敏感。Diffusion：质量主观（CLIP-Score、人评），FP8 量化几乎无感、FP4 略损。LLM：perplexity 客观可测，FP8 几乎无感、INT4 略损。

第四，错误处理。Diffusion：失败重试简单（重跑 N 步），无 session 状态。LLM：失败可能丢 session，多轮对话恢复复杂。

### 4. 工程权衡 / 性能影响
serving 引擎设计差异。LLM：vLLM/SGLang/TRT-LLM 围绕 KV cache 管理 + continuous batching 优化。AIGC：自定义 batch 调度 + LoRA 切换 + 多 model fleet 编排。

显存预算公式。LLM：predictable，weight + KV cache (按上限 max_seq_len × max_batch)。AIGC：static 但峰值高，weight + activation (按 batch × resolution × intermediate states)。

监控指标。LLM：TTFT / TPOT / token throughput。AIGC：TTFI / total time / imgs per sec / image quality。

### 5. 常见追问 / 易错点
第一，能否复用 vLLM 跑 AIGC。不能直接复用：vLLM 设计围绕 PagedAttention + KV cache + continuous batching，diffusion 没这套。diffusers 库 + 自研 serving 是 AIGC 主流。

第二，LLM serving 工程师转 AIGC 学什么。学 diffusion 数学（去噪流程）、多 step iterative 调度、LoRA / ControlNet 等推理时插件机制、3D VAE（视频）。约 2-4 周从基础到能上手。

第三，AIGC 工程师转 LLM 学什么。学 KV cache 形态、continuous batching、attention kernel 优化、量化算法（GPTQ/AWQ）。约 2-4 周。

第四，融合方向。多模态模型（LLaVA / Gemini）同时处理 text + image，serving 上是 LLM-style autoregressive 但 input 含 image patch。介于两者之间。

### 6. 实践建议
跨方向团队：互相 shadow 1-2 周；理解对方常用工具栈（vLLM vs diffusers）；定期 cross-pollination meeting。

新工程师 ramp up：先专注一个方向 3 个月深入，再扩展。同时学两个容易浅。

技术选型：LLM 服务用 vLLM/SGLang；AIGC 用 diffusers + 自研 / ComfyUI / 头部 SaaS API。混合（VLM）选 vLLM 类（带 image input support）。

业务对接：跟产品 / 算法明确 LLM 还是 AIGC，技术 stack 完全不同。

### 7. 30 秒速答
- 7 大差异：执行模式 / 状态管理 / batch / 输出 / 瓶颈类型 / 量化敏感 / 容错
- LLM continuous batching + KV cache 主导 vs AIGC static batch + 无 KV
- Diffusion compute-bound, LLM decode memory-bound
- vLLM 不能直接跑 AIGC，diffusers + 自研 serving 是主流

### 8. 自测 checklist
- [ ] 你能不能列出 7 大工程差异？
- [ ] 你能不能解释为什么 vLLM 难以直接复用到 AIGC？
- [ ] 你能不能讲清 LLM 工程师转 AIGC 的学习路径？
- [ ] 你能不能识别多模态 (VLM) 的中间位置？

## Q23. 多模态联合推理架构（图 + 文 + 音）

> 🔴 专家 · 单模态模型时代结束了——GPT-4V / Gemini / Claude 3.5 都是多模态原生，serving 怎么同时跑文字、图像、音频是新工程问题。

### 1. 核心结论
多模态联合推理（Vision-Language Models, VLM / Audio-Language Models, ALM）的工程模式：（1）独立 encoder（vision tower / audio tower）把非文本输入编码成 embedding；（2）projector / adapter 把跨模态 embedding 映射到 LLM 输入空间；（3）LLM backbone 把统一 sequence 当 text 处理。Serving 架构：vision encoder + LLM 在同一服务里跑，或分离 service（vision encode service + LLM service）。挑战：vision encode 是 compute-bound 异于 LLM decode、跨模态 input shape 动态、量化兼容。主流模型 LLaVA / GPT-4V / Gemini / Qwen-VL / Claude 3.5 Vision。

### 2. 底层原理
VLM 架构。Input image → vision tower (CLIP-L / SigLIP / InternViT) → image embedding [N_patch, d]。Projector (linear / MLP) 把 d 映射到 LLM embed_dim → image token sequence。Image tokens + text tokens 拼接 → LLM forward（standard autoregressive）。

输出。LLM 输出 text token（描述图、回答问题等），跟 text-only LLM 同。

不同 VLM 架构差异。LLaVA 系列：CLIP + MLP projector + LLaMA/Vicuna，简单直接。BLIP-2：Q-Former 作 cross-modal bridge，参数效率高。Qwen-VL：自家 vision tower。Gemini：原生多模态训练（图像 + 文本 + 音频联合训）。

ALM（audio）：跟 vision 类似，audio tower（Whisper encoder）→ projector → LLM。代表 Qwen-Audio / GPT-4o audio mode。

### 3. 关键机制 / 流程 / 数据结构
第一，image preprocessing。Image → resize (224x224 / 336x336 / 448x448) → patches → vision tower forward → embedding。Preprocess CPU 20-50ms。

第二，dynamic image tokens。高分辨率图像可能产生几百到几千 image tokens（patch size 决定）。LLM input 长度变动大，batching 复杂。

第三，serving 架构选择。（a）单实例：vision + LLM 同 instance，简单但 vision tower 闲置（每请求只跑一次）。（b）分离：vision service + LLM service via RPC，资源效率高但延迟 +20-50ms。

第四，跟 vLLM 集成。vLLM 支持 LLaVA / Qwen-VL 等主流 VLM，input 含 image 字段，内部自动 encode。Multimodal vLLM 是 2024 后 active 方向。

### 4. 工程权衡 / 性能影响
显存。Vision tower 几百 MB-几 GB（CLIP-L ~600MB）。LLM 主导显存（70B 140GB BF16）。组合峰值需考虑。

延迟。Vision encode 50-200ms（每图）。LLM prefill 含 image tokens 比纯文本长 → prefill 时间 +。Total: 单图 + 短 text + 100 token output 总延迟比纯 text 长 30-50%。

吞吐。Vision encode batch 内必须 image size 齐（除非 dynamic shape）。LLM 部分 continuous batching 正常。

量化。Vision tower 量化（FP8/INT8）成熟，CLIP 量化是常规操作。LLM 量化（GPTQ/AWQ）正常。两者协同。

### 5. 常见追问 / 易错点
第一，high-res image。原生 CLIP 224x224。高分辨率（1024+）有几种方案：（a）AnyRes（LLaVA-NeXT）切图分块过 CLIP；（b）自家 vision encoder 支持原生高分辨率。

第二，多图 / 视频 input。多图：每图独立 encode → 拼接 tokens。视频：每帧 encode + temporal pooling。视频 input token 数巨大，需 sparse sampling。

第三，audio 处理。Whisper encoder 30s 音频 → 1500 audio tokens。比 image 更多 tokens，LLM context 占用大。

第四，跨模态 hallucination。VLM 容易"看图说瞎话"（描述图里不存在的东西）。模型质量问题，serving 端难解决。

### 6. 实践建议
新业务起步：vLLM + 主流开源 VLM（LLaVA / Qwen-VL）跑通。

显存预算：CLIP-L + 7B LLM = ~16GB BF16；CLIP-L + 70B = ~145GB。

分离 serving：vision encode 单独 service（轻量、可 CPU），LLM 大 fleet。中间用 cache 复用 vision embedding。

监控：分阶段 latency（image preprocess / vision encode / LLM prefill / LLM decode）。

### 7. 30 秒速答
- VLM 架构：vision tower + projector + LLM backbone
- Image preprocess + encode 50-200ms 每图，比纯 text 慢
- 分离 vision service / LLM service 是规模化常见方案
- vLLM 已支持主流 VLM，input 含 image 字段

### 8. 自测 checklist
- [ ] 你能不能讲清 LLaVA 跟 BLIP-2 在 cross-modal bridge 上的差异？
- [ ] 你能不能算 VLM 单图请求的端到端 latency 分布？
- [ ] 你能不能说出 high-res image 处理的两种方案？
- [ ] 你能不能识别 VLM hallucination 的工程边界？

## Q24. AIGC 内容审核与水印（NSFW 过滤 / digital watermark）

> 🟡 进阶 · AIGC 业务上线第一天就要回答两个问题：怎么不让用户生成违规内容、怎么让生成内容能被识别为 AI 产物——审核和水印是 AIGC fleet 的强合规组件。

### 1. 核心结论
AIGC 内容审核分两阶段：（1）prompt 阶段过滤（拒绝违规 prompt）；（2）输出阶段过滤（生成后用 NSFW classifier 拦截）。Watermark 是给生成内容嵌入不可见标识，方便事后追溯 / 区分 AI 内容。法规推动下（如 EU AI Act / 中国生成式 AI 服务管理办法），watermark 从 nice-to-have 变成 must-have。主流方案：Stable Diffusion safety checker（视觉 NSFW 分类）；Stegastamp / WatermarkAnything（不可见水印）；C2PA 标准（content provenance）。

### 2. 底层原理
Prompt 审核。Prompt → classifier（基于 BERT / 自家模型）→ 违规分数（NSFW / 暴力 / 政治敏感 / 涉未成年等）。高于阈值直接 reject。也用 LLM judge（Llama Guard 等）做更复杂判断。

输出审核。Generated image → NSFW classifier（如 NSFW Detector，CNN 几十 MB）→ NSFW score。SDXL 内置 SafetyChecker 是默认实现。视频生成审核类似但要看每帧。

Watermark 类型。Visible watermark：图像角落 logo / 文字，明显但用户体验差。Invisible watermark：DCT/DWT 频域嵌入 / 神经网络方法（Stegastamp / TreeRing），不可见但可解码。Cryptographic（C2PA）：标准化 metadata 签名，证明内容来源。

### 3. 关键机制 / 流程 / 数据结构
第一，prompt 审核 pipeline。Request → prompt classifier → 违规分数 → 若高于阈值返回 error code；低于则进入 generation。

第二，输出审核 pipeline。Generation 完成 → output → NSFW classifier → 若违规，标记 + 不返回给用户 / 替换默认图；合规则正常返回。

第三，watermark 嵌入流程。生成后 image → watermark encoder（如 Stegastamp 神经网络）→ watermarked image。嵌入耗时 ~50-100ms。

第四，watermark 检测。Suspect image → watermark decoder → 解码出 ID → 跟数据库比对，确认是否本平台生成。

### 4. 工程权衡 / 性能影响
NSFW 审核成本。Classifier inference 10-30ms。Generation 已 1-2s，审核 overhead 小（<5%）。

Watermark 嵌入成本。Stegastamp 50-100ms。比 NSFW 高。生产可接受。

False positive / negative 平衡。NSFW classifier 误判率（误杀正常内容）vs 漏检率（放过违规）。生产里阈值调到误杀 < 1% / 漏检 < 5%。

合规跟商业的张力。严审降用户体验（误杀）+ 增加成本（每张审核）；松审增加合规风险。平衡靠业务 / 法务 / 产品 collaboration。

### 5. 常见追问 / 易错点
第一，绕过 NSFW。用户用隐晦 prompt（如 "renaissance painting style"）规避审核。Classifier 必须持续更新对抗。

第二，watermark 鲁棒性。简单 watermark 被 jpeg compress / crop / 截图就失效。Stegastamp / TreeRing 等抗扰动方法更鲁棒但 embedding 时间长。

第三，跨平台识别。各家平台 watermark 各异，难统一识别。C2PA 是行业标准化努力但采用慢。

第四，未成年保护。涉未成年内容是 absolutely red line，必须 zero tolerance。专门 classifier + 法务流程。

### 6. 实践建议
新业务必备：prompt 审核（Llama Guard 或自家 classifier）+ 输出 NSFW 审核 + invisible watermark 三件套。

法规跟进：参考所在司法管辖（EU AI Act / 中国 / 美国 state laws）更新合规要求。

监控：审核拒绝率 / 误杀客诉率 / watermark 嵌入失败率。

应急流程：违规事件触发时能快速：识别影响范围 / 暂停相关功能 / 通知监管。

### 7. 30 秒速答
- 两阶段审核：prompt 阶段拒违规请求 + 输出阶段 NSFW 过滤
- Watermark 主流方案：Stegastamp（神经网络）/ TreeRing / C2PA（标准）
- 性能开销：NSFW 10-30ms / Watermark 50-100ms，可接受
- 合规驱动从 nice-to-have 变 must-have

### 8. 自测 checklist
- [ ] 你能不能讲清 prompt 审核跟输出审核各拦截什么？
- [ ] 你能不能解释 invisible watermark 的两种主流方案？
- [ ] 你能不能说出 NSFW classifier 误杀 / 漏检的 trade-off？
- [ ] 你能不能识别 watermark 被破坏的常见操作？

## Q25. AIGC 单图 / 单视频成本核算与 fleet 容量规划

> 🧭 综合 · LLM 算 tokens/$, AIGC 算 imgs/$ 和 videos/$, 单位经济学差好几个量级——产品定价和 fleet 规划都依赖这个核算。

### 1. 核心结论
AIGC 成本核算单位：图像 $/img、视频 $/video（不像 LLM 是 $/token）。典型数据：SDXL 1024 25 step 在 H100 ~$0.001-0.003/img；Flux dev FP8 ~$0.005-0.015/img；视频生成 5s 720p ~$0.5-2/video。Fleet 容量规划公式：GPU 数 = 峰值 imgs_per_sec × time_per_img / 0.7（70% utilization buffer）。商业定价：图像 $0.01-0.10/img（毛利 5-20x），视频 $2-10/video。新 AIGC 业务的单位经济学跟 LLM serving（$0.5-2 / 1M token）完全不同。

### 2. 底层原理
成本组成。GPU 时间（fleet 成本主体）+ 存储 + 带宽 + CDN + auxiliary services（NSFW / watermark）。GPU 占 70-90%。

GPU 成本计算。GPU 单价 / 小时（H100 $2-4/h on-demand, $1.5/h reserved）× time_per_img / 3600 = $/img。SDXL 1024 25 step H100 ~1.5s → $2/h × 1.5/3600 = $0.0008/img（不含 utilization buffer）。

Fleet 容量公式。Peak QPS × time_per_request / utilization_target = required GPU 数。100 imgs/s × 1.5s / 0.7 = 215 GPUs。

视频成本（H100 5s 720p video ~90s 生成）：$2/h × 90/3600 = $0.05/video on GPU time。加存储 + 带宽 + 转码 fleet 总 $0.5-2。

### 3. 关键机制 / 流程 / 数据结构
第一，按分辨率 / step 分档定价。512×512 cheap / 1024 standard / 2048+ premium。20 step base / 50 step high-quality。

第二，订阅 vs 按需。订阅（$10-30/月 100 imgs）锁定收入，成本可预测。按需（$0.05/img）灵活但需求波动。

第三，free tier 经济学。免费用户 acquisition 用，每用户 cost $0.1-0.5/月。需 paid conversion >5% 才回本。

第四，Spot / Reserved instance。RI 比 On-demand 便宜 40-60%，但需 1-3 年承诺。Spot 便宜 70% 但随时回收。生产里 60% RI + 30% Spot + 10% On-demand 是常见混合。

### 4. 工程权衡 / 性能影响
质量 vs 成本。25 step DPM-Solver $0.001/img；50 step DDIM $0.002/img。LCM 4 step $0.0002/img。质量级配价格 5-10x 差异。

吞吐 vs 成本。Batch=1 latency 优先 vs batch=8 吞吐优先。同 fleet 大小 batch=8 吞吐 5x，成本 / img 降 5x。

显卡选型。H100 $2/h vs A100 $1/h vs 4090 $0.5/h。性能差异：H100 5x A100 2x 4090。Cost-effectiveness 因业务波动而异。

跟 LLM 对比。LLM serving 单位 $0.5-2/1M token；AIGC 单图 $0.001-0.05；视频 $0.5-5。AIGC 单位经济学密度比 LLM 高 100-1000x。

### 5. 常见追问 / 易错点
第一，free tier 滥用。匿名 free 用户用 VPN 重置配额 → 真实 cost 远超预期。Anti-abuse 机制（rate limit + device fingerprint + 实名）必须。

第二，长尾 prompt 不均。大部分 prompt 是热门类型，少部分是 niche。Fleet 容量按峰值规划，长尾会让 utilization 不均。

第三，模型切换成本。SDXL → Flux 换 model，fleet 时间 cost 增 3-5x。商业模型选择影响成本结构。

第四，存储 / 带宽快速涨。每张图 3-5MB，每月千万图 = 30-50TB 存储 + 出向带宽。Cold tier + CDN cache 优化必须。

### 6. 实践建议
新业务起步：单 fleet H100 + SDXL FP8，跟踪 $/img 实际数。定价 cost × 5-10 倍。

成本控制：（1）热门 prompt cache（同 prompt + seed 复用结果）；（2）LCM/Turbo 路径给免费用户，标准给付费；（3）spot/RI 混合；（4）存储 lifecycle。

容量规划：peak QPS 估算（业务历史 / DAU × 频次）→ fleet 大小；预留 20-30% buffer 应对突发。

财务报表：按月统计 fleet cost / 生成图数 / $/img / 毛利率，跟产品 / 财务 collaborative。

### 7. 30 秒速答
- 成本单位：$/img（图像）/ $/video（视频），不是 $/token
- SDXL 1024 H100 ~$0.001-0.003/img；Flux ~$0.01；视频 $0.5-2
- Fleet 容量 = peak QPS × time / 0.7
- 商业定价 cost × 5-10x，毛利率 80-90%

### 8. 自测 checklist
- [ ] 你能不能算 1000 万 imgs/月 SDXL 服务的 fleet GPU 数？
- [ ] 你能不能解释 RI / Spot / On-demand 混合的最优比例？
- [ ] 你能不能讲清 AIGC vs LLM 单位经济学的量级差异？
- [ ] 你能不能识别 free tier 滥用的工程防御？
