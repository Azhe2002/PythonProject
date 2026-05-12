# AI 生成人脸检测 — 数据驱动分支权重学习方案

## 背景与动机

当前检测器 (`ai_face_detector/detector.py`) 使用**手工设定的固定权重**融合四个分支：

| 分支 | 手工权重 | 含义 |
|---|---|---|
| 物理噪声分支 | 0.46 | 传感器噪声模式、CFA 周期、噪声-亮度关系 |
| 频域残差分支 | 0.32 | FFT 高频比例、谱斜率、方向性、JPEG 块效应 |
| RGB 语义分支 | 0.10 | 清晰度、边缘密度、饱和度、纹理能量 |
| 人脸光滑度分支 | 0.12 | 局部平滑比例、纹理能量、边缘密度 |

公式：`real_score = 0.46*P + 0.32*F + 0.10*R + 0.12*S`

此外 `artifact_boost` 的五个修正项（残差标准差惩罚、通道噪声相关性修正等）也是手工阈值。

**核心问题**：这些权重来自经验估计，未经数据验证。在真实/虚假人脸数据集上学习最优权重，有望显著提升判别率和置信度。

---

## 两种方案概览

```
现有 pipeline                      方案一 (fusion_lr)              方案三 (deep_cnn)
────────────                       ──────────────                  ──────────────
                                   ┌─────────────────┐            ┌─────────────────┐
图像 ──► 21维特征 ──► 4分支 ──►    │ 逻辑回归学习     │            │ 端到端 CNN      │
              手工聚合    手工权重  │ 4分支 → AI概率   │            │ 像素 → AI概率   │
                                   └─────────────────┘            └─────────────────┘
```

---

## 方案一：逻辑回归学习四分支融合权重 (`fusion_lr/`)

### 技术路线

```
训练阶段：
  real_face/ ──► 提取4分支分数(物理/频域/RGB/光滑度) ──► 标签=0
  fake_face/ ──► 提取4分支分数(物理/频域/RGB/光滑度) ──► 标签=1
                            │
                    ┌───────▼───────┐
                    │  逻辑回归训练   │
                    │  学习: w1..w4  │
                    └───────┬───────┘
                            │
推理阶段：                     ▼
  新图像 ──► 4分支分数 ──►  sigmoid(w·x + b) ──► AI生成概率
```

### 关键设计决策

1. **输入特征**：每张训练图像调用现有 `detector._branch_scores()` 提取四个分支的 `real_evidence` 分数，构成 4 维向量 `[p_physical, p_freq, p_rgb, p_smoothness]`。

2. **模型选择**：逻辑回归（`sklearn.linear_model.LogisticRegression`），L2 正则化，`C` 参数通过交叉验证选择。

3. **训练/评估**：
   - 对每个数据集做 5 折交叉验证
   - 记录 AUC、Accuracy、置信度校准曲线
   - 输出学到的四个权重，与手工权重 (0.46, 0.32, 0.10, 0.12) 对比

4. **artifact_boost 处理**：作为额外的一维特征加入，或单独用网格搜索学习其五个子阈值。

5. **可扩展性**：如果数据充足，可尝试在 4 维特征上加入二次交互项（`PolynomialFeatures(degree=2)`），捕捉分支间的协同关系，同时保持模型可解释。

### 预期产出

- 每个数据集学到的**最优四分支权重**
- 与手工权重的性能对比（AUC、F1、ECE 校准误差）
- 权重稳定性分析（三个数据集上的方差）

---

## 方案三：端到端三分支 CNN 训练 (`deep_cnn/`)

### 技术路线

```
训练阶段：
  real_face/ ──► make_torch_inputs() ──► (RGB, 残差, 频域) ──► LightweightMultiBranchNet ──► loss
  fake_face/ ──► make_torch_inputs() ──► (RGB, 残差, 频域) ──► LightweightMultiBranchNet ──► loss
                                                     │
                                               BCEWithLogitsLoss
                                               Adam 优化器
推理阶段：
  新图像 ──► make_torch_inputs() ──► 模型 ──► sigmoid ──► AI生成概率
```

### 网络架构（已有 `torch_net.py`）

```
输入: 256×256 人脸 ROI
  │
  ├── RGB 分支 (3→16→32→64, 128维)
  ├── 残差分支 (1→10→20→40, 40维)
  └── 频域分支 (1→10→20→40, 40维)
       │
       └── 拼接 (208维) ──► FC(96) ──► Dropout(0.25) ──► FC(1)
```

总参数量约 **55K**，属于轻量级网络，适合 CPU 训练和推理。

### 关键设计决策

1. **数据增强**：训练时对每张图像做随机水平翻转、轻微旋转（±5°）、亮度/对比度微调（±5%），增强泛化能力。验证和测试时不增强。

2. **训练策略**：
   - 损失函数：`BCEWithLogitsLoss`
   - 优化器：Adam，初始学习率 1e-3，`ReduceLROnPlateau` 衰减
   - Batch size：16（CPU）/ 64（GPU）
   - Epochs：20~50，EarlyStopping（val loss 连续 5 轮不降则停止）
   - 数据划分：70% 训练 / 15% 验证 / 15% 测试

3. **人脸 ROI 提取**：训练和推理时统一使用 `detect_largest_face()` + `resize_roi(256)` 提取人脸区域，确保与现有 pipeline 一致。

4. **交叉生成器评估**（关键）：如果数据集包含多种生成器来源，训练后对每种生成器分别评估，检验模型是否学到了通用的"真实成像物理特征"而非特定生成器的 artifact。

5. **checkpoint 保存**：
   - `models/torch_detector.pt` — 最佳验证 loss 的权重
   - `models/torch_detector_best_auc.pt` — 最佳验证 AUC 的权重
   - 训练日志（TensorBoard 或 CSV）：loss 曲线、AUC 曲线

### 预期产出

- 训练好的 PyTorch 模型权重（替换手工特征分支）
- 在每个数据集上的性能指标（AUC、F1、Accuracy）
- 跨生成器泛化分析
- 与方案一的性能对比

---

## 数据集方案

### 总体策略

三个训练数据集覆盖 **GAN → StyleGAN2 → Stable Diffusion** 两种生成范式，形成论文中可论述的难度梯度。另设独立跨生成器测试集做最终验证（不参与任何训练）。

```
训练数据（三组）                        验证数据（独立，不入训练）
──────────────────                     ──────────────────────────
data1: 入门 GAN 假脸            ──┐
data2: FFHQ + StyleGAN2         ──┼──► 训练 ──► 最终模型
data3: CelebA + SD              ──┘                          │
                                                             ▼
                                              跨生成器测试集（纯验证）
                                              StyleGAN3 / SDXL /
                                              Midjourney / DALL-E3
                                              各 500~1000 张 + 对应真实人脸
```

### 数据集 1：Real and Fake Face Detection（入门基准）

| 属性 | 说明 |
|---|---|
| 来源 | Kaggle: Real and Fake Face Detection |
| 规模 | ~10K 图像，real/fake 均衡 |
| 真实图像 | 真实人脸照片 |
| 生成图像 | GAN 类生成假脸 |
| 生成范式 | GAN |
| 难度 | ⭐⭐ 低 — 作为快速验证 pipeline 的入门基准 |
| 存放路径 | `database/data1/real/` + `database/data1/fake/` |

**论文叙事**：入门基准，验证方法在较易 GAN 假脸上的基本判别能力。

### 数据集 2：140k Real and Fake Faces（核心验证）

| 属性 | 说明 |
|---|---|
| 来源 | Kaggle: `xhlulu/140k-real-and-fake-faces` |
| 规模 | 140K 图像（70K real + 70K fake） |
| 真实图像 | FFHQ（Flickr-Faces-HQ），高质量真实人脸，1024×1024 |
| 生成图像 | StyleGAN2 生成，1024×1024 |
| 生成范式 | StyleGAN2（GAN 类最优水平） |
| 图像尺寸 | 256×256 或 1024×1024 |
| 难度 | ⭐⭐⭐ 中 — StyleGAN2 人脸质量极高，肉眼难辨 |
| 存放路径 | `database/data2/real/` + `database/data2/fake/` |

**论文叙事**：核心验证集。FFHQ 包含年龄、种族、姿态多样性，StyleGAN2 是 GAN 类最强人脸生成器。在此数据集上的表现是方法判别能力的核心证据。

### 数据集 3：CelebA + Stable Diffusion Face Dataset（跨范式挑战）

| 属性 | 说明 |
|---|---|
| 策略 | 分别获取 real 源和 fake 源，配对构成训练集 |
| **真实图像** | **CelebA（unaligned 版）** — 202K 真实人脸，原始分辨率，未做对齐裁剪 |
| **生成图像** | **Stable Diffusion Face Dataset** — 从 HuggingFace / Kaggle 获取 SD 扩散生成人脸 |
| 规模 | 各取约 5GB（real ~80K 张，fake ~50K 张），数量均衡化后使用 |
| 生成范式 | 扩散模型（Stable Diffusion），与 data1/data2 的 GAN 范式形成对比 |
| 难度 | ⭐⭐⭐⭐ 高 — 扩散模型 artifact 模式与 GAN 截然不同 |
| 存放路径 | `database/data3/real/` + `database/data3/fake/` |

**获取指引**：

| 组件 | 渠道 |
|---|---|
| CelebA（unaligned） | `mmlab.ie.cuhk.edu.hk/projects/CelebA.html`、HuggingFace `datasets/celaba` |
| SD Face Dataset | HuggingFace 搜索 "stable-diffusion-faces"、Kaggle 搜索 "AI generated faces stable diffusion" |

**论文叙事**：跨生成范式挑战。data1 和 data2 的假脸均来自 GAN 体系，data3 的假脸来自扩散模型。两个范式下的检测表现的差异与泛化能力是论文关键亮点。

### 三个数据集对比

| 维度 | data1 | data2 | data3 |
|---|---|---|---|
| 真实来源 | 数据集自带 | FFHQ | CelebA |
| 伪造来源 | GAN（早代） | StyleGAN2 | Stable Diffusion |
| 生成范式 | GAN | GAN | 扩散模型 |
| 规模 | ~10K | 140K | ~130K（均衡后） |
| 难度 | 入门 | 核心 | 挑战 |
| 论文角色 | 可行性验证 | 核心评测 | 跨范式泛化 |

### 跨生成器验证测试集（独立，不入训练）

三个训练数据集训练完成后，在以下独立测试集上做最终泛化评估：

| 生成器 | 数据来源 | 数量 | 目的 |
|---|---|---|---|
| StyleGAN3 | 公开采样集 | 500~1000 张 | 测试对未见过的同范式生成器的泛化 |
| Stable Diffusion XL | 社区采样 / 自生成 | 500~1000 张 | 测试对同范式升级版的泛化 |
| Midjourney | 社区公开人脸 | 500~1000 张 | 测试对商业闭源生成器的泛化 |
| DALL-E 3 | API 生成 | 500~1000 张 | 测试对另一商业模型的泛化 |

真实人脸对照统一使用 FFHQ 测试集的未见过子集（~2000 张）。该测试集**任何阶段都不参与训练或超参选择**，仅在最终评估时使用。

---

## 项目结构

```
C_DEF/
│
├── METHODOLOGY.md                     # 本文档
│
├── ai_face_detector/                  # 原始检测器（不动）
│   ├── detector.py                    # 手工权重融合 + 四分支
│   ├── features.py                    # 21维特征提取
│   ├── face.py                        # 人脸定位 + EXIF
│   ├── torch_net.py                   # 三分支 CNN 定义
│   ├── visualization.py               # 可视化工具
│   ├── manual_roi.py                  # 手动框选
│   └── cli.py                         # 命令行入口
│
├── fusion_lr/                         # 【方案一】逻辑回归融合权重
│   ├── config.json                    # 训练超参数
│   ├── extract_branch_scores.py       # 从图像集提取四分支分数
│   ├── train.py                       # 逻辑回归训练 + 交叉验证
│   ├── evaluate.py                    # 评估 + 生成报告
│   └── models/                        # 保存学到的权重
│       └── lr_weights_dataX.json      # 各数据集的最优权重
│
├── deep_cnn/                          # 【方案三】端到端 CNN 训练
│   ├── config.json                    # 训练超参数
│   ├── train.py                       # CNN 训练主程序
│   ├── evaluate.py                    # 评估 + 生成报告
│   ├── dataset.py                     # PyTorch Dataset 数据加载
│   └── models/                        # 保存训练好的 .pt 权重
│       └── torch_detector_dataX.pt    # 各数据集的最优模型
│
├── cross_gen_test/                    # 跨生成器验证测试集（独立，不入训练）
│   ├── real/                          # FFHQ 未见过子集 (~2000张)
│   ├── stylegan3/                     # StyleGAN3 生成
│   ├── sdxl/                          # Stable Diffusion XL 生成
│   ├── midjourney/                    # Midjourney 生成
│   └── dalle3/                        # DALL-E 3 生成
│
├── database/                          # 共享训练数据（.gitignore 排除）
│   ├── data1/                         # Real and Fake Face Detection
│   │   ├── real/                      # 真实人脸
│   │   └── fake/                      # GAN 生成假脸
│   ├── data2/                         # 140k Real and Fake Faces
│   │   ├── real/                      # FFHQ 真实人脸
│   │   └── fake/                      # StyleGAN2 生成假脸
│   └── data3/                         # CelebA + SD Face
│       ├── real/                      # CelebA unaligned 真实人脸
│       └── fake/                      # Stable Diffusion 生成假脸
│
├── models/                            # 共享模型权重
│   └── .gitkeep
│
├── outputs/                           # 原始 GUI/CLI 输出
│   ├── app.py                         # Tkinter GUI
│   ├── requirements.txt
│   └── ...
│
├── .gitignore
└── run_gui.bat
```

### 数据流

```
database/data1/{real,fake}  ──┬──► fusion_lr/extract_branch_scores.py ──► fusion_lr/train.py
database/data2/{real,fake}  ──┤
database/data3/{real,fake}  ──┤
                              │
                              └──► deep_cnn/train.py  (直接读图像)
                              │
                              │
cross_gen_test/               │   最终评估阶段（不参与训练）
  real/ + stylegan3/ + ...  ──┴──► fusion_lr/evaluate.py
                              ──► deep_cnn/evaluate.py
```

### 方案一 v1：四分支逻辑回归融合 (`fusion_lr/`)

| 数据集 | 指标 | 手工基线 | 方案一 v1 (4分支LR) |
|---|---|---|---|
| data1 | AUC | 0.5036 | 0.5409 |
| data1 | F1 | 0.0062 | 0.3087 |
| data1 | ECE | 0.1256 | 0.0085 |
| data2 | AUC | 0.5561 | 0.5713 |
| data2 | F1 | 0.0000 | 0.5706 |
| data2 | ECE | 0.1805 | 0.0018 |

**结论**：四分支特征表达力不足，方案一 v1 上限 AUC≈0.57。

### 方案一 v2：跳过分支聚合 (`fusion_lr_v2/`)

2026-05-12 新增。直接使用 21 维原始特征，对比三种建模方法。

| 方法 | 特征 | 模型 | AUC | F1 | Acc | ECE |
|---|---|---|---|---|---|---|
| 手工基线 | 4分支 | 硬编码规则 | 0.5561 | 0.0000 | 0.5000 | 0.1805 |
| lr_21dim | 21维原始 | LR(CV选C) | 0.6382 | 0.6104 | 0.5992 | 0.0105 |
| lr_hybrid | 21维+6维物理 | LR(CV选C) | 0.6420 | 0.6118 | 0.6018 | 0.0115 |
| **xgb_21dim** | **21维原始** | **XGBoost** | **0.7324** | **0.6681** | **0.6691** | 0.0149 |

**关键发现**：
1. 分支聚合是信息瓶颈 — 跳过它 AUC 从 0.556→0.638
2. 手工物理特征收益可忽略 — lr_hybrid 仅比 lr_21dim 高 0.004
3. 非线性是关键 — XGBoost AUC=0.732，为方案一框架下最优
4. 两个模型依赖不同的特征子集（LR 偏纹理，XGBoost 偏频域/噪声）

#### 跨生成器泛化（data2 训练模型 → 各生成器）

| 测试生成器 | 指标 | 手工基线 | 方案一 v1 | 方案一 v2 (XGB) | 方案三 (CNN) |
|---|---|---|---|---|---|
| StyleGAN3 | AUC | ? | ? | ? | ? |
| SDXL | AUC | ? | ? | ? | ? |
| Midjourney | AUC | ? | ? | ? | ? |
| DALL-E 3 | AUC | ? | ? | ? | ? |

#### 跨范式泛化（data3 训练模型 → GAN 生成器）

| 测试生成器 | 指标 | 手工基线 | 方案一 v1 | 方案一 v2 (XGB) | 方案三 (CNN) |
|---|---|---|---|---|---|
| StyleGAN2 | AUC | ? | ? | ? | ? |
| StyleGAN3 | AUC | ? | ? | ? | ? |

注：cross_gen_test/ 已弃用，跨生成器评估待后续恢复。

---

## 评估指标定义

| 指标 | 全称 | 含义 |
|---|---|---|
| AUC | Area Under ROC Curve | 整体判别能力，不依赖阈值 |
| F1 | F1-Score | 精确率与召回率的调和平均，阈值=0.5 |
| ECE | Expected Calibration Error | 概率校准质量：预测 80% 是否真的对应 80% 真实概率 |
| 权重方差 | Weight Variance | 三个数据集上学到的权重的标准差，反映稳定性 |
| 跨范式 AUC 衰减 | Cross-Paradigm AUC Drop | data2→data3 或 data3→data2 的 AUC 下降幅度，反映范式间泛化难度 |

---

## 当前进度 (2026-05-12)

| # | 行动项 | 状态 |
|---|---|---|
| 1 | 下载 data1 (Kaggle Real and Fake Face) | ✅ 完成 |
| 2 | 下载 data2 (140k Real and Fake Faces) | ✅ 完成 |
| 3 | 下载 data3 (CelebA + SD Face) | ⏳ 未开始 |
| 4 | 方案一 v1：4分支LR融合验证 | ✅ 完成 (AUC≤0.57) |
| 5 | 方案一 v2：21维直连 + XGBoost | ✅ 完成 (AUC=0.732) |
| 6 | cross_gen_test 跨生成器集 | ❌ 已弃用 |
| 7 | 方案三：deep_cnn 端到端训练 | ⏳ 下一步 |
| 8 | 汇总对比报告 | ⏳ 待方案三完成后 |

## 下一步行动

1. 实现 `deep_cnn/dataset.py` + `train.py` — 端到端 LightweightMultiBranchNet 训练
2. 利用 torch 2.9.1+CUDA 在 data2 100K 训练集上训练
3. 与方案一 v2 XGBoost 基线 (AUC=0.732) 对比
4. 视结果决定是否需要 data3 跨范式验证
