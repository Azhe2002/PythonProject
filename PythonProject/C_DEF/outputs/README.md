# AI 生成人脸检测识别 Python 工程

本工程根据研究计划书中的技术路线实现：以人脸 ROI 为核心，融合 RGB 语义、频域残差、物理成像噪声三类证据，读入图像后输出“AI 生成概率”，并提供图形化界面。

> 说明：当前工程默认包含一个可直接运行的物理启发式初始模型。如果准备好真实人脸与 AI 生成人脸数据集，可通过训练脚本生成 `models/physics_detector.joblib`，GUI 和 CLI 会自动优先使用训练权重。

## 工程结构

```text
.
├── app.py                         # Tkinter 图形化界面
├── ai_face_detector/
│   ├── detector.py                # 检测器主流程与概率融合
│   ├── face.py                    # 图像读取、EXIF、人脸定位
│   ├── manual_roi.py              # 手动人脸框校验与裁剪
│   ├── features.py                # RGB/频域/物理噪声特征
│   ├── visualization.py           # 标注图、残差热力图、频域图
│   ├── train.py                   # 训练入口
│   ├── torch_net.py               # 轻量级三分支 CNN
│   ├── train_deep.py              # PyTorch 深度模型训练入口
│   └── cli.py                     # 命令行入口
├── requirements.txt
└── run_gui.bat
```

## 安装依赖

```powershell
pip install -r requirements.txt
```

当前环境已检测到 `numpy`、`opencv-python`、`Pillow`、`scikit-learn`、`torch`、`matplotlib` 可用，因此可以直接运行。

## 启动图形界面

```powershell
python app.py
```

也可以双击：

```text
run_gui.bat
```

界面流程：

1. 点击“选择图像”。
2. 如果自动人脸框位置不准，点击“手动框选人脸”，在图像上拖拽矩形框。
3. 点击“开始检测”。
4. 查看 AI 生成概率、人脸框、四分支证据、解释说明。
5. 可切换“标注图 / 残差热力图 / 频域响应图”。
6. 点击“导出报告”会保存到 `outputs/图像名/`。

## 命令行检测

```powershell
python -m ai_face_detector.cli path\to\face.jpg --save-dir outputs
```

指定手动人脸框，坐标格式为原图像素 `x,y,w,h`：

```powershell
python -m ai_face_detector.cli path\to\face.jpg --box 120,80,360,420
```

输出 JSON：

```powershell
python -m ai_face_detector.cli path\to\face.jpg --json
```

## 训练自定义模型：特征模型

准备如下数据目录：

```text
data/
├── train/
│   ├── real/     # 真实相机拍摄人脸
│   └── fake/     # AI 生成或换脸、修补、扩散生成图像
└── val/
    ├── real/
    └── fake/
```

训练：

```powershell
python -m ai_face_detector.train --data data --output models/physics_detector.joblib
```

训练完成后再次运行 GUI，系统会自动加载 `models/physics_detector.joblib`。

## 训练自定义模型：三分支 CNN

如果希望更贴近计划书中的“RGB + 频域 + 物理噪声”轻量级多分支网络，可训练 PyTorch 版本：

```powershell
python -m ai_face_detector.train_deep --data data --output models/torch_detector.pt --epochs 20 --batch-size 16
```

当不存在 `models/physics_detector.joblib` 且存在 `models/torch_detector.pt` 时，GUI 会自动加载 PyTorch 权重。命令行也可显式指定：

```powershell
python -m ai_face_detector.cli path\to\face.jpg --model models\torch_detector.pt
```

## 检测原理

- 人脸定位：优先使用 OpenCV Haar Cascade 检测最大人脸，未检测到时使用中心区域估计。
- RGB 语义分支：清晰度、边缘密度、饱和度、纹理能量等。
- 频域残差分支：FFT 高频比例、谱斜率、方向性、JPEG 块效应等。
- 物理噪声分支：噪声残差强度、残差熵、噪声与亮度相关性、CFA 周期痕迹、RGB 通道噪声相关性等。
- 人脸光滑度分支：局部平滑比例、纹理能量、边缘密度、残差强度等，用于识别异常光滑的人脸区域。

## 实验规则校准

- 如果 RGB 语义分支真实证据达到或接近 100%，系统会输出警告：这可能是 AI 图像经过对抗优化后的伪装信号，大概率为 AI 生成。
- 如果人脸光滑度分支检测到 AI 风险达到 70% 或以上，系统会输出警告，并大幅提高 AI 生成概率。
- 如果检测到 EXIF 信息，系统会输出提示：根据实验规则，大概率为真实拍摄。
- 如果二者同时出现，系统会同时输出警告和提示，并建议结合原始文件来源、物理噪声分支和频域分支复核。
- 如果最终判定置信度低于 50%，系统只输出低置信度提示，保留算法计算出的 AI 概率与结论，不做固定概率校准。

当前启发式基础融合权重为：物理噪声分支 `0.46`，频域残差分支 `0.32`，RGB 语义分支 `0.10`，人脸光滑度分支 `0.12`。

启发式模式适合演示完整流程和初步筛查，不应作为司法或高风险场景的最终判断。正式研究应使用覆盖不同相机、压缩质量、生成器类型的数据集进行训练、交叉生成器测试和鲁棒性评估。
