# Latent Diffusion Model Project

This project is an implementation of Latent Diffusion Models (LDMs), refactored for clarity, modularity, and ease of use. The codebase separates model architectures, training logic, and utility scripts to facilitate research and development in generative modeling.

---

## 依赖关系 (Dependencies)

在运行此项目之前，请确保已安装以下依赖项。建议使用虚拟环境（如 Conda 或 venv）来管理项目依赖。

```bash
pip install torch torchvision torchaudio
pip install diffusers
pip install transformers
pip install accelerate
pip install torch-ema
pip install torch-fidelity
pip install timm
pip install lpips
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install lmdb
pip install opencv-python
pip install kagglehub
```

---

## 项目结构 (Project Structure)

项目采用模块化的目录结构，清晰地分离了不同的功能组件：

```
/Diffusion-Model-old
|
|-- README.md                   # 本文档 (This file)
|-- utils.py                    # 通用工具函数 (General utility functions)
|
|-- data/                       # 数据处理与加载 (Data processing and loading)
|   |-- __init__.py
|   |-- download_celeba_hq.py   # 使用 kagglehub 下载 CelebA-HQ 数据集
|   |-- face_crop.py            # 人脸裁剪脚本
|   |-- init_dataset.py         # 数据集类定义 (e.g., CelebaHQDataset)
|   `-- ...
|
|-- diffusion/                  # 核心扩散模型组件 (Core diffusion model components)
|   |-- __init__.py
|   |-- DenoiseDiffusion.py     # 核心去噪扩散过程 (Denoising diffusion process)
|   `-- sampler.py              # 采样器实现 (Sampler implementations)
|
|-- models/                     # 神经网络结构定义 (Neural network architectures)
|   |-- __init__.py
|   |-- Unet.py                 # U-Net 模型
|   |-- DiT.py                  # Diffusion Transformer 模型
|   |-- VAVAE.py                # 变分自编码器 (Variational Autoencoder)
|   `-- VQGAN.py                # 矢量量化生成对抗网络 (Vector Quantized GAN)
|
|-- modules/                    # 模型中可重用的子模块 (Reusable sub-modules for models)
|   |-- __init__.py
|   |-- autoencoderkl.py        # KL-Autoencoder 模块
|   |-- resnet.py               # ResNet 模块
|   |-- perceptual_module.py    # 感知损失模块 (Perceptual loss module)
|   `-- ...
|
|-- scripts/                    # 高层应用脚本 (High-level application scripts)
|   |-- train/                  # 各类模型的训练脚本 (Training scripts for models)
|   |   |-- train_unet.py
|   |   |-- train_dit.py
|   |   `-- ...
|   |-- sample_ldm.py           # LDM 采样脚本
|   |-- compute_fid.py          # 计算 FID 分数以评估模型
|   |-- check_recon_effect.py   # 检查自编码器的重建效果
|   `-- ...
|
|-- trainer/                    # 模型训练器 (Model trainers)
|   |-- __init__.py
|   |-- Unet_trainer.py         # U-Net 训练逻辑
|   |-- DiT_trainer.py          # DiT 训练逻辑
|   |-- AutoencoderKL_trainer.py # AutoencoderKL 训练逻辑
|   `-- ...
|
`-- .idea/                      # IDE 配置文件 (IDE configuration files)
```

---

## 如何使用 (How to Use)

### 1. 环境设置 (Environment Setup)

首先，克隆项目并安装所需的依赖包。

```bash
# 克隆仓库
git clone <your-repository-url>
cd Diffusion-Model-old

# (可选但推荐) 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 安装依赖
pip install -r requirements.txt # (如果提供了 requirements.txt)
# 或者手动安装
pip install torch torchvision diffusers transformers accelerate torch-ema torch-fidelity timm lpips scikit-learn matplotlib tqdm lmdb opencv-python kagglehub
```

### 2. 数据准备 (Data Preparation)

本项目支持 CelebA-HQ 等数据集。

- **下载数据**: 使用 `data/download_celeba_hq.py` 脚本从 Kaggle 下载数据集。
- **预处理**:
    - 使用 `data/face_crop.py` 对人脸图像进行裁剪，以获得更好的模型输入。
    - 使用 `scripts/save_latent_to_lmdb.py` 或 `scripts/save_latent_to_lmdb_from_pretrained.py` 将图像数据通过预训练的 VAE 编码为潜在向量，并保存为 LMDB 格式，以加速训练过程。

### 3. 模型训练 (Model Training)

所有训练脚本都位于 `scripts/train/` 目录下。你可以根据需求选择相应的脚本来训练不同的模型。

**示例：训练 U-Net**

```bash
python scripts/train/train_unet.py
```

- **分布式训练**: 脚本已集成 `torch.distributed`，可以方便地进行多 GPU 训练。
- **超参数**: 在启动训练前，请检查并修改对应 `trainer` 目录下的训练器文件或训练脚本中的超参数，如学习率、批量大小、数据集路径、模型保存路径等。

### 4. 生成样本 (Generating Samples)

使用 `scripts/` 目录下的采样脚本从已训练好的模型生成图像。

**示例：使用 LDM 生成图像**

```bash
python scripts/sample_ldm.py
```

- **检查点**: 在运行采样脚本之前，请确保在脚本中正确指定了预训练模型（如 U-Net 和 VAE）的检查点路径。
- **采样方法**: `scripts/sample_ldm_dpm_solver.py` 提供了使用 DPM-Solver 的更快速采样方法。

### 5. 模型评估 (Model Evaluation)

- **重建效果**: 运行 `scripts/check_recon_effect.py` 来可视化自编码器（VAE, VQGAN）的图像重建质量。
- **FID 分数**: 运行 `scripts/compute_fid.py` 来计算生成样本的 FID 分数，以定量评估模型的生成质量。



该项目在上次提交的基础上经过重构，关系更加明晰。



至是，工程已毕，言尽于此。
