# Diffusion Model Playground

This repository documents my learning journey and practical implementations of diffusion models, starting from simple Python scripts and evolving into a more structured project.

## Project Overview

This project explores various aspects of diffusion models, including:

- **Denoising Diffusion Probabilistic Models (DDPM)**
- **Latent Diffusion Models (LDM)**
- **Autoencoders (VAE, VQ-GAN, AutoencoderKL)** for latent space representation.
- **U-Net and Transformer (DiT)** as noise prediction networks.

## File Descriptions

- `README.md`: Project overview, setup instructions, and documentation.
- `.gitignore`: Specifies files and directories to be ignored by Git (e.g., datasets, model checkpoints, cache files).
- `init_dataset.py`: Contains dataset classes (e.g., `CelebaHQDataset`, `LatentDataset`) and PyTorch transforms for loading and preprocessing training data.
- `utils.py`: A collection of helper functions used across the project, such as model loading from checkpoints, distributed training initialization, and visualization helpers.

### Core Model Directories

- **`AutoEncoder/`**: Contains implementations and trainers for all autoencoder-related models.
  - `VAVAE_trainer.py`: Trainer for a custom Variational Autoencoder (VAE) combined with a GAN-style discriminator for sharper results.
  - `AutoencoderKL_trainer.py`: Trainer designed for fine-tuning the `AutoencoderKL` model from Diffusers.
  - `VQGAN_trainer.py`: Trainer for a Vector-Quantized Generative Adversarial Network (VQ-GAN).
  - `modules/`: Contains the actual model definitions for the autoencoders (`VA_VAE.py`, `VQGAN.py`, etc.).

- **`diffusion_and_unet/`**: Holds the core logic for the diffusion process and the U-Net model.
  - `DenoiseDiffusion.py`: Implements the core DDPM forward (noising) and reverse (denoising) processes.
  - `Unet.py`: The U-Net model architecture, which serves as the primary noise predictor in many diffusion setups.
  - `Unet_trainer.py`: The trainer class for training the U-Net model on latent representations.
  - `sampler.py`: Implements various sampling algorithms (e.g., DDPM, DDIM) to generate images from pure noise.

- **`Transformer/`**: Contains the implementation of the Diffusion Transformer (DiT).
  - `DiT.py`: The Diffusion Transformer (DiT) model architecture, an alternative to the U-Net.
  - `DiT_trainer.py`: The trainer class for the DiT model.

- **`modules/`**: A directory for shared, general-purpose neural network modules (e.g., `ResNet` blocks) that can be reused by various models in the project.

### Scripts

- **`scripts/`**: Contains all executable scripts for training, sampling, and other utilities.
  - `train/`: A sub-directory with dedicated scripts for training each major model (`train_unet.py`, `train_vavae.py`, `train_dit.py`, etc.).
  - `sample_ldm.py`: A script to generate images from a trained Latent Diffusion Model.
  - `compute_fid.py`: A script to calculate the Fréchet Inception Distance (FID) score to quantitatively evaluate generated image quality.
  - `face_crop.py`: A utility script for data preprocessing, such as face detection and cropping using a segmentation model.
  - `save_latent_to_lmdb.py`: A powerful script to pre-calculate and save image latents to an LMDB (Lightning Memory-Mapped Database) for significantly faster training of the diffusion model.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Diffusers
- `pip install -r requirements.txt` (You should create this file based on your environment).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Set up your data:** Download the necessary datasets (e.g., CelebA-HQ) and update the placeholder paths (e.g., `/path/to/your/...`) in the scripts.

### Training

Training scripts are in `scripts/train/`. You can launch a training job like so:

```bash
# Example for training the U-Net
python scripts/train/train_unet.py
```

### Sampling

Sampling scripts are in the `scripts/` directory. For example:

```bash
python scripts/sample_ldm.py
```
