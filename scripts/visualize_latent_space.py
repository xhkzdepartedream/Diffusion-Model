import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

def extract_latents(vae, dataloader, device='cuda', max_batches=50):
    """提取一批图像对应的潜在向量和标签"""
    vae.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader)):
            if i >= max_batches:
                break
            x = x.to(device)
            z = vae.encode(x).latent_dist.mean  # 如果你想用 sample 可改为 .rsample()
            latents.append(z.cpu())
            labels.extend(y.cpu())
    return torch.cat(latents).numpy(), np.array(labels)

def visualize_latents(latents, labels=None, method='tsne', title='Latent Space'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    z_2d = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=5) if labels is not None else plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5)
    plt.title(title)
    if labels is not None:
        plt.colorbar(scatter, label="Class")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 用法示例
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from your_vae_model import YourVAE  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = YourVAE().to(device)
    vae.load_state_dict(torch.load('vae_checkpoint.pth', map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 提取 latent
    latents, labels = extract_latents(vae, dataloader, device=device, max_batches=50)

    # 可视化 PCA / t-SNE
    visualize_latents(latents, labels, method='pca', title='PCA of Latent Space')
    visualize_latents(latents, labels, method='tsne', title='t-SNE of Latent Space')
