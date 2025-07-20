def get_latent_tensor(index, lmdb_path = "../data/latents.lmdb"):
    """
    根据提供的索引从LMDB数据库中获取对应的潜在表示，并转换为PyTorch张量。
    """
    env = lmdb.open(lmdb_path, readonly = True, lock = False)
    key = f"{index:06d}".encode("ascii")
    with env.begin() as txn:
        buf = txn.get(key)
        if buf is None:
            raise ValueError(f"No latent found for index {index}.")
        # 加载张量时直接映射到目标设备
        latent_tensor = torch.load(BytesIO(buf), map_location = device)
    env.close()
    return latent_tensor


def main(DiTorNot: bool, load_pretrained: bool, n: int):
    dit_path = "/data1/yangyanliang/Diffusion-Model/DiT_rope_epoch_500.pth"
    unet_path = "path/to/your/unet2_epoch_300.pth"
    autoencoderkl_path = "./autoencoderkl_success2.pth"
    vae_path = "path/to/your/vavae16c32d_test_40.pth"

    # --- 模型初始化 ---
    if DiTorNot:
        model = DiT(input_size = 32, patch_size = 2, input_ch = 4, n_ch = 768, n_blocks = 12, num_heads = 8,
                    learn_sigma = False, pe = "rope", attn_dropout = 0, mlp_dropout = 0).to(device)
        model = load_model_from_checkpoint(dit_path, 'dit', device, model)
    else:
        model = Unet(input_ch = 4, n_ch = 128, ch_mults = [1, 1, 2, 2, 1],
                     is_attn = [False, False, True, True, False], n_blocks = 3).to(device)
        model = load_model_from_checkpoint(unet_path, 'unet', device, model)

    # --- VAE 初始化 ---
    if load_pretrained:
        vae = load_model_from_checkpoint(autoencoderkl_path, 'autoencoderkl', device)
        vae.eval()
    else:
        vae = VAE(input_size = 256, input_ch = 3, base_ch = 128, ch_mults = [1, 2, 4, 4],
                  has_attn = [False, False, False, True], latent_ch = 32, n_blocks = 2).to(device)
        vae = load_model_from_checkpoint(vae_path, 'vae', device, vae)

    print("[INFO] Model loading completed.")

    # --- 采样 --- 
    evaluator = sampler(model, device, use_cos = False, prediction_type = 'v_prediction')
    print(
        f"[INFO] Sampler initialized. Learn sigma: {evaluator.learn_sigma}, Prediction type: {evaluator.diffusion.prediction_type}")

    shape = [1, 4, 32, 32]
    # 直接在目标设备上创建初始随机张量
    for i in range(n):
        xt = torch.randn(shape, device = device)
        print(f"[INFO] Initial latent shape: {xt.shape}, Device: {xt.device}")

        # 采样器应返回与模型设备一致的张量
        z = evaluator.sample_x0(shape = shape, steps = 750, use_ddim = True, leap_steps_num = 5,
                                eta = 0.01, xt = xt)
        print(f"[INFO] Sampled latent z shape: {z.shape}, Device: {z.device}")

        # --- 解码和可视化 ---
        with torch.no_grad():
            # 确保 vae 和 z 在同一设备上
            if load_pretrained:
                # AutoencoderKL 的 decode 返回一个带 .sample 属性的对象
                x0 = vae.decode(z.to(device)).sample
            else:
                # 自定义 VAE 的解码逻辑可能不同，这里仅为示例
                # 您可能需要调整此部分以匹配您的 VAE 实现
                print("[WARNING] Custom VAE decode logic might need adjustment.")
                x0, _, _, _ = vae(z.to(device))

        print(f"[INFO] Decoded image shape: {x0.shape}, Device: {x0.device}")
        evaluator.show_tensor_image(x0)


if __name__ == '__main__':
    print("[INFO] Progarm start.")
    import torch
    import lmdb
    from io import BytesIO
    # 模块导入
    from models.DiT import DiT
    from models.VAVAE import VAE
    from models.Unet import Unet
    from diffusion.sampler import sampler
    from utils import load_model_from_checkpoint

    # --- 设备设置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    main(DiTorNot = True, load_pretrained = True, n = 8)
