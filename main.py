if __name__ == '__main__':
    from diffusion import *
    import torch
    from tqdm import tqdm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("[INFO] Initialization completed.")

    # model, loss = launcher.train(is_train = True, resume = True)
    # print(loss)

    checkpoint = torch.load(r"E:\DL\model_epoch_300.pth", map_location = torch.device('cpu'))
    loaded_state_dict = checkpoint['model_state_dict']
    # 处理加载的 state_dict，去掉 `module.` 前缀
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        if k.startswith("module."):
            new_key = k[len("module."):]  # 移除 "module." 前缀
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # 加载新的 state_dict 到模型
    model = Unet(3, 64, [1, 2, 2, 4], [False, False, True, True], n_blocks = 1).to(device)
    model.load_state_dict(new_state_dict)

    print("[INFO] Model loading completed.")
    evaluator = sampler(model)
    shape = (1, 3, 64, 64)
    xt = torch.randn(shape, device = device)
    evaluator.visualize_ddim_process(750, 250, shape, 'ddim_06_eta0', eta = 0.0, xt = xt)
    evaluator.visualize_ddim_process(750, 250, shape, 'ddim_06_eta0.01', eta = 0.01, xt = xt)
    evaluator.visualize_ddim_process(750, 250, shape, 'ddim_06_eta0.03', eta = 0.03, xt = xt)
    evaluator.visualize_ddim_process(750, 250, shape, 'ddim_06_eta0.05', eta = 0.05, xt = xt)
    evaluator.visualize_ddim_process(750, 250, shape, 'ddim_06_eta0.1', eta = 0.1, xt = xt)
    evaluator.visualize_ddim_process(750, 250, shape, 'ddim_06_eta0.3', eta = 0.3, xt = xt)