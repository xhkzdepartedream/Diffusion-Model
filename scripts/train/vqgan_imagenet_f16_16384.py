import torch
from omegaconf import OmegaConf
import torch
from torch.serialization import add_safe_globals





def initialize_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    import importlib
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if "params" in config:
        return cls(**config["params"])
    else:
        return cls()


# 1. 加载配置
config_path = "/path/to/your/config.yaml"
ckpt_path = "/path/to/your/last.ckpt"
config = OmegaConf.load(config_path)

# 2. 创建模型实例
model = initialize_from_config(config.model)

# 3. 加载权重

ckpt = torch.load(ckpt_path, map_location = "cpu", weights_only = False)
model.load_state_dict(ckpt["state_dict"], strict = False)
model.eval().cuda()  # 如果有 GPU

print("Done.")
# 4. 转为 eval 模式并放到 GPU

