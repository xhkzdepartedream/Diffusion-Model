import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gather(x: torch.Tensor, t: torch.Tensor):
    x=x.to(device)
    res = x.gather(-1, t)
    return res.reshape(-1, 1, 1, 1)
