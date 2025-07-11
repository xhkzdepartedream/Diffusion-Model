import os
from Transformer import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from init_dataset import *


class TransformerSampler:
    def __init__(self, model, sos_token_id, eos_token_id, max_len = 1024, device = 'cuda'):
        self.model = model
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_len = max_len
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size = 1, temperature = 1.0):
        # 初始化输入序列，起始都是<sos>
        input_seq = torch.full((batch_size, 1), self.sos_token_id, dtype = torch.long, device = self.device)

        for _ in range(self.max_len):
            # 前向传播
            logits = self.model(input_seq)  # (B, L, vocab_size)
            logits = logits[:, -1, :]  # 取最后一个位置的输出 (B, vocab_size)
            # 调温度，做softmax概率分布
            probs = F.softmax(logits / temperature, dim = -1)
            # 按概率采样下一个token
            next_token = torch.multinomial(probs, num_samples = 1)  # (B,1)
            # 拼接到序列后面
            input_seq = torch.cat([input_seq, next_token], dim = 1)  # (B, L+1)
            # 如果全部batch都生成了eos，提前终止采样
            if (next_token == self.eos_token_id).all():
                break

        return input_seq


def load_model_from_checkpoint(path, model, device = 'cuda', prefix_to_strip = "_orig_mod."):
    checkpoint = torch.load(path, map_location = device)
    loaded_state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        new_key = k[10:]
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    return model


if __name__ == '__main__':
    model = TransformerModel(1026)
    model = load_model_from_checkpoint("/path/to/your/transformer_epoch_25.pth",
                                       model, 'cuda', 't')
    sampler = TransformerSampler(model, sos_token_id = 0, eos_token_id = 1024, max_len = 1024, device = 'cuda')
    sampled_seq = sampler.sample(batch_size = 1, temperature = 1.0)
    print(sampled_seq.shape)
    sampled_seq=sampled_seq[1:]
    sampled_seq.reshape((32,32))

