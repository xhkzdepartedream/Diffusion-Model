import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你有 B 张图像，每张图像编码成 256 个 token
# 每个 token 是 [0, 511] 之间的整数
# shape: (B, 256)
z_q_indices = torch.randint(0, 512, (8, 256))  # 示例 batch



class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model = 512, nhead = 8, num_layers = 6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1025, d_model))  # 固定序列长度
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = 2048,
            batch_first = True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_seq):
        B, L = input_seq.shape

        token_embed = self.token_embedding(input_seq)
        pos_embed = self.pos_embedding[:, :L, :]

        x = token_embed + pos_embed

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(input_seq.device)
        memory = torch.zeros(B, 1, x.size(-1), device = x.device)

        out = self.transformer(x, memory, tgt_mask = tgt_mask)
        return self.output_head(out)
