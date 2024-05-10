import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import drop_path, trunc_normal_


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0,
        drop_path: float = 0,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y):
        x1 = self.ln_1(x)
        x1, _ = self.self_attn(x1, x1, x1)
        x = x + self.drop_path(x1)
        x1 = self.ln_2(x)

        x1 = x1.unsqueeze(1).permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        y, attn_W = self.cross_attn(x1, y, y)
        x = x + self.drop_path(y.squeeze(0))

        x1 = self.ln_3(x)
        x = x + self.drop_path(self.mlp(x1))
        return x, attn_W


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        input_width: int,
        width: int,
        layers: int,
        heads: int,
        output_width: int,
        max_length=20,
        dropout: float = 0,
        drop_path: float = 0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.max_length = max_length
        # stochastic depth decay rule
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.encoder_proj = nn.Linear(input_width, width)
        self.input_proj = nn.Linear(input_width, width)
        self.resblocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    width,
                    heads,
                    dropout=0,
                    drop_path=drop_path_rate[i],
                )
                for i in range(layers)
            ]
        )
        self.fc = nn.Linear(width, output_width)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, decoder_input, encoder_outputs):
        decoder_input = self.input_proj(decoder_input)
        encoder_outputs = self.encoder_proj(encoder_outputs)

        attention_weights = []
        outputs = []

        for i in range(self.max_length):
            for n, block in enumerate(self.resblocks):
                output, attn_W = block(decoder_input, encoder_outputs)
                decoder_input = output

            attention_weights.append(attn_W)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        attention_weights = torch.stack(attention_weights, dim=0)
        return outputs, attention_weights
