import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLPDecoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_length=20,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            QuickGELU(),
            nn.Linear(hidden_size, max_length * output_size, bias=True),
        )
        self.output_size = output_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, decoder_input, encoder_outputs):
        outputs = self.mlp(decoder_input)
        outputs = outputs.reshape(outputs.shape[0], -1, self.output_size)
        return outputs, None
