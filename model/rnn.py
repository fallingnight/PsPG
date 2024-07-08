import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x1 = x
        self_attention_output, _ = self.self_attention(x, x, x)
        x = x + self_attention_output
        x = self.layer_norm(x)
        x = self.fc(x)
        x = x + x1
        x = self.layer_norm(x)

        return x


class AttentionDecoderGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_length=20,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.max_length = max_length
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.encoder_proj = nn.Linear(input_size, hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads=num_heads)
        self.encoder_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, input_size)
        self.output_proj = nn.Linear(input_size, output_size)
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
        h = torch.zeros(decoder_input.size(0), self.gru.hidden_size).to(
            decoder_input.device
        )
        encoder_outputs = self.encoder_proj(encoder_outputs)
        attention_weights = []
        outputs = []

        for i in range(self.max_length):
            h = self.gru(decoder_input, h)
            h = self.self_attention(h)

            h1 = h.unsqueeze(1).permute(1, 0, 2)
            context = encoder_outputs.permute(1, 0, 2)
            context, attention_weights_t = self.encoder_attention(h1, context, context)
            attention_weights.append(attention_weights_t)

            output = h + context.squeeze(0)
            output = self.fc(output)

            outputs.append(output)
            decoder_input = output

        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.output_proj(outputs)
        attention_weights = torch.stack(attention_weights, dim=0)
        return outputs, attention_weights


class AttentionDecoderLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_length=20,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.max_length = max_length
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.encoder_proj = nn.Linear(input_size, hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads=num_heads)
        self.encoder_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, input_size)
        self.output_proj = nn.Linear(input_size, output_size)
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
        h = torch.zeros(decoder_input.size(0), self.lstm.hidden_size).to(
            decoder_input.device
        )
        c = torch.zeros(decoder_input.size(0), self.lstm.hidden_size).to(
            decoder_input.device
        )

        encoder_outputs = self.encoder_proj(encoder_outputs)
        attention_weights = []
        outputs = []

        for i in range(self.max_length):
            h, c = self.lstm(decoder_input, (h, c))
            h = self.self_attention(h)

            h1 = h.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
            attention_outputs, attention_weights_t = self.encoder_attention(
                encoder_outputs, h1, h1
            )
            attention_weights.append(attention_weights_t)
            context = torch.mean(attention_outputs, dim=1)

            output = h + context
            output = self.fc(output)

            outputs.append(output)
            decoder_input = output
        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.output_proj(outputs)
        attention_weights = torch.stack(attention_weights, dim=0)
        return outputs, attention_weights
