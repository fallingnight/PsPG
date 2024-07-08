import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .prompt_learner import ContextOptimization, PsPG_LP, LinearProbe


class ECA_combined(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avgpool(x)
        avg_y = self.conv(avg_y.permute(0, 2, 1))
        max_y = self.maxpool(x)
        max_y = self.conv(max_y.permute(0, 2, 1))
        y = avg_y + max_y
        y = y.permute(0, 2, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class pspg(nn.Module):
    def __init__(self, cfg, image_encoder, text_encoder, prompt_learner):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.prompt_learner = prompt_learner
        self.output_dim = self.cfg.DIM_PROJECTION
        self.image_projection_layer = None
        self.text_projection_layer = None
        self.channel_attention = None

        if isinstance(self.prompt_learner, PsPG_LP):
            self.channel_attention = ECA_combined(50)
            channel_params = sum(p.numel() for p in self.channel_attention.parameters())
            print(f"Attn parameters: {channel_params:.2f}")

        image_encoder_dim = self.get_encoder_dim("image")
        text_encoder_dim = self.get_encoder_dim("text")

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.cfg.TEMPERATURE)
        )
        """

        if image_encoder_dim is not None:
            self.image_projection_layer = nn.Parameter(
                torch.empty(image_encoder_dim, self.output_dim)
            )
            trunc_normal_(self.image_projection_layer, std=0.02)
        if text_encoder_dim is not None:
            self.text_projection_layer = nn.Parameter(
                torch.empty(text_encoder_dim, self.output_dim)
            )
            trunc_normal_(self.text_projection_layer, std=0.02)
        """

    def get_encoder_dim(self, type="image"):
        encoder_dim = None
        if type == "image":
            if hasattr(self.image_encoder, "dim_out"):
                encoder_dim = self.image_encoder.dim_out
            elif hasattr(self.image_encoder, "output_dim"):
                encoder_dim = self.image_encoder.output_dim
        elif type == "text":
            if hasattr(self.text_encoder, "output_dim"):
                encoder_dim = self.text_encoder.output_dim

        return encoder_dim

    @property
    def network_name(self):
        name = ""
        name += "pspg-{}".format(self.cfg.VISUAL.NAME)
        return name

    @property
    def dtype(self):
        return self.logit_scale.dtype

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {"logit_scale"}
        if hasattr(self.text_encoder, "no_weight_decay"):
            for k in self.text_encoder.no_weight_decay():
                no_weight_decay.add("text_encoder." + k)

        if hasattr(self.image_encoder, "no_weight_decay"):
            for k in self.image_encoder.no_weight_decay():
                no_weight_decay.add("image_encoder." + k)

        return no_weight_decay

    @torch.jit.ignore
    def froze_clip_params(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def froze_prompt_params(self):
        for param in self.prompt_learner.parameters():
            param.requires_grad = False

    def encode_image(self, image):
        x, x2 = self.image_encoder(image)
        if self.image_projection_layer is not None:
            x = x @ self.image_projection_layer

        x1 = x / x.norm(dim=-1, keepdim=True)

        return x1, x2, x

    def encode_text(self, text, embed=None):
        x, _ = self.text_encoder(text, embed)
        if self.text_projection_layer is not None:
            x = x @ self.text_projection_layer

        x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_dualcoop(self, image_features, text_features):
        if self.image_projection_layer is not None:
            image_features = image_features @ self.image_projection_layer
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output[:, : c // 2]
        w_half = F.softmax(output_half, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        output = 5 * (output * w).sum(-1)

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_class]
        logits = output.view(b, 2, c // 2)
        return logits

    def encode_pspglp(self, image_features, image_global, text_features, pair_features):
        pair_logits = None

        if self.image_projection_layer is not None:
            image_features = image_features @ self.image_projection_layer
        image_features = image_features.permute(0, 2, 1)  # 64,50,512
        image_features = torch.cat(
            [image_global.unsqueeze(1), image_features[:, 1:]], dim=1
        )
        if self.channel_attention is not None:
            image_features = self.channel_attention(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        if pair_features is not None:
            pair_logits = logit_scale * image_features @ pair_features.t()
            pair_logits = pair_logits.sum(1)  # 64,91

        output = logit_scale * image_features @ text_features.t()
        output = output.sum(1)
        bs, cls = output.shape
        logits = output.view(bs, 2, cls // 2)
        return logits, pair_logits

    def forward(self, image, text=None):
        image_features, channel_features, attn_feature = self.encode_image(image)
        if text is not None:
            text_features = self.encode_text(text)
        clip_logits = None
        lp_logits = None
        pair_logits = None
        pair_features = None

        if self.training and not self.cfg.ENABLE_LP:
            logit_scale = self.logit_scale.exp()
            clip_logits = logit_scale * image_features @ text_features.t()
        elif self.prompt_learner is not None:
            if isinstance(self.prompt_learner, LinearProbe):
                lp_logits = self.prompt_learner(image_features)
            else:
                if isinstance(self.prompt_learner, PsPG_LP):
                    (
                        pos_token,
                        neg_token,
                        pos_embed,
                        neg_embed,
                        pair_token,
                        pair_embed,
                    ) = self.prompt_learner(attn_feature)
                else:
                    (
                        pos_token,
                        neg_token,
                        pos_embed,
                        neg_embed,
                        pair_token,
                        pair_embed,
                    ) = self.prompt_learner()
                if pos_embed is None or neg_embed is None:
                    pos_features = self.encode_text(pos_token)
                    neg_features = self.encode_text(neg_token)
                else:
                    pos_features = self.encode_text(pos_token, pos_embed)
                    neg_features = self.encode_text(neg_token, neg_embed)

                if pair_token is not None and pair_embed is not None:
                    pair_features = self.encode_text(pair_token, pair_embed)

                if isinstance(self.prompt_learner, ContextOptimization):
                    lp_features = torch.cat([pos_features, neg_features], dim=0)
                    lp_logits = self.encode_dualcoop(channel_features, lp_features)
                elif isinstance(self.prompt_learner, PsPG_LP):
                    lp_features = torch.cat([pos_features, neg_features], dim=0)
                    lp_logits, pair_logits = self.encode_pspglp(
                        channel_features, image_features, lp_features, pair_features
                    )
                else:
                    lp_features = torch.cat((pos_features, neg_features), dim=0)
                    logit_scale = self.logit_scale.exp()
                    lp_logits = logit_scale * image_features @ lp_features.t()
                    bs, cls = lp_logits.shape
                    lp_logits = lp_logits.view(bs, 2, cls // 2)

        return clip_logits, lp_logits, pair_logits
