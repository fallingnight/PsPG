from copy import deepcopy
import torch
import torch.nn as nn
import pdb

from .text_encoder import preprocess_text
from .rnn import AttentionDecoderGRU, AttentionDecoderLSTM
from .transformer import TransformerDecoder
from .mlp import MLPDecoder


class FixedPrompt(nn.Module):
    def __init__(self, class_names, cfg):
        super().__init__()
        self.class_names = class_names
        self.pos_prompts = [
            f"Findings suggesting {class_name}." for class_name in class_names
        ]
        self.neg_prompts = [
            f"No evidence of {class_name}." for class_name in class_names
        ]
        self.pos_token = preprocess_text(self.pos_prompts, cfg)
        self.neg_token = preprocess_text(self.neg_prompts, cfg)

    def forward(self):
        return self.pos_token, self.neg_token, None, None, None, None

    @torch.jit.ignore
    def set_device(self, device):
        self.pos_token = {
            "input_ids": self.pos_token["input_ids"].to(device),
            "attention_mask": self.pos_token["attention_mask"].to(device),
        }
        self.neg_token = {
            "input_ids": self.neg_token["input_ids"].to(device),
            "attention_mask": self.neg_token["attention_mask"].to(device),
        }


# modified from COOP & DualCOOP
class ContextOptimization(nn.Module):

    def __init__(
        self,
        classnames,
        text_encoder,
        cfg,
    ):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.PROMPT.COOP_N_CTX_POS
        n_ctx_neg = cfg.PROMPT.COOP_N_CTX_NEG
        ctx_init_pos = cfg.PROMPT.COOP_POS_INIT.strip()
        ctx_init_neg = cfg.PROMPT.COOP_NEG_INIT.strip()
        ctx_dim = text_encoder.ln_final.weight.shape[0]
        self.token_type = cfg.LANG.NAME
        prefix_len = self.get_prefix_len()

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = preprocess_text([ctx_init_pos], cfg)
            prompt_neg = preprocess_text([ctx_init_neg], cfg)
            with torch.no_grad():
                embedding_pos = text_encoder.get_embedding(prompt_pos)
                embedding_neg = text_encoder.get_embedding(prompt_neg)
            ctx_vectors_pos = embedding_pos[0, prefix_len : prefix_len + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, prefix_len : prefix_len + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.PROMPT.COOP_CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.PROMPT.COOP_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts_pos = preprocess_text(prompts_pos, cfg)
        tokenized_prompts_neg = preprocess_text(prompts_neg, cfg)
        with torch.no_grad():
            embedding_pos = text_encoder.get_embedding(tokenized_prompts_pos)
            embedding_neg = text_encoder.get_embedding(tokenized_prompts_neg)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        if prefix_len == 1:
            self.register_buffer("token_prefix_pos", embedding_pos[:, :prefix_len, :])
            self.register_buffer("token_prefix_neg", embedding_neg[:, :prefix_len, :])
        self.register_buffer(
            "token_suffix_pos", embedding_pos[:, prefix_len + n_ctx_pos :, :]
        )
        self.register_buffer(
            "token_suffix_neg", embedding_neg[:, prefix_len + n_ctx_neg :, :]
        )

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.tokenized_prompts_pos = tokenized_prompts_pos
        self.tokenized_prompts_neg = tokenized_prompts_neg

    def get_prefix_len(self):
        return 0 if "BERT" in self.token_type else 1

    @torch.jit.ignore
    def set_device(self, device):
        self.tokenized_prompts_pos = {
            "input_ids": self.tokenized_prompts_pos["input_ids"].to(device),
            "attention_mask": self.tokenized_prompts_pos["attention_mask"].to(device),
        }
        self.tokenized_prompts_neg = {
            "input_ids": self.tokenized_prompts_neg["input_ids"].to(device),
            "attention_mask": self.tokenized_prompts_neg["attention_mask"].to(device),
        }

    def forward(self):
        prefix_len = self.get_prefix_len()
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        ctx_pos = ctx_pos.expand(self.n_cls, -1, -1)
        ctx_neg = ctx_neg.expand(self.n_cls, -1, -1)

        prefix_pos = self.token_prefix_pos if prefix_len == 1 else None
        prefix_neg = self.token_prefix_neg if prefix_len == 1 else None
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        if prefix_len == 1:
            prompts_pos = torch.cat(
                [
                    prefix_pos,  # (n_cls, 1, dim)
                    ctx_pos,  # (n_cls, n_ctx, dim)
                    suffix_pos,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    prefix_neg,  # (n_cls, 1, dim)
                    ctx_neg,  # (n_cls, n_ctx, dim)
                    suffix_neg,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            prompts_pos = torch.cat(
                [
                    ctx_pos,  # (n_cls, n_ctx, dim)
                    suffix_pos,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    ctx_neg,  # (n_cls, n_ctx, dim)
                    suffix_neg,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        tokenized_prompts_pos = self.tokenized_prompts_pos
        tokenized_prompts_neg = self.tokenized_prompts_neg

        return (
            tokenized_prompts_pos,
            tokenized_prompts_neg,
            prompts_pos,
            prompts_neg,
            None,
            None,
        )


class PsPG_LP(nn.Module):

    def __init__(
        self,
        classnames,
        text_encoder,
        cfg,
    ):
        super().__init__()
        self.input_dim = text_encoder.output_dim
        self.output_dim = text_encoder.ln_final.weight.shape[0]
        self.max_length = cfg.PROMPT.DECODER_MAX_LENGTH
        self.hidden_size = cfg.PROMPT.DECODER_HIDDEN
        self.num_head = cfg.PROMPT.DECODER_NUM_HEADS
        self.dropout = cfg.PROMPT.DECODER_DROP_OUT
        self.max_context = cfg.LANG.CONTEXT_LENGTH
        self.layers = cfg.PROMPT.DECODER_LAYERS
        self.drop_path = cfg.PROMPT.DECODER_DROP_PATH

        self.cfg = cfg
        self.device = None
        self.tokenized_prompt = None

        tokenized_classnames = preprocess_text(classnames, cfg)
        print(classnames)
        if cfg.PROMPT.ENABLE_PREFIX:
            self.register_buffer(
                "cls_length",
                tokenized_classnames["input_ids"].argmax(dim=-1) - 1,
                persistent=False,
            )
        with torch.no_grad():
            x, x1 = text_encoder(tokenized_classnames)
        self.register_buffer("cls_feature", x, persistent=False)
        self.register_buffer(
            "cls_allfeatures", x1[:, 1 : self.max_length + 1], persistent=False
        )
        temp_text = " ".join(["X"] * self.max_length)
        token = preprocess_text([temp_text], cfg)
        token_text = token["input_ids"]
        with torch.no_grad():
            embedding = text_encoder.get_embedding(token)
        self.register_buffer(
            "prefix_embed",
            embedding[0, 0].repeat(self.cls_feature.shape[0], 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "suffix_embed",
            embedding[0, token_text.argmax(dim=-1) :].repeat(
                self.cls_feature.shape[0], 1, 1
            ),
            persistent=False,
        )
        if cfg.PROMPT.ENABLE_PREFIX:
            prompts = [temp_text + " " + name + "." for name in classnames]
            tokenized_prompts = preprocess_text(prompts, cfg)
            self.tokenized_prompt = tokenized_prompts
            with torch.no_grad():
                embedding_template = text_encoder.get_embedding(tokenized_prompts)
            self.register_buffer(
                "prefix_template",
                embedding_template[:, :1, :],
                persistent=False,
            )
            self.register_buffer(
                "suffix_template",
                embedding_template[:, 1 + self.max_length :, :],
                persistent=False,
            )

        decoder_type = cfg.PROMPT.DECODER_TYPE.lower()
        if decoder_type == "gru":
            self.pos_decoder = AttentionDecoderGRU(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
            self.neg_decoder = AttentionDecoderGRU(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
        elif decoder_type == "lstm":
            self.pos_decoder = AttentionDecoderLSTM(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
            self.neg_decoder = AttentionDecoderLSTM(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
        elif decoder_type == "transformer":
            self.pos_decoder = TransformerDecoder(
                self.input_dim,
                self.hidden_size,
                self.layers,
                self.num_head,
                self.output_dim,
                self.max_length,
                self.dropout,
                self.drop_path,
            )
            self.neg_decoder = TransformerDecoder(
                self.input_dim,
                self.hidden_size,
                self.layers,
                self.num_head,
                self.output_dim,
                self.max_length,
                self.dropout,
                self.drop_path,
            )
        elif decoder_type == "mlp":
            self.pos_decoder = MLPDecoder(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
            )
            self.neg_decoder = MLPDecoder(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
            )

    @torch.jit.ignore
    def set_device(self, device):
        self.device = device
        if self.tokenized_prompt is not None:
            self.tokenized_prompt = {
                "input_ids": self.tokenized_prompt["input_ids"].to(device),
                "attention_mask": self.tokenized_prompt["attention_mask"].to(device),
            }

    @property
    def dtype(self):
        return self.cls_feature.dtype

    def get_pairs(self, outputs):
        combinations = torch.combinations(torch.arange(outputs.shape[0]), r=2).to(
            outputs.device
        )
        combined_features = torch.zeros(
            combinations.shape[0], self.max_context, outputs.shape[2]
        ).to(outputs.device)
        tokenized_pairs = None
        if not self.cfg.PROMPT.ENABLE_PREFIX:
            for i, (j, k) in enumerate(combinations):
                combined_features[i] = torch.cat(
                    [
                        self.prefix_embed[0],
                        outputs[j],
                        outputs[k],
                        self.suffix_embed[0, : -self.max_length],
                    ],
                    dim=0,
                )
            pair_texts = [
                " ".join(["X"] * (self.max_length * 2))
            ] * combined_features.shape[0]
            tokenized_pairs = preprocess_text(pair_texts, self.cfg)
            tokenized_pairs = {
                "input_ids": tokenized_pairs["input_ids"].to(self.device),
                "attention_mask": tokenized_pairs["attention_mask"].to(self.device),
            }
        else:
            pair_texts = []
            for i, (j, k) in enumerate(combinations):
                suffix_length = max(
                    self.max_context - 1 - self.max_length * 2 - self.cls_length[j], 0
                )
                combined_features[i] = torch.cat(
                    [
                        self.prefix_embed[0],
                        outputs[j],
                        self.suffix_template[j, : self.cls_length[j], :],
                        outputs[k],
                        self.suffix_template[k, :suffix_length, :],
                    ],
                    dim=0,
                )
                pair_texts.append(
                    " ".join(
                        ["X"]
                        * (
                            self.max_length * 2
                            + self.cls_length[j]
                            + self.cls_length[k]
                        )
                    )
                )
            tokenized_pairs = preprocess_text(pair_texts, self.cfg)
            tokenized_pairs = {
                "input_ids": tokenized_pairs["input_ids"].to(self.device),
                "attention_mask": tokenized_pairs["attention_mask"].to(self.device),
            }
        return combined_features, tokenized_pairs

    def forward(self, image_features):
        # image_features: (bs,dim)
        image_features = image_features.unsqueeze(0).repeat(
            self.cls_feature.shape[0], 1, 1
        )  # (bs,cls,dim)
        output_pos, _ = self.pos_decoder(self.cls_feature, image_features)
        output_neg, _ = self.neg_decoder(self.cls_feature, image_features)
        prompt_pairs = None
        tokenized_pairs = None
        if self.cfg.PROMPT.ENABLE_PAIRLOSS and self.training:
            prompt_pairs, tokenized_pairs = self.get_pairs(output_pos)

        pos_texts = []
        neg_texts = []
        pos_indexs = torch.zeros(self.cls_feature.shape[0]).long()
        neg_indexs = torch.zeros(self.cls_feature.shape[0]).long()

        for i in range(self.cls_feature.shape[0]):
            pos_texts.append(" ".join(["X"] * self.max_length))
            neg_texts.append(" ".join(["X"] * self.max_length))
        tokenized_pos = preprocess_text(pos_texts, self.cfg)
        tokenized_neg = preprocess_text(neg_texts, self.cfg)
        tokenized_pos = {
            "input_ids": tokenized_pos["input_ids"].to(self.device),
            "attention_mask": tokenized_pos["attention_mask"].to(self.device),
        }
        tokenized_neg = {
            "input_ids": tokenized_neg["input_ids"].to(self.device),
            "attention_mask": tokenized_neg["attention_mask"].to(self.device),
        }
        if not self.cfg.PROMPT.ENABLE_PREFIX:
            prompts_pos = torch.cat(
                [
                    self.prefix_embed,  # (cls, 1, dim)
                    output_pos,  # (cls, length, dim)
                    self.suffix_embed,  # (cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    self.prefix_embed,  # (cls, 1, dim)
                    output_neg,  # (cls, length, dim)
                    self.suffix_embed,  # (cls, *, dim)
                ],
                dim=1,
            )
        else:
            prompts_pos = torch.cat(
                [
                    self.prefix_template,  # (cls, 1, dim)
                    output_pos,  # (cls, length, dim)
                    self.suffix_template,  # (cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    self.prefix_template,  # (cls, 1, dim)
                    output_neg,  # (cls, length, dim)
                    self.suffix_template,  # (cls, *, dim)
                ],
                dim=1,
            )
            tokenized_pos = self.tokenized_prompt
            tokenized_neg = self.tokenized_prompt

        return (
            tokenized_pos,
            tokenized_neg,
            prompts_pos,
            prompts_neg,
            tokenized_pairs,
            prompt_pairs,
        )


class LinearProbe(nn.Module):

    def __init__(self, class_names, text_encoder, cfg):
        super().__init__()
        self.class_names = class_names
        self.input_dim = text_encoder.output_dim
        self.device = None
        self.linear_head = nn.Linear(self.input_dim, len(self.class_names))

    @torch.jit.ignore
    def set_device(self, device):
        self.device = device

    def forward(self, image_features):
        output = self.linear_head(image_features)
        return output
