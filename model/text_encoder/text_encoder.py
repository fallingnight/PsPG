import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BertTokenizer
from clip.simple_tokenizer import SimpleTokenizer


def preprocess_text(texts, cfg):
    #     if model.context_length is None:
    #         model = model.module
    if "BERT" in cfg.LANG.NAME:
        _tokenizer = BertTokenizer.from_pretrained(cfg.LANG.NAME)
        result = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.LANG.CONTEXT_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }

    else:
        _tokenizer = SimpleTokenizer()
        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts
        ]
        result = torch.zeros(len(all_tokens), cfg.LANG.CONTEXT_LENGTH, dtype=torch.long)
        attention_mask = torch.zeros(
            len(all_tokens), cfg.LANG.CONTEXT_LENGTH, dtype=torch.long
        )

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > cfg.LANG.CONTEXT_LENGTH:
                tokens = tokens[: cfg.LANG.CONTEXT_LENGTH]
                tokens[-1] = eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)
            attention_mask[i, 1 : len(tokens)] = 1
        return {"input_ids": result, "attention_mask": attention_mask}


class TextEncoderBert(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = BertModel.from_pretrained(self.model_name)
        self.cfg = BertConfig.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.output_dim = self.cfg.hidden_size

    def get_embedding(self, input):
        input_ids = input["input_ids"]
        weights = self.model.get_input_embeddings().weight
        return F.embedding(input_ids, weights)

    def forward(self, input_ids, input_embeds=None):
        if input_embeds is not None:
            output = self.model(
                attention_mask=input_ids["attention_mask"], inputs_embeds=input_embeds
            )
        else:
            output = self.model(**input_ids)
        x = output.last_hidden_state
        x = x[:, 0]
        # x = torch.mean(x, dim=1)
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model_name = "clip"
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.output_dim = self.text_projection.shape[-1]

    def get_embedding(self, text):
        text = text["input_ids"]
        return self.token_embedding(text).type(
            self.dtype
        )  # [batch_size, n_ctx, transformer.width]

    def forward(self, text, embed=None):
        text = text["input_ids"]
        if embed is not None:
            x = embed + self.positional_embedding.type(self.dtype)
        else:
            x = self.token_embedding(text).type(
                self.dtype
            ) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x1 = x @ self.text_projection
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, x1
