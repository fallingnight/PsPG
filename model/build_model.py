from datetime import datetime
import torch

from .image_encoder import SwinTransformer
from .text_encoder import TextEncoder, TextEncoderBert
from .prompt_learner import (
    FixedPrompt,
    ContextOptimization,
    PsPG_LP,
    LinearProbe,
)
from .pspg import pspg

from clip import clip
import torch.nn.functional as F


def load_clip(visual_name, cfg):
    backbone_name = visual_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.ROOT)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(
        state_dict or model.state_dict(),
        dropout=cfg.MODEL.DROP_RATE,
        drop_path=cfg.MODEL.DROP_PATH_RATE,
    )

    return model


def load_swin(cfg):
    visual_encoder = SwinTransformer(
        num_classes=0,
        img_size=cfg.INPUT.SIZE,
        patch_size=cfg.MODEL.VISUAL.SWIN.PATCH_SIZE,
        in_chans=cfg.MODEL.VISUAL.SWIN.IN_CHANS,
        embed_dim=cfg.MODEL.VISUAL.SWIN.EMBED_DIM,
        depths=cfg.MODEL.VISUAL.SWIN.DEPTHS,
        num_heads=cfg.MODEL.VISUAL.SWIN.NUM_HEADS,
        window_size=cfg.MODEL.VISUAL.SWIN.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.VISUAL.SWIN.MLP_RATIO,
        qkv_bias=cfg.MODEL.VISUAL.SWIN.QKV_BIAS,
        qk_scale=cfg.MODEL.VISUAL.SWIN.QK_SCALE,
        drop_rate=cfg.MODEL.DROP_RATE,
        attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
        drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
        ape=cfg.MODEL.VISUAL.SWIN.APE,
        patch_norm=cfg.MODEL.VISUAL.SWIN.PATCH_NORM,
        use_checkpoint=False,
    )
    if cfg.MODEL.VISUAL.PRETRAINED is not None:
        pretrained_path = cfg.MODEL.VISUAL.PRETRAINED
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
        model_dict = visual_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        visual_encoder.load_state_dict(model_dict)

    return visual_encoder


def build_backbone(cfg):
    image_encoder = None
    text_encoder = None
    clip_model = None
    image_encoder_name = cfg.MODEL.VISUAL.NAME
    text_encoder_name = cfg.MODEL.LANG.NAME
    print(f"Creating visual: {image_encoder_name}\t text:{text_encoder_name}")
    if "swin" in image_encoder_name.lower():
        image_encoder = load_swin(cfg)
    else:
        clip_model = load_clip(image_encoder_name, cfg).float()
        image_encoder = clip_model.visual

    if "BERT" in text_encoder_name:
        text_encoder = TextEncoderBert(text_encoder_name)
    else:
        if clip_model is None:
            clip_model = load_clip("RN50", cfg).float()
        text_encoder = TextEncoder(clip_model)

    return image_encoder, text_encoder


def build_prompt_learner(cfg, classnames, text_encoder):
    name = cfg.MODEL.PROMPT.NAME.lower()
    prompt_learner = None
    if name == "fixed":
        prompt_learner = FixedPrompt(classnames, cfg=cfg.MODEL)
    elif name == "coop":
        prompt_learner = ContextOptimization(classnames, text_encoder, cfg=cfg.MODEL)
    elif name == "pspg":
        prompt_learner = PsPG_LP(classnames, text_encoder, cfg=cfg.MODEL)
    elif name == "linear":
        prompt_learner = LinearProbe(classnames, text_encoder, cfg=cfg.MODEL)
    return prompt_learner


def build_pspg(cfg, classnames):
    image_encoder, text_encoder = build_backbone(cfg)
    prompt_learner = build_prompt_learner(cfg, classnames, text_encoder)
    model = pspg(cfg.MODEL, image_encoder, text_encoder, prompt_learner)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/(1e6):.2f}M")
    prompt_learner_params = sum(p.numel() for p in prompt_learner.parameters())
    print(f"Prompt Learner parameters: {prompt_learner_params/(1e6):.2f}M")

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    prompt_learner.set_device(device)

    network_name = (
        model.network_name
        if hasattr(model, "network_name")
        else cfg.MODEL.VISUAL.NAME + "_" + cfg.MODEL.LANG.NAME
    )
    model_name = f"{cfg.DATASET.NAME}-{network_name}"

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")

    if not cfg.EVAL_ONLY:
        model_name += f"bs{cfg.DATALOADER.TRAIN.BATCH_SIZE}-e{cfg.OPTIM.MAX_EPOCH}-{formatted_time}"

    return model, model_name
