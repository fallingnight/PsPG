from torch import optim


def build_optimizer(cfg, model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_params(model, cfg, skip, skip_keywords)

    opt_lower = cfg.OPTIM.NAME.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=cfg.OPTIM.MOMENTUM,
            nesterov=cfg.OPTIM.SGD_NESTEROV,
            dampening=cfg.OPTIM.SGD_DAMPNING,
            lr=cfg.OPTIM.LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=cfg.OPTIM.EPS,
            betas=cfg.OPTIM.ADAM_BETAS,
            lr=cfg.OPTIM.LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(
            parameters,
            lr=cfg.OPTIM.LR,
            momentum=cfg.OPTIM.MOMENTUM,
            alpha=cfg.OPTIM.RMSPROP_ALPHA,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )

    return optimizer


def get_params(model, cfg, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    if not cfg.MODEL.ENABLE_LP:
        try:
            model.froze_prompt_params()
        except:
            model.module.froze_prompt_params()
    if not cfg.TRAIN.FINETUNE_CLIP:
        try:
            model.froze_clip_params()
        except:
            model.module.froze_prompt_params()
    try:
        all_params = model.named_parameters()
    except:
        all_params = model.module.named_parameters()
    for name, param in all_params:
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords(name, skip_keywords)
        ):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def check_keywords(name, keywords):
    is_in_name = False
    for keyword in keywords:
        if keyword in name:
            is_in_name = True
    return is_in_name
