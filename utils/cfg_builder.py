from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = "./output"
_C.CHECKPOINT = "weights/PsPG.pth"
_C.START_AFRESH = False
_C.EVAL_ONLY = False
_C.SEED = -1
_C.USE_CUDA = True
_C.VERBOSE = True


_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
_C.INPUT.INTERPOLATION = "bicubic"
_C.INPUT.NO_TRANSFORM = False
_C.INPUT.PIXEL_MEAN = [0.5305947661399841, 0.5322656035423279, 0.5315520763397217]
_C.INPUT.PIXEL_STD = [0.23302224278450012, 0.2326543629169464, 0.23269867897033691]
_C.INPUT.CROP_SCALE = (0.8, 1.0)
_C.INPUT.ROTATE_DEGREES = (-10, 10)
_C.INPUT.AFFINE_DEGREES = (-10, 10)
_C.INPUT.AFFINE_TRANSLATE = (0.0625, 0.0625)
_C.INPUT.AFFINE_SCALE = (0.9, 1.1)
_C.INPUT.HORIZONTAL_FLIP_PROB = 0.5
_C.INPUT.COLOR_JITTER_BRIGHTNESS = (0.8, 1.2)
_C.INPUT.COLOR_JITTER_CONTRAST = (0.8, 1.2)
_C.INPUT.ENABLE_CUTMIX = False
_C.INPUT.ENABLE_MIXUP = False

_C.DATASET = CN()
_C.DATASET.ROOT = ""
_C.DATASET.NAME = ""
_C.DATASET.VAL_SPLIT = "val"
_C.DATASET.TEST_SPLIT = "test"
_C.DATASET.TRAIN_SPLIT = "train"
_C.DATASET.TEST_FILE_PATH = None
_C.DATASET.VAL_FILE_PATH = None
_C.DATASET.TRAIN_FILE_PATH = None

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.TRAIN = CN()
_C.DATALOADER.TRAIN.BATCH_SIZE = 64
_C.DATALOADER.TRAIN.SHUFFLE = True
_C.DATALOADER.TRAIN.USE_LABEL = True
_C.DATALOADER.TRAIN.PROPORTION = 1.0

_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.BATCH_SIZE = 64
_C.DATALOADER.TEST.SHUFFLE = False

_C.DATALOADER.VAL = CN()
_C.DATALOADER.VAL.SHUFFLE = False
_C.DATALOADER.VAL.BATCH_SIZE = 64


_C.MODEL = CN()
_C.MODEL.ROOT = "weights"
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.ATTN_DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.DIM_PROJECTION = 512
_C.MODEL.TEMPERATURE = 0.07
_C.MODEL.NUM_HEADS = 8

_C.MODEL.ENABLE_LP = True

_C.MODEL.VISUAL = CN()
_C.MODEL.VISUAL.NAME = "RN50"
_C.MODEL.VISUAL.PRETRAINED = None


# swin parameters (we do not use them finally)
_C.MODEL.VISUAL.SWIN = CN()
_C.MODEL.VISUAL.SWIN.PATCH_SIZE = 4
_C.MODEL.VISUAL.SWIN.IN_CHANS = 3
_C.MODEL.VISUAL.SWIN.EMBED_DIM = 96
_C.MODEL.VISUAL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.VISUAL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.VISUAL.SWIN.WINDOW_SIZE = 7
_C.MODEL.VISUAL.SWIN.MLP_RATIO = 4.0
_C.MODEL.VISUAL.SWIN.QKV_BIAS = True
_C.MODEL.VISUAL.SWIN.QK_SCALE = None
_C.MODEL.VISUAL.SWIN.APE = False
_C.MODEL.VISUAL.SWIN.PATCH_NORM = True

_C.MODEL.LANG = CN()
_C.MODEL.LANG.NAME = "CLIP"
_C.MODEL.LANG.PRETRAINED = None
_C.MODEL.LANG.TOKENIZER = "clip"
_C.MODEL.LANG.CONTEXT_LENGTH = 77
_C.MODEL.LANG.WIDTH = 1024
_C.MODEL.LANG.HEADS = 16
_C.MODEL.LANG.LAYERS = 12

_C.MODEL.PROMPT = CN()
_C.MODEL.PROMPT.NAME = "pspg"

_C.MODEL.PROMPT.COOP_N_CTX_POS = 16
_C.MODEL.PROMPT.COOP_N_CTX_NEG = 16
_C.MODEL.PROMPT.COOP_POS_INIT = ""  # "Findings suggesting"
_C.MODEL.PROMPT.COOP_NEG_INIT = ""  # "No evidence of"
_C.MODEL.PROMPT.COOP_CSC = False

_C.MODEL.PROMPT.DECODER_TYPE = "gru"
_C.MODEL.PROMPT.DECODER_HIDDEN = 512
_C.MODEL.PROMPT.DECODER_MAX_LENGTH = 16
_C.MODEL.PROMPT.DECODER_NUM_HEADS = 8
_C.MODEL.PROMPT.DECODER_DROP_OUT = 0.2
_C.MODEL.PROMPT.DECODER_LAYERS = 2
_C.MODEL.PROMPT.DECODER_DROP_PATH = 0.1
_C.MODEL.PROMPT.ENABLE_PAIRLOSS = False
_C.MODEL.PROMPT.ENABLE_PREFIX = False

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.ASL_GAMMA_POS = 1.0
_C.MODEL.LOSS.ASL_GAMMA_NEG = 2.0

# Optimization
_C.OPTIM = CN()
_C.OPTIM.NAME = "sgd"
_C.OPTIM.LR = 1e-4
_C.OPTIM.WEIGHT_DECAY = 0.05
_C.OPTIM.EPS = 1e-8
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = True
_C.OPTIM.RMSPROP_ALPHA = 0.99
_C.OPTIM.ADAM_BETAS = (0.9, 0.999)
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "cosine"

_C.OPTIM.MAX_EPOCH = 50
_C.OPTIM.WARMUP_EPOCHS = 5
_C.OPTIM.DECAY_EPOCHS = 30
_C.OPTIM.DECAY_RATE = 0.1
_C.OPTIM.WARMUP_LR = 1e-6
_C.OPTIM.MIN_LR = 5e-6

_C.OPTIM.EMA = False
_C.OPTIM.EMA_DECAY = 0.9998

_C.TRAIN = CN()
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.VAL_FREQ_IN_EPOCH = 1000
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.FINETUNE_CLIP = False


_C.TEST = CN()
_C.TEST.SAVE_PRED = False
_C.TEST.THRESHOLD = 0.5


def parse_cfg(cfg, args):
    if args.evaluate:
        cfg.EVAL_ONLY = args.evaluate
    if args.verbose:
        cfg.VERBOSE = args.verbose
    if args.checkpoint:
        cfg.CHECKPOINT = args.checkpoint
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed:
        cfg.SEED = args.seed

    if args.prompt:
        cfg.MODEL.PROMPT.NAME = args.prompt
    if args.decoder_type:
        cfg.MODEL.PROMPT.DECODER_TYPE = args.decoder_type
    if args.decoder_hidden:
        cfg.MODEL.PROMPT.DECODER_HIDDEN = args.decoder_hidden
    if args.decoder_max_length:
        cfg.MODEL.PROMPT.DECODER_MAX_LENGTH = args.decoder_max_length
    if args.decoder_dropout:
        cfg.MODEL.PROMPT.DECODER_DROP_OUT = args.decoder_dropout
    if args.decoder_droppath:
        cfg.MODEL.PROMPT.DECODER_DROP_PATH = args.decoder_droppath
    if args.pretrained_visual:
        cfg.MODEL.VISUAL.PRETRAINED = args.pretrained_visual
    if args.pretrained_lang:
        cfg.MODEL.LANG.PRETRAINED = args.pretrained_lang
    if args.gamma_pos:
        cfg.MODEL.LOSS.ASL_GAMMA_POS = args.gamma_pos
    if args.gamma_neg:
        cfg.MODEL.LOSS.ASL_GAMMA_NEG = args.gamma_neg

    if args.dataset_dir:
        cfg.DATASET.ROOT = args.dataset_dir

    if args.val_batch_size:
        cfg.DATALOADER.VAL.BATCH_SIZE
    if args.train_batch_size:
        cfg.DATALOADER.TRAIN.BATCH_SIZE
    if args.proportion:
        cfg.DATALOADER.TRAIN.PROPORTION = args.proportion
    if args.input_size:
        cfg.INPUT.SIZE = (args.input_size, args.input_size)

    if args.print_freq:
        cfg.TRAIN.PRINT_FREQ = args.print_freq
    if args.val_freq:
        cfg.TRAIN.VAL_FREQ_IN_EPOCH = args.val_freq
    if args.threshold:
        cfg.TEST.THRESHOLD = args.threshold
    if args.save_pred:
        cfg.TEST.SAVE_PRED = args.save_pred
    if args.start_afresh:
        cfg.START_AFRESH = args.start_afresh

    if args.optim:
        cfg.OPTIM.NAME = args.optim
    if args.lr:
        cfg.OPTIM.LR = args.lr
    if args.max_epochs:
        cfg.OPTIM.MAX_EPOCH = args.max_epochs
    if args.warmup_epochs:
        cfg.OPTIM.WARMUP_EPOCH = args.warmup_epochs

    if args.ema:
        cfg.OPTIM.EMA = args.ema
    if args.ema_decay:
        cfg.OPTIM.EMA_DECAY = args.ema_decay

    if args.finetune_clip:
        cfg.TRAIN.FINETUNE_CLIP = args.finetune_clip

    if args.test_file_path:
        cfg.DATASET.TEST_FILE_PATH = args.test_file_path
    if args.train_file_path:
        cfg.DATASET.TRAIN_FILE_PATH = args.train_file_path
    if args.val_file_path:
        cfg.DATASET.VAL_FILE_PATH = args.val_file_path


def get_cfg(args):
    cfg = _C

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    parse_cfg(cfg, args)

    cfg.freeze()

    return cfg
