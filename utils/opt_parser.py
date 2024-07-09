import argparse


def parse_opt():
    parser = argparse.ArgumentParser(
        description="PsPG train and evaluation python script"
    )
    # config related
    parser.add_argument(
        "-nc",
        "--network_config_file",
        dest="config_file",
        type=str,
        help="path to network config file",
    )
    parser.add_argument(
        "-dc",
        "--dataset_config_file",
        dest="dataset_config_file",
        type=str,
        help="path to dataset config file",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        metavar="PATH",
        help="path to checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained_visual",
        default=None,
        type=str,
        metavar="PATH",
        help="path to pretrained visual backbone (default: none)",
    )

    parser.add_argument(
        "--pretrained_lang",
        default=None,
        type=str,
        metavar="PATH",
        help="path to pretrained lanuage backbone (default: none)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="print verbose model info"
    )
    parser.add_argument("--seed", type=int, help="set seed")

    # data related
    parser.add_argument("--dataset_dir", type=str, help="path to dataset root")

    parser.add_argument(
        "--input_size", default=224, type=int, help="input image size (default: 224)"
    )

    # log and save related
    parser.add_argument(
        "--output_dir", default="", type=str, help="save path of log file and model "
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="frequency to print the log during the training (default: 100 (iter))",
    )

    parser.add_argument(
        "--val_freq",
        type=int,
        help="val in epochs per n iters, set -1 to disable",
    )

    # test and validate related
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model only",
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="pspg",
        help="model name only for val saving",
    )

    parser.add_argument(
        "--save_pred",
        dest="save_pred",
        action="store_true",
        help="save predictions to json file",
    )

    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="the threshold for val",
    )

    parser.add_argument(
        "--start_afresh",
        dest="start_afresh",
        action="store_true",
        help="do not load the training progress (include optim, lr scheduler, etc), start training afresh",
    )

    # decoder
    parser.add_argument(
        "--prompt",
        dest="prompt",
        type=str,
        help="prompt name (fixed/coop/pspg)",
    )

    parser.add_argument(
        "--decoder_type",
        dest="decoder_type",
        type=str,
        help="decoder type (gru/lstm)",
    )

    parser.add_argument(
        "--decoder_hidden",
        dest="decoder_hidden",
        type=int,
        help="hidden size of prompt decoder",
    )

    parser.add_argument(
        "--decoder_max_length",
        dest="decoder_max_length",
        type=int,
        help="max seq length of prompt decoder",
    )

    parser.add_argument(
        "--decoder_dropout",
        dest="decoder_dropout",
        type=float,
        help="dropout rate of prompt decoder's final fc",
    )

    parser.add_argument(
        "--decoder_droppath",
        dest="decoder_droppath",
        type=float,
        help="droppath rate of prompt decoder",
    )

    # asl loss

    parser.add_argument(
        "--gamma_neg",
        dest="gamma_neg",
        type=float,
        help="the gamma neg for asymmetric loss",
    )

    parser.add_argument(
        "--gamma_pos",
        dest="gamma_pos",
        type=float,
        help="the gamma pos for asymmetric loss",
    )

    # training

    parser.add_argument(
        "-p",
        "--proportion",
        dest="proportion",
        type=float,
        default=1.0,
        help="the proportion of data used for training, from 0. to 1.",
    )

    parser.add_argument(
        "--train_batch_size",
        dest="train_batch_size",
        type=int,
        help="the batch size for training",
    )
    parser.add_argument(
        "--val_batch_size",
        dest="val_batch_size",
        type=int,
        help="the batch size for test/val",
    )

    parser.add_argument(
        "--max_epochs", dest="max_epochs", type=int, help="the max epochs"
    )
    parser.add_argument("--lr", dest="lr", type=float, help="learning rate")

    parser.add_argument("--optim", dest="optim", type=str, help="optimizer")

    parser.add_argument(
        "--finetune_clip",
        dest="finetune_clip",
        action="store_true",
        help="specify if finetuning the whole clip",
    )

    parser.add_argument(
        "--warmup_epochs",
        dest="warmup_epochs",
        type=int,
        default=5,
        help="warm up epochs",
    )

    # additions

    parser.add_argument(
        "--ema",
        dest="ema",
        action="store_true",
        help="enable model ema",
    )

    parser.add_argument(
        "--ema_decay",
        dest="ema_decay",
        type=float,
        default=0.9998,
        help="enable model ema",
    )

    # dataset_map_files

    parser.add_argument(
        "--test_file_path",
        dest="test_file_path",
        type=str,
        help="path of the map file of test dataset",
    )

    parser.add_argument(
        "--val_file_path",
        dest="val_file_path",
        type=str,
        help="path of the map file of val dataset",
    )

    parser.add_argument(
        "--train_file_path",
        dest="train_file_path",
        type=str,
        help="path of the map file of train dataset",
    )

    return parser


parser = parse_opt()
args = parser.parse_args()
