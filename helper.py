import os
import shutil

import torch


class AverageMeter(object):
    """Computes and stores the average value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_logger(cfg, model_name):
    """create/open file to store log info"""

    log_folder = os.path.join(cfg.OUTPUT_DIR, model_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logfile_path = os.path.join(log_folder, "log.log")
    if os.path.exists(logfile_path):
        logfile = open(logfile_path, "a")
    else:
        logfile = open(logfile_path, "w")
    return logfile


def write_logfile(content, logfile):

    print(content, file=logfile, flush=True)


def load_checkpoint(pretrained, model, optimizer, scheduler):

    if pretrained is not None and os.path.exists(pretrained):
        print("... loading weights from %s" % pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        return model, optimizer, scheduler, epoch
    else:
        raise ValueError("pretrained is missing or its path does not exist")


def load_model_only(pretrained, model):
    """only model, except optimzer, scheduler, etc"""

    if pretrained is not None and os.path.exists(pretrained):
        print("... loading weights from %s" % pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        try:
            epoch = checkpoint["epoch"]
        except:
            epoch = None
        try:
            state_dict = checkpoint["state_dict"]
        except:
            return model, epoch
        unmatched_keys = [
            "prompt_learner.token_prefix_pos",
            "prompt_learner.token_prefix_neg",
            "prompt_learner.token_suffix_pos",
            "prompt_learner.token_suffix_neg",
        ]
        compatible_dict = {
            k: v for k, v in state_dict.items() if k not in unmatched_keys
        }
        missing_keys = model.load_state_dict(compatible_dict, strict=False)
        print(missing_keys)
        return model, epoch
    else:
        raise ValueError("pretrained is missing or its path does not exist")


def save_checkpoint(state, epoch, is_best, filepath="", model_name=""):

    output_dir = filepath
    filepath = os.path.join(filepath, model_name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    torch.save(state, os.path.join(filepath, "checkpoint.pth"))
    if (epoch + 1) % 10 == 0:
        shutil.copyfile(
            os.path.join(filepath, "checkpoint.pth"),
            os.path.join(filepath, str(epoch + 1) + "_checkpoint.pth"),
        )
    if is_best:
        shutil.copyfile(
            os.path.join(filepath, "checkpoint.pth"),
            os.path.join(filepath, "best.pth"),
        )
    shutil.copyfile(
        os.path.join(filepath, "checkpoint.pth"), os.path.join(output_dir, "last.pth")
    )
