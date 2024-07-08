import random
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from model import build_pspg
from torch.cuda.amp import autocast
from utils.cfg_builder import get_freeze_cfg
from helper import (
    load_model_only,
)


def threshold_to_binary(y_pred, threshold):
    binary_pred = (y_pred > threshold).astype(int)
    return binary_pred


def generate_json_output(y_pred, y_label):
    num_samples = y_pred.shape[0]

    json_data = []

    for i in range(num_samples):
        sample_data = {
            "id": i + 1,
            "probabilities": y_pred[i, :].tolist(),
            "labels": y_label[i, :].tolist(),
        }

        json_data.append(sample_data)

    return json_data


def build_transform(cfg):
    interpolation_method = InterpolationMode[cfg.INTERPOLATION.upper()]
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.SIZE, interpolation=interpolation_method),
            transforms.ToTensor(),
            transforms.Normalize(
                tuple(cfg.PIXEL_MEAN),
                tuple(cfg.PIXEL_STD),
            ),
        ]
    )
    return transform


def inference(img_path, model, cfg):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    model.eval()

    transform = build_transform(cfg=cfg.INPUT)
    img = Image.open(img_path).convert("RGB")
    img = transform(img)

    Softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        img = img.to(device).unsqueeze(0)
        with autocast():
            _, outputs, _ = model(img)
        if outputs.dim() == 3:
            outputs = Softmax(outputs).cpu()[:, 0]

    y_pred = outputs.numpy()
    y_label = threshold_to_binary(y_pred, cfg.TEST.THRESHOLD)
    json_data = generate_json_output(y_pred, y_label)
    print(json_data)
    return json_data


def get_inference(img_path, classnames):
    cfg = get_freeze_cfg()
    seed = cfg.SEED
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    model, _ = build_pspg(cfg, classnames)
    if cfg.CHECKPOINT is not None:
        model, epoch = load_model_only(cfg.CHECKPOINT, model)
    else:
        raise FileNotFoundError("please fill the checkpoint path!")
    return inference(img_path, model, cfg)


LABELS = [
    "Pericardial Effusion, Calcification",
    "Cardiac Malposition",
    "Cardiomegaly",
    "No Finding",
    "Pneumothorax",
    "Cavity, Cyst",
    "Mediastinal Lesion",
    "Masses, Nodules",
    "Rib Fracture",
    "Pulmonary Arterial and/or Venous Hypertension",
    "Consolidation",
    "Interstitial lung disease",
    "Pleural Effusion",
    "Pleural Thickening, Adhesions, Calcification",
    "Scoliosis",
    "Clavicular Fracture",
    "Obstructive Atelectasis",
    "Obstructive Emphysema",
]
get_inference("G:/myCXR/datasets/images/1.jpg", LABELS)
