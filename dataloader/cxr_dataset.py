import json
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from .data_aug import RandomAugmentation


def build_transform(dataset_type, cfg):
    lower_type = dataset_type.lower()
    interpolation_method = InterpolationMode[cfg.INTERPOLATION.upper()]
    transform = None
    if cfg.NO_TRANSFORM:
        return transform
    if "train" in lower_type:
        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(img_size)
                transforms.Resize(cfg.SIZE, interpolation=interpolation_method),
                RandomAugmentation(cfg, interpolation_method),
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(cfg.PIXEL_MEAN),
                    tuple(cfg.PIXEL_STD),
                ),
            ]
        )
    elif "val" in lower_type or "test" in lower_type:
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


class MultilabelCXR(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
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
            "Interstitial Lung Disease",
            "Pleural Effusion",
            "Pleural Thickening, Adhesions, Calcification",
            "Scoliosis",
            "Clavicular Fracture",
            "Obstructive Atelectasis",
            "Obstructive Emphysema",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data[index]["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        text = self.data[index]["text"]
        labels = torch.Tensor(self.data[index]["labels"])
        target = labels

        return img, text, target

    def name(self):
        return "MultilabelCXR"


class ChexpertDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data[index]["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        text = self.data[index]["text"]
        labels = torch.Tensor(self.data[index]["labels"])
        target = labels

        return img, text, target

    def name(self):
        return "Chexpert"


class PadChestDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "normal",
            "COPD signs",
            "unchanged",
            "chronic changes",
            "cardiomegaly",
            "aortic elongation",
            "scoliosis",
            "vertebral degenerative changes",
            "interstitial pattern",
            "pleural effusion",
            "air trapping",
            "aortic atheromatosis",
            "costophrenic angle blunting",
            "pneumonia",
            "apical pleural thickening",
            "vascular hilar enlargement",
            "alveolar pattern",
            "infiltrates",
            "laminar atelectasis",
            "fibrotic band",
            "kyphosis",
            "increased density",
            "pacemaker",
            "callus rib fracture",
            "pseudonodule",
            "calcified granuloma",
            "volume loss",
            "nodule",
            "atelectasis",
            "hilar congestion",
            "hemidiaphragm elevation",
            "sternotomy",
            "suboptimal study",
            "NSG tube",
            "hiatal hernia",
            "heart insufficiency",
            "bronchiectasis",
            "vertebral anterior compression",
            "suture material",
            "bronchovascular markings",
            "hilar enlargement",
            "diaphragmatic eventration",
            "endotracheal tube",
            "nipple shadow",
            "central venous catheter via jugular vein",
            "consolidation",
            "metal",
            "emphysema",
            "gynecomastia",
            "calcified densities",
            "goiter",
            "dual chamber device",
            "osteosynthesis material",
            "flattened diaphragm",
            "aortic button enlargement",
            "tracheostomy tube",
            "supra aortic elongation",
            "central venous catheter via subclavian vein",
            "mammary prosthesis",
            "single chamber device",
            "pulmonary mass",
            "pleural thickening",
            "tracheal shift",
            "granuloma",
            "osteopenia",
            "descendent aortic elongation",
            "hypoexpansion",
            "bullas",
            "hyperinflated lung",
            "tuberculosis sequelae",
            "superior mediastinal enlargement",
            "sclerotic bone lesion",
            "lobar atelectasis",
            "pulmonary fibrosis",
            "mediastinic lipomatosis",
            "rib fracture",
            "hypoexpansion basal",
            "azygos lobe",
            "vascular redistribution",
            "mastectomy",
            "surgery neck",
            "central venous catheter",
            "minor fissure thickening",
            "ground glass pattern",
            "calcified adenopathy",
            "dai",
            "adenopathy",
            "pulmonary edema",
            "artificial heart valve",
            "reservoir central venous catheter",
            "mediastinal enlargement",
            "axial hyperostosis",
            "cavitation",
            "non axial articular degenerative changes",
            "pneumothorax",
            "pectum excavatum",
            "vertebral compression",
            "calcified pleural thickening",
            "humeral fracture",
            "multiple nodules",
            "exclude",
            "surgery breast",
            "costochondral junction hypertrophy",
            "clavicle fracture",
            "vertebral fracture",
            "lung metastasis",
            "osteoporosis",
            "mediastinal mass",
        ]
        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data[index]["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        text = self.data[index]["text"]
        labels = torch.Tensor(self.data[index]["labels"])
        target = labels

        return img, text, target

    def name(self):
        return "PadChest"


class ChestXray14Dataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Fibrosis",
            "Consolidation",
            "Emphysema",
            "No Finding",
            "Infiltration",
            "Mass",
            "Effusion",
            "Pneumothorax",
            "Edema",
            "Hernia",
            "Atelectasis",
            "Pleural Thickening",
            "Pneumonia",
            "Cardiomegaly",
            "Nodule",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data[index]["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        text = self.data[index]["text"]
        labels = torch.Tensor(self.data[index]["labels"])
        target = labels

        return img, text, target

    def name(self):
        return "ChestXray"


class VindrCXRDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Aortic enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Clavicle fracture",
            "Consolidation",
            "Emphysema",
            "Enlarged PA",
            "Interstitial lung disease",
            "Infiltration",
            "Lung Opacity",
            "Lung cavity",
            "Lung cyst",
            "Mediastinal shift",
            "Nodule/Mass",
            "Pleural effusion",
            "Pleural thickening",
            "Pneumothorax",
            "Pulmonary fibrosis",
            "Rib fracture",
            "Other lesion",
            "COPD",
            "Lung tumor",
            "Pneumonia",
            "Tuberculosis",
            "Other disease",
            "No finding",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.data[index]["filepath"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        text = self.data[index]["text"]
        labels = torch.Tensor(self.data[index]["labels"])
        target = labels

        return img, text, target

    def name(self):
        return "Vindr"
