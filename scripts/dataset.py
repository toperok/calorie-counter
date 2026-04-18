import os
import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

class DishDataset(Dataset):

    def __init__(
            self,
            config,
            transforms,
            ds_type="train"
    ) -> None:
        if ds_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.VAL_DF_PATH)
        self.image_cfd = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

        ingr_df = pd.read_csv(config.INGR_CSV_PATH)
        self.id_to_name = dict(zip(ingr_df["id"].astype(int), ingr_df["ingr"]))

        self.img_dir = config.IMG_DIR
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        row_ingrs = str(row["ingredients"]).split(";") 
        ingr_names = []
        for item in row_ingrs:
            match = re.search(r'(\d+)', item)
            if match:
                clean_id = int(match.group(1))
                ingr_names.append(self.id_to_name[clean_id])
        
        text_ingrs = ", ".join(ingr_names)

        mass = float(row["total_mass"])
        label = float(row["total_calories"])


        img_path = os.path.join(self.img_dir, str(row["dish_id"]), "rgb.png")
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]

        return {
            "label": label,
            "image": image,
            "text": text_ingrs,
            "mass": mass
        }

def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    masses = torch.tensor([item["mass"] for item in batch], dtype=torch.float32)

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=64)
    
    return {
        "label": labels,
        "image": images,
        "mass": masses,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }

def get_transforms(config, ds_type="traim"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        return A.Compose([
            A.Resize(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                fill=0,
                p=0.7
            ),
            A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(int(0.07 * cfg.input_size[1]),
                                       int(0.15 * cfg.input_size[1])),
                    hole_width_range=(int(0.1 * cfg.input_size[2]),
                                      int(0.15 * cfg.input_size[2])),
                    fill=0,
                    p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.7),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2()
        ], seed=config.SEED)
    else:
        return A.Compose([
            A.Resize(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2()
        ])