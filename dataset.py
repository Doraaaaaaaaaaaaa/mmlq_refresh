from torch.utils.data import Dataset
import os
import pickle
import socket
import string
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from augmentation import get_tsfms

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings(action="ignore", category=UserWarning)

if socket.gethostname() in ["ntu-webank-gpu01"]:
    DATASET_DIR = "/ssd2/zhiwei"
else:
    DATASET_DIR = None  # 从 config 读取


def add_padding(img):
    img = np.array(img)
    h, w, c = img.shape
    if h > w:
        img2 = np.ones((h, h, c), dtype="uint8") * 255
        start = (h - w) // 2
        img2[:, start : start + w, :] = img
    elif w > h:
        img2 = np.ones((w, w, c), dtype="uint8") * 255
        start = (w - h) // 2
        img2[start : start + h, :, :] = img
    else:
        img2 = img
    img2 = Image.fromarray(img2)
    return img2


def remove_digits(s):
    return "".join(filter(lambda x: not x.isdigit(), s))


def remove_punc(s):
    return s.translate(str.maketrans("", "", string.punctuation))


def _resolve_split_csv(data_dir, split, config):
    dataset_name = config.get("dataset")
    candidates = []

    if dataset_name:
        candidates.append(f"{split}_{dataset_name}.csv")
    candidates.append(f"{split}.csv")

    for filename in candidates:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find {split} split csv in {data_dir}. Tried: {', '.join(candidates)}"
    )


def _resolve_comment_pkl(data_dir):
    candidates = [
        os.path.join(data_dir, "AVA_Comments_Full.pkl"),
        os.path.join(data_dir, "ava_captions", "AVA_Comments_Full.pkl"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find AVA_Comments_Full.pkl. Tried: {', '.join(candidates)}"
    )


def _normalize_image_id(image_id):
    if isinstance(image_id, str):
        return int(image_id.replace(".jpg", ""))
    return int(image_id)


class AVADataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        train_tsfms, test_tsfms = get_tsfms(config)

        data_dir = DATASET_DIR or config["ava_dataset_dir"]

        if split == "train":
            self.df = pd.read_csv(_resolve_split_csv(data_dir, "train", config))
            self.transform = train_tsfms
        elif split == "val":
            self.df = pd.read_csv(_resolve_split_csv(data_dir, "val", config))
            self.transform = test_tsfms
        elif split == "test":
            self.df = pd.read_csv(_resolve_split_csv(data_dir, "test", config))
            self.transform = test_tsfms
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.padding = config["padding"]
        self.images_path = os.path.join(data_dir, "images")
        with open(_resolve_comment_pkl(data_dir), "rb") as f:
            self.caption_dict = pickle.load(f)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int):
        row = self.df.iloc[item]
        image_id_int = _normalize_image_id(row["image_id"])
        image_path = os.path.join(self.images_path, f"{image_id_int}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.padding:
            image = add_padding(image)
        image = self.transform(image)

        caption = " ".join(self.caption_dict.get(image_id_int, ["no comment"]))

        dos = row[
            [
                "score2",
                "score3",
                "score4",
                "score5",
                "score6",
                "score7",
                "score8",
                "score9",
                "score10",
                "score11",
            ]
        ].values.astype("float32")
        dos = dos / dos.sum()

        return {
            "image_id": image_id_int,
            "image": image,
            "caption": caption,
            "dos": dos,
        }
