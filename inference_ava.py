import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from model import MultiModalQueryNetwork as MMQN
from dataset import AVADataset
from tqdm import tqdm
# from metrics import cal_metrics
from scipy.stats import pearsonr, spearmanr
from dataset import add_padding, remove_digits, remove_punc
import numpy as np
import pickle
from augmentation import get_tsfms
from loss import EMDLoss
import random
import math
from metrics import cal_metrics

DATASET_DIR="/home/zhiwei/datasets/AVA"

def main():
    checkpoint_path = "/hdd1/zhiwei/mmiaa/save/2023-12-17-01-38-13/checkpoint_4.pt"
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]
    device = torch.device("cuda:0")
    config["inference_comment"] = 1.0

    model = MMQN(config=config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = AVADataset(config, "test")

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=12
    )

    y_pred = []
    y_true = []
    image_ids = []
    with torch.no_grad():
        for batch_idx, samples in tqdm(enumerate(test_loader)):
            samples["image"] = samples["image"].to(device)
            samples["dos"] = samples["dos"].to(device)
            image_ids.extend(samples["image_id"].tolist())

            with autocast():
                preds = model(samples)

            y_pred.append(preds)
            y_true.append(samples["dos"])
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        y_true = torch.cat(y_true, dim=0).cpu().numpy()

    mos_pred = np.dot(y_pred, np.arange(1, 11))
    mos = np.dot(y_true, np.arange(1, 11))
    
    results = []
    for i in range(len(image_ids)):
        emd = EMDLoss(dist_r=1)(torch.from_numpy(y_pred[i]).unsqueeze(0), torch.from_numpy(y_true[i]).unsqueeze(0))
        cur_result = [image_ids[i], emd.item(), mos_pred[i], mos[i]] + list(y_pred[i]) + list(y_true[i])
        results.append(cur_result)

    column_names = ["image", "emd", "mos_pred", "mos"] + [f"dos_pred_{i}" for i in range(1, 11)] + [f"dos_{i}" for i in range(1, 11)]
    results_df = pd.DataFrame(results, columns=column_names)
    results_df.to_csv("inference_ava.csv", index=False)

    metrics = cal_metrics(y_pred, y_true)
    print(f"{[round(m, 3) for m in metrics]}")

class AVADataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        train_tsfms, test_tsfms = get_tsfms(config)
        if split == "train":
            self.df = pd.read_csv(os.path.join(DATASET_DIR, f"train_{config['dataset']}.csv"))
            self.transform = train_tsfms
        elif split == "test":
            self.df = pd.read_csv(os.path.join(DATASET_DIR, f"test_{config['dataset']}.csv"))
            self.transform = test_tsfms
        self.padding = config["padding"]
        self.images_path=os.path.join(DATASET_DIR, "images")
        if isinstance(config["inference_comment"], float):
            with open(os.path.join(DATASET_DIR, "ava_captions/AVA_Comments_Full.pkl"), "rb") as f:
                self.caption_dict = pickle.load(f)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) :
        row = self.df.iloc[item]
        image_id = int(row["image_id"])
        image_path = f"{self.images_path}/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        if self.padding:
            image = add_padding(image)
        image = self.transform(image)

        if isinstance(self.config["inference_comment"], float):
            all_captions = self.caption_dict[image_id]
            num_captions = len(all_captions)
            indices = random.sample(range(num_captions), math.ceil(num_captions * self.config["inference_comment"]))
            all_captions = [all_captions[i] for i in sorted(indices)]
        elif isinstance(self.config["inference_comment"], str):
            all_captions = [self.config["inference_comment"]]

        caption = " ".join(all_captions)

        dos = (row[2:12].values/row[2:12].sum()).astype("float32")

        return {
            "image_id": image_id,
            "image": image,
            "caption": caption,
            "dos": dos
        }

if __name__ == "__main__":
    main()