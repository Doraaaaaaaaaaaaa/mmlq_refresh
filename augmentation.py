from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor

# CLIP ViT-L/14 的标准化参数
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def get_tsfms(config):
    if config["augmentation"] == 0:
        train_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
    elif config["augmentation"] == 1: # add horizontal flip
        train_tsfms = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
    elif config["augmentation"] == 2: # add random crop
        train_tsfms = Compose([
            Resize((272, 272)),
            RandomCrop((224, 224)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
    elif config["augmentation"] == 3: # add both horizontal flip and random crop
        train_tsfms = Compose([
            Resize((272, 272)),
            RandomCrop((224, 224)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
    elif config["augmentation"] == 4: # inception
        train_tsfms = Compose([
            Resize((299, 299)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
        test_tsfms = Compose([
            Resize((299, 299)),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])

    return train_tsfms, test_tsfms