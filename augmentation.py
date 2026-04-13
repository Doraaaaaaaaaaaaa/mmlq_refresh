from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor

def get_tsfms(config):
    if config["augmentation"] == 0:
        train_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif config["augmentation"] == 1: # add horizontal flip
        train_tsfms = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif config["augmentation"] == 2: # add random crop
        train_tsfms = Compose([
            Resize((272, 272)),
            RandomCrop((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif config["augmentation"] == 3: # add both horizontal flip and random crop
        train_tsfms = Compose([
            Resize((272, 272)),
            RandomCrop((224, 224)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_tsfms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif config["augmentation"] == 4: # inception
        train_tsfms = Compose([
            Resize((299, 299)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_tsfms = Compose([
            Resize((299, 299)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return train_tsfms, test_tsfms