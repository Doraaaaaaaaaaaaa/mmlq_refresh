"""环境和数据检查脚本"""
import sys, yaml, torch

print("=== 环境 ===")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("显存:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")

print("\n=== 依赖 ===")
import transformers; print("transformers:", transformers.__version__)
import scipy; print("scipy:", scipy.__version__)
import tqdm; print("tqdm:", tqdm.__version__)
import tensorboard; print("tensorboard: OK")
import PIL; print("Pillow:", PIL.__version__)

print("\n=== 配置 ===")
with open("config.yml", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("batch_size:", config["batch_size"])
print("epochs:", config["epochs"])
print("lr:", config["lr"])
print("accum_steps:", config["accum_steps"])

print("\n=== 数据集 ===")
from dataset import AVADataset
train_ds = AVADataset(config, "train")
test_ds  = AVADataset(config, "test")
print("train size:", len(train_ds))
print("test size:", len(test_ds))
sample = train_ds[0]
print("image shape:", sample["image"].shape)
print("dos shape:", sample["dos"].shape)
print("caption[:80]:", sample["caption"][:80])

print("\n=== 模型 ===")
from model import ImprovedIAAModel
model = ImprovedIAAModel(config)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量:     {total/1e6:.1f}M")
print(f"可训练参数量: {trainable/1e6:.1f}M")

print("\n✅ 全部检查通过，可以开始训练！")
