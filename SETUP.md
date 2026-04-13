# 环境配置指南 - MMLQ-IAA

## 1. 创建虚拟环境

### 使用 conda (推荐)
```bash
# 创建Python 3.9环境
conda create -n mmlq python=3.9
conda activate mmlq
```

### 或使用 venv
```bash
python -m venv mmlq_env
# Windows
mmlq_env\Scripts\activate
# Linux/Mac
source mmlq_env/bin/activate
```

## 2. 安装PyTorch

**重要**: 先根据你的CUDA版本安装PyTorch

### 检查CUDA版本
```bash
nvidia-smi
```

### 安装PyTorch (根据CUDA版本选择)
```bash
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CPU版本 (不推荐，训练会很慢)
pip install torch torchvision
```

## 3. 安装LAVIS (BLIP-2)

**重要**: 必须从官方GitHub仓库安装

```bash
# 克隆LAVIS仓库
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS

# 安装LAVIS
pip install -e .

# 返回项目目录
cd ..
```

## 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

## 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from lavis.models.clip_vit import create_clip_vit_L; print('LAVIS OK')"
```

## 5. 准备数据集

需要下载AVA数据集：
- 图像文件
- 评分CSV文件
- 评论文件 (AVA_Comments_Full.pkl)

修改 `dataset.py` 中的 `DATASET_DIR` 路径指向你的数据集位置。

## 6. 开始训练
```bash
python main.py
```
