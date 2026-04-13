#!/bin/bash
# 快速环境配置脚本

echo "=== MMLQ-IAA 环境配置 ==="

# 1. 创建conda环境
echo "步骤1: 创建conda环境..."
conda create -n mmlq python=3.9 -y
conda activate mmlq

# 2. 安装PyTorch (CUDA 11.8)
echo "步骤2: 安装PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装LAVIS
echo "步骤3: 克隆并安装LAVIS..."
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
cd ..

# 4. 安装其他依赖
echo "步骤4: 安装其他依赖包..."
pip install -r requirements.txt

# 5. 验证安装
echo "步骤5: 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from lavis.models.clip_vit import create_clip_vit_L; print('LAVIS: OK')"

echo "=== 环境配置完成 ==="
echo "使用 'conda activate mmlq' 激活环境"
