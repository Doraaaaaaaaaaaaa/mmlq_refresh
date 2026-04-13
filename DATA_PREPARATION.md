# 数据集准备指南

## 需要的数据集

### 1. AVA (Aesthetic Visual Analysis) 数据集

#### 1.1 AVA图像和评分
- **下载地址**: http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460
- **内容**:
  - 约255,000张图像
  - 每张图像的美学评分分布（1-10分）

#### 1.2 AVA Comments (文本评论)
- **GitHub仓库**: https://github.com/V-Sense/Aesthetic-Image-Captioning-ICCVW-2019
- **重要**: 作者使用的是**清洗之前**的原始数据
- **文件**: `AVA_Comments_Full.txt`

## 数据集目录结构

```
../datasets/AVA/
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── ava_captions/
│   ├── AVA_Comments_Full.txt      # 原始评论文件
│   └── AVA_Comments_Full.pkl      # 处理后的pickle文件（运行脚本生成）
├── train_hlagcn.csv               # 训练集划分
└── test_hlagcn.csv                # 测试集划分
```

## 数据准备步骤

### 步骤1: 下载AVA图像和评分

1. 从 http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460 下载
2. 解压到 `../datasets/AVA/images/`

### 步骤2: 下载AVA Comments

1. 克隆仓库:
```bash
git clone https://github.com/V-Sense/Aesthetic-Image-Captioning-ICCVW-2019.git
```

2. 找到 `AVA_Comments_Full.txt` 文件
3. 复制到 `../datasets/AVA/ava_captions/AVA_Comments_Full.txt`

**重要**: 使用原始的 `AVA_Comments_Full.txt`，不要使用清洗后的版本

### 步骤3: 处理评论数据

运行清洗脚本生成pickle文件:
```bash
python clean_comment_json.py
```

这会生成 `AVA_Comments_Full.pkl` 文件

### 步骤4: 准备训练/测试划分CSV

需要包含以下列的CSV文件:
- `image_id`: 图像ID
- 列2-12: 评分1到10的投票数

### 步骤5: 修改数据路径

编辑 `dataset.py` 第14-17行，修改 `DATASET_DIR` 为你的数据集路径:
```python
DATASET_DIR = "../datasets"  # 改为你的路径
```
