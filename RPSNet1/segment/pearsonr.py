import numpy as np
import pandas as pd
import seaborn as sns
import os, cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image


def load_and_flatten_images(folder_path):
    flattened_images = []
    filenames = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            # 读取图像，提取第一通道
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)  # 读取彩色图像
            if img is not None:
                flattened_images.append(img.flatten())  # 展平并添加到列表
                filenames.append(filename)

    return np.array(flattened_images)

# 1. 加载训练数据
def load_training_data(label_paths):
    labels = [np.array(Image.open(p)).flatten() for p in label_paths]  # 分割标签（掩码）
    return np.array(labels)

# 2. 计算皮尔逊相关系数
def compute_pearson_correlation(images, labels):
    # 将图像和标签拼接以计算相关性
    data = np.hstack((images, labels))
    correlation_matrix = np.corrcoef(data, rowvar=False)
    return correlation_matrix

# 3. 绘制热图
def plot_heatmap(correlation_matrix, features):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        xticklabels=features,
        yticklabels=features
    )
    plt.title("Correlation Heatmap")
    plt.show()

# 示例数据加载
#image_paths = ["path_to_image1.jpg", "path_to_image2.jpg", "path_to_image3.jpg"]  # 图像路径
#label_paths = ["path_to_mask1.png", "path_to_mask2.png", "path_to_mask3.png"]    # 标签路径
image_paths = 'D:/algorithm/yolov5-v7.0-2/data/2024-12-04-json-txt/images50'
label_paths = 'D:/algorithm/yolov5-v7.0-2/data/2024-12-04-json-txt/masks50'
# 归一化处理
scaler = MinMaxScaler()

# 加载数据
images = load_and_flatten_images(image_paths)
labels = load_and_flatten_images(label_paths)
images = scaler.fit_transform(images)  # 对图像数据归一化
labels = scaler.fit_transform(labels)  # 对标签数据归一化

# 计算相关性
features = [f'Pixel_{i}' for i in range(images.shape[1])] + ['Label']
correlation_matrix = compute_pearson_correlation(images, labels)

# 绘制热图
plot_heatmap(correlation_matrix, features)
