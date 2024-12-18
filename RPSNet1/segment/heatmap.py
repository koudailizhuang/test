import os
import cv2
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MaxNLocator

# Step 1: 读取图像并预处理
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

# Step 2: 计算图像间的皮尔逊相关系数
def calculate_pearson_correlation(images):
    # 将每张图像展平（flatten）成一维数组，便于进行相关性计算
    flattened_images = [img.flatten() for img in images]
    flattened_images = np.array(flattened_images)

    # 对每个图像进行标准化
    scaler = StandardScaler()
    flattened_images = scaler.fit_transform(flattened_images)

    # 计算皮尔逊相关系数矩阵
    correlation_matrix = np.corrcoef(flattened_images)
    return correlation_matrix

# Step 3: 绘制热图
def plot_correlation_heatmap(correlation_matrix, filenames):
    path = 'D:/algorithm/yolov5-v7.0-2/data/2024-12-04-json-txt/heatmaps40.png'
    paths = Path(path)
    labels = [f"{i}.jpg" for i in range(1, 41)]
    plt.figure(figsize=(8, 7))
    #x_major_locator = MultipleLocator(2)
    sns.set(font='Times New Roman')
    ax = sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", xticklabels=filenames, yticklabels=filenames,annot_kws={"size": 18, "family": "Times New Roman"})
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True, step=2))
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True, step=2))
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize=10.5)  # heatmap 刻度字体大小 10

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # 设置x轴的间隔
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.title('Correlation Heatmap of Images', loc='center', pad=20, fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlabel('Image names', labelpad=1, fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Image names', labelpad=1, fontdict={'family': 'Times New Roman', 'size': 18})
    #plt.xticks(fontproperties='Times New Roman', size=12, rotation=90)  # 让x轴的文件名竖直显示，防止重叠
    #plt.xticks(np.arange(0, correlation_matrix.shape[1], 2))  # x轴每隔2个位置显示一个刻度
    #plt.yticks(fontproperties='Times New Roman', size=12, rotation=0)   # y轴的文件名水平显示
    #plt.xticks([])
    #plt.yticks([])
    f = paths.with_suffix('.tif')  # filename
    plt.savefig(f, dpi=1000, bbox_inches='tight')
    plt.show()

# 主程序
folder_path = 'D:/algorithm/yolov5-v7.0-2/data/2024-12-04-json-txt/images40'  # 将此路径替换为你的文件夹路径
images, filenames = load_images_from_folder(folder_path)
correlation_matrix = calculate_pearson_correlation(images)
plot_correlation_heatmap(correlation_matrix, filenames)

