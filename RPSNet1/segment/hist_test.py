# 导入所需要的库
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

img = cv2.imread('D:/algorithm/yolov5-v7.0-2/data/output_images_BMP1/yolov10/mask/Camera MV-SUA33GM#0001-0003-Snapshot-20231226104702-9952342697858.JPG')

def count_red_pixels_per_column(image_path, red_threshold=150):
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    # 初始化一个列表来存储每列的红色像素计数
    red_pixel_counts = [0] * width
    signal_maps = [0] * width
    # 遍历图像的每一列
    for x in range(width):
        # 遍历该列的每个像素
        for y in range(height):
            # 获取该像素的RGB值
            r, g, b = image.getpixel((x, y))
            # 如果红色分量超过阈值，则计数增加
            if r > red_threshold and g < 80 and b < 80:
                red_pixel_counts[x] += 1
        if red_pixel_counts[x]>=5:
            signal_maps[x] += 1

    return signal_maps
img_path='D:/algorithm/yolov5-v7.0-2/data/output_images_BMP1/yolov10/mask/Camera MV-SUA33GM#0001-0003-Snapshot-20231226104702-9952342697858.JPG'
signals=count_red_pixels_per_column(img_path)
x = np.linspace(0, 639, 640)

y=np.array(signals)
_ = plt.ylim(-0.2, 1.2)
plt.rcParams['xtick.direction'] = 'in' # 将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in' # 将y轴的刻度线方向设置向内

font = {'family' : 'Times New Roman','weight' : 'normal','size': 12}
plt.xlabel("cutting direction/pixel",font)
plt.ylabel("signal",font)
plt.xticks(np.arange(0, 640, 10))
plt.yticks(np.arange(0, 1, 1))


plt.plot(x, y,'b-',linewidth=1.0)

plt.savefig("信号图.jpg", dpi=800)
#plt.grid()
plt.show()
output_path = img_path[:-4]+'.xlsx'
data = {'X': x, 'Y': y}
df = pd.DataFrame(data)
df.to_excel(output_path, index=False)
