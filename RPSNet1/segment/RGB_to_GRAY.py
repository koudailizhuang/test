import cv2
img_path='D:/algorithm/yolov5-v7.0-2/data/5.JPG'
filepath1 = img_path[:-4] + "_gray" +".JPG"
image = cv2.imread(img_path)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(filepath1, img_gray)  # 将生成图片a存入路径。
