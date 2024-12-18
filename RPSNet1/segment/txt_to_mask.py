
import os
import cv2
import numpy as np
import math
#读取path中对应的txt文件，并将txt文件中的坐标转换为mask图像,存入savepath路径中。
def txtToImg(path, savepath, imgmask=None):
    files = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            files.append(path + "\\" + file)
            # print(file[:-4])     #每个txt的名称
            # print(type(files))     #list

    # file_abs 是每个txt文件的绝对路径
    for file_abs in files:
        #print(file_abs)  # 每个txt文件的绝对路径
        print(file_abs[59:-4])  # 每个txt文件的名称
        # 按行读取txt文件
        f = open(file_abs)
        line = f.readlines()  # 读出line是str类型

        data=[]
        for row in line:
            tmp_list = row.strip('\n').split(' ')  # 按‘，’切分每行的数据
            # tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            data.append(tmp_list)  # 将每行数据插入data中
        #print(data)
        list=[]
        for lis in data:
            lens=len(lis)
            list1=[]
            for i in range(1,lens-1,2):
                x,y=np.float32(lis[i]),np.float32(lis[i+1])
                a,b=(math.floor(x*640),math.floor(y*200))
                list1.append(a)
                list1.append(b)
            list.append(list1)
        #print(list[0])
        # 生成一张512*512像素照片   其中背景全为1
        print(file_abs[59:-4])
        print(path+"\\"+file_abs[59:-4]+'.JPG')
        img = cv2.imread(path+"\\"+file_abs[59:-4]+'.JPG')
        Img = 255*np.ones((240, 480), dtype=np.uint8)
        img2 = np.zeros_like(img)
        #print(img2.shape)
        img2[:, :, 0] = Img
        img2[:, :, 1] = Img
        img2[:, :, 2] = Img
        # Img = Img * 255  # 使得背景全黑
        #print(list[0])
        #print(5/2)
        #print(len(list[0]))
        #n=int(len(list[0])/2)
        #print(type(n))
        #mask = np.array(list[0]).reshape(1,n,2)

        #Img = cv2.polylines(Img, [mask], True, 0)  # 将坐标点连接起来，线段颜色是0
        for i in range(len(list)):
            n = int(len(list[i]) / 2)
            mask = np.array(list[i]).reshape(1,n,2)
            #print(mask)
            Img = cv2.polylines(img2, [mask], True, (0,0,255),1)  # 将坐标点连接起来，线段颜色是0

            masks = cv2.fillPoly(Img, [mask], color=(0,0,255))

            #cv2.fillConvexPoly(imgmask, mask, color=(0, 255, 0))
            #cv2.addWeighted(img2, 1, imgmask, 0.3, 0, img2)
            #cv2.circle(Img, [mask], 0.01, (0, 0, 255), 3)
        #mask_img = 0.9 * masks + 0.1 * img2
        filepath = savepath + "\\" + file_abs[59:-4] + ".JPG"
        filepath1 = savepath + "\\" + file_abs[59:-4] + "1"+".JPG"
        #cv2.imwrite(filepath1, mask_img)  # 将生成图片a存入路径。
        #print(file_abs[66:-4])
        cv2.imwrite(filepath, Img)  # 将生成图片a存入路径。

        f.close()
        '''
        while line:
            str = line.strip('\n')  # 将每行结尾的\n删除
            list = str.split(' ')  # 将str转换诶list

            results = np.array(list, np.float32)  # 将list转换为numpy.ndarray
            print(results)
            
            mask = results.reshape((-1, 1, 2))  # 对序列第一个数设置为x，第二个数设置为y，依次类推
            print(mask)
            a = cv2.polylines(Img, [mask], True, 0)  # 将坐标点连接起来，线段颜色是0
            #cv2.fillPoly(Img, [mask], color=0)  # 候选框中的范围用像素0填充

            filepath = savepath + "\\" + file_abs[35:-4] + ".jpg"
            cv2.imwrite(filepath, a)  # 将生成图片a存入路径。
            line = f.readline()
        '''


txtToImg(r"D:\algorithm\yolov5-v7.0-2\data\2024-12-04-json-txt\images",r"D:\algorithm\yolov5-v7.0-2\data\2024-12-04-json-txt\masks")

