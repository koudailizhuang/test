import os
import shutil

fix_path = "E:/Doc.Zheng/data_algorithm/datasets/data-2023-9-21"
save_path = "E:/Doc.Zheng/data_algorithm/datasets/train"

if not os.path.exists(save_path):
	os.mkdir(save_path)   # 如果文件夹不存在自动创建

for i in os.listdir(fix_path):
    if os.path.isdir(os.path.join(fix_path,i)):
        continue
    if os.path.splitext(i)[-1] == ".json":
        print(os.path.splitext(i)[-1])
        shutil.copy(os.path.join(fix_path, i),save_path)
        shutil.copy(os.path.join(fix_path, os.path.splitext(i)[0]+'.JPG'), save_path)
'''
fix_path = 'E:/Doc.Zheng/data_algorithm/datasets/data'
save_path = 'E:/Doc.Zheng/data_algorithm/datasets/train'
filelist = os.listdir(fix_path)
i = 1

for item in filelist:
    if item.endswith('.json'):
        src = os.path.join(os.path.abspath(fix_path), item)
        dst = os.path.join(os.path.abspath(save_path),item)
        new_name = os.path.join(os.path.abspath(save_path),''+str(i)+'.bmp')
       #复制图像
        shutil.copy(src,dst)
       #重命名
        os.rename(dst, new_name)
        i += 1

        print(src)
        print(new_name)
'''
