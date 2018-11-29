import cv2
import numpy as np
import math
import os

def read_img(img_dir):
    img_data = []
    names = []
    for root, dir, file_names in os.walk(img_dir):
        names = file_names
        for filename in file_names:
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            img_data.append(img)
    return img_data, names
"""
计算图片的信息熵，根据香农信息熵计算公式，三通道
"""
img_dir = "./result"
img_data, names = read_img(img_dir)
img_data = [np.bincount((np.array(img)).reshape(-1))/len((np.array(img)).reshape(-1))
            for img in img_data]
for idx, tmp in enumerate(img_data):
    res = 0
    for i in range(len(tmp)):
        if tmp[i] !=0:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    print(names[idx], res)

"""
example:
fiber_test_22.jpg 5.705488417214355
"""





