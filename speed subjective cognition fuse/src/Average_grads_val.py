import cv2
import numpy as np
import os

def read_img(img_dir):
    img_data = []
    names = []
    for root, dir, file_names in os.walk(img_dir):
        names = file_names
        for filename in file_names:
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path, 0)
            img_data.append(img)
    return img_data, names

img_dir = "../datasets/test2"
img_data, names = read_img(img_dir)

for index, img in enumerate(img_data):
    laplacian = cv2.Laplacian(img, cv2.CV_64F).reshape(-1)
    # laplacian = [i for i in laplacian if i != 0]
    print(names[index], abs(np.sum(laplacian)))











