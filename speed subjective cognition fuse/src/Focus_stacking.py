import os
import cv2
import numpy as np
import time as time
import numba as nb

#三通道均值滤波
@nb.jit(nopython=True)
def jit_ave_filter_ch3(img, out_img, kernel=3):
    w, h = img.shape[:2]
    k = (kernel-1)//2
    # 使用for循环，为了实现numba加速
    for n in range(3):
        for i in range(w):
            for j in range(h):
                # out_img[i, j, n] = \
                #     np.average(img[max(0, i - k):min(w - 1, i + k), max(0, j - k):min(h - 1, j + k), n])
                out_img[i, j, n] = \
                    np.sum(img[max(0, i-k):min(w+1, i+k+1), max(0, j-k):min(h+1, j+k+1), n])\
                    //((min(w+1, i+k+1)-max(0, i-k))*(min(h+1, j+k+1)-max(0, j-k)))
    return out_img

#单通道均值滤波
@nb.jit(nopython=True)
def jit_ave_filter_ch1(img, out_img, kernel=3):
    w, h = img.shape[:2]
    k = (kernel-1)//2
    # 使用for循环，为了实现numba加速
    for i in range(w):
        for j in range(h):
            # out_img[i, j] = \
            #     np.average(img[max(0, i - k):min(w, i + k), max(0, j - k):min(h, j + k)])
            out_img[i, j] = \
                np.sum(img[max(0, i-k):min(w+1, i+k+1), max(0, j-k):min(h+1, j+k+1)])\
                //((min(w+1, i+k+1)-max(0, i-k))*(min(h+1, j+k+1)-max(0, j-k)))
    return out_img

#图片平滑滤波封装，内部使用均值滤波
def smooth_filter(img, channels = 3, kernel = 3):
    if channels == 3:
        w, h, chan = img.shape
        tem_img = np.zeros([w, h, channels], dtype=np.uint8)
        out_img = jit_ave_filter_ch3(img, tem_img, kernel=kernel)
    elif channels == 1:
        w, h = img.shape[:2]
        tem_img = np.zeros([w, h], dtype=np.uint8)
        out_img = jit_ave_filter_ch1(img, tem_img, kernel=kernel)
    return out_img

#最大值滤波，单通道，加速实现
@nb.jit(nopython=True)
def maxcount_filter1(idx_tab, out_idx, cont, kernel=5, idx_mx=7):
    w, h = idx_tab.shape[:2]
    k = (kernel-1)//2
    # 使用for循环，为了实现numba加速
    for i in range(w):
        for j in range(h):
            aw, ah = (idx_tab[max(0, i - k):min(w + 1, i + k + 1), max(0, j - k):min(h + 1, j + k + 1)]).shape[:2]
            for p in range(aw):
                for q in range(ah):
                    for m in range(1, idx_mx+1):
                        if m == (idx_tab[max(0, i - k):min(w + 1, i + k + 1), max(0, j - k):min(h + 1, j + k + 1)])[p][q]:
                            cont[m] += 1
            out_idx[i, j] = np.argmax(cont)
            for idx, ceil in enumerate(cont):
                if ceil != 0:
                    cont[idx] = 0
    return out_idx

# creat fuse index table
def idx_table(img_ser, w=1920, h=1080):
    # 这里求区域方差来确定以代替出现最大次数滤波，说不定会更好
    idx_tab = np.argmax(img_ser, axis=0)
    idx_tab = np.array(idx_tab, dtype=np.uint8)

    idx_tab = smooth_filter(idx_tab, channels=1, kernel=5)
    idx_tab = cv2.resize(idx_tab, (h, w))
    idx_tab = smooth_filter(idx_tab, channels=1, kernel=3)

    idx_tab = idx_tab.transpose((1, 0))
    return idx_tab

#根据索引表融合图像
@nb.jit(nopython=True)
def fuse2_img(img_data, idx_tab, img_out, mx):
    for i in range(idx_tab.shape[0]):
        for j in range(idx_tab.shape[1]):
            img_out[i, j, 0] = img_data[idx_tab[i][j], i, j, 0]
            img_out[i, j, 1] = img_data[idx_tab[i][j], i, j, 1]
            img_out[i, j, 2] = img_data[idx_tab[i][j], i, j, 2]
            # idx_tab visualization
            idx_tab[i, j] = int((idx_tab[i, j]/mx)*255)
    return img_out, idx_tab

# image fuse
def idxtab_fuse2_img(img_data, idx_tab, w = 1920, h = 1080):
    img_out = np.zeros([h, w, 3], dtype=np.uint8)
    mx = np.max(np.max(idx_tab, axis=0))
    img_out, idx_tab = fuse2_img(img_data, idx_tab, img_out, mx)
    return img_out, idx_tab

if __name__ == "__main__":
    img_data = []
    for root, dirs, file_names in os.walk("../datasets/test2/"):
        num = len(file_names)
        for i in range(num):
            pic_path = os.path.join(root, file_names[i])
            image0 = cv2.imread(pic_path)
            img_data.append(image0)
    img_data = np.array(img_data, dtype=np.uint8)
    print(img_data.shape)

    img_ser = np.zeros([len(img_data), 480, 270], dtype=np.uint8)
    for i in range(len(img_data)):
        image = cv2.resize(img_data[i], (480, 270)).transpose((1, 0, 2))
        image = np.sum(image, axis=2)
        #maxcount_filter1
        cont = np.zeros([255 + 1])
        out_image = np.zeros_like(image, dtype=np.uint8)
        image = maxcount_filter1(image, out_image, cont, kernel=3, idx_mx=255)
        img_ser[i, :, :] = image

    start_time1 = time.time()
    idx_tab = idx_table(img_ser)
    img, idx = idxtab_fuse2_img(img_data, idx_tab)
    print("time:", time.time() - start_time1)

    cv2.imwrite("../idx_test_22.jpg", idx)
    cv2.imwrite("../fiber_test_22.jpg", img)

