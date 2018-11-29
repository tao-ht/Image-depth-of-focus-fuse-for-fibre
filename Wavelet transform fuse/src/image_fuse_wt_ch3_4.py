import pywt
import numpy as np
from PIL import Image
import cv2
import time
import os
from numba import vectorize

@vectorize(["float32(float32, float32)"], target='cuda')
def varianceWeight(list1):
    ech = []
    for i in list1:
        ech.append(i.var())
    res = [k/sum(ech) for k in ech]
    return res

# def getVarianceImg(array):
#     k = 3
#     row, col = array.shape
#     varImg = np.zeros((row, col))
#     for i in range(row):
#         for j in range(col):
#             up = i-k if i-k > 0 else 0
#             down = i+k if i+k < row else row
#             left = j-k if j-k > 0 else 0
#             right = j+k if j+k < col else col
#             window = array[up:down, left:right]
#             mean, var = cv2.meanStdDev(window)
#             varImg[i, j] = var
#             return varImg

def testWave(img_ch, wavelet1 = "sym4", lev = 5, w=3): # haar、db4、sym4、bior2.4
    transf = []
    for i in range(len(img_ch)):
        transf.append(pywt.wavedec2(img_ch[i], wavelet1, level=lev))
        if i >= 1:
            assert len(transf[i]) == len(transf[0])

    recWave = []

    for k in range(len(transf[0])):
        cvtArray, cvtArray1, cvtArray2 = [], [], []
        # 低频分量
        if k == 0:
            list1 = [transf[low][0] for low in range(len(transf))]
            coe = varianceWeight(list1)

            lowFreq = np.zeros(transf[0][0].shape)
            row, col = transf[0][0].shape
            for i in range(row):
                for j in range(col):
                    lowFreq[i, j] = sum([coe[m] * transf[m][0][i, j] for m in range(len(transf))])
            recWave.append(lowFreq)
            continue

        # 高频分量
        for array1, array2, array3, array4, array5, array6 in zip(
            transf[0][k],transf[1][k],transf[2][k],transf[3][k],transf[4][k],transf[5][k]):

            array = [array1, array2, array3, array4, array5, array6]

            tmp_row, tmp_col = array[0].shape
            highFreq = np.zeros((tmp_row, tmp_col))

            sum_arr = [sum(map(sum, abs(array[n]))) for n in range(len(array))]
            idx = np.argsort(sum_arr)

            for i in range(tmp_row):
                for j in range(tmp_col):
                    highFreq[i, j] = (array[idx[-1]][i, j] + array[idx[-2]][i, j] + array[idx[-3]][i, j]) / w
            cvtArray.append(highFreq)

        recWave.append(tuple(cvtArray))

    return pywt.waverec2(recWave, wavelet1)

def BGR_2_YIQ(img):
    B, G, R = cv2.split(img)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B
    img = cv2.merge([Y, I, Q])
    return img

def YIQ_to_BGR(Y, I, Q):
    # Y, I, Q = cv2.split(img)
    R = Y + 0.956 * I + 0.624 * Q
    G = Y - 0.272 * I - 0.647 * Q
    B = Y - 1.106 * I + 1.703 * Q
    img = cv2.merge([B, G, R])
    return img

def read_img_dir(file_path):
    img_dir = []
    for root, dirs, file_names in os.walk(file_path):
        for file_name in file_names:
            image_path = os.path.join(root, file_name)
            img_dir.append(image_path)
    return img_dir

if __name__ == '__main__':
    start_time0 = time.time()
    img_dir = read_img_dir("../test/")
    img_data = []
    print(img_dir)
    for dir in img_dir[:6]:
        print(dir)
        img = cv2.imread(dir)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        img = BGR_2_YIQ(img)
        img_data.append(img)
    start_time1 = time.time()
    h, w = img_data[0].shape[:2]
    print(img_data[0].shape)

    Y = [img_data[y][:, :, 0] for y in range(len(img_data))]
    I = [img_data[i][:, :, 1] for i in range(len(img_data))]
    Q = [img_data[q][:, :, 2] for q in range(len(img_data))]
    print(len(Y), Y[0].shape)

    # haar、db4、sym4、bior2.4
    model = ["coif1",  "coif2", "coif3", "coif4", "coif5",
             "sym2", "sym6", "sym10", "sym14", "sym18", "sym20",
             "db1", "db5", "db9", "db13", "db17", "db20",
             "bior1.1", "bior2.2", "bior3.3", "bior4.4", "bior5.5", "bior6.8",
             "rbio1.1", "rbio2.2", "rbio3.3", "rbio4.4", "rbio5.5", "rbio6.8"]#"sym4", , "bior2.4", "haar", "db4", "db10",
    w1 = [i/10 for i in range(29, 30)]
    print(w1)
    for i in model:
        for j in w1:
            for l in [3]:
                Y1 = testWave(Y, wavelet1=i, lev=int(l), w=int(j))
                I1 = testWave(I, wavelet1=i, lev=int(l), w=int(j))
                Q1 = testWave(Q, wavelet1=i, lev=int(l), w=int(j))

                img = YIQ_to_BGR(Y1, I1, Q1)
                img_save_path = "../result/test8_" + i + "_w" + str(j) + "_" + str(l) + ".png"
                cv2.imwrite(img_save_path, img)

                print("time ; %.4f s" % (time.time() - start_time1))







