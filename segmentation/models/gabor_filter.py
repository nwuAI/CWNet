# 创建时间: 2023-10-12 16:03
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn


# def batch_normalization(image):
#     # 计算图像的均值和标准差
#     mean = np.mean(image)
#     std = np.std(image)
#     # 对图像进行归一化
#     normalized_image = (image - mean) / std
#     return normalized_image


# def build_bn_filterssss():
#     bn_filters = []
#     filters = build_filters()
#     for i in range(len(filters)):
#         new_filter =batch_normalization(filters[i])
#         bn_filters.append(new_filter)
#     bn_filters = np.array(bn_filters)
#     bn_filters = torch.tensor(bn_filters)
#     return bn_filters

def get_img(input_path):    # 遍历input_path  获取所有 .jpg  .png后缀文件
    img_paths = []
    for (path, dirs, files) in os.walk(input_path):
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                img_paths.append(path+'/'+filename)
    return img_paths


# 构建Gabor滤波器
def build_filters():
    filters = []
    # ksize = [3, 11]  # gabor尺度，3个
    # lamda = np.pi/2.0         # 波长
    ksize = 3
    # lamda = 1 / math.sqrt(2)    # 波长
    lamda = 0.5
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor方向，共八个
        # print(theta)
        # for k in range(2):
        # cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 1.0, 0, ktype=cv2.CV_32F)
        #  ksize(返回滤波器大小)  sigma(高斯标准差)  theta  lamda(波长) gama(空间纵横比)  psi(相位偏移)
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        # kern /= 1.5*kern.sum()  # 这里   ？？？？？？
        filters.append(kern)
    plt.figure(1)

    # 用于绘制滤波器   画图................................
    # for temp in range(len(filters)):
    #     # plt.subplot(3, 8, temp + 1)
    #     plt.imshow(filters[temp])
    #     plt.show()
    # plt.savefig(str('filter.jpg'), bbox_inches="tight")
    # plt.show()

    return filters


#构建BN后的Gabor滤波器
def build_bn_filters():
    filters = build_filters()
    filters = np.array(filters)
    filters = torch.from_numpy(filters)
    # print(filters.shape)
    filters = torch.unsqueeze(filters, dim=0)
    # print(filters.shape) 【1,8,3,3】
    bn = nn.BatchNorm2d(8)
    # relu = nn.ReLU(inplace=True)
    bn_filters = bn(filters)
    # bn_filters = relu(bn_filters)
    bn_filters = torch.squeeze(bn_filters, dim=0)
    # bn_filters = bn_filters.detach().numpy()
    # bn_filters = np.array(bn_filters)
    return bn_filters


# Gabor特征提取
def getGabor(img, filters):
    res = []  # 滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)    # 创建一个与img同样shape的全零矩阵
        for kern in filters[i]:
            # fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)    # 进行滤波操作  前后形状不变
            fimg = cv2.filter2D(img, -1, kern)    # 进行滤波操作  前后形状不变
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))

    # 用于绘制滤波效果
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(2, 4, temp+1)
        plt.imshow(res[temp], cmap="gray")
    plt.show()
    return res  #


if __name__ == '__main__':
    # img = cv2.imread('C:/Users/Lenovo/Desktop/c.png')
    # filters = build_filters()
    # # print('滤波器', filters[0])
    # getGabor(img, filters)
    # print('**********************************')
    # new_filters = build_bn_filters()
    # # print('new_filters', new_filters[3])
    # # new_filters = np.array(new_filters)
    # # new_filters = torch.tensor(new_filters)
    # print('BN之后滤波器')
    # # new_fil = build_bn_filterssss()
    # # print('new_filters2', new_fil[3])
    # getGabor(img, new_filters)

    # img = cv2.imread('C:/Users/Lenovo/Desktop/d.png')
    img = cv2.imread('../datasets/DRIVE/test/images/01_test.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    plt.show()

    filters = build_filters()
    f = np.array(filters)
    # print("Shape ", f.shape)
    # print(f[2])
    # print(f[4])
    # print(f[6])
    # print("--------------------------------")
    print(f[1])
    print(f[7])
    print("--------------------------------")
    print(f[3])
    print(f[5])
    getGabor(img, filters)
    print('**********************************')
    print('BN之后滤波器')
    new_filters = build_bn_filters()
    f1 = np.array(new_filters)
    # print("Shape ", f1.shape)
    # print(f1[0])
    # print(f1[1])
    # print('new_filters', new_filters[3])
    getGabor(img, new_filters)

