import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files


#                datasets/DRIVE       DRIVE           48                    6
# data_process(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")
def data_process(data_path, name, patch_size, stride, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "ROSE":
        img_path = os.path.join(data_path, mode, "img")
        gt_path = os.path.join(data_path, mode, "gt")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []

    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "ROSE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "CHASEDB1":
            if len(file) == 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))

    img_list = normalization(img_list)   # 可能在这添加预处理*******************

    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)
        save_patch(img_patch, save_path, "img_patch", name)
        save_patch(gt_patch, save_path, "gt_patch", name)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name)
        save_each_image(gt_list, save_path, "gt", name)


def get_square(img_list, name):       # 把img_list中的图片按照数据集名称进行填充操作
    img_s = []
    if name == "DRIVE":  # (584×565)
        shape = 592
    elif name == "ROSE":  # (304x304)
        shape = 304
    elif name == "CHASEDB1":   # (999×960)    STARE (700x605)
        shape = 1008
    elif name == "DCA1":
        shape = 320
    elif name == "START":
        shape = 702
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)  # 左右上下顺序填充0   原图在左上角
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):  # 让图像完全按照 大小为patch_size 步长为stride划分  没有遗漏
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


# save_patch(img_patch, save_path, "img_patch", name)
def save_patch(imgs_list, path, type, name):   # 和下面 save_each_image 方法一样的
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):   # 将图片归一化
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/CHASEDB1", type=str,
                        help='the path of dataset')
    parser.add_argument('-dn', '--dataset_name', default="CHASEDB1", type=str,
                        help='the name of dataset', choices=['DRIVE', 'CHASEDB1', 'STARE', 'CHUAC', 'DCA1', 'ROSE'])
    parser.add_argument('-ps', '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=6,
                        help='the stride of image partition')
    args = parser.parse_args()
    with open('config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  # 为列表类型

    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "training")
    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "test")

    print('end')
