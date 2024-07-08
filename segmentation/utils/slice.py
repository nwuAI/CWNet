import os, sys
import torch
import torch.nn as nn
import numpy as np
from os.path import join
import torch.nn.functional as F

# 创建时间: 2023-11-21 11:16
# 裁剪和拼接图片用的
def slide_inference(model, img, stride=(36, 36), crop_size=(48, 48), num_classes=1, mask=None):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """

    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    # num_classes = self.num_classes

    # 弄一个原尺寸大小的mask来恢复原图？ 全是0的原图大小
    # if h_img < 224 or w_img < 224:
    #     mask = img.new_zeros((batch_size, num_classes, h_img, w_img))
    #     if h_img < 224:
    #         # 图片尺寸小于224x224就将图片补充成224x224
    #         img = F.pad(img, (0, 0, 0, 224 - h_img), 'constant', 0)
    #     if w_img < 224:
    #         img = F.pad(img, (0, 224 - w_img, 0, 0), 'constant', 0)

        # batch_size, _, h_img, w_img = img.size()

    # 填充完之后重新给h_img和w_img赋一下值
    # h_grids和w_grids是控制循环来裁剪拼接图片的
    # //是向下取整
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    # 模型中返回的是六个阶段的特征所以需要有6个preds来分别存储不同阶段的特征
    # clone() 方法用于创建张量的深拷贝，以避免多个列表元素共享相同的内存
    # preds_list = [preds.clone() for _ in range(6)]
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            # crop_seg_logit, feature_map_dict = model(crop_img)
            crop_seg_logit = model(crop_img)
            # F.pad() 是pytorch 内置的 tensor 扩充函数，便于对数据集图像或中间层特征进行维度扩充
            # torch.nn.functional.pad(input, pad, mode=‘constant’, value = 0)
            # input：需要扩充的
            # tensor，可以是图像数据，亦或是特征矩阵数据；
            # pad：扩充维度，预先定义某维度上的扩充参数；
            # mode：扩充方法，有三种模式，分别表示常量（constant），反射（reflect），复制（replicate）；
            # value：扩充时指定补充值，value只在mode = constant有效，即使用value填充在扩充出的新维度位置，而在reflect和replicate模式下，value不可赋值
            # (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2))
            # 分别代表(左边填充数， 右边填充数， 上边填充数， 下边填充数)
            # 这个操作就是将剩余的边缘图和当前的边缘图拼接起来？
            # 返回的是一个数组所以需要对其中的每一个都拼接一下
            preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))


            # count_mat是保存当前位置的元素加和了几次的
            # 之后预测结果/count_mat保证每个位置上的元素权重相等
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(
            count_mat.cpu().detach().numpy()).to(device=img.device)

    preds = preds / count_mat

    return preds