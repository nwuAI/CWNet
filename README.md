## CWNet: Convolution and Wavelet Cooperated CNN-Transformer architecture for Retinal Vessel Segmentation

by Zhan Li, Jianjun Zeng, Yuning Wang, Yisong Han, Chengpeng Nie, Songdi Wu, Qinru Sun


* We provide related codes and configuration files to reproduce the "CWNet: Convolution and Wavelet Cooperated CNN-Transformer architecture for Retinal Vessel Segmentation"

## Introduction
Our paper propose a Convolution and Wavelet Cooperated CNN-transformer architecture for Retinal Vessel Segmentation (CWNet) by integrating convolution, wavelet and transformer together. The
CWNet employs a dual-branch encoder ,and a Transformer as the bottleneck, to capture the
multi-directional features and the global features, respectively.
In addition, we design two modules, the Convolution and
Wavelet Fusion module (CWF) and the Feature Enhancement
module (FE), to better fuse the features of the two encoders
and enhance the decoder’s feature expression ability.

<div align="center">
  <img src="figures/framework.png" width="600" />
</div>


## Citation
```
@article{li2024can,
  title={CWNet: Convolution and Wavelet Cooperated CNN-Transformer architecture for Retinal Vessel Segmentation},
  author={Li, Zhan and Zhang, Chunxia and Zhang, Yongqin and Wang, Xiaofeng and Ma, Xiaolong and Zhang, Hai and Wu, Songdi},
  journal={},
  pages={},
  year={},
  publisher={IEEE}
}
```

## Data Preprocessing
```
python data_process.py -dp DATASET_PATH -dn DATASET_NAME
```

## Train
If you are prompted for no packages, enter pip install * * to install dependent packages
```
python train.py -dp DATASET_PATH
```

## Test
```
python test.py -dp DATASET_PATH -wp WEIGHT_FILE_PATH
```
