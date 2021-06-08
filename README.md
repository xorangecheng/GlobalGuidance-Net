# GG-Net
by Cheng Xue, Lei Zhu, Huazhu Fu, Xiaowei Hu, Xiaomeng Li, Hai Zhang, Pheng Ann Heng


## Introduction

This repository is for our MIA 2021 paper '[Global guidance network for breast lesion segmentation in ultrasound images](https://arxiv.org/abs/2104.01896.pdf)'.
![](figure/arc.pdf)
## Requirement
* Python 
* PyTorch
* torchvision
* numpy
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)
* training set and testing set (you need to prepare them by yourself and groundtruths should be binary masks)
* 
## Training
Train the model with:
   ```shell
   python train.py
   ```
The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by [Zijun Deng](https://github.com/zijundeng/DAF).
   
## Citation
If this repository is useful for your research, please consider citing:
```
@article{xue2021global,
  title={Global guidance network for breast lesion segmentation in ultrasound images},
  author={Xue, Cheng and Zhu, Lei and Fu, Huazhu and Hu, Xiaowei and Li, Xiaomeng and Zhang, Hai and Heng, Pheng-Ann},
  journal={Medical image analysis},
  volume={70},
  pages={101989},
  year={2021},
  publisher={Elsevier}
}
```


### Questions
Please contact 'xchengjlu@gmail.com'
