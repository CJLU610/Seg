# Unet and Unet++: multiple classification using Pytorch

基于 [UNet](https://arxiv.org/pdf/1505.04597.pdf) 和 [UNet++](https://arxiv.org/abs/1807.10165)

Demo数据[百度网盘](https://pan.baidu.com/s/1QrjYFyjVRzjUTumIjiXB9w)提取码：1418；其中images是图像数据集，masks是该数据集对应的标签，test是测试数据，checkpoints是在该数据集上预训练的模型。
## Usage

#### Note :  Python 3

### Dataset
确保你的数据集按以下结构放置:  
```
data
├── images
|   ├── 0a7e06.jpg
│   ├── 0aab0a.jpg
│   ├── 0b1761.jpg
│   ├── ...
|
└── masks
    ├── 0a7e06.png
    ├── 0aab0a.png
    ├── 0b1761.png
    ├── ...
```
Mask是一个单通道类别的标签。 例如，数据集有三个类别，mask应该是值0,1,2作为分类值的8位图像，该图像看起来是黑色的。



### Training

修改config.py中的配置：

```bash
n_channels = 3, # 根据图像修改通道数
n_classes = 3,  # 类别数目，n(类别)+1(背景) or n

model='NestedUNet',# 默认是NestedUNet(unet++)
deepsupervision = True, # unet++时为True,选择unet时改为False

self.images_dir = './data/images'
self.masks_dir = './data/masks'
self.checkpoints_dir = './data/checkpoints'
```

修改train.py中的代码

```bash
 # compute loss
if cfg.deepsupervision:
	with torch.no_grad():
    	masks_preds = net(imgs)
	loss = 0
     
     
 else:
	masks_pred = net(imgs)
```



```bash
python train.py
```



### inference

```base
python inference.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```
如果你想用颜色区分类别，你可以  
```bash
python inference_color.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```

## Tensorboard
实时可视化训练和验证损失，以及模型预测:  
```bash
tensorboard --logdir=runs
```


