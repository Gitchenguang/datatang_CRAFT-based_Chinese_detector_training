# 基于CRAFT的中文字符检测
[English version](https://github.com/datatang-ailab/datatang_CRAFT-based_Chinese_detector_training/blob/master/README.md)
## 简介
-测试环境:Ubuntu 16.04
-介绍: this is a sample of CRAFT training experiment in Chinese dataset, Datatang open source Chinese Mandarin Corpus
-文档更新时间: November 13,2019
## CRAFT说明
 本项目是CRAFT论文的复现[原文](https://arxiv.org/abs/1904.01941)
 CRAFT text detector effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.
 [官方推理项目](https://github.com/clovaai/CRAFT-pytorch)
 [原始训练项目](https://github.com/RubanSeven/CRAFT_keras)

## 安装
### 安装依赖环境(推荐使用虚拟环境Anaconda)
先安装Anaconda
创建虚拟环境
```shell
conda create -n CRAFT python=3.6
```
激活虚拟环境
```shell
conda activate CRAFT
```
安装依赖包
```shell
pip install -r requirements.txt
```

## CRAFT训练
### 总体概览
项目文件简要描述:
|文件夹         |描述            |简介                    |
|:-------------|:---------------|:-----------------------|
|converts      |Script directory|transform data format   |
|module        |Model scripts   |model modules           |
|net           |Model scripts   |VGG16 net backbone      |
|utils         |Script directory|inference&training utils|
|weights       |Model weights   |weights folder          |
|compute_PPR.py|General scripts |analyze trained model   |
|test_json.py  |General scripts |generate demo results   |
|train.py      |General scripts |training program        |
### 训练详解
训练共分为两个步骤：准备数据，运行训练脚本。数据这里使用的包括中文和英文混合训练。
#### 下载与解压数据集
中英文数据使用的分别是CTW数据集和SynthText数据集。下载链接参照下面官网地址：
[CTW官方网站](https://ctwdataset.github.io/)
[SynthText官方网站](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
[SynthText项目链接](https://github.com/ankush-me/SynthText)
**简要介绍**
CTW数据集主要包含32285张图片，共计1018402个汉字，3850个汉字类别，来自清华-腾讯100K数据集和腾讯街景数据集，图片大小为2048x2048。在相关论文《Chinese Text in the Wild》中，清华大学的研究人员以该数据集为基础训练了多种目前业内最先进的深度模型进行字符识别和字符检测。这些模型将作为基线算法为人们提供测试标准。这些图像源于腾讯街景，从中国的几十个不同城市中捕捉得到，不带任何特定目的的偏好。由于其多样性和复杂性，使得该数据集的收集很困难。它包含了平面文本、凸出文本、城市街景文本、乡镇街景文本、弱照明条件下的文本、远距离文本、部分显示文本等。对于每张图像，数据集中都标注了所有中文字符。对每个中文字符，数据集都标注了其真实字符、边界框和 6 个属性以指出其是否被遮挡、有复杂的背景、被扭曲、3D 凸出、艺术化，和手写体等。
SynthText数据集是一个包含单词的自然场景图像组成的数据集，主要运用于自然场景领域中的文本检测，SynthText数据集由80万个图像组成，大约有800万个合成单词实例。每个文本实例都使用其文本字符串，字级和字符级边界框进行注释。SynthText 数据集于 2016 年由牛津大学工程科学系视觉几何组的 Gupta, A. and Vedaldi, A. and Zisserman, A. 在 IEEE 计算机视觉和模式识别会议 (CVPR) 中发布。
#### 数据准备
下载完成的数据集需要转换存储格式，转换格式的脚本存放在**converts**文件夹内。使用时更换脚本中的路径即可。
CTW dataset
```shell
python converts/CTW_convert.py
```
SynthText dataset
```shell
python converts/synth_convert.py
```
#### 训练细节
在[原始项目](https://github.com/RubanSeven/CRAFT_keras)中，网络激活函数使用的sigmoid，和linear进行对比实验，结论如下：
- 训练中sigmoid可以带来更大的loss
- 更容易陷入局部收敛
- 更快的收敛可以缩短训练时长 
我们使用linear代替sigmoid，输入图片调整到800x800尺寸。训练数据集使用英文和中文混合数据，其中可以自行设定中英文比例。

以我们实验经验，用NVIDIA 1080TI/Tesla V100显卡的batch size可以设置为8并且减小图像大小可以适当的增大batch size。在训练阶段，每过30K步学习率乘以0.64有利于loss的稳定下降。 初始学习率设置和batch size有关，batch越大学习率越大比较能收敛，这里batch size=8参考学习率0.00001.
设置好参数之后运行：
```shell
python train.py
```

### 推理过程
下载训练好的权重
[Pan.Baidu]() 
推理过程我们提供字符级、字级、行级三种结果，可以在**inference_util.py**中选择需要的检测类型。
测试并生成demo图、演示热力图以及json结果：
```shell
python test_json.py
```
计算PPR(Pixel Pass Rate)以及MPD，这两个指标是衡量检测结果与ground truth贴合度的指标。PPR是计算检测到的四个顶点和ground truth的欧氏距离在n个像素内的比例，MPD计算预测框与ground truth平均欧距。
```shell
python compute_PPR.py 
```
### 实验总结
使用CTW以及SynthText混合数据训练，在两个数据集的测试集上分别测试，结果如下：
i. Official 是官方发布的模型，Original是对照试验里面原版训练程序，Modification是微调过的训练程序。

|OCR Detection Training| \ |Official|Official|Original |Original |Modification|Modification|
|:--------------------:|:-:|:------:|:------:|:-------:|:-------:|:----------:|:----------:|
|Hardware              | \ | \      | \      |1 1080Ti |1 1080Ti |1 Tesla v100|1 Tesla v100| 
|Data size(Test)       | \ |SynT    |CTW     |SynT     |CTW      |SynT        |CTW         |
|AP                    | \ | \      | \      |\        |\        |\           |\           |
|Pixels Pass Rate(PPR)[IOU@.50:.95]|1|**6.1%**|0%|3.1% |0%       |4.1%        |0%          |
|                      |2  |**32.6%**|**1.4%**|22.5%   |1.1%     |26.8%       |1.2%        |
|                      |3  |**52.6%**|**5.6%**|44.7%   |4.3%     |49.0%       |5.0%        |
|                      |4  |**62.9%**|11.9%  |59.3%    |10.2%    |62.5%       |**12.1%**   |
|                      |5  |69.1%   |18.0%   |69.3%    |17.5%    |**70.7%**   |**21.9%**   |
|                      |6  |73.4%   |22.7%   |75.8%    |23.7%    |**76.0%**   |**28.2%**   |
|                      |7  |76.5%   |26.0%   |**80.1%**|28.7%    |79.6%       |**32.9%**   |
|                      |8  |79.1%   |28.5%   |**83.3%**|32.6%    |82.3%       |**37.0%**   |
|Mean Pixel Diff(MPD)  | \ |3.0     |19.4    |**2.7**  |23.3     |2.8         |**19.2**    |
从试验结果分析：
- 官方发布模型在PPR低时有更好的表现，但在平均贴合度略高，预测结果与ground truth之差的分布方差较大
- 使用sigmoid模型收敛效果并不理想。
- 使用linear模型训练收敛效果整体略高于官方模型
- CTW数据集表现明显劣于SynthText数据集，这和两个数据集差异有很大关系。CTW是中文场景文字数据集，图片多是街道场景，文字多有遮挡，检测难度较大。SynthText数据集采用图片+文字方法生成的图片，里面的文字更加清晰，辨识度高，检测难度低。

## 更多资源
数据堂是一家专业的人工智能数据服务提供商，致力于为全球人工智能企业提供数据获取、处理及数据产品服务，覆盖语音、图像、文本等数据类型，涵盖生物识别、语音识别、自动驾驶、智能家居、智能制造、新零售、OCR场景、智能医疗、智能交通、智能安防、手机娱乐等领域。

获取更多开源数据集，请访问[这里](https://www.datatang.com/webfront/opensource.html)
了解更多商业数据集，请点击[这里](https://www.datatang.com/webfront/datatang_dataset.html)
面向高校及科研群体,数据堂将持续开源更多高质量商业数据集,帮助研究人员拓宽研究领域，丰富研究内容，加速迭代。敬请期待！
