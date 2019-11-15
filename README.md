# CRAFT-based Chinese words detector training
---------------------------------------------------------------
[Chinese version](https://github.com/datatang-ailab//blob/master/README.zh.md)
## Description
---------------------------------------------------------------
-Test environment:Ubuntu 16.04
-Instruction: this is a sample of CRAFT training experiment in Chinese dataset, Datatang open source Chinese Mandarin Corpus
-Update time of document: November 13,2019
## Introdution of CRAFT
---------------------------------------------------------------
 Implementation of CRAFT text detector [paper](https://arxiv.org/abs/1904.01941)
 CRAFT text detector effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.
 [Offical inference project](https://github.com/clovaai/CRAFT-pytorch)
 [Originl training project](https://github.com/RubanSeven/CRAFT_keras)

## Installation
---------------------------------------------------------------
### Install dependence(Recomend Anaconda)
Install anaconda first.
Create conda environment
```shell
conda create -n CRAFT python=3.6
```
Activate conda environment
```shell
conda activate CRAFT
```
Install packages
```shell
pip install -r requirements.txt
```

## training
---------------------------------------------------------------
### General Overview
There are several folders and files under project, and their brief introductions are as follows:
|Folder        |Description     |Remark                  |
|:-------------|:---------------|:-----------------------|
|converts      |Script directory|transform data format   |
|module        |Model scripts   |model modules           |
|net           |Model scripts   |VGG16 net backbone      |
|utils         |Script directory|inference&training utils|
|weights       |Model weights   |weights folder          |
|compute_PPR.py|General scripts |analyze trained model   |
|test_json.py  |General scripts |generate demo results   |
|train.py      |General scripts |training program        |
### Detailed explanation of training
Training phase could be separated into two parts: prepare datasets and run training script. 
#### Download and decompression of data set
Download two datasets from urls below:
[CTW官方网站](https://ctwdataset.github.io/)
[SynthText官方网站](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
[SynthText项目链接](https://github.com/ankush-me/SynthText)
**Introduction**
 CTW provides details of a newly created dataset of Chinese text with about 1 million Chinese characters from 3850 unique ones annotated by experts in over 30000 street view images. This is a challenging dataset with good diversity containing planar text, raised text, text under poor illumination, distant text, partially occluded text, etc. 
 SynthText is a synthetically generated dataset, in which word instances are placed in natural scene images, while taking into account the scene layout.
 The dataset consists of 800 thousand images with approximately 8 million synthetic word instances. Each text instance is annotated with its text-string, word-level and character-level bounding-boxes. 
#### Data preparation
Using follow scripts could converte dataset into specified format.
CTW dataset
```shell
python converts/CTW_convert.py
```
SynthText dataset
```shell
python converts/synth_convert.py
```
#### Training details
In [Originl training project](https://github.com/RubanSeven/CRAFT_keras) author added sigmoid function in network. Contrast experiments we have done prove that:
- Sigmoid function can bring bigger loss during training phase.
- It is easiler to fall in local optimum.
- It can effectively shrinking training time. 
We used linear function instead sigmoid and amplify input image size to 800x800. Datasets have two parts: English data and Chinese data. Training phase will combine two parts of data and English data will be limited under manmade rate of Chinese and English. 

In our experience, set batch size to 8 is fine on NVIDIA 1080TI/Tesla V100 and reducing image size will lead a bigger batch size. During training phase, learning rate multiply by 0.64 after 30K steps will hold a stable decay of loss. Initial learning has a close correlation with batch size, bigger batch size bring a bigger learning rate. In our experence, learning rate set 0.00001 is fine when batch size equal to 8.
Run:
```shell
python train.py
```
### Inference
Download pretrained model
[Pan.Baidu]() 
We provide three kinds detector for inference phase: character level, word level, line level. You can choose one of them in **inference_util.py**
Get demo, heatmap and json file by running:
```shell
python test_json.py
```
Compute PPR(Pixel Rate)
PPR(Pixel Pass Rate) and MPD(Mean Pixel Difference) are indicators to measure how close the predict box and ground truth are. 
```shell
python compute_PPR.py 
```
### Summary
Using CTW and SynthText for training and separate them for testing.
i. Official is official release model, Original is original training model and Modification is modificate training model.

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
analyze:
 - Official model has a good performence on lower PPR
 - Sigmoid function can not bring a good convergence result
 - Linear function shows a better convergence performence 
 - It has a strong relation between dataset and test results. CTW is a hard sample with much shelter and difficult to recognize even for a human.  

## More Resource
---------------------------------------------------------------
Datatang (Beijing) Technology Co., Ltd is a professional AI data service provider and is dedicated to providing data collection, processing and data product services for global artificial intelligence enterprises, covering data types such as speech, image and text, including biometrics, speech recognition, autonomous driving, smart home, smart manufacturing, new retail, OCR scene, smart medical treatment, smart transportation, smart security, mobile phone entertainment, etc.

[For more open source data sets](https://www.datatang.com/webfront/opensource.html)

[For more business data sets](https://www.datatang.com/webfront/datatang_dataset.html)