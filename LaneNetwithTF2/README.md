#  基于卷积神经网络识别车道线

## 一、训练数据集实例

本文的训练图片文件使用 full_CNN_train.p 对应的标签图片文件为 full_CNN_labels.p 其中包含了12764张图片，包括各种场景下的车道线图片。其尺寸为(80,160,3)标签文件大小为(80,160,1)。将其集合到两个.p文件当中。

## 二、训练网络的搭建

训练使用的神经网络模型搭建如下图所示，首先使用批归一化层对输入的训练数据进行归一化处理；每两个卷积后面加上一层池化层，适当使用dropout层防止其过拟合。后面通过上采样层和反卷积层，将其特征进一步还原。如下图即是搭建的网络的模型，其输入网络数据shape与输出shape保持一致。


![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JoeWlqZw==,size_16,color_FFFFFF,t_70.png)