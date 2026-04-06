# AlexNet
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result



---

## 简介
AlexNet 是由 Alex Krizhevsky、Ilya Sutskever 与 Geoffrey Hinton 于 2012 年提出的里程碑式深度卷积神经网络，相关成果发表于《ImageNet Classification with Deep Convolutional Neural Networks》，它在当年的 ImageNet 大规模图像分类竞赛中，将 Top-5 错误率从传统机器学习方法的 26% 左右大幅降至 15.3%，以压倒性优势夺冠，其核心架构由 5 层用于特征提取的卷积层（搭配 ReLU 非线性激活、最大池化操作）与 3 层用于分类输出的全连接层构成，首次在大规模视觉任务中成功落地多项关键技术 —— 包括用 ReLU 激活函数解决深层网络梯度消失问题、Dropout 层抑制过拟合、GPU 并行计算大幅提升训练效率、数据增强扩充训练样本提升泛化能力，不仅彻底打破了传统手工特征 + 机器学习的计算机视觉技术范式，更直接开启了深度学习在计算机视觉乃至整个人工智能领域的爆发式发展，其卷积堆叠的网络设计思路也为后续 VGG、GoogLeNet、ResNet 等经典 CNN 模型奠定了核心范式。
## 架构
AlexNet的核心架构为**8层带可学习参数的端到端深度卷积神经网络**，整体分为「5层卷积特征提取模块」和「3层全连接分类模块」两大核心部分，原论文标准输入为227×227分辨率的3通道RGB图像，最终输出对应分类类别的预测概率，具体结构与设计如下：
-  **特征提取模块（5个卷积层）**：第1层采用11×11大卷积核（步长4）输出96通道特征图，搭配ReLU激活与3×3步长2的重叠最大池化，完成边缘、纹理等基础视觉特征提取与尺寸压缩；第2层用5×5卷积核输出256通道特征图，同样搭配ReLU激活与最大池化，提取更复杂的组合特征；第3、4层均为3×3卷积核，输出384通道特征图，仅搭配ReLU激活无池化，堆叠提取深层语义特征；第5层用3×3卷积核输出256通道特征图，接ReLU激活与最大池化，最终输出256×6×6的高维特征图。
-  **分类输出模块（3个全连接层）**：先通过flatten将卷积输出的特征图展平为一维向量，前两层全连接层均为4096维输出，搭配ReLU激活与Dropout（随机丢弃率0.5）抑制过拟合，完成高维特征的筛选映射；最后一层全连接层为输出层，维度匹配分类任务的类别数（原论文ImageNet任务为1000维），输出各类别的预测得分。

该架构首次在大规模视觉任务中落地ReLU激活、Dropout、重叠池化等关键设计，解决了深层网络梯度消失与过拟合难题，其卷积堆叠提取特征+全连接层分类的范式，也成为了后续所有经典CNN模型的核心设计基础。
<img width="1115" height="320" alt="image" src="https://github.com/user-attachments/assets/2cf22eed-0dfd-4fe6-aa0e-f664fdb3b021" />
<img width="541" height="713" alt="image" src="https://github.com/user-attachments/assets/9f08ade7-f8d4-4bb8-bb71-f66f0f685ff6" />


**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 227×227，我们会对网络结构做微小适配（主要是缩小卷积核和步长），但核心架构（5 卷积 + 3 全连接 + ReLU + Dropout）完全保留。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
AlexNet, a landmark deep convolutional neural network proposed in 2012 by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, with its findings published in "ImageNet Classification with Deep Convolutional Neural Networks," achieved a significant victory in the ImageNet large-scale image classification competition. It drastically reduced the Top-5 error rate from around 26% for traditional machine learning methods to 15.3%. Its core architecture consists of 5 convolutional layers for feature extraction (with ReLU non-linear activation and max pooling) and 3 fully connected layers for classification output. It was the first to successfully implement several key technologies in large-scale vision tasks—including using ReLU activation to address the vanishing gradient problem in deep networks, using Dropout layers to suppress overfitting, significantly improving training efficiency through GPU parallel computing, and data augmentation to expand training samples and improve generalization ability. This not only completely broke away from the traditional hand-crafted feature + The machine learning paradigm in computer vision technology directly triggered the explosive development of deep learning in computer vision and even the entire field of artificial intelligence. Its convolutional stacking network design also laid the core paradigm for subsequent classic CNN models such as VGG, GoogLeNet, and ResNet.

## Architecture
AlexNet's core architecture is an **8-layer end-to-end deep convolutional neural network with learnable parameters**, divided into two main parts: a "5-layer convolutional feature extraction module" and a "3-layer fully connected classification module". The original paper's standard input was a 227×227 resolution 3-channel RGB image, and the final output was the predicted probability of the corresponding classification category. The specific structure and design are as follows:

- **Feature Extraction Module (5 Convolutional Layers)**: Layer 1 uses an 11×11 large convolutional kernel (stride 4) to output a 96-channel feature map, combined with ReLU activation and 3×3 overlapping max pooling with a stride of 2, to extract basic visual features such as edges and textures and compress their size. Layer 2 uses a 5×5 convolutional kernel to output a 256-channel feature map, also combined with ReLU activation and max pooling, to extract more complex combined features. Layers 3 and 4 both use 3×3 convolutional kernels to output a 384-channel feature map, combined with only ReLU activation and no pooling, stacking to extract deep semantic features. Layer 5 uses a 3×3 convolutional kernel to output a 256-channel feature map, followed by ReLU activation and max pooling, finally outputting a 256×6×6 high-dimensional feature map.

- **Classification Output Module (3 Fully Connected Layers)**: First, the feature map output by the convolution is flattened into a one-dimensional vector through flattening. The first two fully connected layers are 4096-dimensional outputs, and ReLU activation and Dropout (random drop rate of 0.5) are used to suppress overfitting and complete the filtering and mapping of high-dimensional features. The last fully connected layer is the output layer, and the dimension matches the number of categories in the classification task (1000 dimensions for the ImageNet task in the original paper), outputting the prediction scores for each category.
<img width="1115" height="320" alt="image" src="https://github.com/user-attachments/assets/2cf22eed-0dfd-4fe6-aa0e-f664fdb3b021" />

**Note:** We use the CIFAR-10 dataset, which is a 10-class dataset. Unlike the original paper, the image size of CIFAR-10 (32×32) is much smaller than the 227×227 in the original paper. We will make minor adaptations to the network structure (mainly reducing the convolution kernels and stride), but the core architecture (5 convolutions + 3 fully connected layers + ReLU + Dropout) will be completely retained.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Hinton's students Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here. If you'd like to learn more about this dataset, please see the link: https://cloud.tencent.com/developer/article/2150614

---
## 原文章 | Original article
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).
