# mobile-face
try on face recognition with mobile deep learning architecture

## Requirements
1. pytorch 0.3.0
2. opencv3-python
3. dlib 0.19.0
4. numpy 
5. pandas

## Current architecture
1. sphereface
2. resnet
3. mobilenetv2
4. angle loss
5. AM loss (implmeanted on Pytorch)

## Reference

1. code
* [sphereface](https://github.com/clcarwin/sphereface_pytorch)
* [AM softmax](https://github.com/happynear/AMSoftmax)
* [AM softmax tensorflow](https://github.com/Joker316701882/Additive-Margin-Softmax)
* [MTCNN](https://github.com/TropComplique/mtcnn-pytorch)

2. paper
* [sphereface](https://arxiv.org/abs/1704.08063)
* [AM softmax](https://arxiv.org/abs/1801.05599)
* [resnet](https://arxiv.org/abs/1512.03385)
* [mobilenetv2](https://arxiv.org/abs/1801.04381)
* [MTCNN](https://arxiv.org/abs/1604.02878)


Thanks to the [carwin](https://github.com/clcarwin), this code is heavily based on sphereface_pytorch
