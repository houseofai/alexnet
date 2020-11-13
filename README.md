# AlexNet
This project intends to replicate the AlexNet CNN network by strictly following the original paper.

___

### 1. Dataset

Currently, the project is configured to be trained on *imagenette/full-size-v2*, a dataset delivered by Tensorflow. From the documentation:

> Imagenette is a subset of 10 easily classified classes from the Imagenet dataset. It was originally prepared by Jeremy Howard of FastAI. The objective behind putting together a small version of the Imagenet dataset was mainly because running new ideas/algorithms/experiments on the whole Imagenet take a lot of time. 

Link: https://github.com/fastai/imagenette 

**TODO: Find ImageNet 2010**  
Please drop me an email if you know where to find this specific dataset  

### 2. Prerequisites

- Python 3.7.9
- Tensorflow 2.3.1
```
pip install tensorflow-gpu==2.3.1
```

### 3. Setup
1. Clone the repo:

 ```
 git clone https://github.com/houseofai/alexnet.git
 ```

2. Install the 3rd party packages
```
cd alexnet/
pip install -r requirements.txt
```

### 4. Training
To train the model, launch the following command:
 ```
 python train.py
 ```

 To test the configuration on a smaller dataset:
 ```
 python train.py --conf=test
 ```

### 5. Predict

To predict class probabilities on an image:
```
python train.py --mode=predict --image=/path/to/image
```


### 6. Bibliographies: ImageNet Classification with Deep Convolutional Neural Networks
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf