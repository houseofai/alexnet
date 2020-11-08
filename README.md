# AlexNet
This project intend to replicate the AlexNet CNN network by stricly following the original paper.

___
#### TODO

- [ ] 2 Dataset: ImageNet 2010
- [X] 3.1 ReLU Nonlinearity
- [X] 3.2 Training on Multiple GPUs: To split model on two GPUs
- [X] 3.3 Local Response Normalization
- [X] 3.4 Overalapping Pooling
- [X] 3.5 Overall Architecture
- [ ] 4.1 Data Augmentation:
- [ ] Altering intensities of the RGB channels
- [ ]   Patch and horizontal flip for testing
- [X] 4.2 Dropout
- [X] 5 Details of learning:
- [X]   Learning Rate Decay

- [X] Predict task

*Note: Learning Rate Decay is a manual process as described on the paper. I have implemented early stopping when the manual decay should happen.*

### 1. Dataset

Currently, the project is configured to be trained on *imagenet-a*, a dataset delivered by Tensorflow. From the documentation:

> ImageNet-A is a set of images labelled with ImageNet labels that were obtained by collecting new data and keeping only those images that ResNet-50 models fail to correctly classify. 

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