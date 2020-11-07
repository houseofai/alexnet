# AlexNet
This project intend to replicate the AlexNet CNN network by stricly following the original paper.

### TODO

- [ ] 2 Dataset: Find ImageNet 2010
- [X] 3.1 ReLU Nonlinearity
- [X] 3.2 Training on Multiple GPUs: To split model on two GPUs
- [X] 3.3 Local Response Normalization
- [X] 3.4 Overalapping Pooling
- [X] 3.5 Overall Architecture
- [ ] 4.1 Data Augmentation:
- [ ]   Altering intensities of the RGB channels
- [ ]   Patch and horizontal flip for testing
- [X] 4.2 Dropout
- [ ] 5 Details of learning:
- [ ]   Learning Rate Decay

- [X] Predict task

### Dataset

### Prerequisites

- Python 3.7.9
- Tensorflow 2.3.1
```
pip install tensorflow-gpu==2.3.1
```

### Setup
1. Clone the repo:

 ```
 git clone https://github.com/houseofai/alexnet.git
 ```

2. Install the 3rd party packages
```
cd alexnet/
pip install -r requirements.txt
```

### Training
To train the model, launch the following command:
 ```
 python train.py
 ```

 To test the configuration on a smaller dataset:
 ```
 python train.py --conf=test
 ```

### Predict

**TODO**

### Main Folders

├── checkpoints         Checkpoints from training

├── config              Configuration files

|       ├── original.yml

|       └── test.yml

├── docker              Files to build a Docker image

|       ├── Dockerfile

├── dataloader.py

└── setup.sh
