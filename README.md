# AlexNet
This project intend to replicate the AlexNet CNN network by stricly following the original paper.

### Prerequisites

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
│   ├── original.yml
│   └── test.yml
├── docker              Files to build a Docker image
│   ├── Dockerfile
│   ├── dataloader.py
│   └── setup.sh
