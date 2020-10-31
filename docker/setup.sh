#!/bin/sh
#apt-get update
#apt-get -y install git aria2

# Download Dataset
#mkdir -p ~/tensorflow_datasets/imagenet_a/0.1.0
#cd ~/tensorflow_datasets/imagenet_a/0.1.0
#aria2c -x 16 -s 16 https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
#tar -xvf

# Download project
git clone https://github.com/houseofai/alexnet.git
cd alexnet/
pip install -r requirements.txt
#nohup python train.py > log/training.log &
