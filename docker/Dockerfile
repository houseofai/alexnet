FROM tensorflow/tensorflow:2.3.1-gpu

COPY . .

# Update the environments
RUN apt-get update
RUN pip install --upgrade pip

# Install git
RUN apt-get -y install git

# Download dataset
RUN pip install tensorflow-datasets==4.0.1
RUN python dataloader.py

# Install Git requirements
RUN pip install -r https://raw.githubusercontent.com/houseofai/alexnet/main/requirements.txt

WORKDIR /root

# PROJECT
ENTRYPOINT /bin/bash
