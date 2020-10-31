FROM tensorflow/tensorflow:2.3.1-gpu

COPY . .

# Update the environments
RUN apt-get update
RUN pip install --upgrade pip

# Install git
RUN apt-get -y install git

# Download dataset
RUN pip install tensorflow-datasets==4.0.1
#RUN python dataloader.py

WORKDIR /root

# PROJECT
#CMD git clone https://github.com/houseofai/alexnet.git
ENTRYPOINT /bin/bash
#ENTRYPOINT pip install -r /root/alexnet/requirements.txt
