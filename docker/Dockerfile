FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update

# Install Python, pip3
RUN apt-get -y install git

RUN git clone https://github.com/houseofai/alexnet.git

WORKDIR /alexnet

RUN pip install -r requirements.txt
