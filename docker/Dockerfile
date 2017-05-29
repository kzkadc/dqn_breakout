FROM nvidia/cuda:8.0-cudnn5-devel

LABEL maintainer "kzkadc"

# pythonなどのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
	checkinstall \
	build-essential \
	curl \
	python \
	python-dev \
	git \
	libhdf5-dev \
	cmake \
	zlib1g-dev \
	libjpeg-dev \
	xvfb \
	libav-tools \
	xorg-dev \
	python-opengl \
	libboost-all-dev \
	libsdl2-dev \
	swig \
	ca-certificates && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*
# pipのインストール
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py
# pythonの各種パッケージをインストール
RUN pip --no-cache-dir install \ 
	scikit-learn \
	numpy \
	scipy \
	h5py \
	chainer \
	gym[atari]
