#!/bin/bash
set -e
sudo apt-get update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt install -y \
	build-essential \
	git \
	autoconf \
	libtool \
	unzip \
	gcc-4.8 \
	g++-4.8 \
	gfortran \
	gfortran-4.8 \
	nasm \
	make \
	automake \
	pkg-config \
	pandoc \
	python-dev \
	libssl-dev \
	python-pip

wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz
tar -xvzf cmake-3.12.3.tar.gz
cd cmake-3.12.3
./bootstrap
make -j
sudo make install

pip install -U pip "setuptools==36.2.0" wheel --user
pip install pypandoc numpy==1.15.0 --user


#export mxnet_variant=mkl
# CPU Build
mkdir $HOME/cpu
cd $HOME/cpu
git clone --recursive https://github.com/access2rohit/incubator-mxnet.git
cd $HOME/cpu/incubator-mxnet
git checkout temp_lts

mv python/setup.py python/setup.py.bak
cp tools/pip/setup.py python/
cp tools/pip/MANIFEST.in python/
cp -r tools/pip/doc python/

source tools/staticbuild/build.sh cpu pip
source tools/staticbuild/build_wheel.sh


# cu100 Build
mkdir $HOME/cu100
cd $HOME/cu100
git clone --recursive https://github.com/access2rohit/incubator-mxnet.git
cd $HOME/cu100/incubator-mxnet
git checkout temp_lts

mv python/setup.py python/setup.py.bak
cp tools/pip/setup.py python/
cp tools/pip/MANIFEST.in python/
cp -r tools/pip/doc python/

source tools/staticbuild/build.sh cu100 pip
source tools/staticbuild/build_wheel.sh


# cu101 Build
mkdir $HOME/cu101
cd $HOME/cu101
git clone --recursive https://github.com/access2rohit/incubator-mxnet.git
cd $HOME/cu101/incubator-mxnet
git checkout temp_lts

mv python/setup.py python/setup.py.bak
cp tools/pip/setup.py python/
cp tools/pip/MANIFEST.in python/
cp -r tools/pip/doc python/

source tools/staticbuild/build.sh cu101 pip
source tools/staticbuild/build_wheel.sh
