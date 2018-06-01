FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-tk \
        rsync \
        software-properties-common \
        unzip \
        wget \
        && \
    apt-get clean

ADD . /data-sets
WORKDIR /data-sets
RUN wget 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
RUN tar -xzf 'aclImdb_v1.tar.gz'

RUN pip3 install --upgrade pip setuptools cython nltk sklearn

RUN pip3 --no-cache-dir install --upgrade  \
        h5py \
        matplotlib \
        numpy \
        gensim \
        pandas \
        scipy \
        tflearn \
        toolz

RUN (echo "import nltk" ; echo "nltk.download('reuters')" ; echo "nltk.download('stopwords')" ; echo "nltk.download('punkt')") | python3
RUN (echo "from sklearn.datasets import fetch_20newsgroups" ; echo "fetch_20newsgroups(subset='all')") | python3


RUN pip3 --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl

ADD . /code
WORKDIR /code

CMD ["/bin/bash"]