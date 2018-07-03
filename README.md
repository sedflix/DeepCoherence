# DeepCoherence

It is kind of based on [**Text Coherence Analysis Based on Deep Neural Network**](https://arxiv.org/abs/1710.07770)  
by *Baiyun Cui, Yingming Li, Yaqing Zhang, Zhongfei Zhang*

Based on the origial data provied by [Baiyun Cui](mailto:baiyunc@yahoo.com) 

## Usage 

## Download the code

```
git https://github.com/geekSiddharth/DeepCoherence.git
cd DeepCoherence
```

## Download the glove embedding 

Download the embeddings using `download.sh` script

```
$/DeepCoherence> sh download.sh
$/DeepCoherence> ls -sh glove/
total 2.1G
332M glove.6B.100d.txt  662M glove.6B.200d.txt  990M glove.6B.300d.txt  164M glove.6B.50d.txt
```
## Environment

### With docker/nvidia-docker

Make sure you have Docker installed on your system.

#### Building the image

```
docker build -t coherence .
```

#### Start a container

```
sudo docker run -it -v "$PWD":/src coherence
```

##### For GPU user:
- Use **nvidia-docker** instead of docker for building and running the image 

### With pip

- After that you can install the required packages using:
```
pip install  -r requirements.txt
```

## Pre-processing and training

### Preprocess data

Based on script and data provided by Baiyun Cui. It reads data from data2 folder and stores it in processed folder
```
$/DeepCoherence> cd data/cui
$/cui> python data_preprocess.py 
```
### Training

You can change the model info in `model.py`(Like hidden dimension, max sequence length, filter size etc ) and training info(like batch sizes, epoch etc) in `train.py`
```
$/DeepCoherence> python train.py
```

## Predict

Open `predict.py`, and add file path of  model_weight_file and enter the input sentences and then
```
$/DeepCoherence> python predcit.py
```

### Details about the dataset:

data/training 100: original train documents for accident dataset
    /testing 100:  original test documents for accident dataset

data2: original train/dev/test and their permutations for accident dataset
data3: original train/dev/test and their permutations for earthquake dataset.

- accident dataset as an example to run the code.


## Citing 

```
@article{DBLP:journals/corr/abs-1710-07770,
  author    = {Baiyun Cui and
               Yingming Li and
               Yaqing Zhang and
               Zhongfei Zhang},
  title     = {Text Coherence Analysis Based on Deep Neural Network},
  journal   = {CoRR},
  volume    = {abs/1710.07770},
  year      = {2017},
  url       = {http://arxiv.org/abs/1710.07770},
  archivePrefix = {arXiv},
  eprint    = {1710.07770},
  timestamp = {Wed, 01 Nov 2017 19:05:42 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1710-07770},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```