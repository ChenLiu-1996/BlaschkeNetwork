# Blaschke Network

## Background
Relevant papers for Blaschke decomposition:
1. [Multiscale decompositions of Hardy spaces](https://arxiv.org/pdf/2101.05311).
2. [On Complex Analytic tools, and the Holomorphic Rotation methods](https://arxiv.org/pdf/2210.01949).

The analytical decomposition in this code base was based on:
[Carrier frequencies, holomorphy and unwinding](https://arxiv.org/pdf/1606.06475) and
adapted from [the official MATLAB codebase](https://github.com/hautiengwu/BlaschkeDecomposition).

## Preparation

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name bnet pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda activate bnet
conda install -c anaconda scikit-image pillow matplotlib seaborn tqdm -y
python -m pip install tinyimagenet
python -m pip install phate
python -m pip install einops
python -m pip install wfdb
python -m pip install numpy==1.26
```


### Dataset
#### (1D) PTB-XL
The PTB-XL ECG dataset consisting four subsets can be downloaded [here](https://physionet.org/content/ptb-xl/1.0.3/).

```
wget -O data/ptbxl.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
```
Unzip and rename the folder as `PTBXL`.

#### (1D) CSN
The CSN(Chapman-Shaoxing-Ningbo) ECG dataset can be downloaded [here](https://physionet.org/content/ecg-arrhythmia/1.0.0/).

```
wget -O data/csn.zip https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
```
Unzip and rename the folder as `CSN`.


#### (2D) MNIST, CIFAR, STL
Most 2D image datasets (MNIST, CIFAR-10, CIFAR-100, STL-10) can be directly downloaded via the torchvision API as you run the training code. However, for the following datasets, additional effort is required.

#### (2D) ImageNet data
NOTE: In order to download the images using wget, you need to first request access from http://image-net.org/download-images.
```
cd data/
mkdir imagenet && cd imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

#### The following lines are instructions from Facebook Research. https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset.
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

```
