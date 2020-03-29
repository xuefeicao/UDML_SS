## Introduction 
This is offitial implementation of our recent paper on unsupervised metric learning (UDML-SS). 

In this work, we explore the task of unsupervised metric
learning. We propose to generate pseudo-labels for deep metric learning
directly from the clustering assignment and introduce an unsupervised
deep metric learning (UDML) scheme. By sampling the positive and negative
pairs based on the cluster assignments, we are able to learn feature
embedding together with cluster assignments progressively. Moreover,
we discovered that the training can be regularized well by adding selfsupervised
(SS) loss to the loss terms. In particular, we propose to regularize
the training process by predicting image rotations. Our method
(UDML-SS) iteratively cluster embeddings using traditional clustering
algorithm (e.g., k-means), and sampling training pairs based on the cluster
assignment, while optimizing self-supervised pretext task in a multitask
fashion. The proposed method outperforms the current state of the
art by a signicant margin on all the standard benchmarks.

### Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

   first 98 classes as train set and last 98 classes as test set

- [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)

  first 100 classes as train set and last 100 classes as test set

- [Stanford-Online-Products](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502 images of 11,316 classes for testing

## Dataset
The structure of the dataset should be like

```
CUB_200_2011
|_ train.txt
|_ test.txt
|_ train
    |_ 001.Black_footed_Albatross
        |_ <im-1-name>.jpg
        |_ ...
        |_ <im-N-name>.jpg
    |_ 002.Laysan_Albatross 
        |_ <im-1-name>.jpg
        |_ ...
        |_ <im-N-name>.jpg
    |_ ...
|_ test
    |_ ...
```
The "train.txt" or "test.txt" contains the samples for training and testcd, which is like
```
train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0075_21715.jpg 0
train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0012_21961.jpg 0
train/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0077_21986.jpg 0
...
```
You need to provide a DATA_ROOT in scripts/run_train_test.sh. And your data should be stored in 
${DATA_ROOT}. So if you want to run our algorithm in three benchmark datasets. You will have three data folders Cars196, CUB_200_2011, Products in ${DATA_ROOT}.

## Reproduce our work
```
cd scripts
./all_weight.sh (you need to change the DATA_ROOT and ROOT in scripts/run_train_test.sh to your data root and outpur dir respectively.)
```
We also provided a sample testing output of using UDML-SS on CUB dataset in scripts/out.

## Run with your own dataset
Our code can be easily extended to run on other datasets. There are two steps needed to achieve this goal. Frist, give your dataset a name. Second, register this dataset in Dict defined in data/dataset.py by providing its folder name. 

### Requirements
* Python >= 3.5
* PyTorch >= 1.0

### Open-source repos
This library contains code that has been adapted and modified from the following open-source repos:
- https://github.com/bnu-wangxun/Deep_Metric
- https://github.com/vadimkantorov/metriclearningbench

## Acknowledgements
Thank you to Ser-Nam Lim from Facebook AI and Facebook AI. This project was completed during my internship at Facebook. 

## Citing our work
```latex
@article{cao2019unsupervised,
  title={Unsupervised Deep Metric Learning via Auxiliary Rotation Loss},
  author={Cao, Xuefei and Chen, Bor-Chun and Lim, Ser-Nam},
  journal={arXiv preprint arXiv:1911.07072},
  year={2019}
}
```


 
    
