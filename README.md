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



## Reproduce our work


## Run with your own dataset



### Requirements
* Python >= 3.5
* PyTorch >= 1.0

### Open-source repos
This library contains code that has been adapted and modified from the following open-source repos:
- https://github.com/bnu-wangxun/Deep_Metric
- https://github.com/vadimkantorov/metriclearningbench

## Acknowledgements
Thank you to Ser-Nam Lim from Facebook AI and Facebook AI. This project was completed during my part-time work at Facebook. 

## Citing our work
```latex
@article{cao2019unsupervised,
  title={Unsupervised Deep Metric Learning via Auxiliary Rotation Loss},
  author={Cao, Xuefei and Chen, Bor-Chun and Lim, Ser-Nam},
  journal={arXiv preprint arXiv:1911.07072},
  year={2019}
}
```


 
    
