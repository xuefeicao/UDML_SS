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


## Citing our work
```latex
@article{cao2019unsupervised,
  title={Unsupervised Deep Metric Learning via Auxiliary Rotation Loss},
  author={Cao, Xuefei and Chen, Bor-Chun and Lim, Ser-Nam},
  journal={arXiv preprint arXiv:1911.07072},
  year={2019}
}
```




## Acknowledgements
Thank you to Ser-Nam Lim from Facebook AI and Facebook AI. This project was completed during my part-time work at Facebook. 


 
    
