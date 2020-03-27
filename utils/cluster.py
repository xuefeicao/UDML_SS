# coding=utf-8
from __future__ import absolute_import, print_function
import mkl
mkl.get_max_threads()
import faiss

import numpy as np
import os
import torch

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import shutil
# from sklearn.mixture import GaussianMixture

dic = {
    'cub': "CUB_200_2011",
    'car': "Cars196",
    'product': "Products",
    'shop': "InShopClothes",
    'cifar': "Cifar10"
}


def cluster_(features,  labels, n_clusters):
    centers = []
    center_labels = []
    for label in set(labels):
        X = features[labels == label]
        kmeans = KMeans(n_clusters=n_clusters, random_state=None).fit(X)
        center_ = kmeans.cluster_centers_
        centers.extend(center_)
        center_labels.extend(n_clusters*[label])
    centers = np.conjugate(centers)
    centers = normalize(centers)
    return centers, center_labels


def normalize(X):
    norm_inverse = np.diag(1/np.sqrt(np.sum(np.power(X, 2), 1)))
    X_norm = np.matmul(norm_inverse, X)
    return X_norm

def create_fake_labels(train_features, train_labels, args, init_centers="k-means++"):
    n_clusters = args.num_clusters
    root = args.data_root
    root = os.path.join(root, dic[args.data])
    save_dir = args.save_dir
    os.makedirs(os.path.join(save_dir, dic[args.data]), exist_ok=True)
    
    with open(os.path.join(root, "train.txt")) as f:
        alldata = f.readlines()
        if os.path.exists(os.path.join(root, "test.txt")):
            shutil.copyfile(os.path.join(root, "test.txt"), os.path.join(save_dir, dic[args.data], "test.txt"))
        if os.path.exists(os.path.join(root, "gallery.txt")):
            shutil.copyfile(os.path.join(root, "gallery.txt"), os.path.join(save_dir, dic[args.data], "gallery.txt"))
        if os.path.exists(os.path.join(root, "query.txt")):
            shutil.copyfile(os.path.join(root, "querry.txt"), os.path.join(save_dir, dic[args.data], "querry.txt"))
    train_names = []
    train_data_labels = []
    for x in alldata:
        name, l = x.split()
        train_names.append(os.path.join(root, name))
        train_data_labels.append(int(l))
    if args.rot_only:
        train_data_labels = np.random.permutation(train_data_labels)
        new_data = [" ".join([image, str(label)]) for image, label in zip(train_names, train_data_labels)]
        with open(os.path.join(save_dir, dic[args.data], "train_1.txt"), 'w') as f:
            for item in new_data:
                f.write("%s\n" % item)

        return
    

    train_labels = [x.item() for x in train_labels]
    print(train_features.size(), type(train_features))

    # if init_centers == "k-means++":
    assert(train_labels == train_data_labels)
    # norm = train_features.norm(dim=1, p=2, keepdim=True)
    # train_features = train_features.div(norm.expand_as(train_features))
    train_features = train_features.numpy().astype(np.float32)
    if (args.data in ["product","cifar"]) and ((n_clusters > 1000) or ("ResNet" in args.net)):
        
        # Change faiss seed at each k-means so that the randomly picked
        # initialization centroids do not correspond to the same feature ids
        # from an epoch to another.
        if ("ResNet" in args.net):
            kmeans = faiss.Kmeans(train_features.shape[1], n_clusters, niter=100)
            kmeans.min_points_per_centroid = 3
            if init_centers != "k-means++":
                kmeans.centroids = init_centers
            kmeans.train(train_features)
            _, I = kmeans.index.search(train_features, 1)
            kmeans.cluster_centers_ = kmeans.centroids
        else:
            kmeans = faiss.Clustering(train_features.shape[1], n_clusters)
            kmeans.min_points_per_centroid = 3
            kmeans.seed = np.random.randint(1234)
            kmeans.niter = 20
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = False
            flat_config.device = 0
            index = faiss.GpuIndexFlatL2(res, train_features.shape[1], flat_config)
            
            if init_centers != "k-means++":
                print(init_centers.shape)
                faiss.copy_array_to_vector(init_centers.ravel(), kmeans.centroids)
        
            # perform the training
            kmeans.train(train_features, index)
            _, I = index.search(train_features, 1)
            kmeans.cluster_centers_ = faiss.vector_to_array(kmeans.centroids)
        
        kmeans.labels_ = [int(n[0]) for n in I]
        
        

    else:
        print("use old")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0, n_jobs=10, init=init_centers, max_iter=100, precompute_distances=True).fit(train_features)
    print("finish cluster")
    new_data = [" ".join([image, str(label)]) for image, label in zip(train_names, kmeans.labels_)]
    dic_label = {}
    for i in kmeans.labels_:
        if i not in dic_label:
            dic_label[i] = 1
        else:
            dic_label[i] += 1
    print("label", len([i for i in dic_label if dic_label[i] == 1]), len(dic_label))
    
    with open(os.path.join(save_dir, dic[args.data], "train_1.txt"), 'w') as f:
        for item in new_data:
            f.write("%s\n" % item)
    print(adjusted_rand_score(train_labels, kmeans.labels_), kmeans.cluster_centers_.shape)
    return kmeans.cluster_centers_

        





















