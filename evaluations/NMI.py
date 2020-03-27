
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from utils import to_numpy


def NMI(X, ground_truth, n_cluster=3):
    X = [to_numpy(x) for x in X]
    X = np.array(X).astype(np.float32)
    ground_truth = np.array(ground_truth)

    kmeans = KMeans(n_clusters=n_cluster, n_jobs=5, random_state=0, max_iter=100).fit(X)
    I = kmeans.labels_
    print(n_cluster, "num_clusters")
    print('K-means done')
    nmi = normalized_mutual_info_score(ground_truth, I.reshape((-1)))
    return nmi


def main():
    label = [1, 2, 3]*2

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])

    print(NMI(X, label))

if __name__ == '__main__':
    main()
