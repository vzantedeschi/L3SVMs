# ------------------------------------------------------------------ IMPORTS
import numpy as np
import random

from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

def random_landmarks(x,n):
    m = x.shape[0]
    landmarks = x[random.sample(range(m),min(m,n))]

    return landmarks

def pca_landmarks(x,n):
    pca = PCA(n).fit(x)

    return pca.components_

def get_unit_vectors(landmarks,clusterer=None):
    if clusterer is not None:
        land_clusters = clusterer.predict(landmarks)
        centroids = clusterer.cluster_centers_
        land_centroids = centroids[land_clusters]
    else:
        land_centroids = np.zeros(landmarks.shape)
    return csr_matrix(normalize(landmarks-land_centroids).transpose())