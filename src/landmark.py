# ------------------------------------------------------------------ IMPORTS
import numpy as np
import random

from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

# ----------------------------------------------------------- LAND SELECTION
def random_landmarks(x,n):
    m = x.shape[0]
    landmarks = x[random.sample(range(m),min(m,n))]

    return landmarks

def pca_landmarks(x,n):
    pca = PCA(n).fit(x)

    return pca.components_