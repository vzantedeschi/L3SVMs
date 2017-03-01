#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import numpy as np
import os

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize,scale

# -------------------------------------------------------------- I/0 FUNCTIONS

def dict_to_array(l):
    data,indices,indptr = [],[],[0]

    ptr = 0

    for d in l:

        for k,v in d.items():
            
            data.append(v)
            indices.append(k-1)

        ptr += len(d.keys())
        indptr.append(ptr)

    return csr_matrix((data, indices, indptr))

def array_to_dict(a,**kwargs):

    results = []
    r,c = a.shape

    try:
        a = a.tolil()
    except:
        pass

    if kwargs:
        clusters = kwargs['clusters']
        L = kwargs['land']

        for i in range(r):
            k = clusters[i]
            results.append({})
            for j in range(c):
                results[i][k*L+j+1] = float(a[i,j])
            # results[i][k*L+c+1] = 1
    else:

        for i in range(r):
            results.append({})
            for j in range(c):
                results[i][j+1] = float(a[i,j])
            # results[i][c+1] = 1

    return results

# ----------------------------------------------------------------- DATASET LOADERS

def load_csr_matrix(filename,y_pos=0):
    with open(filename,'r') as in_file:
        data,indices,indptr = [],[],[0]

        labels = []
        ptr = 0

        for line in in_file:
            line = line.split(None, 1)
            if len(line) == 1: 
                line += ['']
            label = line[y_pos]
            features = line[-1-y_pos]
            labels.append(float(label))

            f_list = features.split()
            for f in f_list:

                k,v = f.split(':')
                data.append(float(v))
                indices.append(float(k)-1)

            ptr += len(f_list)
            indptr.append(ptr)

        return labels,csr_matrix((data, indices, indptr))

def load_sparse_dataset(name,norm=False,y_pos=0):

    y,x = load_csr_matrix(name,y_pos)

    if norm:
        return y,normalize(x)
    else:
        return y,scale(x,with_mean=False)

def load_dense_dataset(name,norm=False,y_pos=0):
    dataset = np.loadtxt(name)
    if y_pos == -1:
        x,y = np.split(dataset,[-1],axis=1)
    else:
        y,x = np.split(dataset,[1],axis=1)

    if norm:
        return np.squeeze(y).tolist(),csr_matrix(normalize(x))
    else:
        return np.squeeze(y).tolist(),csr_matrix(scale(x))

# ------------------------------------------------------------------- ARG PARSER


def get_args(prog,testf_required=True,nb_clusters=1,nb_landmarks=10,linear=True,pca=False,nb_iterations=1,verbose=False,y_pos=0,nb_cv=5):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # positional arguments
    parser.add_argument('train_file',help='path of the training set')

    if testf_required:
        parser.add_argument('test_file',help='path of the testing set')

    # optional arguments
    parser.add_argument("-y", "--labelindex", type=int, dest='y_pos', default=y_pos, choices=[-1,0],
                        help='index of the labels in the file')
    parser.add_argument("-n", "--nbclusters", type=int, dest='nb_clusters', default=nb_clusters,
                        help='number of clusters')
    parser.add_argument("-l", "--nblands", type=int, dest='nb_landmarks', default=nb_landmarks,
                        help='number of landmarks')
    parser.add_argument("-i", "--nbiter", type=int, dest='nb_iterations', default=nb_iterations,
                        help='number of times the learning is repeated')
    parser.add_argument("-c", "--cv", type=int, dest='nb_cv', default=nb_cv,
                        help='number of folds for cross-validation')
    parser.add_argument("-r", "--rbfk", dest='linear', action="store_false",
                        help='if set, the rbf projection is used, otherwise the dot product')
    parser.add_argument("-o", "--normalize", dest='norm', action="store_true",
                        help='if set, the dataset is normalized, otherwise rescaled to stdev=1')
    parser.add_argument("-p", "--pca", dest='pca', action="store_true",
                        help='if set, the landmarks are selected as the principal components, otherwise randomly')
    parser.add_argument("-v", "--verbose", dest='verbose', action="store_true",
                        help='if set, verbose mode')

    return parser.parse_args()