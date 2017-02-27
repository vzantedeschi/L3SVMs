#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import csv
import numpy as np
import os

from liblinearutil import *
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize,scale

# -------------------------------------------------------------- I/0 FUNCTIONS

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def dict_to_csv(my_dict,header,filename):

    make_directory(os.path.dirname(filename))

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(header)
        for key, value in my_dict.items():
            writer.writerow([key, value])

def csv_to_dict(filename):

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader,None)
        my_dict = {row[0]:eval(row[1]) for row in reader}

    return my_dict

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
DATAPATH = "./datasets/"

def load_csr_matrix(filename):
    with open(filename,'r') as in_file:
        data,indices,indptr = [],[],[0]

        labels = []
        ptr = 0

        for line in in_file:
            line = line.split(None, 1)
            if len(line) == 1: 
                line += ['']
            label, features = line
            labels.append(float(label))

            f_list = features.split()
            for f in f_list:

                k,v = f.split(':')
                data.append(float(v))
                indices.append(float(k)-1)

            ptr += len(f_list)
            indptr.append(ptr)

        return labels,csr_matrix((data, indices, indptr))

def load_dataset(name,norm=False):

    if name == "letter":
        m = 16000
        y,x = load_csr_matrix(DATAPATH+"letter-recognition.data.sparse")
        train_y,test_y = y[:m],y[m:]
        train_x,test_x = x[:m],x[m:]

    else:
        if name == "svmguide1":
            train_path = DATAPATH+name
            test_path = DATAPATH+name+'.t'

        elif name == "ijcnn1":
            train_path = DATAPATH+name+'.tr'
            test_path = DATAPATH+name+'.t'

        elif name == "mnist":
            train_path = DATAPATH+name+'_train.csv.sparse'
            test_path = DATAPATH+name+'_test.csv.sparse'


        train_y,train_x = load_csr_matrix(train_path)
        
        test_y,test_x = load_csr_matrix(test_path)

    if norm:
        return train_y,normalize(train_x),test_y,normalize(test_x)
    else:
        return train_y,scale(train_x,with_mean=False),test_y,scale(test_x,with_mean=False)

def load_dense_dataset(dataset_name,norm=False):
    dataset = np.loadtxt(DATAPATH+dataset_name+".txt")
    if dataset_name == "sonar":
        x,y = np.split(dataset,[-1],axis=1)
    elif dataset_name == "ionosphere":
        x,y = np.split(dataset,[-1],axis=1)
    elif dataset_name == "heart-statlog":
        y,x = np.split(dataset,[1],axis=1)
    elif dataset_name == "liver":
        x,y = np.split(dataset,[-1],axis=1)
        y[y==2] = -1
    else:
        raise Exception("Unknown dataset: please implement a loader.")

    if norm:
        return csr_matrix(normalize(x)),y
    else:
        return csr_matrix(scale(x)),y

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog,dataset_name="svmguide1",nb_clusters=1,nb_landmarks=10,linear=True,pca=False,nb_iterations=1):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset", dest='dataset_name', default=dataset_name,
                        help='dataset name')
    parser.add_argument("-n", "--nbclusters", type=int, dest='nb_clusters', default=nb_clusters,
                        help='number of clusters')
    parser.add_argument("-l", "--nblands", type=int, dest='nb_landmarks', default=nb_landmarks,
                        help='number of landmarks')
    parser.add_argument("-o", "--normalize", dest='norm', action="store_true",
                        help='if set, the dataset is normalized')
    parser.add_argument("-i", "--nbiter", type=int, dest='nb_iterations', default=nb_iterations,
                        help='number of times the learning is repeated')
    parser.add_argument("-r", "--rbfk", dest='linear', action="store_false",
                        help='if set, the rbf kernel is used')
    parser.add_argument("-p", "--pca", dest='pca', action="store_true",
                        help='if set, the landmarks are selected as the principal components')

    return parser.parse_args()