import time
import statistics

from sklearn.model_selection import KFold

import l3svms
from src.utils import *

args = get_args(__file__,False)
TRAIN = args.train_file

LAND = args.nb_landmarks # default 10
CLUS = args.nb_clusters # default 1
NORM = args.norm # default False
LIN = args.linear # default True
PCA_BOOL = args.pca # default False
ITER = args.nb_iterations # default 1
VERB = args.verbose # default False
YPOS = args.y_pos # default 0
CV = args.nb_cv # default 5

verboseprint = print if VERB else lambda *a, **k: None

verboseprint("{}-fold cross-validation on {}: {} clusters, {} landmarks".format(CV,TRAIN,CLUS,LAND))

if LIN:
    verboseprint("linear kernel")
else:
    verboseprint("rbf kernel")

if NORM:
    verboseprint("normalized dataset")
else:
    verboseprint("scaled data")

t1 = time.time()

# load dataset
try:
    Y,X = load_sparse_dataset(TRAIN,norm=NORM,y_pos=YPOS)
except:
    Y,X = load_dense_dataset(TRAIN,norm=NORM,y_pos=YPOS)

Y = np.asarray(Y)

t2 = time.time()
verboseprint("dataset loading time:",t2-t1,"s")

if PCA_BOOL:
    if LAND > train_x.shape[1]:
        raise Exception("When using PCA, the nb landmarks must be at most the nb of features")
    verboseprint("landmarks = principal components")
else:
    verboseprint("random landmarks")

verboseprint("--------------------\n")
cross_acc_list = []
cross_time_list = []

for it in range(ITER):

    splitter = KFold(n_splits=CV,shuffle=True,random_state=it)
    splitter.get_n_splits(X)

    acc_list,time_list = [],[]
    for train_index,test_index in splitter.split(X):
        train_x,test_x = X[train_index],X[test_index]
        train_y,test_y = Y[train_index].tolist(),Y[test_index].tolist()

        acc,time = l3svms.learning(train_x,train_y,test_x,test_y,verboseprint,CLUS,PCA_BOOL,LIN,LAND)
        acc_list.append(acc)
        time_list.append(time)

    cross_time_list.append(statistics.mean(time_list))
    cross_acc_list.append(statistics.mean(acc_list))

print("Mean accuracy (%), mean stdev (%), mean time (s) over {} iterations:".format(ITER))
try:
    print(statistics.mean(cross_acc_list),statistics.stdev(cross_acc_list),statistics.mean(cross_time_list))
except:
    print(cross_acc_list[0],0.,cross_time_list[0])