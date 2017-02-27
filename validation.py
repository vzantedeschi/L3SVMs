import time
import statistics

import l3svms

from src.utils import *

args = get_args(__file__)
LAND = args.nb_landmarks # default 10
CLUS = args.nb_clusters # default 1
DATASET = args.dataset_name # default svmguide1
NORM = args.norm # default False
LIN = args.linear # default True
PCA_BOOL = args.pca # default False
ITER = args.nb_iterations # default 1
VERB = args.verbose # default False

verboseprint = print if VERB else lambda *a, **k: None

verboseprint("learning on {}: {} clusters, {} landmarks".format(DATASET,CLUS,LAND))

if LIN:
    verboseprint("linear kernel")
else:
    verboseprint("rbf kernel")
    CENT = False
if NORM:
    verboseprint("normalized dataset")
else:
    verboseprint("scaled data")

t1 = time.time()
# load dataset
train_y,train_x,test_y,test_x = load_dataset(DATASET,norm=NORM)
t2 = time.time()
verboseprint("dataset loading time:",t2-t1,"s")

if PCA_BOOL:
    if LAND > train_x.shape[1]:
        raise Exception("When using PCA, the nb landmarks must be at most the nb of features")
    verboseprint("landmarks = principal components")
else:
    verboseprint("random landmarks")

verboseprint("--------------------\n")
acc_list = []
time_list = []

for it in range(ITER):

    acc,time = l3svms.learning(train_x,train_y,test_x,test_y,verboseprint,CLUS,PCA_BOOL,LIN,LAND)
    acc_list.append(acc)
    time_list.append(time)

print("Mean accuracy (%), mean stdev (%), mean time (s) over {} iterations:".format(ITER))
try:
    print(statistics.mean(acc_list),statistics.stdev(acc_list),statistics.mean(time_list))
except:
    print(acc_list[0],0.,time_list[0])