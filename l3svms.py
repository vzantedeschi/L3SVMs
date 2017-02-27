import time
import statistics

from liblinearutil import *
from sklearn import cluster

from src.landmark import *
from src.projection import *
from src.utils import *

args = get_args(__file__)
LAND = args.nb_landmarks # default 10
CLUS = args.nb_clusters # default 1
DATASET = args.dataset_name # default svmguide1
NORM = args.norm # default False
LIN = args.linear # default True
PCA_BOOL = args.pca # default False

print("learning on {}: {} clusters, {} landmarks".format(DATASET,CLUS,LAND))

if LIN:
    print("linear kernel")
else:
    print("rbf kernel")
    CENT = False
if NORM:
    print("normalized dataset")
else:
    print("scaled data")

t1 = time.time()
# load dataset
train_y,train_x,test_y,test_x = load_dataset(DATASET,norm=NORM)
t2 = time.time()
print("dataset loading time:",t2-t1)

if PCA_BOOL:
    if LAND > train_x.shape[1]:
        raise Exception("When using PCA, the nb landmarks must be at most the nb of features")
    print("landmarks = principal components")
else:
    print("random landmarks")

acc_list = []
for it in range(2):

    t2 = time.time()
    if CLUS > 1:
        # get clusterer
        clusterer = cluster.MiniBatchKMeans(n_clusters=CLUS)
        train_clusters = clusterer.fit(train_x).labels_
        test_clusters = clusterer.predict(test_x)
    else:
        clusterer = None
        train_clusters = None
        test_clusters = None
    t3 = time.time()
    print("clustering time:",t3-t2)

    # select landmarks
    if PCA_BOOL:
        landmarks = pca_landmarks(train_x.toarray(),LAND)
    else:
        landmarks = random_landmarks(train_x,LAND)
    # centered kernel
    u = None

    t2 = time.time()
    print("landmarks selection time:",t2-t3)
    t2 = time.time()

    # project data
    # tr_x = project(train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)
    tr_x = parallelized_projection(-1,train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)

    t3 = time.time()
    print("projection time:",t3-t2)
    t3 = time.time()

    # tuning
    best_C,_ = train(train_y, tr_x, '-C -s 2 -B 1')

    t4 = time.time()
    print("tuning time:",t4-t3)
    # training
    m = train(train_y, tr_x, '-c {} -s 2 -B 1'.format(best_C))
    assert m.nr_feature == LAND*CLUS

    t5 = time.time()
    print("training time:",t5-t4)

    te_x = parallelized_projection(-1,test_x,landmarks,clusters=test_clusters,unit_vectors=u,linear=LIN)
    # te_x = project(test_x,landmarks,clusters=test_clusters,unit_vectors=u,linear=LIN)
    p_label,p_acc,p_val = predict(test_y, te_x, m)

    acc_list.append(p_acc[0])

    t6 = time.time()
    print("testing time:",t6-t5)

    print(evaluations(test_y,p_label),t6-t1)

try:
    print(statistics.mean(acc_list),statistics.stdev(acc_list))
except:
    print(acc_list[0],0.)
