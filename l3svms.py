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
    verboseprint("clustering time:",t3-t2,"s")

    # select landmarks
    if PCA_BOOL:
        landmarks = pca_landmarks(train_x.toarray(),LAND)
    else:
        landmarks = random_landmarks(train_x,LAND)
    # centered kernel
    u = None

    t2 = time.time()
    verboseprint("landmarks selection time:",t2-t3,"s")
    t2 = time.time()

    # project data
    # tr_x = project(train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)
    tr_x = parallelized_projection(-1,train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)

    t3 = time.time()
    verboseprint("projection time:",t3-t2,"s")
    t3 = time.time()

    # tuning
    if LIN:
        tr_x = parallelized_projection(-1,train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)
        best_C,_ = train(train_y, tr_x, '-C -s 2 -B 1 -q')
        best_G = None
    else:
        best_G,best_C,best_acc = 0,0,0
        for g in [10**i for i in range(-3,3)]:
            tr_x = parallelized_projection(-1,train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN,gamma=g)
            c,p = train(train_y, tr_x, '-C -s 2 -B 1 -q')
            if p > best_acc:
                best_C = c
                best_G = g
                best_acc = p

        tr_x = parallelized_projection(-1,train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN,gamma=best_G)
        print("Best Gamma =",best_G,"\n")

    t4 = time.time()
    verboseprint("tuning time:",t4-t3,"s")
    # training
    model = train(train_y, tr_x, '-c {} -s 2 -B 1 -q'.format(best_C))
    assert model.nr_feature == LAND*CLUS

    t5 = time.time()
    verboseprint("training time:",t5-t4,"s")

    te_x = parallelized_projection(-1,test_x,landmarks,clusters=test_clusters,unit_vectors=u,linear=LIN)
    # te_x = project(test_x,landmarks,clusters=test_clusters,unit_vectors=u,linear=LIN)
    p_label,p_acc,p_val = predict(test_y, te_x, model)

    acc_list.append(p_acc[0])

    t6 = time.time()
    verboseprint("testing time:",t6-t5,"s")
    time_list.append(t6-t5)

    verboseprint("iteration {} results: (accuracy,mean squared error,squared correlation coefficient), learning time".format(it))
    verboseprint(evaluations(test_y,p_label),t6-t2)

    print("-------------------\n")

print("Mean accuracy (%), mean stdev (%), mean time (s) over {} iterations:".format(ITER))
try:
    print(statistics.mean(acc_list),statistics.stdev(acc_list),statistics.mean(time_list))
except:
    print(acc_list[0],0.,time_list[0])