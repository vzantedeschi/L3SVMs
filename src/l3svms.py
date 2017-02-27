import time

from liblinearutil import *
from sklearn import cluster

from src.landmark import *
from src.projection import *

def learning(train_x,train_y,test_x,test_y,printf=print,CLUS=1,PCA_BOOL=False,LIN=True,LAND=10):

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
    printf("clustering time:",t3-t2,"s")

    # select landmarks
    if PCA_BOOL:
        landmarks = pca_landmarks(train_x.toarray(),LAND)
    else:
        landmarks = random_landmarks(train_x,LAND)
    # centered kernel
    u = None

    t2 = time.time()
    printf("landmarks selection time:",t2-t3,"s")
    t2 = time.time()

    # project data
    # tr_x = project(train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)
    tr_x = parallelized_projection(-1,train_x,landmarks,clusters=train_clusters,unit_vectors=u,linear=LIN)

    t3 = time.time()
    printf("projection time:",t3-t2,"s")
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
    printf("tuning time:",t4-t3,"s")
    # training
    model = train(train_y, tr_x, '-c {} -s 2 -B 1 -q'.format(best_C))
    assert model.nr_feature == LAND*CLUS

    t5 = time.time()
    printf("training time:",t5-t4,"s")

    te_x = parallelized_projection(-1,test_x,landmarks,clusters=test_clusters,unit_vectors=u,linear=LIN)
    # te_x = project(test_x,landmarks,clusters=test_clusters,unit_vectors=u,linear=LIN)
    p_label,p_acc,p_val = predict(test_y, te_x, model)

    t6 = time.time()
    printf("testing time:",t6-t5,"s")

    printf("iteration results: (accuracy,mean squared error,squared correlation coefficient), learning time")
    printf(evaluations(test_y,p_label),t6-t2)

    print("-------------------\n")

    return p_acc[0],t6-t5