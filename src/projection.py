#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS

from functools import partial
import multiprocessing as mp
from contextlib import closing

from sklearn.metrics import pairwise

from src.utils import array_to_dict

def project(x,landmarks,clusters=None,unit_vectors=None,linear=True,gamma=1):

    # project on landmark space
    if unit_vectors is not None:
        if linear:
            projection = x.dot(unit_vectors-landmarks.transpose())
        else:
            raise Exception("centered rbf kernel not supported")
    else:
        if linear:
            projection = x.dot(landmarks.transpose())
        else:
            projection = pairwise.rbf_kernel(x,landmarks,gamma=gamma)

    if clusters is not None:
        assert len(clusters) == x.shape[0]
        return array_to_dict(projection,clusters=clusters,land=landmarks.shape[0])
    else:
        return array_to_dict(projection)

def project_with_id(indexes,x,landmarks,clusters=None,unit_vectors=None,linear=True,gamma=1):
    a = x[indexes]
    # project on landmark space
    if unit_vectors is not None:
        if linear:
            projection = a.dot(unit_vectors-landmarks.transpose())
        else:
            raise Exception("centered rbf kernel not supported")
    else:
        if linear:
            projection = a.dot(landmarks.transpose())
        else:
            projection = pairwise.rbf_kernel(a,landmarks,gamma=gamma)

    if clusters is not None:
        return array_to_dict(projection,clusters=clusters[indexes],land=landmarks.shape[0])
    else:
        return array_to_dict(projection)

def parallelized_projection(nb_jobs,x,landmarks,clusters=None,unit_vectors=None,linear=True,gamma=1):
    if nb_jobs <= 0:
        nb_jobs = mp.cpu_count()-1

    m = x.shape[0]
    size = int(m/nb_jobs)

    partial_project = partial(project_with_id,x=x,clusters=clusters,landmarks=landmarks,unit_vectors=unit_vectors,linear=linear,gamma=gamma)

    with closing(mp.Pool(nb_jobs)) as pool:

        out = pool.map(partial_project,[range(s,min(m,s+size)) for s in range(0,m,size)])

        return [e for o in out for e in o]