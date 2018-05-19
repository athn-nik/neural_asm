#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import operator
import pickle
import glob
import scipy.io
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

"""
Computation of purity score with sklearn.
"""
def purity_score(y_true, y_pred):
    """Purity score

    To compute purity, each cluster is assigned to the class which is most frequent
    in the cluster [1], and then the accuracy of this assignment is measured by counting
    the number of correctly assigned documents and dividing by the number of documents.abs

    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters

    Returns:
        float: Purity score

    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
       y_true = np.random.randint(5, size=(20,1))
       y_pred = np.random.randint(5, size=(20,1))
    the way to call it
    """
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_labeled_voted[y_pred==cluster] = winner

    return accuracy_score(y_true, y_labeled_voted)


#c_n=np.load('../corrupt_nouns.npy').item()
#n_seq=np.load('./nouns_seq.npy').item()
dataset=sys.argv[1]#AP
for sv in ['50','100','150','200','225','250']:
    with open('./dict_similarities/sim_'+sv+'_'+dataset+'.pkl', 'rb') as handle:
        s_m = pickle.load(handle)
    with open('./dict_similarities/nouns_in_'+dataset+'.pkl','rb') as n_s:
        nouns_in_dataset=pickle.load(n_s)
    sim_matrix=np.zeros((len(s_m),len(s_m)))
    nouns=[]
    for idx_no,no in enumerate(nouns_in_dataset):
        sim_matrix[idx_no,:]=s_m[no]
    if dataset=='essli':
        taxonomy=3
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/'+dataset+'_'+str(taxonomy)+'/*')
    else:
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/'+dataset+'/*')
    labels_dict={}
    print(flst)
    for f_index,f in enumerate(flst):
        with open(f,'r') as category:
            for line in category:
                noun=line.strip()
                labels_dict[noun]=f_index
    labels_true=[]
    for bn in nouns_in_dataset:
        labels_true+=[labels_dict[bn]]

    sim_matrix[range(sim_matrix.shape[0]), range(sim_matrix.shape[0])] = 1.0
    #convert array appropriately
    i_lower = np.tril_indices(len(s_m), -1)
    sim_matrix[i_lower] = sim_matrix.T[i_lower]
    symmetric=np.allclose(sim_matrix, sim_matrix.T, atol=1e-8)
    #scipy.io.savemat('./sm_'+sv+'.mat', mdict={'arr': sim_matrix})
    # print(symmetric)
    maxs=np.amax(sim_matrix)
    mins=np.amin(sim_matrix)
    sim_matrix = np.add(10.0,sim_matrix)
    model=SpectralClustering(n_clusters=len(flst),affinity='precomputed').fit(sim_matrix)
    #model = AffinityPropagation(preference=-50,affinity='precomputed').fit(sim_matrix)
    labels=model.labels_
    #labels=clstr.labels_
    print(labels)
    l_t= np.reshape(labels_true,(len(s_m),1))
    l_p=np.reshape(labels,(len(s_m),1))
    p=purity_score(l_t,l_p)
    print("Purity score for "+sv+" stable voxels is: "+str(p)+"\n")
