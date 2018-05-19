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
import gensim
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import cPickle as pickle
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
if sys.argv[2] == '1' :
    print(sys.argv[3]+' '+sys.argv[4])
for sv in ['50','100','150','200','225','250']:
    if sys.argv[2] == 'no_smell':
        with open('./dict_similarities/sim_'+sv+'_'+dataset+'no_smell.pkl', 'rb') as handle:
            s_m = pickle.load(handle)
        with open('./dict_similarities/nouns_in_'+dataset+'no_smell.pkl','rb') as n_s:
            nouns_in_dataset=pickle.load(n_s)
    else:
        with open('./dict_similarities/sim_'+sv+'_'+dataset+'.pkl', 'rb') as handle:
            s_m = pickle.load(handle)
        with open('./dict_similarities/nouns_in_'+dataset+'.pkl','rb') as n_s:
            nouns_in_dataset=pickle.load(n_s)
    nouns=[]

    if sys.argv[2] == '3':
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/senses/sensi_vision_hear_other/*')
    elif sys.argv[2] == '2':
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/senses/sensi_audition_other/*')
    elif sys.argv[2] == '1':
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/senses/'+
        sys.argv[3]+'_'+sys.argv[4]+'/*')
    elif sys.argv[2] == 'all':
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/senses/'+dataset+'/*')
    elif sys.argv[2] == 'no_smell':
        flst=glob.glob('/home/nathan/Desktop/diploma/data/task_categorization/senses/'+dataset+'_no_smell/*')
    labels_dict={}
    sendict=dict()
    for f_index,f in enumerate(flst):
        with open(f,'r') as category:
            for line in category:
                fn = f.split('/')
                sense = fn[-1]
                sense = sense[:-4]
                noun=line.strip()
                if sense not in sendict :
                    sendict[sense]=[noun]
                else:
                    sendict[sense].append(noun)
                labels_dict[noun]=f_index
    labels_true = []
    if sys.argv[2] == '1':
        sn1_sn2_idx = []
        sn1_sn2_nouns = []
        if sys.argv[3] == 'vision' and sys.argv[4] == 'audition' :
            sense_1 = 'Hearing'
            sense_2 = 'Sight'
        if sys.argv[3] == 'vision' and sys.argv[4] == 'touch' :
            sense_1 = 'Sight'
            sense_2 = 'Touch'
        if sys.argv[3] == 'vision' and sys.argv[4] == 'taste' :
            sense_1 = 'Sight'
            sense_2 = 'Taste'
        if sys.argv[3] == 'audition'and sys.argv[4] == 'taste' :
            sense_1 = 'Hearing'
            sense_2 = 'Taste'
        if sys.argv[3] == 'audition' and sys.argv[4] == 'touch' :
            sense_1 = 'Hearing'
            sense_2 = 'Touch'
        if sys.argv[3] == 'touch' and sys.argv[4] == 'taste' :
            sense_1 = 'Touch'
            sense_2 = 'Taste'
        for nn_idx,nn in enumerate(nouns_in_dataset):
            if (nn in sendict[sense_1]) or (nn in sendict[sense_2]):
                sn1_sn2_idx += [nn_idx]
                sn1_sn2_nouns += [nn]
        tot_n=len(sn1_sn2_nouns)
        sim_matrix=np.zeros((tot_n,tot_n))
        for key in s_m.keys():
            if key not in sn1_sn2_nouns:
                del s_m[key]
        for key in s_m:
            s_m[key] = [i for j, i in enumerate(s_m[key]) if j in sn1_sn2_idx]
        for idx_no,no in enumerate(sn1_sn2_nouns):
            sim_matrix[idx_no,:]=s_m[no]
        for bn in sn1_sn2_nouns:
            labels_true+=[labels_dict[bn]]
    else:
        tot_n=len(s_m)
        sim_matrix=np.zeros((tot_n,tot_n))
        for idx_no,no in enumerate(nouns_in_dataset):
            sim_matrix[idx_no,:]=s_m[no]
        for bn in nouns_in_dataset:
            labels_true+=[labels_dict[bn]]
    sim_matrix[range(sim_matrix.shape[0]), range(sim_matrix.shape[0])] = 1.0
    #convert array appropriately
    i_lower = np.tril_indices(tot_n, -1)
    sim_matrix[i_lower] = sim_matrix.T[i_lower]
    symmetric=np.allclose(sim_matrix, sim_matrix.T, atol=1e-8)
    #scipy.io.savemat('./sm_'+sv+'.mat', mdict={'arr': sim_matrix})
    maxs=np.amax(sim_matrix)
    mins=np.amin(sim_matrix)
    sim_matrix = np.add(-mins+10.0*(maxs-mins),sim_matrix)
    sim_matrix = np.divide(maxs-mins,sim_matrix)

    model=SpectralClustering(n_clusters=len(flst),affinity='precomputed').fit(sim_matrix)
    #model = AffinityPropagation(preference=-50,affinity='precomputed').fit(sim_matrix)
    labels=model.labels_
    #labels=clstr.labels_
    l_t= np.reshape(labels_true,(tot_n,1))
    l_p=np.reshape(labels,(tot_n,1))
    p=purity_score(l_t,l_p)
    print("Purity score for "+sv+" stable voxels is: "+str(p)+"\n")
