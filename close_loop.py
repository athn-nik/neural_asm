# -*- coding: utf-8 -*-

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import pickle
from sklearn import linear_model
from scipy.stats import spearmanr,pearsonr,ttest_ind,ttest_rel
import json
from scipy import spatial
import gensim
from data_input import train_set,test_set,forbidden,fuse_input
from sklearn.decomposition import PCA
from operator import add
#from data_input import fuse_input
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sknn.mlp import Regressor, Layer
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
def plot_fn(plot_data1,plot_data2):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    y1 = tsne.fit_transform(plot_data1)
    y2 = tsne.fit_transform(plot_data2)
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    fig1.scatter(y1[:, 0], y1[:, 1], c='b', marker="s", label='ours')
    for label, x, y in zip(nns, y1[i:(i+1), 0], y1[:, 1]):
        fig1.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    fig2.scatter(y2[:, 0], y2[:, 1], c='r', marker="o", label='w2vec')
    for label, x, y in zip(nns, y2[:, 0], y2[:, 1]):
        fig2.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    #plt.legend(loc='upper left')

#glove_file = datapath('/home/nathan/Desktop/diploma/code/utils/embeddings/glove.42B.300d_raw.txt')
tmp_file = get_tmpfile('/home/nathan/Desktop/diploma/code/utils/embeddings/glove.42B.300d.txt')

#glove2word2vec(glove_file, tmp_file)
#model = KeyedVectors.load_word2vec_format(tmp_file)
svd = TruncatedSVD(n_components=5,  random_state=42)

dim_red=0
pca_dim=5
pca = PCA(n_components=pca_dim)
load_from_scratch = 0
dataset = 'men'# 'men'

if load_from_scratch:
    w2vec_whole = gensim.models.KeyedVectors.load_word2vec_format(\
    '../utils/embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)

    whole_set = train_set+test_set
    all_wds = [[x[0],x[1]] for x in whole_set]
    flatten = [item for sublist in all_wds for item in sublist]
    unique_test = set(flatten)
    whole_set_words =list(unique_test)

    w2vec=dict()
    for word in whole_set_words:
        print(word)
        if word=='theatre':
            word='theater'
        if word=='harbour':
            word='harbor'
        if word=='colour':
            word='color'
        w2vec[word]=  w2vec_whole[word]
    with open('w2vec_ws353.pkl', 'wb') as f:
        pickle.dump(w2vec, f, pickle.HIGHEST_PROTOCOL)
else:
    with open('w2vec_'+dataset+'.pkl', 'rb') as f:
        w2vec = pickle.load(f)

bsl_vectors=np.zeros((599,300))
widx=0
w_seq=[]
if dim_red:
    for x in w2vec:
        w_seq+=[x]
        bsl_vectors[widx,:]=w2vec[x]
        widx+=1
    bsl_vectors_reduced = pca.fit_transform(bsl_vectors)
    assert bsl_vectors_reduced.shape == (599,pca_dim)
    w2vec=dict()
    for w_id,x in enumerate(w_seq):
        w2vec[x] = bsl_vectors_reduced[w_id,:]
skip_or_cooc='sem_feat.txt'#or _glove
feat='cooc'
embeds = np.load('/home/nathan/Desktop/diploma/code/utils/embeddings/embds_men.npy').item()
i=0
forbidden=['drive','eat','break','run']
verbs=['see' ,'say' ,'taste' ,'wear' ,'open' ,'run','neared','eat','hear' ,'drive' \
,'ride' ,'touch' ,'break','enter','move' ,'listen' ,'approach' ,'fill','clean','lift', 'rub' ,
'smell' ,'fear' ,'push','manipulate' ]
partition=1
early_fusion = 0
late_fusion = not early_fusion
fuse_way = 'add'

for stable_voxels in [50,100,150,200,225,250]:
    print("----------------------------------------------")
    print("Results for the %d stablest voxels",stable_voxels)
    print("----------------------------------------------")
    sum_pear=sum_spear=sum_pear_l=sum_spear_l=sum_pear_m=sum_spear_m=sum_pear_h=sum_spear_h=0
    sum_bsl_s=sum_bsl_p=bsl_sum_pear_l=bsl_sum_spear_l=bsl_sum_pear_h=bsl_sum_spear_h=0
    sum_pear_hl=sum_spear_hl=0
    sum_pear_l_fus=sum_spear_l_fus=sum_pear_h_fus=sum_spear_h_fus=sum_pear_hl_fus=sum_spear_hl_fus=0
    sum_pear_fused=sum_spear_fused=0
    how_many=9
    num_parts=range(1,how_many+1)
    for part in num_parts:
        weights_extracted=np.loadtxt('../train_all/'+str(stable_voxels)+\
        'st_vox'+'/coeffs'+str(part)+'.txt',dtype=float)
        if early_fusion:
            train_data=np.zeros((len(train_set),stable_voxels+300))
            train_data1=np.zeros((len(train_set),stable_voxels+300))
            train_data2=np.zeros((len(train_set),stable_voxels+300))
        #elif dim_red:
        #    train_data = np.zeros((len(train_set),pca_dim))
        else:
            train_data=np.zeros((len(train_set),stable_voxels))
            train_data1=np.zeros((len(train_set),stable_voxels))
            train_data2=np.zeros((len(train_set),stable_voxels))

        targets=np.zeros((len(train_set),1))
        i=0
        mini=float("inf")
        maxi=-1
        for x in train_set:
            e1_t=np.append(embeds[x[0]],1)
            e2_t=np.append(embeds[x[1]],1)
            e1_t=e1_t.reshape(26,1)
            e2_t=e2_t.reshape(26,1)
            pred_1_t=np.dot(weights_extracted,e1_t)
            pred_2_t=np.dot(weights_extracted,e2_t)
            if early_fusion:
                try:
                    w_v1 = w2vec[x[0]]
                except KeyError:
                    w_v1 = np.random.uniform(0.0,1.0,300)
                try:
                    w_v2 = w2vec[x[1]]
                except KeyError:
                    w_v2 = np.random.uniform(0.0,1.0,300)
                pred_1_t=np.append(pred_1_t,w_v1)
                pred_2_t=np.append(pred_2_t,w_v2)
            diff_t=abs(pred_2_t-pred_1_t)**2
            dist_t=diff_t.reshape(len(diff_t))
            train_data1[i,:]=pred_1_t.reshape(stable_voxels)
            train_data2[i,:]=pred_2_t.reshape(stable_voxels)
            targets[i,0]=float(x[2])
            if float(x[2])>maxi:
                maxi=float(x[2])
            if float(x[2])<mini:
                mini=float(x[2])
            i+=1
        if dim_red:
            train_data1=pca.fit_transform(train_data1)
            train_data2=pca.fit_transform(train_data2)
        train_data = abs(train_data1-train_data2)**2
        targets=(targets-mini)/(maxi-mini)
        model = linear_model.LinearRegression(fit_intercept=True,normalize=True)
        #model = Regressor(layers=[Layer("Tanh", units=stable_voxels),Layer("Tanh", units=20),Layer("Tanh", units=5),Layer("Linear")],
        #               learning_rate=0.01,n_iter=30,n_stable=8,batch_size=8,learning_rule='nesterov')
        #model.fit(train_data, targets)
        model.fit(train_data,targets)
        mle_est=model.coef_
        bias=model.intercept_
        bias=np.array(bias)
        mle_est=np.append(mle_est,bias)
        mini=float("inf")
        maxi=-1
        mini2=float("inf")
        maxi2=-1
        sum1=0
        estimated_similarity=[]
        real=[]
        bsl=[]
        if early_fusion:
            test_data=np.zeros((len(test_set),stable_voxels+300))
            test_data1=np.zeros((len(test_set),stable_voxels+300))
            test_data2=np.zeros((len(test_set),stable_voxels+300))
        #elif dim_red:
        #    test_data=np.zeros((len(test_set),pca_dim))
        else:
            test_data=np.zeros((len(test_set),stable_voxels))
            test_data1=np.zeros((len(test_set),stable_voxels))
            test_data2=np.zeros((len(test_set),stable_voxels))
        j=0
        for x in test_set:
            e1_te=np.append(embeds[x[0]],1)
            e1_te=e1_te.reshape(26,1)#26
            e2_te=np.append(embeds[x[1]],1)
            e2_te=e2_te.reshape(26,1)

            try:
                bsl_pair=(1-spatial.distance.cosine(w2vec[x[0]],w2vec[x[1]]))
                bsl.append([bsl_pair])
            except KeyError:
                bsl.append([0.5])

            pred_1_te=np.dot(weights_extracted,e1_te)
            pred_2_te=np.dot(weights_extracted,e2_te)
            if early_fusion:
                try:
                    w_v1 = w2vec[x[0]]
                except KeyError:
                    w_v1 = np.random.uniform(0.0,1.0,300)
                try:
                    w_v2 = w2vec[x[1]]
                except KeyError:
                    w_v2 = np.random.uniform(0.0,1.0,300)
                pred_1_te=np.append(pred_1_te,w_v1)
                pred_2_te=np.append(pred_2_te,w_v2)
            diff_te=abs(pred_2_te-pred_1_te)**2
            diff_te=np.append(diff_te,1)
            helper=diff_te.reshape(len(diff_te))
            test_data1[j,:]=pred_1_te.reshape(stable_voxels)
            test_data2[j,:]=pred_2_te.reshape(stable_voxels)

            #test_data[j,:]=helper
            j+=1
            #est_sim=np.dot(diff_te,mle_est)
            #estimated_similarity.append(est_sim)
            real.append(float(x[2]))
            # plt.figure(1)

            # plt.plot(w2vec[x[0]])

            # plt.plot(pred_1_te)

            #plt.plot(w2vec[x[1]])

            # plt.plot(pred_2_te)
            # plt.title(str(x[0])+" "+str(x[1]))
            # plt.show()
            #if est_sim>maxi2:
            #    maxi2=est_sim
            #if est_sim<mini2:
            #    mini2=est_sim
            if float(x[2])>maxi:
                maxi=float(x[2])
            if float(x[2])<mini:
                mini=float(x[2])
  #Normalization of similarities
  #estimated_similarity=[(val-mini2)/(maxi2-mini2) for val in estimated_similarity]
        # Preparation of score for baseline and our model
        pca2 = PCA(n_components=pca_dim+1)
        if dim_red:
            test_data1=pca2.fit_transform(test_data1)
            test_data2=pca2.fit_transform(test_data2)
        test_data = abs(test_data1-test_data2)**2
        #predictions=model.predict(test_data)
        #for i in predictions:
            #print(i)
        #    estimated_similarity.append(i[0])
        for i in range(test_data.shape[0]):
            if not dim_red:
                est_sim=np.dot(np.append(test_data[i,:],1),mle_est)
            else:
                est_sim=np.dot(test_data[i,:],mle_est)
            estimated_similarity.append(est_sim)
        #print(len(estimated_similarity))
        real=[(val_real-mini)/(maxi-mini) for val_real in real]
        bsl=[x[0] for x in bsl]
        c=0
        for i in range(len(real)):
            #print(real[i],estimated_similarity[i],bsl[i])
            if abs(estimated_similarity[i]-real[i])<abs(bsl[i]-real[i]):
                c+=1
        #print(c*1.0/len(test_set))
        real_low=[x for x in real if x<0.1]
        real_low_index=[real.index(x) for x in real if x<0.1]
        real_high=[x for x in real if x>0.85]
        real_high_index=[real.index(x) for x in real if x>0.85]
        #print(len(real_high))
        #print(len(real_low))
        # plot_data1 = np.zeros((len(real_high_index)*2,stable_voxels))
        # plot_data2 = np.zeros((len(real_high_index)*2,300))
        # k = 0
        # nns = []
        # for id in real_high_index:
        #     plot_data1[k,:] = test_data1[id,:]
        #     plot_data2[k,:] = w2vec[test_set[id][0]]
        #     k+=1
        #     plot_data1[k,:] = test_data2[id,:]
        #     plot_data2[k,:] = w2vec[test_set[id][1]]
        #     k+=1
        #     print(test_data1[id,:])
        #     print(test_data2[id,:])
        #     print(w2vec[test_set[id][0]])
        #     print(w2vec[test_set[id][1]])
        #     nns += [test_set[id][0]]
        #     nns += [test_set[id][1]]
        estima_low= [estimated_similarity[idx] for idx in real_low_index]
        estima_high= [estimated_similarity[idx] for idx in real_high_index]
        bsl_low= [bsl[idx] for idx in real_low_index]
        bsl_high= [bsl[idx] for idx in real_high_index]
        real=(np.array(real)).reshape(len(test_set),1)
        estimated_similarity=(np.array(estimated_similarity)).reshape(len(test_set),1)
        bsl=(np.array(bsl)).reshape(len(test_set),1)
        pps = zip(real.tolist(),estimated_similarity.tolist(),bsl.tolist())
        print(len(pps))
        pps = sorted(pps)
        real_srd = [pps[0] for x in pps]
        estim_srd = [pps[1] for x in pps]
        bsl_srd = [pps[2] for x in pps]
        plt.hist(real_srd,bins=10)
        plt.show()
        plt.hist(estim_srd,bins=10)
        plt.show()
        plt.hist(bsl_srd,bins=10)
        plt.show()
        #print("Number of low and high similar pairs : ")
        #print(len(real_low_index))
        #print(len(real_high_index))
        ########################################################################
        ########################################################################
        # Calculation of scores
        # All dataset
        ########################################################################
        ########################################################################

        res=spearmanr(estimated_similarity,real)[0]
        res2=pearsonr(estimated_similarity,real)[0]
        #bsl=np.array(bsl)
        #
        bsl_s=spearmanr(bsl,real)[0]
        bsl_p=pearsonr(bsl,real)[0]
        if late_fusion:
            res_fusion_pearsonr = pearsonr(fuse_input(estimated_similarity,bsl,fuse_way),real)[0]
            res_fusion_spearman = spearmanr(fuse_input(estimated_similarity,bsl,fuse_way),real)[0]
            sum_spear_fused+=res_fusion_pearsonr[0]
            sum_pear_fused+=res_fusion_spearman
            fus_lh_sp=spearmanr(fuse_input(bsl_low,estima_low,fuse_way)+\
                                fuse_input(bsl_high,estima_high,fuse_way),real_low+real_high)[0]
            fus_lh_pe=pearsonr(fuse_input(bsl_low,estima_low,fuse_way)+\
                                fuse_input(bsl_high,estima_high,fuse_way),real_low+real_high)[0]
            fus_l_pe=pearsonr(fuse_input(estima_low,bsl_low,fuse_way),real_low)[0]
            fus_h_pe=pearsonr(fuse_input(estima_high,estima_high,fuse_way),real_high)[0]
            fus_re_l_sp=spearmanr(fuse_input(bsl_low,estima_low,fuse_way),real_low)[0]
            fus_re_h_sp=spearmanr(fuse_input(bsl_high,estima_high,fuse_way),real_high)[0]
            sum_pear_hl_fus += fus_lh_pe
            sum_spear_hl_fus += fus_lh_sp

            sum_pear_h_fus += fus_h_pe
            sum_spear_h_fus += fus_re_h_sp
            sum_pear_l_fus += fus_l_pe
            sum_spear_l_fus += fus_re_l_sp

        sum_pear+=res2[0]
        sum_spear+=res

        ########################################################################
        ########################################################################
        # Low and high subsets
        ########################################################################
        ########################################################################
        re_lh_sp=spearmanr(estima_low+estima_high,real_low+real_high)[0]
        re_lh_pe=pearsonr(estima_low+estima_high,real_low+real_high)[0]
        bsl_lh_sp=spearmanr(bsl_low+bsl_high,real_low+real_high)[0]
        bsl_lh_pe=pearsonr(bsl_low+bsl_high,real_low+real_high)[0]

        #map(max, zip(L1, L2))
        re_l_sp=spearmanr(estima_low,real_low)[0]
        re_h_sp=spearmanr(estima_high,real_high)[0]
        re_l_pe=pearsonr(estima_low,real_low)[0]
        re_h_pe=pearsonr(estima_high,real_high)[0]

        bsl_re_l_sp=spearmanr(bsl_low,real_low)[0]
        bsl_re_h_sp=spearmanr(bsl_high,real_high)[0]
        bsl_re_l_pe=pearsonr(bsl_low,real_low)[0]
        bsl_re_h_pe=pearsonr(bsl_high,real_high)[0]


        # High+Low similar
        sum_pear_hl += re_lh_pe
        sum_spear_hl += re_lh_sp
        # Low similar
        sum_pear_l+=re_l_pe
        sum_spear_l+=re_l_sp
        # High similar
        sum_pear_h += re_h_pe
        sum_spear_h += re_h_sp

        bsl_sum_pear_l+=bsl_re_l_pe
        bsl_sum_spear_l+=bsl_re_l_sp

        bsl_sum_pear_h+=bsl_re_h_pe
        bsl_sum_spear_h+=bsl_re_h_sp
        #
        t, p = ttest_rel(bsl_low+bsl_high, estima_low+estima_high)#, equal_var=False)
        print("ttest_ind for low+high:            t = %g  p = %g" % (t, p))
        #
        t, p = ttest_rel(estima_low, bsl_low)#, equal_var=False)
        print("ttest_ind for low:            t = %g  p = %g" % (t, p))
        #
        t, p = ttest_rel(estima_high, bsl_high)#, equal_var=False)
        print("ttest_ind for high:            t = %g  p = %g" % (t, p))

    #  print("Spearman:",res)
    #  print("Pearson:",res2[0])
    #
    #  print("Spearman(low correlated words):",re_l_sp)
    #  print("Pearson:(low correlated words)",re_l_pe)
    #
    #  #print("Spearman(med correlated words):",re_m_sp)
    #  #print("Pearson(med correlated words):",re_m_pe)
    #
    #  print("Spearman(highly correlated words):",re_h_sp)
    #  print("Pearson(highly correlated words):",re_h_pe)

    print("High and low")
    print("category\tpearson \t spearman")
    print("predicted\t"+str(sum_pear_hl/how_many)+'\t'+str(sum_spear_hl/how_many))
    print("fused\t"+str(sum_pear_hl_fus/how_many)+'\t'+str(sum_spear_hl_fus/how_many))
    print("baseline\t"+str(bsl_lh_pe)+'\t'+str(bsl_lh_sp))
    print("High")
    print("category\tpearson \t spearman")
    print("predicted\t"+str(sum_pear_h/how_many)+'\t'+str(sum_spear_h/how_many))
    print("fused\t"+str(sum_pear_h_fus/how_many)+'\t'+str(sum_spear_h_fus/how_many))
    print("baseline\t"+str(bsl_sum_pear_h/how_many)+'\t'+str(bsl_sum_spear_h/how_many))

    print("Low")
    print("category\tpearson \t spearman")
    print("predicted\t"+str(sum_pear_l/how_many)+'\t'+str(sum_spear_l/how_many))
    print("fused\t"+str(sum_pear_l_fus/how_many)+'\t'+str(sum_spear_l_fus/how_many))
    print("baseline\t"+str(bsl_sum_pear_l/how_many)+'\t'+str(bsl_sum_spear_l/how_many))

    print("Whole Dataset")
    print("category\tpearson \tspearman")
    print("predicted\t"+str(sum_pear/how_many)+'\t'+str(sum_spear/how_many))
    print("fused\t"+str(sum_pear_fused/how_many)+'\t'+str(sum_spear_fused/how_many))
    print("baseline\t"+str(bsl_p)+'\t'+str(bsl_s))

    # print("Spearman avg",sum_spear/how_many)
    # print("Pearson avg",sum_pear/how_many)
    # print("Spearman avg Baseline ",bsl_s)
    # print("Pearson avg Baseline ",bsl_p)
    #
    # print("Spearman avg(low correlated words):",sum_spear_l/how_many)
    # print("Pearson avg:(low correlated words)",sum_pear_l/how_many)
    #
    # print("Spearman avg baseline(low correlated words):",bsl_sum_spear_l)
    # print("Pearson avg baseline:(low correlated words)",bsl_sum_pear_l)
    # #print("Spearman avg(med correlated words):",sum_spear_m/9)
    # #print("Pearson avg (med correlated words):",sum_pear_m/9)
    #
    # print("Spearman avg(highly correlated words):",sum_spear_h/how_many)
    # print("Pearson avg(highly correlated words):",sum_pear_h/how_many)
    # #
    # print("Spearman avg baseline(highly correlated words):",bsl_sum_spear_h)
    # print("Pearson avg baseline:(higlhy correlated words)",bsl_sum_pear_h)
