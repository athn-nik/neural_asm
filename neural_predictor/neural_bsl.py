#!/usr/bin/env python3
'''
Multiple Linear regression to obtain weights for prediction
inputs: semantic features vectors for nine participants
        P1-P9
output: cvi activation of voxel v for intermediate semantic feature i
(1-25) sensor-motor verbs
'''

#from extr_page import noun,sem_feat
import timeit
import scipy.io
from sklearn import linear_model
from scipy.stats.stats import pearsonr
import numpy as np
import itertools
from scipy import spatial
import glob
import sys
import re
from heapq import nlargest
import matplotlib
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Regressor, Layer

########################################################################
##############Cosine similarity computation for evaluation##############
########################################################################


def evaluation(i1,p1,i2,p2,metric):
 #print("Cosine Similarity Calculation...")
  #Normalize vectors
    '''i1[:]= [x*x for x in i1]
    magni1=np.sum(i1)
    i1[:]= [x/magni1 for x in i1]
    i2[:]= [x*x for x in i2]
    magni2=np.sum(i2)
    i2[:]= [x/magni2 for x in i2]
    p1[:]= [x*x for x in p1]
    magnp1=np.sum(p1)
    p1[:]= [x/magnp1 for x in p1]
    p2[:]= [x*x for x in p2]
    magnp2=np.sum(p2)
    p2[:]= [x/magnp2 for x in p2]'''
 #print(spatial.distance.cosine(p1,i2),spatial.distance.cosine(p2,i1),spatial.distance.cosine(i2,p2),spatial.distance.cosine(i1,p1))
    if metric=='cosine':
        bad=2-spatial.distance.cosine(p1,i2)-spatial.distance.cosine(p2,i1)
        good=2-spatial.distance.cosine(i2,p2)-spatial.distance.cosine(i1,p1)
    elif metric=='pearson':
        bad=scipy.stats.pearsonr(p1,i2)+scipy.stats.pearsonr(p2,i1)
        good=scipy.stats.pearsonr(i2,p2)+scipy.stats.pearsonr(i1,p1)
    else:
        print("You have given wrong parameter regarding similarity metric!")
        print("give pearson or cosine")
        sys.exit()
    if (bad<=good):
        return 1
    else :
        return 0

if __name__ == '__main__':

 ###################################################################
 ######Load semantic features and handle execution requests#########
 ###################################################################
    skip_or_cooc='../sem_feat.txt'
    if skip_or_cooc=='sem_feat_skipgram.txt':
        latent_dims=300
    elif skip_or_cooc=='../sem_feat.txt':
        latent_dims=25
    elif skip_or_cooc=='sem_feat_glove.txt':
        latent_dims=50
    stable_voxels=500
    sem_feat=np.loadtxt(skip_or_cooc,dtype=float)
    noun=np.loadtxt('../noun.txt',dtype=bytes).astype(str)
    test_pairs=set(itertools.combinations(list(range(60)),2))
    test_pairs=list(test_pairs)
    #outFile=open("../outputs/"+str("cooccur")+"_vox_"+str(stable_voxels), 'w') #w for truncating
    help=re.findall("\d",sys.argv[1])
    if help==[]:
        no_parts=list(range(1,10))
    else:
        no_parts=list(help)
    if (sys.argv[2]=='-tr'):
        train_option=1
    elif (sys.argv[2]=='-notr'):
        train_option=0
    help=float(sys.argv[3])
    var=float(len(test_pairs))
    help=help*var
    test_pairs=test_pairs[0:int(help)]
 #print(help)
    if (len(sys.argv)>4 and sys.argv[4]=='-st_vox'):
        calc_st=1
    elif (len(sys.argv)<=4):
        calc_st=0
#########################################################################
#Start computations for every participant(1-9) for every test pair(1770)#
#########################################################################
    alpha=[]
    save_coefs=0
    tot_acc=0
    reg_par=[(0.8,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (0.9,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (0.8,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0,2.0),
 (1.0,2.0,5.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (1.0,5.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (1.2,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),
 (290.0,1.0,5.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0)]

    for parts in no_parts:
        print("Processing data for Participant "+str(parts))
        mat = scipy.io.loadmat('../../data/FMRI/data-science-P'+str(parts)+'.mat')
        #outFile.write("Participant "+str(parts)+"\n")
        #outFile.write("Test Words             Cosine similarity\n")
        acc=0
        alpha=[]
        for test_words in test_pairs:
   ##############################################################
   ###############Data Split and merge formatting################
   ##############################################################
            test_1=noun[test_words[0]]
            test_2=noun[test_words[1]]
   #print("Combination of test words are "+str(test_1)+" "+str(test_2))

   #it goes to 2nd trial and accesses i'th voxel
   #trials are 60 concrete nouns*6 times=360
   #extract data and noun for that data from .mat file
            print('Data reading & processing starts...')
            length=len(mat['data'][0].item()[0])
   #trial data are 6x60=360-2x6=348(test words excluded)
            fmri_data_for_trial=np.zeros((348,length))
            fmri_data_raw=np.zeros((360,length))
            noun_for_trial=[]
            test_data1=np.zeros((6,length))
            test_data2=np.zeros((6,length))
            k=0
            j=0
            colToCoord=np.zeros((length,3))
            coordToCol=np.zeros((mat['meta']['dimx'][0][0][0][0],mat['meta']['dimy'][0][0][0][0],mat['meta']['dimz'][0][0][0][0]))
            colToCoord=mat['meta']['colToCoord'][0][0]
            coordToCol=mat['meta']['coordToCol'][0][0]
            t1=0
            t2=0
            for x in range (0,360):
                fmri_data_raw[k,:]=mat['data'][x][0][0]
                k+=1
                if mat['info'][0][x][2][0]==test_1:
                    test_data1[t1,:]=mat['data'][x][0][0]
                    t1+=1
                elif mat['info'][0][x][2][0]==test_2:
                    test_data2[t2,:]=mat['data'][x][0][0]
                    t2+=1
                else:
                    fmri_data_for_trial[j,:]=mat['data'][x][0][0]
                    noun_for_trial=noun_for_trial+[mat['info']['word'][0][x][0]]
                    j+=1
            k=0
            tempo=np.zeros((58,6),dtype=int)
   #test1_trials=np.zeros((1,6))
   #test2_trials=np.zeros((1,6))
#   noun_set = set()
#   result = []
#   for item in noun_for_trial:
#    if item not in noun_set:
#      noun_set.add(item)
#      result.append(item)
            for x in noun:
                if ((x!=test_1) and (x!=test_2)):
                    tempo[k,:]=[i for i, j in enumerate(noun_for_trial) if j == x]
                    k+=1
    #elif x==test_1:
    # test1_trials=[i for i, j in enumerate(noun_for_trial) if j == x]
    #else:
    # test2_trials=[i for i, j in enumerate(noun_for_trial) if j == x]
            combs=set(itertools.combinations([0,1,2,3,4,5],2))
            combs=list(combs)
            #print('Data reading & processing ends...')
   ########################################################################
   #################Voxel Stability Selection Starts#######################
   ########################################################################
            #print('Voxel Selection starts...')
   #print(test_pairs.index(test_words))
            if (calc_st):
                vox=np.zeros((length,6,58))
                fd=open('/home/n_athan/Desktop/diploma/code/stable_voxels/st_vox'+str(parts)+'.pkl','wb')
                #print(fmri_data_for_trial[tempo[0,:],0])
                stab_score=np.zeros((length))
                for x in range(0,length):#voxel
                    sum_vox=0
                    for y in range(0,58):#noun
                        vox[x,0,y]=fmri_data_for_trial[tempo[y,0],x]
                        vox[x,1,y]=fmri_data_for_trial[tempo[y,1],x]
                        vox[x,2,y]=fmri_data_for_trial[tempo[y,2],x]
                        vox[x,3,y]=fmri_data_for_trial[tempo[y,3],x]
                        vox[x,4,y]=fmri_data_for_trial[tempo[y,4],x]
                        vox[x,5,y]=fmri_data_for_trial[tempo[y,5],x]
                        # compute the correlation
                    for z in combs:
                        sum_vox+=pearsonr(vox[x,z[0],:],vox[x,z[1],:])[0]
                    stab_score[x]=sum_vox/15#no of possible correlations
    #stab_vox=nlargest(500,range(len(stab_score)),stab_score.take)
                stab_vox=np.argsort(stab_score)[::-1][:stable_voxels]
                np.savetxt('./stable_voxels/st_vox'+str(parts)+'/'+noun[test_words[0]]+'_'+noun[test_words[1]]+'.txt',stab_vox,fmt='%d')
            else:
                stab_vox=np.loadtxt('../stable_voxels/st_vox'+str(parts)+'_'+str(stable_voxels)+'.txt',dtype=int)
                print('I loaded the voxels NOT calculated them!')
            #print('Voxel Selection ends...')
   #################################################################
   ########Data preproccesing and mean normalization################
   #################################################################
            #print('Mean normalization and global representation construction starts...')
            test_data1=np.sum(test_data1,axis=0)
            test_data1/=6
            test_data2=np.sum(test_data2,axis=0)
            test_data2/=6
            #print(test_data1.shape)
            #test_data1=test_data1[0,stab_vox]
            #test_data2=test_data2[0,stab_vox]

            fmri_data_proc=np.zeros((58,stable_voxels))
            fmri_data_final=np.zeros((58,stable_voxels))
            for x in range(0,58):
                fmri_data_proc[x,:] =fmri_data_for_trial[tempo[x,0],stab_vox]+fmri_data_for_trial[tempo[x,1],stab_vox]+fmri_data_for_trial[tempo[x,3],stab_vox]+fmri_data_for_trial[tempo[x,2],stab_vox]+fmri_data_for_trial[tempo[x,4],stab_vox]+fmri_data_for_trial[tempo[x,5],stab_vox]
                fmri_data_proc[x,:]/=6
            mean_data=np.sum(fmri_data_proc,axis=0)+test_data1[stab_vox]+test_data2[stab_vox]
            mean_data/=60

            fmri_data_final=np.zeros((58,stable_voxels))
            mean_data=np.tile(mean_data,(58,1))
            fmri_data_final=fmri_data_proc-mean_data
            #for x in range(0,58):
            # fmri_data_final[x,:]=fmri_data_proc[x,:]-mean_data
            test_data1=(test_data1[stab_vox]-mean_data[0,:])
            test_data2=(test_data2[stab_vox]-mean_data[0,:])
            yy=fmri_data_final
            yy=np.append(yy,test_data1)
            yy=np.append(yy,test_data2)
            fmri_data_final /= np.std(yy)
            test_data1 /= np.std(yy)
            test_data2 /= np.std(yy)

            test_data1=test_data1.reshape((stable_voxels,1))
            test_data2=test_data2.reshape((stable_voxels,1))

            #print(test_data1.shape)
            #print(test_data2.shape)
            #print('Mean normalization and global representation construction ends...')
            #########################################################################
            ##########################Training section###############################
            #########################################################################
            #print('Training starts...')
            mle_est=np.zeros((stable_voxels,latent_dims+1))#zeros 25
            semantic=np.zeros((58,latent_dims))
            sem_feat=np.array(sem_feat)
            temp=np.ones((60,latent_dims+1))
            temp[:,:-1]=sem_feat
            k=0
            for x in range(60):
                if ((noun[x]!=test_1) and (noun[x]!=test_2)) :
                    semantic[k,:]=sem_feat[x,:]
                    k+=1
            bias=[]
            if (train_option):
                model = linear_model.RidgeCV(reg_par[int(parts)-1],fit_intercept=True,normalize=False)#####Ridge(alpha=0.5)
                #     #Here we have to do this for 58/60 for all possible combinations!!
                model.fit(semantic,fmri_data_final)
                mle_est=model.coef_ #TODO remove [x,:
                bias=model.intercept_
                bias=np.array(bias)
                bias=np.reshape(np.array(bias),(stable_voxels,1))
                mle_est=np.append(mle_est,bias,1)

                # model = Regressor(layers=[Layer("Tanh", units=10),Layer("Tanh", units=5),Layer("Linear")],
                #                 learning_rate=0.01,n_iter=30,n_stable=8,learning_rule='nesterov')
                # model.fit(semantic, fmri_data_final)

                if save_coefs:
                    np.savetxt('./mle_estimates/'+skip_or_cooc+'/coeffs'+str(parts)+str(stable_voxels)+'.txt',mle_est,fmt='%f')
            else:
                mle_est=np.loadtxt('./mle_estimates/'+skip_or_cooc+'/coeffs'+str(parts)+str(stable_voxels)+'.txt',dtype=float)
            #print('Training ends..')
   #######################################################################
   #####################Evaluation section################################
   #######################################################################
            #print('Evaluation starts...')
            i1=test_data1
            i1=i1.reshape((1,stable_voxels))
            i2=test_data2
            i2=i2.reshape((1,stable_voxels))
            #   #we want to found the noun
            #   #the noun contained in info
            #   #mat['info'][0][i][2] contains the word i want for what the trial is set
            idx1=np.where(noun==test_1)
            idx2=np.where(noun==test_2)
            sf1=temp[idx1,:]
            sf2=temp[idx2,:]
            sf1=sf1.reshape((latent_dims+1,1))
            sf2=sf2.reshape((latent_dims+1,1))
            p1=np.dot(mle_est,sf1)
            p2=np.dot(mle_est,sf2)
            #p1=model.predict(sf1)
            #p2=model.predict(sf2)
            acc=acc+evaluation(i1,p1,i2,p2,'cosine')
            print('Evaluation ends...')
            #outFile.write(str(test_1)+" "+str(test_2)+"             "+str(acc)+"\n")
        #TODO uncomment for proper use CAUTION
        accuracy=acc/(len(test_pairs))
        #print('The accuracy for Participant'+str(parts)+'is'+str(accuracy*100)+' %')
        tot_acc+=accuracy
        print("\n"+"Total Accuracy "+str(accuracy)+"\n")
    print("\n"+"Total Accuracy "+str(tot_acc/9)+"\n")
