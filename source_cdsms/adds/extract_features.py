import os,sys,csv
import numpy as np 


with open("../example_data/sample.cols", "r") as ffl:
 for nol,_ in enumerate(ffl):
  pass
print(nol)


sem_feat=np.zeros((nol,25))
with open("./index.ws5.index", "r") as f:
 for no_line,line in enumerate(f):
  #remove whitespace and split them to spaces
  if no_line>=74:
    each = line.strip().split()
    each=each[:74]
    each_filt=[x.split(',') for x in each]
    for x in each_filt:
     if int(x[0])==1 or int(x[0])==2:
      sem_feat[no_line-74][0]+=int(x[1])
     elif int(x[0])<=5:
      sem_feat[no_line-74][1]+=int(x[1])
     elif int(x[0])<=8: 
      sem_feat[no_line-74][2]+=int(x[1])
     elif int(x[0])<=11:
      sem_feat[no_line-74][3]+=int(x[1])
     elif int(x[0])<=14:
      sem_feat[no_line-74][4]+=int(x[1])
     elif int(x[0])<=17:
      sem_feat[no_line-74][5]+=int(x[1])
     elif int(x[0])<=20:
      sem_feat[no_line-74][6]+=int(x[1])
     elif int(x[0])<=23:
      sem_feat[no_line-74][7]+=int(x[1])
     elif int(x[0])<=26:
      sem_feat[no_line-74][8]+=int(x[1])
     elif int(x[0])<=29:
      sem_feat[no_line-74][9]+=int(x[1])
     elif int(x[0])<=32:
      sem_feat[no_line-74][10]+=int(x[1])
     elif int(x[0])<=35:
      sem_feat[no_line-74][11]+=int(x[1])
     elif int(x[0])<=38:
      sem_feat[no_line-74][12]+=int(x[1])
     elif int(x[0])<=41:
      sem_feat[no_line-74][13]+=int(x[1])
     elif int(x[0])<=44:
      sem_feat[no_line-74][14]+=int(x[1])
     elif int(x[0])<=47:
      sem_feat[no_line-74][15]+=int(x[1])
     elif int(x[0])<=50:
      sem_feat[no_line-74][16]+=int(x[1])
     elif int(x[0])<=53:
      sem_feat[no_line-74][17]+=int(x[1])
     elif int(x[0])<=56:
      sem_feat[no_line-74][18]+=int(x[1])
     elif int(x[0])<=59:
      sem_feat[no_line-74][19]+=int(x[1])
     elif int(x[0])<=62:
      sem_feat[no_line-74][20]+=int(x[1])
     elif int(x[0])<=65:
      sem_feat[no_line-74][21]+=int(x[1])
     elif int(x[0])<=68:
      sem_feat[no_line-74][22]+=int(x[1])
     elif int(x[0])<=71:
      sem_feat[no_line-74][23]+=int(x[1])
     elif int(x[0])<=74:
      sem_feat[no_line-74][24]+=int(x[1])
#Case of men dataset only
#sem_feat[747][9]=1#drive
#sem_feat[748][7]=1#eat
#sem_feat[749][12]=1#break
#sem_feat[750][6]=1#run
for i in range(nol):
# sem_feat/=np.sum(sem_feat**2,axis=1)[:,None]
  norm=np.sum(sem_feat[i]**2)
#  print(norm,i)
  norm=np.sqrt(norm)
  sem_feat[i]/=norm
#  x1=np.sum((sem_feat[i]**2))
#  print(np.sqrt(x1))
embds={}

with open("../example_data/sample.voc", "r") as f:
 for no_line,line in enumerate(f):
  if no_line>=74:
    li=line.strip()
    embds[li]=sem_feat[no_line-74,:]
print (embds)
np.save('embds_battig.npy', embds) 

# Load


