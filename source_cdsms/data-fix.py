#Extraxt semantic features fi(w)
#!/usr/bin/env python
import os
import sys 
import re
from sys import argv
import numpy as np

#outFile.write("\n"+"Total Accuracy "+str(accuracy)+"alpha = "+str(alpha)+"\n")
out=open('./bla.txt','w')
with open("./example_data/sample.voc","r") as inf:
 for line in inf:

  line=line.split('-')
  if len(line)>1:
   out.write(line[0]+'\n')
   out.write(line[1].split(" ")[1]+'\n')
for x in range(6075):
 out.write(str(x)+'\n')
out.close()



