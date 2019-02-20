#making some code to plot an ROC curve for this neural net
from matplotlib import pyplot as plt
import os
import sys
import numpy as np

path = "."
if(len(sys.argv) > 1):
    path = sys.argv[1]
h = open(os.path.join(path, "hout.csv"), 'r')
s = open(os.path.join(path, "sout.csv"), 'r')

#throw the data into arrays
#formatting my data like so
#[happy_confidence, ...]
datah = []
first = True
for line in h:
    if(first):
        first = False
        continue
    vals = line.split(',')
    if(vals[1] == '0'):
        datah.append(1-float(vals[2]))
    else:
        datah.append(float(vals[2]))
h.close()

datas = []
first = True
for line in s:
    if(first):
        first = False
        continue
    vals = line.split(',')
    if(vals[1] == '0'):
        datas.append(1-float(vals[2]))
    else:
        datas.append(float(vals[2]))
s.close()

#now getting it ready to graph
#for each confidence threshold compute false-positive rate (x)
#and true positive rate (y)
datax = []
datay = []
max_diff = 0
tm = 0
tprm = 0
fprm = 0
#varying our threshold t
num_points = 1000
for T in range(0,num_points+1):
    t = T/num_points
    tpr = 0
    for val in datah:
        if(val >= t):
            tpr += 1
    tpr = tpr/len(datah)
    datay.append(tpr)
    fpr = 0
    for val in datas:
        if(val >= t):
            fpr += 1
    fpr = fpr/len(datas)
    datax.append(fpr)
    if(tpr-fpr > max_diff):
        max_diff = tpr-fpr
        tm = t
        tprm = tpr
        fprm = fpr

print("Best Results at threshold = "+str(tm))
print("TPR = "+str(tprm))
print("FPR = "+str(fprm))
#plot it
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.plot(datax, datay)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle='--')
plt.tight_layout()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
