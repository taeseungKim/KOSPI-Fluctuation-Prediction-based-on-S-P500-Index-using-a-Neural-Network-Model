import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

kospi = pd.read_csv('/home/tskim/바탕화면/kospi(97.10_17.8).csv')['Close'] #코스피 종가
kospi = np.array(kospi.dropna()) #remove Null value
kospi = kospi[3574:4768]
k_train = kospi[:956]
k_test = kospi[956:]

sp = pd.read_csv('/home/tskim/바탕화면/S&P500(97.10_17.8).csv')['Close'] #S&P500 종가
sp = np.array(sp.dropna()) #remove Null value
sp = sp[3574:4768]
sp_train = sp[:956]
sp_test = sp[956:]

rl = []

for i in range(0, len(kospi) - 1): # label
    if kospi[i] < kospi[i + 1]:
        rl.append(0)
    else:
        rl.append(1)

sp_ud_train = []

for i in range(0, len(sp) - 1):
    if sp[i] < sp[i + 1]:
        sp_ud_train.append(0)
    else:
        sp_ud_train.append(1)

del rl[0]
del sp_ud_train[1192]

rl = np.array(rl).reshape(-1,1)
sp_ud_train = np.array(sp_ud_train).reshape(-1,1)

ab = np.hstack((rl,sp_ud_train))

def cross(ab):
    sum = 0
    for i in range(5):
        a=ab[(i)*int(len(ab)/5):(i+1)*int(len(ab)/5)]
        sum+=accuracy_score(a[:,:1],a[:,1:])
        print(accuracy_score(a[:,:1],a[:,1:]))
    mean = sum/5
    return mean
print(cross(ab))
print(accuracy_score(rl,sp_ud_train))