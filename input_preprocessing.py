from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from math import log
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import cross_val_score
from collections import OrderedDict
import tensorflow as tf

n_train_begin = 0
n_train_end = 3111

n_test_begin = 3111
n_test_end = 3853

kospi = np.array((pd.read_csv('/home/tskim/바탕화면/kospi(00.1.1_16.2.12).csv')).dropna()) #미국 주가 제외 x
sp = np.array((pd.read_csv('/home/tskim/바탕화면/s&p(00.1.1_16.2.12).csv')).dropna())

kospi_input_dic = OrderedDict() #input dictionary
sp_input_dic = OrderedDict()
for i in range(len(kospi)):
    kospi_input_dic["".join(kospi[:, :1][i])] = kospi[:, 1:7][i]

for i in range(len(sp)):
    sp_input_dic["".join(sp[:,:1][i])] = sp[:,1:7][i]

sl = [] # s&p close price change
for i in range(0,len(sp)-1): # s&p500 close price change
    sl.append(sp[i+1,4:5] - sp[i,4:5])

del sp_input_dic[sp[0,0]]# first value remove for append
sp = np.delete(sp,0,axis=0)

i=0
for k,v in sp_input_dic.items():#하루 전 종가등락 dic에 병합
    sp_input_dic[k] = np.append(sp_input_dic[k],sl[i])
    i=i+1

rl_past = [] # kospi close price change for target on One day before

for i in range(0, len(kospi) - 1): # label
    rl_past.append(kospi[i+1,4:5] - kospi[i, 4:5])

del kospi_input_dic[kospi[0,0]]
kospi = np.delete(kospi,0,axis=0)

a=0
for k,v in kospi_input_dic.items():#하루 전 종가등락 dic에 병합
   kospi_input_dic[k] = np.append(kospi_input_dic[k],rl_past[a])
   a=a+1
##################### 한국 날짜 미국 날짜 같은 날만 남기고 제거
input = OrderedDict()
date = [] # 한국 미국 겹치는 날짜
for k,v in kospi_input_dic.items():
    arr = []
    if k in sp_input_dic:
        arr.append(kospi_input_dic[k])
        arr.append(sp_input_dic[k])
        input[k] = arr
        date.append(k)
#####################making target
input_kospi = []
input_sp = []
for k,v in input.items():
    input_kospi.append(v[0])
    input_sp.append(v[1])

answer=[] #answer
for i in range(len(input)-1):
    if input_kospi[i+1][3] > input_kospi[i][3]:
        answer.append(1) #종가 상승시 1
    else:
        answer.append(0) #종가 하락,유지시 0
a = answer[:]
s=[] #미국 종가 등락 baseline classification용
for i in range(len(input_sp)-1):
    if input_sp[i+1][3] > input_sp[i][3]:
        s.append(1) #종가 상승시 1
    else:
        s.append(0) #종가 하락,유지시 0
del a[0]
del s[-1]

print(" s&p, kospi direct comparison acc :  " + str(accuracy_score(a,s))) #0.60591

def make_z_score(input):
    open = []
    high = []
    low = []
    close = []
    adjClose = []
    volume = []
    ud = []
    for i in input:
        open.append(i[0])
        high.append(i[1])
        low.append(i[2])
        close.append(i[3])
        adjClose.append(i[4])
        volume.append(i[5])
        ud.append(i[6])
    open_z = stats.zscore(open)
    high_z = stats.zscore(high)
    low_z = stats.zscore(low)
    close_z = stats.zscore(close)
    adjClose_z = stats.zscore(adjClose)
    volume_z = stats.zscore(volume)
    ud_z = stats.zscore(ud)
    a_a = []
    for i in range(0,len(input)):
        k_a = []
        k_a.append(open_z[i])
        k_a.append(high_z[i])
        k_a.append(low_z[i])
        k_a.append(close_z[i])
        k_a.append(adjClose_z[i])
        k_a.append(volume_z[i])
        k_a.append(ud_z[i])
        a_a.append(k_a)
    return a_a

k_input_z = make_z_score(input_kospi)
sp_input_z = make_z_score(input_sp)


input_z = OrderedDict()
i = 0
for k,v in input.items():
    all = []
    all.append(k_input_z[i])
    all.append(sp_input_z[i])
    input_z[k] = all
    i=i+1
#######################################################여기까지
temp = []
input_z.popitem() #answer와 맞추기위해 마지막 value삭제
del date[-1]

for k,v in input_z.items(): # 2개 array의 kospi,s&p를 학습하기위해 하나의 array로 합침
    for i in v[1]:
        v[0].append(i)
    temp.append(v[0])
train_X = []
train_Y = []
answer_n = [] #0 = [0,1] 1 = [1,0]
for i in answer:
    if i == 0:
        answer_n.append([0,1])
    else:
        answer_n.append([1,0])

for i in range(n_train_begin,n_train_end):
    train_X.append(temp[i])
    train_Y.append(answer_n[i])

test_X = []
test_Y = []
for i in range(n_test_begin+1,n_test_end):
    test_X.append(temp[i])
    test_Y.append(answer_n[i])
print("train_date : " + date[n_train_begin] +" ~ " + date[n_train_end])
print("test_date : " + date[n_test_begin+1] +" ~ " + date[n_test_end-1])
#####################