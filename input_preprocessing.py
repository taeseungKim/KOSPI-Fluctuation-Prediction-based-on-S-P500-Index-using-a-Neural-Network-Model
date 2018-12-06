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
import input_prepro_function as func

n_train_begin = 0
n_train_end = 3351

n_test_begin = 3351
n_test_end = 4306

kospi = np.array((pd.read_csv('/home/tskim/바탕화면/kospi.csv')).dropna()) #미국 주가 제외 x
sp = np.array((pd.read_csv('/home/tskim/바탕화면/s&p500.csv')).dropna())

# make numpy(kospi,s&p) dictionary
kospi_input_dic = func.getInDic(kospi)
sp_input_dic = func.getInDic(sp)

# get answer
kospi_updown = func.getCloseChange(kospi_input_dic)

# for direct comparison with kospi
sp_updown = func.getCloseChange(sp_input_dic)

# make feature(당일 종가 - 전일 종가)
sp_close_minus = func.getCloseMinus(sp_input_dic)
kospi_close_minus = func.getCloseMinus(kospi_input_dic)

# merge 6feature with 당일 종가-전일 종가
sp_input_dic_closeMinus = func.mergeDics(sp_input_dic,sp_close_minus)
kospi_input_dic_closeMinus = func.mergeDics(kospi_input_dic,kospi_close_minus)

# zscore(feature)
kospi_input_dic_closeMinus_z = func.makeZscore(kospi_input_dic_closeMinus)
sp_input_dic_closeMinus_z = func.makeZscore(sp_input_dic_closeMinus)

#merge kospi feature with s&p feature
final_input = func.mergeDics(kospi_input_dic_closeMinus_z,sp_input_dic_closeMinus_z)
# -> 이때 final_input 안의 value들은 key값에 있는 날짜 다음날의 종가 등락에 영향을 주는 features

#answer와 input날짜에 맞게 병합 (answer를 final_input의 날짜에 맞추기 위한 병합)
answerPlusFinalInput = func.mergeDics(final_input,kospi_updown)
sp_answerPlusFinalInput = func.mergeDics(final_input,sp_updown)

kospi_updown_adjusted = OrderedDict() # final answer
final_input_adjusted = OrderedDict() # final input
for k,v in answerPlusFinalInput.items():
    kospi_updown_adjusted[k] = v[14:]
    final_input_adjusted[k] = v[:14]

sp_updown_adjusted = OrderedDict() # direct comparison을 위한 변수
for k,v in sp_answerPlusFinalInput.items():
    sp_updown_adjusted[k] = v[14:]

sp_answer = [] # make sp_updown_adjusted integer
for k,v in sp_updown_adjusted.items():
    a=[]
    a.append(int(v[0]))
    a.append(int(v[1]))

    sp_answer.append(a)

#for baseline target (total size 4306)
sp_direct_com = func.makeOneNum(sp_answer)

# for direct comparison (test size)
sp_direct_com_final = []
for i in range(n_test_begin,n_test_end):
    sp_direct_com_final.append(sp_direct_com[i])

temp = []
date = []
for k,v in final_input_adjusted.items():
    date.append(k)
    temp.append(v)

answer = [] # make kospi_updown_adjusted integer
for k,v in kospi_updown_adjusted.items():
    a=[]
    a.append(int(v[0]))
    a.append(int(v[1]))

    answer.append(a)

#for direct comparison (total size 4306)kospi
kospi_all_com = func.makeOneNum(answer)

train_X = []
train_Y = []

for i in range(n_train_begin,n_train_end):
    train_X.append(temp[i])
    train_Y.append(answer[i])

test_X = []
test_Y = []

for i in range(n_test_begin,n_test_end):
    test_X.append(temp[i])
    test_Y.append(answer[i])

train_Y_oneNum = func.makeOneNum(train_Y)
test_Y_oneNum = func.makeOneNum(test_Y)

print("train_date : " + date[n_train_begin] +" ~ " + date[n_train_end-1])
print("test_date : " + date[n_test_begin] +" ~ " + date[n_test_end-1])