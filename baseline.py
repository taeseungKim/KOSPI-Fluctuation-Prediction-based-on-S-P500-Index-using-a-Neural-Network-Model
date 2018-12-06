from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import input_preprocessing as preprocessingData

train_X = preprocessingData.train_X
train_Y = preprocessingData.train_Y_oneNum

test_X = preprocessingData.test_X
test_Y = preprocessingData.test_Y_oneNum

# test set 한정
direct_com_K_real = preprocessingData.test_Y_oneNum
direct_com_S_real = preprocessingData.sp_direct_com_final

#전체 데이터
# direct_com_K = preprocessingData.kospi_all_com
# direct_com_S = preprocessingData.sp_direct_com

logit = LogisticRegression()
bayes = GaussianNB()
decision = tree.DecisionTreeClassifier()

logit.fit(train_X,train_Y)
bayes.fit(train_X,train_Y)
decision.fit(train_X,train_Y)

sum = 0
sum_n = 0
for i in preprocessingData.test_Y_oneNum: # 0, 1 비율
    if i == 0:
        sum +=1
    else:
        sum_n += 1

direct_com_K_de = np.delete(direct_com_K_real,0) #미국과 비교날짜를 맞추기 위해 삭제
direct_com_S_de = np.delete(direct_com_S_real,-1)

# direct_com_K_d = np.delete(direct_com_K,0)
# direct_com_S_d = np.delete(direct_com_S,-1)
#
# date = preprocessingData.date
# del date[0] #한국 날짜에 맞추기 위해 하루 삭제

# a = 5
# results = []
# for i in range(a):
#     print(date[i*int(len(date)/a)] + "~" + date[(i+1)*int(len(date)/a)])
#     print("direct comparison : " + str(accuracy_score(direct_com_K_d[i*int(len(date)/a):(i+1)*int(len(date)/a)],
#                                                       direct_com_S_d[i*int(len(date)/a):(i+1)*int(len(date)/a)])))
#     results.append(accuracy_score(direct_com_K_d[i*int(len(date)/a):(i+1)*int(len(date)/a)],
#                                                       direct_com_S_d[i*int(len(date)/a):(i+1)*int(len(date)/a)]))
# print(results)
# print("mean : " + str(np.mean(results)))
# print("standard deviation : " + str(np.std(results)))
print("direct comparison : " + str(accuracy_score(direct_com_K_de,direct_com_S_de)))
print("only choose 1 : " + str(sum_n/(sum+sum_n)))
print("only choose 0 : " + str(sum/(sum+sum_n)))
print("logistic acc : " + str(logit.score(test_X,test_Y)))
print("bayesian acc : " + str(bayes.score(test_X,test_Y)))
print("decision tree acc : " + str(decision.score(test_X,test_Y)))
