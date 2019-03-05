from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from real_swag_paper import input_pre_minMax_z as preprocessingData
from sklearn.svm import SVC
from itertools import product

C = [1,10,100,1000]
k = ["linear","rbf"]

items = []
items.append(C)
items.append(k)
items_s = list(product(*items))

train_X = preprocessingData.train_X
train_Y = preprocessingData.train_Y_oneNum

vali_X = preprocessingData.vali_X
vali_Y = preprocessingData.vali_Y_oneNum

total_train_X = preprocessingData.total_train_X
total_train_Y = preprocessingData.total_train_Y_oneNum

test_X = preprocessingData.test_X
test_Y = preprocessingData.test_Y_oneNum

# for c,k in items_s:
#
#     svm = SVC(C=c, kernel=k)
#
#     svm.fit(train_X, train_Y)
#
#     print("C : " + str(c) + " kernel : " + str(k) + " vali svm acc : " + str(svm.score(vali_X,vali_Y)))

svm = SVC(C=100)

svm.fit(total_train_X, total_train_Y)

print("svm acc : " + str(svm.score(test_X,test_Y)))
#vali
# print("vali svm acc : " + str(svm.score(vali_X,vali_Y)))
# print("vali logistic acc : " + str(logit.score(vali_X,vali_Y)))
# print("vali decision tree acc : " + str(decision.score(vali_X,vali_Y)))
