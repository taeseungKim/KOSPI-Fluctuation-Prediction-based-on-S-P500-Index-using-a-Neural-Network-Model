from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from real_swag_paper import input_pre_minMax_z as preprocessingData
from sklearn.svm import SVC

train_X = preprocessingData.train_X
train_Y = preprocessingData.train_Y_oneNum

vali_X = preprocessingData.vali_X
vali_Y = preprocessingData.vali_Y_oneNum

total_train_X = preprocessingData.total_train_X
total_train_Y = preprocessingData.total_train_Y_oneNum

test_X = preprocessingData.test_X
test_Y = preprocessingData.test_Y_oneNum

# test set 한정
direct_com_K_real = preprocessingData.test_Y_oneNum
direct_com_S_real = preprocessingData.sp_direct_com_final


svm = SVC(C=5.0)
logit = LogisticRegression(C=2.0)
#pruning decision tree
decision = tree.DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5, min_samples_leaf = 5, max_depth= 5)

# svm.fit(train_X, train_Y)
# logit.fit(train_X,train_Y)
# decision.fit(train_X,train_Y)

svm.fit(total_train_X, total_train_Y)
logit.fit(total_train_X,total_train_Y)
decision.fit(total_train_X,total_train_Y)

sum = 0
sum_n = 0
for i in preprocessingData.test_Y_oneNum: # 0, 1 비율
    if i == 0:
        sum +=1
    else:
        sum_n += 1

direct_com_K_de = np.delete(direct_com_K_real,0) #미국과 비교날짜를 맞추기 위해 삭제
direct_com_S_de = np.delete(direct_com_S_real,-1)

print("direct comparison : " + str(accuracy_score(direct_com_K_de,direct_com_S_de)))
print("only choose 1 : " + str(sum_n/(sum+sum_n)))
print("only choose 0 : " + str(sum/(sum+sum_n)))
print("svm acc : " + str(svm.score(test_X,test_Y)))
print("logistic acc : " + str(logit.score(test_X,test_Y)))
print("decision tree acc : " + str(decision.score(test_X,test_Y)))
#vali
# print("vali svm acc : " + str(svm.score(vali_X,vali_Y)))
# print("vali logistic acc : " + str(logit.score(vali_X,vali_Y)))
# print("vali decision tree acc : " + str(decision.score(vali_X,vali_Y)))
