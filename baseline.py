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
train_Y = preprocessingData.answer[preprocessingData.n_train_begin:preprocessingData.n_train_end]

test_X = preprocessingData.test_X
test_Y = preprocessingData.answer[preprocessingData.n_test_begin:preprocessingData.n_test_end]

direct_com_K = preprocessingData.a_t
direct_com_S = preprocessingData.s_t

logit = LogisticRegression()
bayes = GaussianNB()
decision = tree.DecisionTreeClassifier()

logit.fit(train_X,train_Y)
bayes.fit(train_X,train_Y)
decision.fit(train_X,train_Y)

sum = 0
sum_n = 0
for i in preprocessingData.answer: # 0, 1 비율
    if i == 0:
        sum +=1
    else:
        sum_n += 1

print("direct comparison : " + str(accuracy_score(direct_com_K,direct_com_S)))
print("only choose 1 : " + str(sum_n/(sum+sum_n)))
print("only choose 0 : " + str(sum/(sum+sum_n)))
print("logistic acc : " + str(logit.score(test_X,test_Y)))
print("bayesian acc : " + str(bayes.score(test_X,test_Y)))
print("decision tree acc : " + str(decision.score(test_X,test_Y)))
