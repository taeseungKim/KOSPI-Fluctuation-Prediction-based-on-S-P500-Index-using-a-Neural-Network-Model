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

kospi = pd.read_csv('/home/tskim/바탕화면/kospi(97.10_17.8).csv')['Close'] #코스피 종가
kospi = np.array(kospi)
kospi_5days_forRE = kospi[3570:4768]
kospi_5days_forRM = kospi[3569:4768]
kospi_all = kospi[3574:4768]

sp = pd.read_csv('/home/tskim/바탕화면/S&P500(97.10_17.8).csv')['Close'] #S&P500 종가
sp = np.array(sp.dropna()) #remove Null value
sp = sp[3574:4768]

rl = []

for i in range(0, len(kospi_all) - 1): # label
    if kospi_all[i] < kospi_all[i + 1]:
        rl.append(0)
    else:
        rl.append(1)
al = []

for i in range(0, len(sp) - 1): # s&p close price up&down
     al.append(sp[i + 1]-sp[i])

del al[1192]
del rl[0]

def fnRSI(m_Df, m_N): #RSI

    m_Df = pd.DataFrame(m_Df)
    U = np.where(m_Df.diff(1) > 0, m_Df.diff(1), 0)
    D = np.where(m_Df.diff(1) < 0, m_Df.diff(1) *(-1), 0)

    AU = pd.DataFrame(U).rolling( window=m_N, min_periods=m_N).mean()
    AD = pd.DataFrame(D).rolling( window=m_N, min_periods=m_N).mean()
    RSI = AU.div(AD+AU) *100
    RSI = np.array(RSI.dropna())
    return RSI
def fnEma(s, n):

    s = np.array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema
def fnRoc(k_c,period):
    Roc=[]
    for i in range(len(k_c)-period):
        Roc.append(((k_c[i+period]-k_c[i])/k_c[i])*100)

    return Roc
def fnMom(k_c,period):
    Mom=[]
    for i in range(len(k_c)-period):
        Mom.append((k_c[i+period]/k_c[i])*100)

    return Mom

n = 5 #이동평균에 쓴 날짜
last_n = 1193

RSI_train = np.delete(fnRSI(kospi_5days_forRE,n),last_n).reshape(-1,1)
RSI_train_z = stats.zscore(RSI_train)
EMA_train = np.delete(fnEma(kospi_5days_forRE,n),last_n).reshape(-1,1)
EMA_train_z = stats.zscore(EMA_train)
ROC_train = np.delete(fnRoc(kospi_5days_forRM,n),last_n).reshape(-1,1)
ROC_train_z = stats.zscore(ROC_train)
MOM_train = np.delete(fnMom(kospi_5days_forRM,n),last_n).reshape(-1,1)
MOM_train_z = stats.zscore(MOM_train)
al_z = stats.zscore(al).reshape(-1,1) #z_score normalize

RSI_train_z = np.delete(RSI_train_z,0).reshape(-1,1)
EMA_train_z = np.delete(EMA_train_z,0).reshape(-1,1)
ROC_train_z = np.delete(ROC_train_z,0).reshape(-1,1)
MOM_train_z = np.delete(MOM_train_z,0).reshape(-1,1)

technical_index_train = np.hstack((RSI_train_z,al_z,EMA_train_z,ROC_train_z,MOM_train_z)) # feature merge

colors = ['red' if l ==0 else 'black' for l in rl] #등락을 color로 입력

fig=plt.figure()

# ax = fig.add_subplot(111,projection='3d') #3 dimension graph
# ax.set_xlabel('al_z')
# ax.set_ylabel('RSI_train_z')
# ax.set_zlabel('EMA_train_z')
# ax.scatter(al_z,RSI_train_z,EMA_train_z,color=colors)
# plt.xlabel('al')
# plt.ylabel('MOM')
# plt.scatter(al_z,MOM_train_z,color=colors)  #2 dimension graph
plt.show()

clf = LogisticRegression()
scores = cross_val_score(clf,technical_index_train,rl,cv=5) # 5fold cv
print(scores)
print(scores.mean())
