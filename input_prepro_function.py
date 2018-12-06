import numpy as np
from scipy import stats
from collections import OrderedDict

def getInDic(input): # get it in dictionary
    input_dic = OrderedDict()
    for i in range(len(input)):
        input_dic["".join(input[:,:1][i])] = input[:,1:][i]
    return input_dic

def getCloseChange(input):
    change_dic = OrderedDict()
    values = []
    keys = []
    for k,v in input.items():
        keys.append(k)
        values.append(v)
    for i in range(len(values)-1):
        if values[i+1][3] > values[i][3]:
            change_dic[keys[i]] = [1,0]
        else:
            change_dic[keys[i]] = [0,1]
        i+=1
    return change_dic

def getCloseMinus(input): #당일 종가 - 전일 종가
    change_dic = OrderedDict()
    values = []
    keys = []
    for k,v in input.items():
        keys.append(k)
        values.append(v)
    for i in range(len(values) - 1):
        change_dic[keys[i+1]] = values[i + 1][3] - values[i][3]
    return change_dic

def mergeDics(inp1,inp2): # compare with inp1,inp2 and Remove unequals and merge
    dic = OrderedDict()
    for k,v in inp1.items():
        if k in inp2:
            arr = np.append(inp1[k],inp2[k])
            dic[k] = arr
    return dic

def makeZscore(input):
    open = []
    high = []
    low = []
    close = []
    adjClose = []
    volume = []
    ud = []
    for k,v in input.items():
        open.append(v[0])
        high.append(v[1])
        low.append(v[2])
        close.append(v[3])
        adjClose.append(v[4])
        volume.append(v[5])
        ud.append(v[6])
    open_z = stats.zscore(open)
    high_z = stats.zscore(high)
    low_z = stats.zscore(low)
    close_z = stats.zscore(close)
    adjClose_z = stats.zscore(adjClose)
    volume_z = stats.zscore(volume)
    ud_z = stats.zscore(ud)
    a_a = OrderedDict()
    i=0
    for k,v in input.items():
        k_a = []
        k_a.append(open_z[i])
        k_a.append(high_z[i])
        k_a.append(low_z[i])
        k_a.append(close_z[i])
        k_a.append(adjClose_z[i])
        k_a.append(volume_z[i])
        k_a.append(ud_z[i])
        a_a[k]=k_a
        i+=1
    return a_a
def makeOneNum(input):
    input_oneNum = []
    for i in input:
        if i == [1,0]:
            input_oneNum.append(1)
        else:
            input_oneNum.append(0)
    return input_oneNum