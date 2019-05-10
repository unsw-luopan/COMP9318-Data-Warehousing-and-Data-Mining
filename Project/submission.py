# Import your files here...
#project comp9318
#ZHIDONG LUO z5181142         PAN LUO  z5192086
import re
import pandas as pd
import numpy as np
import math
import copy

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    file_name_symbol = 'Symbol_File'
    file_name_state = 'State_File'
    file_symbol = open(file_name_symbol)
    file_state = open(file_name_state)
    L = []
    for line in file_symbol:
        L.append(line.strip('\n'))
    LL = []
    for line in file_state:
        LL.append(line.strip('\n'))
    # extract the list after 42211 row
    extract_symbol = L[(int(L[0]) + 1):]
    extract_state = LL[(int(LL[0]) + 1):]
    total_symbol = int(L[0])
    total_state = int(LL[0])
    prob_matrix_sym = prob_matrix_symbol(extract_symbol, total_symbol, total_state)
    prob_matrix_sta = prob_matrix_state(extract_state, total_state)
    matrix_pi = prob_matrix_sta[-2]  # 初始概率Π
    index = total_state + 1
    state = LL[1:index]
    # print(state)

    query = extract_queryfile()
##    print(query)
    index2 = int(L[0]) + 1
    symbol = L[1:index2]
    state = [i for i in range(len(state))]
    O = []
    for i in range(len(query)):
        obser = queryIndex(query[i], symbol, index2)
        new_ob = Q1(matrix_pi, state, obser, prob_matrix_sym, prob_matrix_sta, total_state)
        O.append(new_ob)
    return (O)


def prob_matrix_symbol(extract_symbol, total_symbol, total_state):
    # symbol
    ################################################
    # 构建一个二维数组  长度为23  每个元素为0 1 2 。。。个数的list
    num_of_each_symbol = [0] * (int(extract_symbol[-1].split(' ')[0]) + 1)
    num_of_transfer = [0] * (int(extract_symbol[-1].split(' ')[0]) + 1)
    # 统计symbol 的个数有多少
    for item in extract_symbol:
        for i in range(len(num_of_each_symbol)):
            if int(item.split(' ')[0]) == i:
                num_of_each_symbol[i] += 1
    # symbol
    # 统计转移次数的总和
    for item in extract_symbol:
        for i in range(len(num_of_transfer)):
            if int(item.split(' ')[0]) == i:
                num_of_transfer[i] += int(item.split(' ')[2])
    num_of_transfer.append(total_state - 1)
    num_of_transfer.append(0)
    #################################################
    # symbol  K·N的矩阵
    prob_matrix_symbol = [0] * total_state
    # 用来 存转移概率的矩阵
    for i in range(len(prob_matrix_symbol)):
        prob_matrix_symbol[i] = [0] * (total_symbol + 1)
    ##symbol
    for item in extract_symbol:
        ##        for i in range(len(prob_matrix_symbol)):
        ##            if int(item.split(' ')[0]) == i:
        ##                for j in range(len(prob_matrix_symbol[0])):
        ##                    if int(item.split(' ')[1]) == j:
        prob_matrix_symbol[int(item.split(' ')[0])][int(item.split(' ')[1])] = (int(item.split(' ')[2]) + 1) / (
                num_of_transfer[int(item.split(' ')[0])] + total_symbol + 1)

    for i in range(len(prob_matrix_symbol) - 2):
        prob_matrix_symbol[i][-1] = (1 / (num_of_transfer[i] + total_symbol + 1))

    for i in range(len(prob_matrix_symbol) - 2):
        for j in range(len(prob_matrix_symbol[0])):
            if prob_matrix_symbol[i][j] == 0:
                prob_matrix_symbol[i][j] = 1 / (num_of_transfer[i] + total_symbol + 1)
    # df = pd.DataFrame(prob_matrix_symbol)
    # df.to_csv("prob_matrix_symbol.csv", index=False, sep=',', header=None)
    return prob_matrix_symbol


def prob_matrix_state(extract_state, total_state):
    # state
    ################################################
    num_of_each_state = [0] * (int(extract_state[-1].split(' ')[0]) + 1)
    num_of_transfer2 = [0] * total_state
    # 统计state 的个数有多少
    for item in extract_state:
        for i in range(len(num_of_each_state)):
            if int(item.split(' ')[0]) == i:
                num_of_each_state[i] += 1
    # state
    for item in extract_state:
        for i in range(len(num_of_transfer2)):
            if int(item.split(' ')[0]) == i:
                num_of_transfer2[i] += int(item.split(' ')[2])
    ####################################################
    # state   K·K的矩阵
    prob_matrix_state = [0] * total_state
    # 用来 存转移概率的矩阵
    for i in range(len(prob_matrix_state)):
        prob_matrix_state[i] = [0] * total_state
    ##state
    for item in extract_state:
        ##        for i in range(len(prob_matrix_state)):
        ##            if int(item.split(' ')[0]) == i:
        ##                for j in range(len(prob_matrix_state[0])):
        ##                    if int(item.split(' ')[1]) == j:
        prob_matrix_state[int(item.split(' ')[0])][int(item.split(' ')[1])] = (int(item.split(' ')[2]) + 1) / (
                num_of_transfer2[int(item.split(' ')[0])] + total_state - 1)
    for i in range(len(prob_matrix_state) - 1):
        for j in range(len(prob_matrix_state[0])):
            if prob_matrix_state[i][j] == 0:
                prob_matrix_state[i][j] = 1 / (num_of_transfer2[i] + total_state - 1)
    # start 对应的列为0  end对应的行为0
    for i in range(len(prob_matrix_state)):
        for j in range(len(prob_matrix_state[0])):
            prob_matrix_state[i][-2] = 0
            prob_matrix_state[-1][j] = 0
    ##state
    # df = pd.DataFrame(prob_matrix_state)
    # df.to_csv("prob_matrix_state.csv", index=False, sep=',', header=None)

    return prob_matrix_state


def split_item(text):
    text = text.replace('(', 'lbracket')
    text = text.replace(')', 'rbracket')
    LL = re.split(r'(,|/|-|&|lbracket|rbracket| )', text)
    L = []
    for item in LL:
        if item == 'lbracket':
            item = '('
        if item == 'rbracket':
            item = ')'
        if item != '' and item != ' ':
            L.append(item)
    return L


def extract_queryfile():
    file_name = 'Query_File'
    file = open(file_name)
    L = []
    for line in file:
        L.append(line.strip('\n'))
    state_set = []
    for item in L:
        state_set.append(split_item(item))
    return state_set


def queryIndex(query, symbol, index2):
    index = []
    for item in query:
        for j in range(len(symbol)):
            if item == symbol[j]:
                index.append(j)
                break
        ##            elif item not in symbol:
        else:
            index.append(index2 - 1)

    return index

def Q1(matrix_pi, state, obser, prob_matrix_symbol, prob_matrix_state, total_state):
    pr = [[0 for col in range(len(state))] for row in range(len(obser))]
    path = [[0 for col in range(len(state))] for row in range(len(obser))]
    for i in range(len(state)):
        pr[0][i] = matrix_pi[i] * prob_matrix_symbol[state[i]][obser[0]]
        path[0][i] = i
    for i in range(1, len(obser)):  # 循环t-1次，初始已计算
        max_item = [0 for i in range(len(state))]
        for j in range(len(state)):  # 循环隐藏状态数，计算当前状态每个隐藏状态的概率
            item = [0 for i in state]
            for k in range(len(state)): # 再次循环隐藏状态数，计算选定隐藏状态的前驱状态为各种状态的概率
                p = pr[i - 1][k] * prob_matrix_symbol[state[j]][obser[i]] * prob_matrix_state[state[k]][
                    state[j]]
                item[state[k]] = p  #item列表计算Oi对应的state[k]的概率（取Oi的一行）
                max_item[state[j]] = max(item) #返回一行Oi的最大概率值，用于计算下一行
                path[i][state[j]] = item.index(max(item))
            pr[i] = max_item  #记录第i行O的最大概率值
        result = []
        p = pr[len(obser) - 1].index(max(pr[len(obser) - 1]))
        result.append(p)
##    print(np.array(pr))
    # print(result)
    # 从最后一个状态开始倒着寻找前驱节点
    for i in range(len(obser) - 1, 0, -1):
        result.append(path[i][p])
        p = path[i][p]
    result.reverse()
    result.append(total_state - 1)
    result.insert(0, total_state - 2)
    result.append(math.log(max(pr[-1]) * prob_matrix_state[result[-2]][result[-1]]))
    return result

def max_k(list, k):
    m = sorted(list,reverse=True)

    return m[:k]


def top_k(matrix_pi, state, obser, prob_matrix_symbol, prob_matrix_state, total_state, kv):
    pr = [[[0 for i in range(kv)] for col in range(len(state))] for row in range(len(obser))]
    path = [[[0 for i in range(kv)] for col in range(len(state))] for row in range(len(obser))]
    # path 记录max_p 对应概率处的路径 i 行 j列 （i,j）记录第i个时间点 j隐藏状态最大概率的情况下 其前驱状态
    for i in range(len(state)):
        for n in range(kv):
            pr[0][i][n] = matrix_pi[i] * prob_matrix_symbol[state[i]][obser[0]]
            path[0][i][n] = i
    for i in range(1, len(obser)):  # 循环t-1次，初始已计算
        max_item = [0 for i in range(len(state))]
        for j in range(len(state)):  # 循环隐藏状态数，计算当前状态每个隐藏状态的概率
            item = [[0 for i in range(kv)] for i in state]
            for k in range(len(state)):  # 再次循环隐藏状态数，计算选定隐藏状态的前驱状态为各种状态的概率
                for n in range(kv):
                    p = pr[i - 1][k][n] * prob_matrix_symbol[state[j]][obser[i]] * prob_matrix_state[state[k]][
                        state[j]]
                    item[state[k]][n] = p  # item列表计算Oi对应的state[k]的概率,计算次数为total state的数量
            tmp=[]
            tmp2 = []
            if i == 1:
                for y in range(len(item)):
                    tmp.append(item[y][0])
                    max_item[state[j]]=max_k(tmp,kv)
                for m in range(kv):
                    tmp2.append(item.index(max_k(item, kv)[m]))
                    path[i][state[j]] = tmp2
            else:
                max_item[state[j]] = big(item, kv)  # 返回item中的最大概率值，将该值赋给pr[i][k]，最终的max_item矩阵为Oi一行的值
                if j<len(state) - 2:
                    path[i][state[j]] = find_erweipath(item,max_item[state[j]],kv)
            pr[i] = max_item  # 记录第i行O的概率值
        #print(path[i])
    #print(np.array(pr))
    result=[]
    prnew=copy.deepcopy(pr)
    p = find_erweipath(pr[-1],big(pr[-1],kv),kv)  # pr矩阵最后一行对应的最大值所在的index

    #print(path)
    # 从最后一个状态开始倒着寻找前驱节点
    a = []
    b=[]
    for i in range(len(p)):
        a.append(p[i][0])
        b.append(p[i][1])
    #print(a,b)

    for i in range(len(a)):
        last_line=a[i]
        result.append(a[i])
        last_from=b[i]
        for j in range(len(path)-1,1,-1):
            result.append(path[j][last_line][last_from][0])
            c=last_line
            d=last_from
            last_line=path[j][c][d][0]
            last_from=path[j][c][d][1]
        result.append(path[1][last_line][last_from])
    L1=[]
    splitnumber=len(result)/kv
    pr4=big(prnew[-1],kv)

    for i in range(0,len(result),int(splitnumber)):
        a=result[i:i+int(splitnumber)]
        a.reverse()
        a.append(total_state - 1)
        a.insert(0, total_state - 2)
        L1.append(a)
    for i in range(len(L1)):
        L1[i].append(math.log(pr4[i] * prob_matrix_state[L1[i][-2]][L1[i][-1]]))

    return L1

def find_erweipath(double, single,kv):
    tmp = []
    LL = double
    L = single
    for j in range(len(L)):
        for i in range(len(LL)):
            for k in range(len(LL[i])):
                if L[j] == LL[i][k]:
                    if len(tmp)<kv:
                        tmp.append([i, k])
                        LL[i][k] = 0
    return tmp


def big(L, kv):
    a = []
    for i in range(len(L)):
        if isinstance(L[i], list):
            a.append(L[i])
    LL = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            LL.append(a[i][j])
    b = sorted(LL)[-kv:]
    return b[::-1]




# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):  # do not change the heading of the function
    file_name_symbol = 'Symbol_File'
    file_name_state = 'State_File'
    file_symbol = open(file_name_symbol)
    file_state = open(file_name_state)
    L = []
    for line in file_symbol:
        L.append(line.strip('\n'))
    LL = []
    for line in file_state:
        LL.append(line.strip('\n'))
    # extract the list after 42211 row
    extract_symbol = L[(int(L[0]) + 1):]
    extract_state = LL[(int(LL[0]) + 1):]
    total_symbol = int(L[0])
    total_state = int(LL[0])
    prob_matrix_sym = prob_matrix_symbol(extract_symbol, total_symbol, total_state)
    prob_matrix_sta = prob_matrix_state(extract_state, total_state)
    matrix_pi = prob_matrix_sta[-2]  # 初始概率Π
    index = total_state + 1
    state = LL[1:index]


    query = extract_queryfile()
    index2 = int(L[0]) + 1
    symbol = L[1:index2]
    state = [i for i in range(len(state))]
    P=[]
    O = []
    for i in range(len(query)):
        obser = queryIndex(query[i], symbol, index2)
        new_ob = top_k(matrix_pi, state, obser, prob_matrix_sym, prob_matrix_sta, total_state,k)
        for item in new_ob:
            O.append(item)
    return (O)
##State_File = './toy_example/State_File'
##Symbol_File = './toy_example/Symbol_File'
##Query_File = './toy_example/Query_File'
##k = 2
##top_k_result = top_k_viterbi(State_File, Symbol_File, Query_File, k)
##
##for row in top_k_result:
##    print(row)


def prob_matrix_symbol_q3(extract_symbol, total_symbol, total_state,alpha):
    # symbol
    ################################################
    # 构建一个二维数组  长度为23  每个元素为0 1 2 。。。个数的list
    num_of_each_symbol = [0] * (int(extract_symbol[-1].split(' ')[0]) + 1)
    num_of_transfer = [0] * (int(extract_symbol[-1].split(' ')[0]) + 1)
    # 统计symbol 的个数有多少
    for item in extract_symbol:
        for i in range(len(num_of_each_symbol)):
            if int(item.split(' ')[0]) == i:
                num_of_each_symbol[i] += 1
    # symbol
    # 统计转移次数的总和
    for item in extract_symbol:
        for i in range(len(num_of_transfer)):
            if int(item.split(' ')[0]) == i:
                num_of_transfer[i] += int(item.split(' ')[2])
    num_of_transfer.append(total_state - 1)
    num_of_transfer.append(0)
    #################################################
    # symbol  K·N的矩阵
    prob_matrix_symbol = [0] * total_state
    # 用来 存转移概率的矩阵
    for i in range(len(prob_matrix_symbol)):
        prob_matrix_symbol[i] = [0] * (total_symbol + 1)
    ##symbol
    for item in extract_symbol:
        ##        for i in range(len(prob_matrix_symbol)):
        ##            if int(item.split(' ')[0]) == i:
        ##                for j in range(len(prob_matrix_symbol[0])):
        ##                    if int(item.split(' ')[1]) == j:
        prob_matrix_symbol[int(item.split(' ')[0])][int(item.split(' ')[1])] = (int(item.split(' ')[2]) + alpha) / (
                num_of_transfer[int(item.split(' ')[0])] + alpha*(total_symbol + 1))

    for i in range(len(prob_matrix_symbol) - 2):
        prob_matrix_symbol[i][-1] = (alpha / (num_of_transfer[i] + alpha*(total_symbol + 1)))

    for i in range(len(prob_matrix_symbol) - 2):
        for j in range(len(prob_matrix_symbol[0])):
            if prob_matrix_symbol[i][j] == 0:
                prob_matrix_symbol[i][j] = alpha / (num_of_transfer[i] + alpha*(total_symbol + 1))
    # df = pd.DataFrame(prob_matrix_symbol)
    # df.to_csv("prob_matrix_symbol.csv", index=False, sep=',', header=None)
    return prob_matrix_symbol


def prob_matrix_state_q3(extract_state, total_state,beta):
    # state
    ################################################
    num_of_each_state = [0] * (int(extract_state[-1].split(' ')[0]) + 1)
    num_of_transfer2 = [0] * total_state
    # 统计state 的个数有多少
    for item in extract_state:
        for i in range(len(num_of_each_state)):
            if int(item.split(' ')[0]) == i:
                num_of_each_state[i] += 1
    # state
    for item in extract_state:
        for i in range(len(num_of_transfer2)):
            if int(item.split(' ')[0]) == i:
                num_of_transfer2[i] += int(item.split(' ')[2])
    ####################################################
    # state   K·K的矩阵
    prob_matrix_state = [0] * total_state
    # 用来 存转移概率的矩阵
    for i in range(len(prob_matrix_state)):
        prob_matrix_state[i] = [0] * total_state
    ##state
    for item in extract_state:
        ##        for i in range(len(prob_matrix_state)):
        ##            if int(item.split(' ')[0]) == i:
        ##                for j in range(len(prob_matrix_state[0])):
        ##                    if int(item.split(' ')[1]) == j:
        prob_matrix_state[int(item.split(' ')[0])][int(item.split(' ')[1])] = (int(item.split(' ')[2]) + beta) / (
                num_of_transfer2[int(item.split(' ')[0])] + beta*(total_state - 1))
    for i in range(len(prob_matrix_state) - 1):
        for j in range(len(prob_matrix_state[0])):
            if prob_matrix_state[i][j] == 0:
                prob_matrix_state[i][j] = beta / (num_of_transfer2[i] + beta*(total_state - 1))
    # start 对应的列为0  end对应的行为0
    for i in range(len(prob_matrix_state)):
        for j in range(len(prob_matrix_state[0])):
            prob_matrix_state[i][-2] = 0
            prob_matrix_state[-1][j] = 0
    ##state
    # df = pd.DataFrame(prob_matrix_state)
    # df.to_csv("prob_matrix_state.csv", index=False, sep=',', header=None)

    return prob_matrix_state


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    file_name_symbol = 'Symbol_File'
    file_name_state = 'State_File'
    file_symbol = open(file_name_symbol)
    file_state = open(file_name_state)
    L = []
    for line in file_symbol:
        L.append(line.strip('\n'))
    LL = []
    for line in file_state:
        LL.append(line.strip('\n'))
    # extract the list after 42211 row
    extract_symbol = L[(int(L[0]) + 1):]
    extract_state = LL[(int(LL[0]) + 1):]
    total_symbol = int(L[0])
    total_state = int(LL[0])
    prob_matrix_sym = prob_matrix_symbol_q3(extract_symbol, total_symbol, total_state,alpha=0.00000001)
    prob_matrix_sta = prob_matrix_state_q3(extract_state, total_state,beta=0.000008)
    matrix_pi = prob_matrix_sta[-2]  # 初始概率Π
    index = total_state + 1
    state = LL[1:index]
    # print(state)

    query = extract_queryfile()
    index2 = int(L[0]) + 1
    symbol = L[1:index2]
    state = [i for i in range(len(state))]
    O = []
    #aa = 0
    for i in range(len(query)):
        obser = queryIndex(query[i], symbol, index2)
        # for i in range(len(obser)):
        #     if obser[i]==44211:
        #         aa=aa+1
        new_ob = Q1(matrix_pi, state, obser, prob_matrix_sym, prob_matrix_sta, total_state)
        O.append(new_ob)
        # print(query[i],obser)
    #print(aa)
    return (O)
