import numpy as np
import math
import random
N = 0
epsilon = 0.001
gamma = 0.001
def t_one_hot_encodin(line):
    line = line.replace(' ', '')
    t = np.zeros(3)
    t[0] = line[10]
    t[1] = line[12]
    t[2] = line[12]
    return t
def one_hot_encoding(line):
    line = line.replace(' ', '')
    x = np.zeros(24)
    if   (line[0] == 'A'):
        x[0] = 1
    elif (line[0] == 'B'):
        x[1] = 1
    elif (line[0] == 'C'):
        x[2] = 1
    elif (line[0] == 'D'):
        x[3] = 1
    elif (line[0] == 'E'):
        x[4] = 1
    elif (line[0] == 'F'):
        x[5] = 1
    else:
        x[6] = 1
    if   (line[1] == 'X'):
        x[7] = 1
    elif (line[1] == 'R'):
        x[8] = 1
    elif (line[1]  == 'S'):
        x[9] = 1
    elif (line[1]  == 'A'):
        x[10] = 1
    elif (line[1]  == 'H'):
        x[11] = 1
    else:
        x[12] = 1
    if (line[2] == 'X'):
        x[13] = 1
    elif (line[2] == 'O'):
        x[14] = 1
    elif (line[2] == 'I'):
        x[15] = 1
    elif (line[2] == 'C'):
        x[16] = 1
    k = 0
    while k + 3 < len(line) - 3:
        x[3+k] = line[3+k]
        k += 1
    return x
def prepareData(pathToFile, x_1, t_1, N):
    f = open(pathToFile, 'r')
    lines = f.readlines()
    lines = lines[1:]
    N = 0
    for line in lines:
        x_1.append(one_hot_encoding(line))
        t_1.append(t_one_hot_encodin(line))
        N += 1
    x_toConv = []
    for i in range(len(x_1)):
        x_conv = []
        x_conv.append(x_1[i])
        x_toConv.append(x_conv)
    x_1 = np.array(x_toConv)
    return x_1, t_1, N



def main():
    w = [[random.normalvariate(0, math.sqrt(1 / 24)) for i in range(3)] for j in range(24)]
    w_cur, w_prev = w, w
    x_1 = []
    t_1 = []
    N = 0
    x_1, t_1, N = prepareData('flare.data2', x_1, t_1, N)
    x = np.array(x_1)
    t = np.array(t_1)

    ind = np.arange(N)
    ind_prm = np.random.permutation(ind)
    train_ind = ind_prm[:np.int32(0.8 * N)]
    test_ind = ind_prm[np.int32(0.8 * N):]
    x_train = np.array(x[train_ind])
    t_train = t[train_ind]
    N_train = x_train.shape[0]
    x_test = x[test_ind]
    t_test = t[test_ind]
    N_test = N - N_train
    step = 0

    while (step < N_train):
        k = 0
        nabla_w_E = 0
        while k < N_train:
            cur = np.array(t[k].transpose() - np.dot(x[k], w_cur))
            nabla_w_E += np.dot(x[k].transpose(), cur)
            k += 1
        nabla_w_E = (-1 / N_train) * nabla_w_E
        w_prev = w_cur
        w_cur = w_prev - gamma * nabla_w_E
        step += 1
        if step > 20:
            if np.linalg.norm(w_cur - w_prev) < epsilon or np.linalg.norm(nabla_w_E) < epsilon:
                break
    sum = 0
    i = 0
    while i < N_train:
        sum += pow(np.linalg.norm((t_train[i]) - np.dot(x_train[i],w_cur)), 2)
        i += 1
    error_train = (sum / (2 * N_train))
    print('Ошибка модели на обучающей выборке' + str(error_train))
    sum = 0
    i = 0
    while i < N_test:
        sum += pow(np.linalg.norm(t_test[i] - np.dot(x_test[i],w_cur)), 2)
        i += 1
    error_test = (sum /(2*N_test))
    print('Ошибка модели на тестовой выборке' + str(error_test))
main()

