import matplotlib.pyplot as plt
import numpy as np
N = 100
maxLevel = 20
maxRound = 10
x = np.linspace(0, 1, N)
z = 0.05 * np.sin(2 * np.pi * 3 * x) + 0.25 * np.exp(x)
x_test = np.linspace(0, 1, 10 * N)
z_test = 0.05 * np.sin(2 * np.pi * 3 * x_test) + 0.25 * np.exp(x_test)
def getPlanMatrix(x, m, n):
    F = np.zeros((n, m + 2))
    i = 0
    for i in range(n):
        j = 0
        for j in range(m + 2):
            F[i][j] = x[i] ** j
    return F
def getPolynomialRegression(w, x, m):
    y = 0
    step = 0
    for step in range(m + 2):
        y += w[step] * (x ** step)
    return y

def getModelParametres(F, t):
    w = np.dot(np.dot(np.linalg.inv(np.dot(F.transpose(), F)), F.transpose()), t)
    return w
def getModelParametresReg(F, l, t):
    w = np.dot(np.dot(np.linalg.inv(np.dot(F.transpose(), F) + l * np.eye(len(F.transpose()))), F.transpose()), t)
    return w
def getModel(_isRegNeed):
    level = 0
    bias_list, var_list, global_error_list = [], [], []

    for level in range(maxLevel):
        pred_list = []
        error_list = []
        round = 0
        for round in range(maxRound):
            error = (0.025 * np.random.randn(N))
            error_test = (0.025 * np.random.randn(10 * N))
            pred = np.zeros(10 * N)
            e = np.zeros(10 * N)
            t = z + error
            F = getPlanMatrix(x, level, N)
            if (_isRegNeed == True):
                w = getModelParametresReg(F, 0.005, t)
            else:
                w = getModelParametres(F, t)
            t_test = z_test + error_test
            i = 0
            for i in range(len(x_test)):
                pred[i] = getPolynomialRegression(w, x_test[i], level)
            i = 0
            for i in range(len(pred)):
                e[i] = (pow(pred[i] - t_test[i], 2))
            error_list.append(e)
            pred_list.append(pred)

        var = np.mean(np.var(pred_list, axis=0))
        pred_list_bias = np.mean(pred_list, axis=0)
        e_bias = np.zeros(len(x_test))
        i = 0
        for i in range(len(pred_list_bias)):
            e_bias[i] = pow(pred_list_bias[i] - z_test[i], 2)

        bias = np.mean(e_bias)
        bias_list.append(bias)
        var_list.append(var)
        global_error_list.append(np.mean(error_list))

    return bias_list, var_list, global_error_list

ax_x = np.linspace(0,20,20)

bias_list, var_list, global_error_list = getModel(False)
bias_list_reg, var_list_reg, global_error_list_reg = getModel(True)
print(bias_list)
print(bias_list_reg)
print(var_list)
print(var_list_reg)
print(global_error_list)
print(global_error_list_reg)
fig, axes = plt.subplots(3, 2)

axes[0,0].plot(ax_x, bias_list, label='bias_list')
axes[0,0].legend(shadow=True, fontsize=8)
axes[0,0].set_title('Смещение')
axes[0,1].plot(ax_x, bias_list_reg, label='bias_list_reg')
axes[0,1].legend(shadow=True, fontsize=8)


axes[1,0].plot(ax_x, var_list, label='var_list')
axes[1,0].legend(shadow=False, fontsize=8)
axes[1,0].set_title('Разброс')
axes[1,1].plot(ax_x, var_list_reg, label='var_list_reg')
axes[1,1].legend(shadow=False, fontsize=8)

axes[2,0].plot(ax_x, global_error_list, label='global_error_list')
axes[2,0].legend(shadow=False, fontsize=8)
axes[2,0].set_title('Ошибка модели')
axes[2,1].plot(ax_x, global_error_list_reg, label='global_error_list_reg')
axes[2,1].legend(shadow=False, fontsize=8)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
