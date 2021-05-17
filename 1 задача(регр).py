import matplotlib.pyplot as plt
import numpy as np

N = 1000
# M = 100

def generate_wave_set():
    x =np.linspace(0, 1, N)
    y = 20*np.sin(2*np.pi*3*x) + 100*np.exp(x) + 10*np.random.randn(N)
    return x, y

x, y = generate_wave_set()

def generate_degree_list(m):
    degree_list = []
    for i in range(1, m + 1):
        degree_list.append(i)
    return degree_list

def regr(m):
    degree_list = generate_degree_list(m)
    w_list = []
    err = []
    for ix, degree in enumerate(degree_list):
        a = np.ones(x.shape[0])
        dlist = [np.ones(x.shape[0])] + \
                    [x**n for n in range(1, degree + 1)]
        X = np.array(dlist).T
        X = np.array(X)
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        w_list.append((degree, w))
        y_hat = np.dot(w, X.T)
        err.append(np.mean((y - y_hat)**2))

    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'ro', markersize=0.8)
    plt.plot(x, y_hat, 'g')
    plt.subplot(1, 2, 2)
    x_ = np.arange(1, len(degree_list)+1, 1)
    plt.plot(x_, err)
    plt.show()


regr(1)
regr(8)
regr(100)