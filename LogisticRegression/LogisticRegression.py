import  numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def LogisticRegression():
    data = loadtxtAndcsv_data("data2.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]

    plot_data(X, y)  # 作图

    X = mapFeature(X[:, 0], X[:, 1])  # 映射为多项式
    initial_theta = np.zeros((X.shape[1], 1))  # 初始化theta
    initial_lambda = 0.1  # 初始化正则化系数，一般取0.01,0.1,1.....

    J = costFunction(initial_theta, X, y, initial_lambda)  # 计算一下给定初始化的theta和lambda求出的代价J

    print(J)  # 输出一下计算的值，应该为0.693147
    # result = optimize.fmin(costFunction, initial_theta, args=(X,y,initial_lambda))    #直接使用最小化的方法，效果不好
    '''调用scipy中的优化算法fmin_bfgs（拟牛顿法Broyden-Fletcher-Goldfarb-Shanno）
    - costFunction是自己实现的一个求代价的函数，
    - initial_theta表示初始化的值,
    - fprime指定costFunction的梯度
    - args是其余测参数，以元组的形式传入，最后会将最小化costFunction的theta返回 
    '''
    result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X, y, initial_lambda))
    p = predict(X, result)  # 预测
    print('train :%f%%' % np.mean(np.float64(p == y) * 100))  # 与真实值比较，p==y返回True，转化为float

    X = data[:, 0:-1]
    y = data[:, -1]


    plotDecisionBoundary(result, X, y)  # 画决策边界

def costFunction(initial_theta, X, y, initial_lambda):
    m = len(y)
    J = 0
    h = sigmoid(np.dot(X,initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0
    temp = np.dot(np.transpose(theta1),theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y), np.log(1 - h)) + temp * initial_lambda/2)/m

    return J


def gradient(initial_theta, X, y, initial_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))

    h = sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0
    grad = np.dot(np.transpose(X), h - y) / m + initial_lambda / m * theta1
    return grad


def sigmoid(z):
    h = np.zeros((len(z), 1))
    h = 1.0/(1.0+np.exp(-z))
    return h

def mapFeature(X1, X2):
    degree = 3
    out = np.ones((X1.shape[0], 1))

    for i in np.arange(1, degree+1):
        for j in range(i+1):
            temp = X1**(i-j)*(X2**j)
            out = np.hstack((out, temp.reshape(-1, 1)))
    return out

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def loadnpy_data(fileName):
    return np.load(fileName)

def plot_data(X, y):
    pos = np.where(y==1)    #找到y==1的坐标位置
    neg = np.where(y==0)    #找到y==0的坐标位置
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')        # red o
    plt.plot(X[neg, 0], X[neg, 1], 'bo')        # blue o
    plt.title(u"double plot")
    plt.show()

def plotDecisionBoundary(theta, X, y):
    pos = np.where(y == 1)  # 找到y==1的坐标位置
    neg = np.where(y == 0)  # 找到y==0的坐标位置
    # 作图
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')  # red o
    plt.plot(X[neg, 0], X[neg, 1], 'bo')  # blue o
    plt.title(u"descide side")

    # u = np.linspace(30,100,100)
    # v = np.linspace(30,100,100)

    u = np.linspace(-1, 1.5, 50)  # 根据具体的数据，这里需要调整
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeature(u[i].reshape(1, -1), v[j].reshape(1, -1)), theta)  # 计算对应的值，需要map

    z = np.transpose(z)
    plt.contour(u, v, z, [0, 0.01], linewidth=2.0)  # 画等高线，范围在[0,0.01]，即近似为决策边界
    # plt.legend()
    plt.show()


def predict(X, theta):
    m = X.shape[0]
    p = np.zeros((m, 1))
    p = sigmoid(np.dot(X, theta))  # 预测的结果，是个概率值

    for i in range(m):
        if p[i] > 0.5:  # 概率大于0.5预测为1，否则预测为0
            p[i] = 1
        else:
            p[i] = 0


    return p



def testLogisticRegression():
    LogisticRegression()


if __name__ == "__main__":
    testLogisticRegression()