import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


#two different data file format
TXT_AND_CSV = 0
NPY = 1


def plot_X1_X2(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

def plot_J(J_history, num_iters):
    x = np.arange(1, num_iters+1)
    plt.plot(x, J_history)
    plt.xlabel(u'num_iters')
    plt.ylabel(u'loss')
    plt.title(u'loss change progress')
    plt.show()

def linerRegression(fileName, fileType = TXT_AND_CSV, alpha = 0.01, num_iters = 400):
    print('load data from %s'% (fileName))
    data = []
    if fileType == TXT_AND_CSV:
        data = loadtxtAndcsv_data(fileName, ',', np.float64)
    else:
        data = loadnpy_data(fileName)

    X = data[:, 0:-1]
    y = data[:, -1]
    m = len(y)
    col = data.shape[1]

    X, mu, sigma = featureNormaliza(X)
    plot_X1_X2(X)
    X = np.hstack((np.ones((m,1)),X))


    print('excute gradient')

    theta = np.zeros((col, 1))
    y = y.reshape(-1, 1)
    theta, J_history = gradientDscent(X, y, theta, alpha, num_iters)
    plot_J(J_history, num_iters)

    return mu, sigma, theta

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType )

def loadnpy_data(fileName):
    return np.load(fileName)

def computerCost(X, y, theta):
    m = len(y)
    J = 0
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m)
    return J

def gradientDscent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)
    temp = np.matrix(np.zeros((n, num_iters)))
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = np.dot(X,theta)
        temp[:, i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))
        theta = temp[:, i]
        J_history[i] = computerCost(X, y, theta)
        print('.')
    return theta, J_history

def featureNormaliza(X):
    X_norm = np.array(X)
    width = X.shape[1]
    mu = np.zeros((1, width))
    sigma = np.zeros((1, width))
    mu = np.mean(X_norm, 0)
    sigma = np.std(X_norm, 0)
    for i in range(width):
        X_norm[:, i] = (X_norm[:, i]-mu[i])/sigma[i]

    return X_norm, mu, sigma

def predict(mu,sigma,theta):
    result = 0
    predict = np.array([1650, 3])
    norm_predict = (predict - mu)/sigma
    final_predict = np.hstack((np.ones((1)), norm_predict))
    result = np.dot(final_predict, theta)
    return result

def testLinearRegression():
    fileName = 'data.txt'
    mu, sigma, theta = linerRegression(fileName)
    print('\ntheta:\n', theta)
    print('\npredict\n', predict(mu, sigma, theta))

if __name__ == '__main__':
    testLinearRegression()