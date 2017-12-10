import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


def linearRegression():
    data = loadtxtAndcsv_data("data.txt", ",", np.float64)
    X = np.array(data[:, 0:-1], dtype=np.float64)
    y = np.array(data[:, -1], dtype=np.float64)

    scaler = StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)

    temp = np.array([1650, 3])
    print(temp)
    x_test = scaler.transform(temp)

    model = linear_model.LinearRegression()
    model.fit(x_train, y)

    result = model.predict(x_test)
    print(model.coef_)
    print(model.intercept_)
    print(result)


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def loadnpy_data(fileName):
    return np.load(fileName)


if __name__ == "__main__":
    linearRegression()