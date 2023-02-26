import numpy as np
import pandas as pd

df = pd.read_csv('data.txt')

# Вектор входные значения
X = df.drop('Y', axis=1).to_numpy()

# Вектор выходных значений
Y = df['Y'].to_numpy()

print(df)


class NeuralNetwork():
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter
        self.w = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
        self.w_ = np.array([0.2, 0.3])
        self.intercept = 0

    def f(self, x):
        return (2 / (1 + np.exp(-x))) - 1

    def df(self, x):
        return 0.5 * (1 + x) * (1 - x)

    def output(self, x):
        out = np.array([self.f(val) for val in (x @ self.w.T)])  # + self.intercept
        y_pred = self.f(out @ self.w_)
        return y_pred, out

    def fit(self, x, y):
        for _ in range(self.n_iter):
            for row, result in zip(x, y):
                y_pred, out = self.output(row)
                e_tmp = y_pred - result
                local_grad = e_tmp * self.df(y_pred)

                self.w_[0] = self.w_[0] - self.eta * local_grad * out[0]
                self.w_[1] = self.w_[1] - self.eta * local_grad * out[1]

                local_grad_ = self.w_ * local_grad * self.df(out)

                self.w[0, :] = self.w[0, :] - np.array([row]) * local_grad_[0] * self.eta
                self.w[1, :] = self.w[1, :] - np.array([row]) * local_grad_[1] * self.eta

        return self

    def predict(self, x):
        y_pred, out = self.output(x)
        return y_pred


nn = NeuralNetwork(0.01, 10).fit(X, Y)

print(df)

for i in range(df.shape[0]):
    print(nn.predict(df.iloc[i].tolist()[0:3]), df.iloc[i].to_list()[-1])

print(nn.w, nn.w_)
