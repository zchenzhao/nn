import numpy as np
from sklearn.metrics import mean_squared_error

w1 = np.array([[0.15, 0.2], [0.25, 0.3]])
w2 = np.array([[0.4, 0.45], [0.5, 0.55]])

b1 = 0.35
b2 = 0.6

learning_rate = 0.5

temp_x = None

def forward_pass(x):
    global temp_x
    z1 = w1@x + b1
    z2 = 1/(1+np.e**(-z1))
    temp_x = z2
    z3 = w2@z2 + b2
    y_hat = 1/(1+np.e**(-z3))

    return y_hat

def backward_pass(x, y, y_hat):
    de_dx = - (y - y_hat) * (1 - y_hat) * y_hat
    de_dw2 = de_dx.reshape((2, 1))@x.reshape((1, 2))

    w2_updated = w2 - learning_rate * de_dw2

    return w2_updated

x = np.array([0.05, 0.1])
y_hat = forward_pass(x)

y = np.array([0.01, 0.99])

w2_updated = backward_pass(temp_x, y, y_hat)
print(w2_updated)


