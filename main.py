import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

x = np.array(
    [[0, 0],
    [0, 1],
    [1, 0],
    [1, 1]]
)

y = np.array(
    [[0],
     [1],
     [1],
     [1]]
)

ws_inp_hidd = np.random.rand(2, 2)
ws_hidd_out = np.random.rand(2, 1)


lr = 0.1

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative(a):
    return a*(1-a)

hiss = []

for i in range(1000):
    z_hidd = x.dot(ws_inp_hidd)
    a_hidd = sigmoid(z_hidd)

    z = a_hidd.dot(ws_hidd_out)
    a = sigmoid(z)

    error = y - a
    loss = np.mean(0.5*((error)**2))
    hiss.append(loss)
    
    ws_hidd_out += a_hidd.T.dot(error*derivative(a))*lr
    ws_inp_hidd += x.T.dot(error*derivative(a)*sigmoid(a_hidd))*lr

x = np.array([[0, 1]])
z_hidd = x.dot(ws_inp_hidd)
a_hidd = sigmoid(z_hidd)


z = a_hidd.dot(ws_hidd_out)
a = sigmoid(z)
print(a)


plt.plot(hiss)
plt.show()
