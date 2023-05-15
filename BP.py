from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from numpy import *
# Load the dataset and split into training/testing sets
x = np.random.random((100,1)) * 10
print(max(x[:,0]))
x1 = x[:,0]
# x2 = x[:,1]
# y = x**6 + x**5 + x**4 + x**3 + x**2 + x
# y = np.sin(x**2)*np.cos(x) - 1
y = sin(x1)+sin(x1+x1**2)
# y = log(x1+1) + log(x1**2 + 1)
# y= sin(x1)+sin(x2**2)
# y = x1**x2
# y = x1**4 - x1**3 + 0.5* x2**2 - x2
# y = x1 + x2
# y = np.sin(x)


# Initialize the MLPRegressor model with desired parameters
# Note: you can experiment with different values for these parameters
w1 = 20
w2 = 200
mlp = MLPRegressor(hidden_layer_sizes=(w1,w2), activation='relu', solver='lbfgs', max_iter=10000, learning_rate_init=0.001)
R2 = []
for i in range(10):
    # Train the model on the training set
    mlp.fit(x, y)

    # Predict the output for test set and calculate mean squared error
    y_pred = mlp.predict(x)
    # print(y)
    # print(y_pred)
    r2 = r2_score(y, y_pred)
    R2.append(r2)
    print("R^2 score: ", r2)
# print(R2)
print('nod',w1 + w2)
print('para',2 * w1 + w1 + w1 * w2 + w2)
print(np.mean(R2))