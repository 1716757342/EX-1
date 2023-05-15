###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# main.py: From here, launch deep symbolic regression tasks. All
# hyperparameters are exposed (info on them can be found in train.py). Unless
# you'd like to impose new constraints / make significant modifications,
# modifying this file (and specifically the get_data function) is likely all
# you need to do for a new symbolic regression task.

###############################################################################
# Dependencies
###############################################################################

from train import train
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from numpy import *

###############################################################################
# Main Function
###############################################################################

# A note on operators:
# Available operators are: '*', '+', '-', '/', '^', 'sin', 'cos', 'tan',
#   'sqrt', 'square', and 'c.' You may also include constant floats, but they
#   must be strings. For variable operators, you must use the prefix var_.
#   Variable should be passed in the order that they appear in your data, i.e.
#   if your input data is structued [[x1, y1] ... [[xn, yn]] with outputs
#   [z1 ... zn], then var_x should precede var_y.

def main():
    # Load training and test data
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # Perform the regression task
    results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = ['*', '+', '-','/','cos', 'sin','var_x1'],
        # operator_list=['*', '+', 'sin', 'var_x'],
        # operator_list=['*', '+', '-', '/', 'cos', 'sin', 'exp', 'ln', 'sqrt', 'var_x1'],
        min_length = 2,
        max_length = 70,
        type = 'lstm',
        num_layers = 2,
        hidden_size = 250,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'lbfgs',
        inner_lr = 0.1,
        inner_num_epochs = 2,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 1000,
        scale_initial_risk = True,
        batch_size = 1000,
        num_batches = 10000,
        use_gpu = False,
        live_print = True,
        summary_print = True,
        config_prior='./config_prior.json'
    )

    # Unpack results
    epoch_best_rewards = results[0]
    epoch_best_expressions = results[1]
    best_reward = results[2]
    best_expression = results[3]

    # Plot best rewards each epoch
    plt.plot([i+1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.show()



###############################################################################
# Getting Data
###############################################################################

def get_data():
    """Constructs data for model (currently x^3 + x^2 + x)
    """
    X = np.arange(-1, 1.1, 0.01) * 1
    # X = np.random.randn(20) * 2
    # X = (np.random.rand(20) * 2 - 1) * 2
    # X.sort()
    # y = X**7 + X**6 + X**5 + X**4 + X**3 + X**2 + X
    # y = X ** 6 + X ** 5 + X ** 4 + X ** 3 + X ** 2 + X
    # y = X**6 + X**5 + X**4 + X ** 3 + X ** 2 + X
    y = X**9 + X**8 + X**7 + X**6 + X**5 + X**4 + X ** 3 + X ** 2 + X
    # y = np.sin(X**2) * X**3
    # y = np.sin(X**2)*np.cos(X) - 1
    # y = 2.5 * X + 2.7*X**2
    # y = np.sin(X**2) + np.sin(X**2 + X)
    # y = 2.4 * X**2 +
    # plt.plot(X,y)
    # plt.show()
    # y = np.log(X ** 2 + 1) + np.log(X+1)
    # y = np.sin(X**2) * X**3

    x1 = X
    # y = 0.3*X*sin(2*pi*x1)LeNet.py
    # y = pow(x1,3)*exp(-x1)*cos(x1)*sin(x1)*(pow(sin(x1),2)*cos(x1)-1)
    # y = (x1*(x1+1)/2)
    # y = log(x1 + sqrt(pow(x1, 2) + 1))
    # y = 0.13*sin(x1)-2.3
    # y = 3+2.13*log(abs(x1))
    # y = 6.87+11*cos(7.23*pow(x1,3))
    # y = pow(x1,5)-2*pow(x1,3)+x1
    # y = sin(x1**3)*cos(x1**2)-1
    # y = ((pow(x1,6)+pow(x1,5)))/((pow(x1,4)+pow(x1,3)+pow(x1,2)+x1))
    X = X.reshape(X.shape[0], 1)

    # Split randomly
    comb = list(zip(X, y))
    random.shuffle(comb)
    X, y = zip(*comb)

    # Proportion used to train constants versus benchmarking functions
    training_proportion = 0.2
    div = int(training_proportion*len(X))
    X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
    y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
    X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
    y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
    return X_constants, X_rnn, y_constants, y_rnn

if __name__=='__main__':
    main()


###############################################################################
# Getting Data
###############################################################################

# def get_data():
#     """Constructs data for model (currently x^3 + x^2 + x)
#     """
#     # X = np.arange(-1, 1.1, 0.1)
#
#     # X1 = np.arange(-1, 1.1, 0.05) * 1.5
#     # X2 = np.arange(-1, 1.1, 0.05) * 1
#     X1 = (np.random.rand(100) * 2 - 1) * 1.5
#     X2 = (np.random.rand(100) * 2 - 1) * 1.5
#
#     # y = X**7 + X**6 + X**5 + X**4 + X**3 + X**2 + X
#     # # y = X ** 6 + X ** 5 + X ** 4 + X ** 3 + X ** 2 + X
#     # # y = X ** 3 + X ** 2 + X
#     # y = np.sin(X**2)* X**3
#     # y = np.sin(X ** 2) + np.cos(X ** 2)
#     # y = np.sin(X ** 2) * np.cos(X)
#     # y = np.sin(X1**2) * np.cos(X2)
#     y = X1 ** 4 - X1**3 + 0.5 * X2 ** 2 - X2
#     # y = 2 * np.sin(X1) * np.cos(X2)
#     # y = np.sin(X1)+np.sin(X2**2)
#
#
#     num = len(X1)
#     X1 = X1.reshape(num, 1)
#     X2 = X2.reshape(num, 1)
#     X = np.concatenate((X1, X2), axis=1)
#     # print(X)
#     # Split randomly
#     comb = list(zip(X, y))
#     random.shuffle(comb)
#     X, y = zip(*comb)
#
#     # Proportion used to train constants versus benchmarking functions
#     training_proportion = 0.2
#     div = int(training_proportion*len(X))
#     X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
#     y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
#     X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
#     y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
#     return X_constants, X_rnn, y_constants, y_rnn
#
# if __name__=='__main__':
#     main()
