#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from comet_ml import Experiment
from dataset.mnist import load_mnist
from sklearn.metrics import accuracy_score

def softmax(a):
    c = a.max()
    exp_a = np.exp(a - c)
    sum_exp_a = exp_a.sum(axis=1)
    sum_exp_a = sum_exp_a.reshape(1, sum_exp_a.size).transpose(1,0)
    value = exp_a / sum_exp_a
    return value

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    value = -np.sum(t * np.log(y+1e-7)) / batch_size
    return value

def numerical_diff(f, x, i):
    h = 1e-4
    h_vec = np.zeros_like(x)
    h_vec[i] = h
    return (f(x + h_vec) - f(x - h_vec)) / (2*h)

def numerical_diff2(f, x, i, j):
    h = 1e-4
    h_vec = np.zeros_like(x)
    h_vec[i, j] = h
    return (f(x + h_vec) - f(x - h_vec)) / (2*h)

def numerical_gradient(f, x):
    grad = np.zeros_like(x).astype(np.float128)
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            grad[i, j] = numerical_diff2(f, x, i, j)
    return grad

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.w) + self.b
    
    def backward(self, dout):
        self.dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return self.dx

class ReLU:
    def forward(self, x):
        self.mask = (x <= 0)
        self.out = x.copy()
        self.out[self.mask] = 0
        return self.out
    
    def backward(self, dout):
        dout[self.mask] = 0
        self.dx = dout
        return self.dx
    
class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        self.dx = dout * self.out * (1 - self.out)
        return self.dx
    
class SoftmaxWithLoss:
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        self.dx = (self.y - self.t) / batch_size
        return self.dx

class Adam:
    def __init__(self, shape, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def get_update(self, w, dw):
        self.t += 1
        self.m = (self.beta1 * self.m) + (1 - self.beta1) * dw
        self.v = (self.beta2 * self.v) + (1 - self.beta2) * dw**2
        mh = self.m / (1 - self.beta1 ** self.t)
        vh = self.v / (1 - self.beta2 ** self.t)
        w -= self.alpha * (mh / (np.sqrt(vh) + self.epsilon))
        return w
    
class Network:
    def __init__(self, input_size, hidden_size=32, output_size=10):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        self.adam_w1 = Adam(self.w1.shape)
        self.adam_b1 = Adam(self.b1.shape)
        self.adam_w2 = Adam(self.w2.shape)
        self.adam_b2 = Adam(self.b2.shape)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.w1, self.b1)
        self.layers['ReLU'] = ReLU()
        self.layers['Affine2'] = Affine(self.w2, self.b2)
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def get_score(self, x, t):
        y_pred = np.argmax(self.predict(x), axis=1)
        y_true = np.argmax(t, axis=1) if t.ndim != 1 else t
        return accuracy_score(y_true, y_pred)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        self.dw1 = self.layers['Affine1'].dw
        self.db1 = self.layers['Affine1'].db
        self.dw2 = self.layers['Affine2'].dw
        self.db2 = self.layers['Affine2'].db
    
    def update(self):
        self.w1 = self.adam_w1.get_update(self.w1, self.dw1)
        self.b1 = self.adam_b1.get_update(self.b1, self.db1)
        self.w2 = self.adam_w2.get_update(self.w2, self.dw2)
        self.b2 = self.adam_b2.get_update(self.b2, self.db2)

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    net = Network(x_train.shape[1])
    
    train_size = x_train.shape[0]
    batch_size = 100
    batch_count = train_size // batch_size
    epoch = 30
    losses = []
    
    hyper_params = {"epoch": epoch, "batch_size": batch_size}
    experiment = Experiment()
    experiment.log_parameters(hyper_params)
    for param, value in hyper_params.items():
        print(f'{param}: {value}')

    for epoc in range(epoch):
        running_loss = 0
        for i, batch in tqdm(enumerate(range(batch_count))):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            net.gradient(x_batch, t_batch)
            net.update()
            
            loss = net.loss(x_batch, t_batch)
            running_loss += loss
        losses.append(running_loss / i)
        print(f'[{epoc+1}] loss: {losses[-1]}')
        experiment.log_metric('loss', losses[-1].astype(np.float64), step=epoc)
    
    train_score = net.get_score(x_train, t_train)
    test_score = net.get_score(x_test, t_test)
    print(f'train accuracy: {train_score}')
    print(f'test accuracy: {test_score}')
    experiment.log_metric('train_accuracy', train_score)
    experiment.log_metric('test_accuracy', test_score)

main()
