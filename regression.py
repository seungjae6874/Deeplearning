import warnings as w
w.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
print("Packages Loaded")


mnist = input_data.read_data_sets('data/', one_hot = True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("Mnist Loaded")

#mnist가 28*28이니까 한줄 벡터로 피면 784크기
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

#가중치의 행렬은 784*10이어야 w*x의 matrix 곱이 성립
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Logistic regression model
#이게 hypo
actv = tf.nn.softmax(tf.matmul(x,w)+b)

#cost function
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))

#optimizer
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Prediction and Accuracy
#실제 정답값인 y와 내가 계산해서 학습시키는 actv(hypo)값이 같으면 equal로써 1이다.
pred = tf.equal(tf.argmax(actv,1), tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(pred, "float"))

#initailizer
#이걸 초기화 해줘야 w와 b를 Variable로 사용 가능
init = tf.global_variables_initializer()

#Train Model
