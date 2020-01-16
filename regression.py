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

training_epochs = 50 #전체 epoch 즉 전체 데이터셋을 50번 학습시키겠다.
batch_size = 100 #한번 iteration 돌릴때의 data sample set
display_step = 2

#Session
sess = tf.Session()
sess.run(init)
#mini batch
for epoch in range(training_epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples/batch_size)
    #batch의 갯수는 전체 샘플 수를 batchsize로 나누어야 나옴

    for i in range(num_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={ x : batch_xs, y : batch_ys})
        feeds = {x : batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict = feeds)/num_batch

    #display
    if epoch % display_step == 0:
        feed_train = {x:batch_xs, y: batch_ys}
        feed_test = {x : mnist.test.images, y:mnist.test.labels}
        train_acc = sess.run(accr, feed_dict = feed_train)#train의 정확도
        test_acc =sess.run(accr, feed_dict = feed_test)# test정확도
        print("Epoch: %03d/%03d cost : %.9f train_acc : %.3f test_acc : %.3f"
              %(epoch, training_epochs, avg_cost,train_acc, test_acc))
print("All Done")
