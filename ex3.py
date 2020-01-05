import tensorflow as tf
x = [1,2,3]
y = [1,2,3]

#W = tf.Variable(tf.random_normal([1]),name = 'weight')
W = tf.Variable(5.0)
#X = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)
h = W*x

cost = tf.reduce_mean(tf.square(h-y))
#cost = tf.reduce_sum(tf.square(h-y))

#lr = 0.1 #러닝 비율
#gradient = tf.reduce_mean((h-y)*X) #미분
#descent = W - lr*gradient #가중치 W의 감소량
#update = W.assign(descent) #update에 새로운 가중치 descent를 할당



#이 미분계산과 learngin_rate을 설정해주지 않아도 tensorflow에서 자동으로
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost) #cost비용을 최소화시키도록 훈련시키라는 의미




sess = tf.Session()

sess.run(tf.global_variables_initializer())

#for step in range(21):
 #   sess.run(update, feed_dict = {X : x, Y : y})
  #  print(step, sess.run(cost, feed_dict= { X : x, Y: y}),sess.run(W))


for step in range(10):
    print(step,sess.run(W))
    sess.run(train)
