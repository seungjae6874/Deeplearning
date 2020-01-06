import tensorflow as tf
x = [1,2,3]
y = [1,2,3]

W = tf.Variable(tf.random_normal([1]),name = 'weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = W*X

cost = tf.reduce_sum(tf.square(h-y))

lr = 0.1 #러닝 비율
gradient = tf.reduce_mean((h-y)*X) #미분
descent = W - lr*gradient #가중치 W의 감소량
update = W.assign(descent) #update에 새로운 가중치 descent를 할당

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict = {X : x, Y : y})
    print(step, sess.run(cost, feed_dict= { X : x, Y: y}),sess.run(W))
    
