import tensorflow as tf
#x_train = [1,2,3]
#y_train = [1,2,3]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')
hypo = W*X+b
cost = tf.reduce_mean(tf.square(hypo-Y))
opti = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = opti.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    #sess.run(train)
    cost_val,W_val,b_val,_ = sess.run([cost,W,b,train],
                                      feed_dict={X :[1,2,3],
                                                Y : [2,4,6]})
    
    if step % 20 == 0 :
          print(step, cost_val,W_val,b_val)
        
