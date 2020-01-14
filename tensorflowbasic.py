import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
print("Package Loaded")
print ("Current TF version is %s" %(tf.__version__))

#tensor를 실행시키기 위해서는 무조건 Session.run(x)를 통해 실행시켜야한다.
sess = tf.Session()
print("Open Session")

def print_tf(x):
    print("Type is %s" %(type(x)))
    print("Value is %s" %(x))

hello = tf.constant("Hello tensorflow")

print_tf(hello)

#Sess.run
node = sess.run(hello)
print_tf(node)

#operation with fixed inputs
node1 = tf.constant(1)
node2 = tf.constant(2)
node = tf.add(node1,node2)
fnode = sess.run(node)
print_tf(fnode)
