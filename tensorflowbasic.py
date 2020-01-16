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
    print("Value is\n %s" %(x))

hello = tf.constant("Hello tensorflow")
'''
print_tf(hello)

#Sess.run
node = sess.run(hello)
print_tf(node)

#operation with fixed inputs
node1 = tf.constant(4)
node2 = tf.constant(5)
node = tf.add(node1,node2)
fnode = sess.run(node)
print_tf(fnode)

a_mul_b = tf.multiply(node1,node2)
print_tf(sess.run(a_mul_b))
'''
#Variable = 변하는 값 즉 , 학습시키고 싶은 parameter ex) weight or convolution filter
weight = tf.Variable(tf.random_normal([5,2],stddev=0.01))
#크기가 5*2의 랜덤값을 갖는 행렬을 생성한다. 반드시 밑에서 init해줘야함

#Variable은 생성후 한번 반드시 초기화 해줘야 함 
init = tf.global_variables_initializer()
sess.run(init)

weight_out = sess.run(weight)
print(weight_out)

#placeholder 내가 미리 알고있는 값들을 넣어주는 통로 input,ouput data들의 통로
x = tf.placeholder(tf.float32, [None,5])
#대신 placeholder는 data 타입과 행렬 크기는 알려줘야함. 앞의 None은 이 data 갯수가 무한개

#operator를 선언
oper = tf.matmul(x,weight) #weight는 5*2행렬이고 x는 5dimension
print(oper)


#feed dict placeholder = placeholder통로에 내가 넣을 값을 넣어줌
data= np.random.rand(4,5)
oper_out = sess.run(oper, feed_dict = {x: data})
print_tf(oper_out)


