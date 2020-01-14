import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#mnist는 0~9의 숫자들이 그려진 흑백 28*28 크기의 img
'''
print("Tensorflow version is %s" %(tf.__version__))

'''
#Download MNIST

mnist = input_data.read_data_sets('./',one_hot=True)

'''
def print_np(x):
    print('Shape of is %s' %( x.shape,))
    print('Value look like \n %s' %(x))

print_np(mnist.test.images) #test는 실제 test하는 case의 갯수
print_np(mnist.train.labels) #train은 학습시키기위한 case의 갯수
print_np(mnist.validation.labels)#validation은 train과 test사이의 모의고사같은 case
'''
'''
#PLOT TRAIN IMAGES
ntrain = mnist.train.images.shape[0]
nsample = 3

randidx = np.random.randint(ntrain, size = nsample)
for i in randidx:
    imgvec = mnist.train.images[i, :]
    labelvec = mnist.train.labels[i, :]
    img = np.reshape(imgvec, (28, 28))
    label = np.argmax(labelvec) #ONEHOT VECTOR -> LABEL
    plt.matshow(img, cmap = plt.get_cmap('gray'))
    plt.title("[%d] DATA / LABEL IS [%d]" %(i,label))
    print(labelvec)
'''

#GET RANDOM MINIBATCH

ntrain = 10
randindices = np.random.permutation(ntrain)#mnist를 shuffle하는 함수
print(randindices.shape)


#Select minibatch

ntrain = 10
nbatch = 4
niter = ntrain // nbatch +1
for i in range(niter):
    currindices = randindices[i*nbatch:(i+1)*nbatch]
    print('ITERATION : [%d] Batch index: %s' % (i, currindices))

    #Get batch
    xbatch = mnist.train.images[currindices, :]
    ybatch = mnist.train.labels[currindices, :]
    print("Shape of xbatch is %s" %(xbatch.shape,))
    print("Shape of ybatch is %s" %(ybatch.shape,))


#data를 한줄 vector로 펴서 차곡차곡 잘라서 학습시키는 개념을 mnist를 사용해서
#연습해 본것이다

