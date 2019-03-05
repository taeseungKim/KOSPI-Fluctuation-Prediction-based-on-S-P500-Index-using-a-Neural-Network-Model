import tensorflow as tf
from real_swag_paper import input_pre_minMax_z as preprocessingData

import math

import numpy as np

tf.reset_default_graph()

# start_learn_rate = 0.001
global_step = tf.Variable(0,trainable=False)

# learning_rate = tf.train.exponential_decay(start_learn_rate,global_step,100000,0.95,staircase=True)
learning_rate = 0.001
total_epoch = 30
batch_size = 4
n_hidden = 8

train_X = np.array(preprocessingData.total_train_X)
train_Y = np.array(preprocessingData.total_train_Y)

test_X = np.array(preprocessingData.test_X)
test_Y = np.array(preprocessingData.test_Y)

n_input = 14 # 14 input_dimension
n_class = 2 #2
beta = 0.01

#########
# 신경망 모델 구성
#########

X = tf.placeholder(tf.float32, [None,n_input])
Y = tf.placeholder(tf.float32, [None, n_class]) #label

stddev = 0.3

W1 = tf.Variable(tf.random_normal([n_input, n_hidden], mean=0.0, stddev=stddev), name='weight1') # weight
W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden], mean=0.0, stddev=stddev), name='weight2')
W3 = tf.Variable(tf.random_normal([n_hidden, n_class], mean=0.0, stddev=stddev), name='weight3')

b1 = tf.Variable(tf.random_normal([n_hidden],mean=0.0,stddev=stddev),name='biases1') #bias
b2 = tf.Variable(tf.random_normal([n_hidden],mean=0.0,stddev=stddev),name='biases2')
b3 = tf.Variable(tf.random_normal([n_class],mean=0.0,stddev=stddev),name='biases3')

pred1 = tf.nn.relu(tf.matmul(X,W1)+b1)
pred2 = tf.nn.relu(tf.matmul(pred1,W2)+b2)
pred3 = tf.matmul(pred2,W3)+b3

cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred3, labels=Y))

reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
# reg = 0
cost2 = tf.reduce_mean(cost1 + reg * beta)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2,global_step=global_step)

#########
# 신경망 모델 학습
######
with tf.Session() as sess:
    results = []
    for i in range(5):
        sess.run(tf.global_variables_initializer())

        total_batch = math.ceil(int(len(train_X) / batch_size))
        for epoch in range(total_epoch):
            total_cost = 0
            for i in range(total_batch):

                train_X_l = train_X[i * batch_size:(i + 1) * batch_size, :]
                train_Y_l = train_Y[i * batch_size:(i + 1) * batch_size, :]

                _, cost_val = sess.run([optimizer, cost2],
                                       feed_dict={X: train_X_l, Y: train_Y_l})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1),
                  'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

            is_correct = tf.equal(tf.argmax(pred3, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            print('정확도:', sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))

        print('최적화 완료!')

        #########
        # 결과 확인
        #########
        is_correct = tf.equal(tf.argmax(pred3, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도:', sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
        results.append(sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
sess.close()
print(results)
print("mean : " + str(np.mean(results)))
print("standard deviation : " + str(np.std(results)))