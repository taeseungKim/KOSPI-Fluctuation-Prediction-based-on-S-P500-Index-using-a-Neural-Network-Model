import tensorflow as tf
from real_swag_paper import input_pre_minMax_z as preprocessingData
import random as rn

import math

import numpy as np

#Clears the default graph stack and resets the global default graph


# start_learn_rate = 0.001
#
# learning_rate = tf.train.exponential_decay(start_learn_rate,global_step,100000,0.90,staircase=True)

learning_rate = 0.001
total_epoch = 15
batch_size = 4
num_layers_l = [1,2,3,4,5] #hidden layers
n_step = 3
n_hidden = 16 #hidden units이 클수록 복잡한 연산 가능

for num_layers in num_layers_l:
    tf.reset_default_graph()

    global_step = tf.Variable(0,trainable=False)

    train_X = np.array(preprocessingData.train_X)
    train_Y = np.array(preprocessingData.train_Y)

    vali_X = np.array(preprocessingData.vali_X)
    vali_Y = np.array(preprocessingData.vali_Y)

    test_X = np.array(preprocessingData.test_X)
    test_Y = np.array(preprocessingData.test_Y)

    #time step에 맞게 변환
    train_X_f = []
    train_Y_f = []

    test_X_f = []
    test_Y_f = []

    for i in range(0, len(train_X) - n_step + 1):
        _train_X = train_X[i:i + n_step]
        _train_Y = train_Y[i + n_step - 1]
        train_X_f.append(_train_X)
        train_Y_f.append(_train_Y)
    train_X_f_n = np.array(train_X_f)
    train_Y_f_n = np.array(train_Y_f)

    for i in range(0, len(vali_X) - n_step + 1):
        _test_X = vali_X[i:i + n_step]
        _test_Y = vali_Y[i + n_step - 1]
        test_X_f.append(_test_X)
        test_Y_f.append(_test_Y)
    test_X_f_n = np.array(test_X_f)
    test_Y_f_n = np.array(test_Y_f)

    n_input = 14 # 14 input_dimension
    n_class = 2 #label수

    X = tf.placeholder(tf.float32, [None, n_step, n_input])
    Y = tf.placeholder(tf.float32, [None, n_class]) #label

    W = tf.Variable(tf.random_normal([n_hidden, n_class],mean=0.0,stddev=0.3),name='weight')# weight //make Variable instance
    b = tf.Variable(tf.random_normal([n_class],mean=0.0,stddev=0.3),name='biases') #bias

    cells = []
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden) #n_hidden은 output에서의 unit size와 관련
        cells.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
    ###############################################
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # 결과를 Y의 다음 형식과 바꿔야 하기 때문에
    # Y : [batch_size, n_class]
    # outputs 의 형태를 이에 맞춰 변경해야합니다.
    # outputs : [batch_size, n_step, n_hidden]
    #        -> [n_step, batch_size, n_hidden]
    #        -> [batch_size, n_hidden]
    def rem(output):
        outputs = tf.transpose(output, [1, 0, 2])
        out = outputs[-1]
        return out
    output = rem(outputs)

    model = tf.matmul(output, W) + b

    cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))

    reg =  tf.nn.l2_loss(W)

    cost2 = tf.reduce_mean(cost1 + reg * 0.01)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost2,global_step=global_step)

    # 신경망 모델 학습
    with tf.Session() as sess:
        results = []
        for i in range(5):
            sess.run(tf.global_variables_initializer()) #variable initializing
            total_batch = int(math.ceil(len(train_X_f_n)/batch_size))

            for epoch in range(total_epoch):
                total_cost = 0
                for i in range(total_batch):

                    train_X_f_l = train_X_f_n[i*batch_size:batch_size*(i+1),:,:]
                    train_Y_f_l = train_Y_f_n[i*batch_size:batch_size*(i+1),:]
                    # X 데이터를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input] 형태로 변환합니다.
                    # train_X_f_l = train_X_f_l.reshape((batch_size, n_step, n_input)) 얘때매 마지막 꼬다리가 학습 안됨

                    _, cost_val = sess.run([optimizer, cost2],
                                           feed_dict={X: train_X_f_l, Y: train_Y_f_l}) #feed placeholder
                    total_cost += cost_val

                # print('Epoch:', '%04d' % (epoch + 1),
                #       'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
                # is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  # correct_prediction
                # accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                # print('test set 정확도:', sess.run(accuracy, feed_dict={X: test_X_f_n, Y: test_Y_f_n}))

            # print('최적화 완료!')
            ################Test####################################
            is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) #correct_prediction
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

            # print('정확도:', sess.run(accuracy,feed_dict={X: test_X_f_n, Y: test_Y_f_n}))
            results.append(sess.run(accuracy, feed_dict={X: test_X_f_n, Y: test_Y_f_n}))
    sess.close()
    print(results)
    print("layers : " + str(num_layers))
    print("mean : " + str(np.mean(results)))
    print("standard deviation : " + str(np.std(results)))