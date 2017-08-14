# implementation of AlexNet
# Date: 2017/08/03
# Author: Rui Wang
# Language: Python
# import packages
import tensorflow as tf
import numpy as np
# import scipy.io as sio  # scipy doesn't supprot the .mat type that beyond v7.3
import h5py as hy
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def read_YTC_images():
    file = hy.File('YTC_AlexNet_images_1.mat', 'r')  # will exist shape translation
    train_data = file['train_data']
    test_data = file['test_data']
    train_label = file['train_label']
    test_label = file['test_label']
    whole = [train_data, train_label, test_data, test_label]
    return whole

# build this model
def model_construction(images, n_class):
    parameters = []
    # cov_layer_1
    with tf.name_scope('cov1') as scope:
        W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], dtype=tf.float32, stddev=0.1), name='weights')  # 5*5, 1
        cov = tf.nn.conv2d(images, W1, [1, 2, 2, 1], padding='SAME')
        B1 = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), name='bias')
        sum_layer1 = tf.nn.relu(tf.nn.bias_add(cov, B1), name=scope)
        print_activations(sum_layer1)
        parameters += [W1, B1]
    # the first normalization operate
    with tf.name_scope('lrn1') as scope:
        lrn_1 = tf.nn.local_response_normalization(sum_layer1,
                                                   alpha=1e-4,
                                                   beta=0.75,
                                                   depth_radius=2,
                                                   bias=2.0)
    # the first pooling operation
    pool_1 = tf.nn.max_pool(lrn_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')  # 2*2, 2
    print_activations(pool_1)

    # cov_layer_2
    with tf.name_scope('cov2') as scope:
        W2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1), name='weights')
        cov = tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='SAME')
        B2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), name='bias')
        sum_layer2 = tf.nn.relu(tf.nn.bias_add(cov, B2), name=scope)
        print_activations(sum_layer2)
        parameters += [W2, B2]

    # the second normalization operate
    with tf.name_scope('lrn2') as scope:
        lrn_2 = tf.nn.local_response_normalization(sum_layer2,
                                                   alpha=1e-4,
                                                   beta=0.75,
                                                   depth_radius=2,
                                                   bias=2.0)
        # the second pooling operation
    pool_2 = tf.nn.max_pool(lrn_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool_2)

    # cov_layer_3
    with tf.name_scope('cov3') as scope:
        W3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=0.1), name='weights')
        cov = tf.nn.conv2d(pool_2, W3, [1, 1, 1, 1], padding='SAME')
        B3 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='bias')
        sum_layer3 = tf.nn.relu(tf.nn.bias_add(cov, B3), name=scope)
        print_activations(sum_layer3)
        parameters += [W3, B3]

    # cov_layer_4
    with tf.name_scope('cov4') as scope:
        W4 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1), name='weights')
        cov = tf.nn.conv2d(sum_layer3, W4, [1, 1, 1, 1], padding='SAME')
        B4 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), name='bias')
        sum_layer4 = tf.nn.relu(tf.nn.bias_add(cov, B4), name=scope)
        print_activations(sum_layer4)
        parameters += [W4, B4]

    # cov_layer_5
    with tf.name_scope('cov5') as scope:
        W5 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=0.1), name='weights')
        cov = tf.nn.conv2d(sum_layer4, W5, [1, 1, 1, 1], padding='SAME')
        B5 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), name='bias')
        sum_layer5 = tf.nn.relu(tf.nn.bias_add(cov, B5), name=scope)
        print_activations(sum_layer5)
        parameters += [W5, B5]

    # the third pooling operation
    pool_5 = tf.nn.max_pool(sum_layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool_5)

    # the first fully connected layer
    with tf.name_scope('fc1') as scope:
        W6 = tf.Variable(tf.truncated_normal([3*3*128, 1024], dtype=tf.float32, stddev=0.1),
                         name='weights')
        feature_reshape_1 = tf.reshape(pool_5, shape=[-1, 3*3*128])
        B6 = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), name='bias')
        sum_fc1 = tf.nn.relu(tf.matmul(feature_reshape_1, W6)+B6, name=scope)
        print_activations(sum_fc1)
        parameters += [W6, B6]

    # the second fully connected layer
    with tf.name_scope('fc2') as scope:
        W7 = tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=0.1), name='weights')
        B7 = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), name='bias')
        sum_fc2 = tf.nn.relu(tf.matmul(sum_fc1, W7)+B7, name=scope)
        print_activations(sum_fc2)
        parameters += [W7, B7]

    # output layer (soft_max)
    with tf.name_scope('sfm') as scope:
        W8 = tf.Variable(tf.truncated_normal([1024, n_class], dtype=tf.float32, stddev=0.1), name='weights')
        B8 = tf.Variable(tf.constant(0.0, shape=[n_class], dtype=tf.float32), name='bias')
        sfm_forward = tf.matmul(sum_fc2, W8)+B8
        print_activations(sfm_forward)
        parameters += [W8, B7]

    '''all_weights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]),
                             tf.reshape(W3, [-1]), tf.reshape(W4, [-1]),
                             tf.reshape(W5, [-1]), tf.reshape(W6, [-1]),
                             tf.reshape(W7, [-1]), tf.reshape(W8, [-1])], 0)
    all_bias = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]),
                          tf.reshape(B3, [-1]), tf.reshape(B4, [-1]),
                          tf.reshape(B5, [-1]), tf.reshape(B6, [-1]),
                          tf.reshape(B7, [-1]), tf.reshape(B8, [-1])], 0)'''

    return sfm_forward, parameters

def run_AlexNet():
    # read data
    YTC_data = read_YTC_images()
    train_data = YTC_data[0]  # 141*40000
    train_label = YTC_data[1]  # 141*47
    test_data = YTC_data[2]  # 282*40000
    test_label = YTC_data[3]  # 282*47
    train_label = np.reshape(train_label, newshape=[train_label.shape[1]])
    train_label = tf.one_hot(train_label, depth=10)
    test_label = np.reshape(test_label, newshape=[test_label.shape[1]])
    test_label = tf.one_hot(test_label, depth=10)
    # m_data = input_data.read_data_sets("/data/mnist", one_hot=True)
    # define some common parameters
    batch_size_train = 100
    batch_size_test = 20
    n_class = 10
    n_epoch = 60

    # define the main placeholders
    X = tf.placeholder(tf.float32, [None, 48, 48, 3])
    Y_practical = tf.placeholder(tf.float32, [None, n_class])

    # X_test = tf.placeholder(tf.float32, [batch_size_test, 48, 48, 3])
    # Y_practical_test = tf.placeholder(tf.float32, [batch_size_test, n_class])

    lr = tf.placeholder(tf.float32)

    # train the model
    sfm_forward, parameters = model_construction(X, n_class)
    # sfm_forward_test, _ = model_construction(X_test, n_class)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=sfm_forward, labels=Y_practical)
    loss = tf.reduce_mean(cross_entropy)
    train_optimize = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    sum_each_batch_loss = tf.Variable(0.0, tf.float32)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # train phrase
        train_label = train_label.eval()
        test_label = test_label.eval()
        n_batch_train = int(train_data.shape[0] / batch_size_train)  # m_data.train.num_examples
        for i in range(n_epoch):
            for j in range(n_batch_train):
                # X_batch, Y_batch = m_data.train.next_batch(batch_size)
                X_batch = train_data[j*batch_size_train:(j+1)*batch_size_train, :]  # 47*40000
                Y_batch = train_label[j*batch_size_train:(j+1)*batch_size_train, :]  # 47*47
                X_batch = np.reshape(X_batch, [batch_size_train, 48, 48, 3], order="F")   # 47*200*200*1
                '''a = X_batch[5, :] plt.imshow(a) plt.show()'''
                optimize_value, loss_value, net_output = sess.run([train_optimize, loss, sfm_forward],
                                                                  feed_dict={X: X_batch,
                                                                             Y_practical: Y_batch, lr: 0.001})
                net_output_sotfmax = sess.run(tf.nn.softmax(net_output))
                if (i+1) % 2 == 0:
                    correct = tf.equal(tf.argmax(net_output_sotfmax, 1), tf.argmax(Y_batch, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.0
                    accuracy_value = sess.run(accuracy)
                    print("Step %d: batch-train %d, accuracy rate is %f:" % (i-1, j, accuracy_value))
                sum_each_batch_loss = sess.run(tf.add(sum_each_batch_loss, loss_value))
            print("Step %d: loss value is %f" % (i, sum_each_batch_loss / n_batch_train))

        # test phrase
        # correct_num = 0
        n_batch_test = int(test_data.shape[0] / batch_size_test)  # m_data.test.num_examples
        for k in range(n_batch_test):
            # X_batch_test, Y_batch_test = m_data.test.next_batch(batch_size)
            X_batch_test = test_data[k * batch_size_test:(k + 1) * batch_size_test, :]  # 5*40000
            Y_batch_test = test_label[k * batch_size_test:(k + 1) * batch_size_test, :]  # 5*47
            X_batch_test = np.reshape(X_batch_test, [batch_size_test, 48, 48, 3], order="F")  # 5*200*200*1
            sfm_forward_value = sess.run(sfm_forward, feed_dict={X: X_batch_test, Y_practical: Y_batch_test})
            sfm_output = tf.nn.softmax(sfm_forward_value)
            predict_able = tf.equal(tf.argmax(sfm_output, 1), tf.argmax(Y_batch_test, 1))
            predict_result = tf.reduce_mean(tf.cast(predict_able, tf.float32))*100.0
            predict_accuracy = sess.run(predict_result)
            # correct_num += predict_result.eval()
            print("Batch-test %d: the classification accuracy is: %f" % (k+1, predict_accuracy))

def main(argv):
    run_AlexNet()
if __name__ == "__main__":
    tf.app.run()











