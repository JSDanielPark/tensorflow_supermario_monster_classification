import tensorflow as tf
import numpy as np

from os import listdir
from scipy import misc

tf.set_random_seed(777)


train_0_path = './dataset/train/0/'
train_1_path = './dataset/train/1/'
test_0_path = './dataset/test/0/'
test_1_path = './dataset/test/1/'

input_size = 120*120
output_size = 1

x_stack = np.empty(0).reshape(0, input_size)
y_stack = np.empty(0).reshape(0, output_size)

li = listdir(train_0_path)
for file in li:
    f = misc.imread(train_0_path + file)
    f = np.reshape(f, [input_size])
    x_stack = np.vstack([x_stack, f])
    y_stack = np.vstack([y_stack, [0]])

li = listdir(train_1_path)
for file in li:
    f = misc.imread(train_1_path + file)
    f = np.reshape(f, [input_size])
    x_stack = np.vstack([x_stack, f])
    y_stack = np.vstack([y_stack, [1]])


# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, input_size])
X_img = tf.reshape(X, [-1, 120, 120, 1])
Y = tf.placeholder(tf.float32, [None, output_size])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
L3 = tf.nn.conv2d(L3, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
print L3

L3 = tf.reshape(L3, [-1, 256 * 8 * 8])


W4 = tf.get_variable("W4", shape=[256 * 8 * 8, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[625, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1]))


hypothesis = tf.sigmoid(tf.matmul(L4, W5) + b5)


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(15):
    avg_cost = 0

    for i in range(100):
        feed_dict = {X: x_stack, Y: y_stack, keep_prob: 0.7}
        c, _, h = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
    print('Acc: {}'.format(h))

print('Learning Finished!')

x_stack = np.empty(0).reshape(0, input_size)
y_stack = np.empty(0).reshape(0, output_size)

li = listdir(test_0_path)
for file in li:
    f = misc.imread(test_0_path + file)
    f = np.reshape(f, [input_size])
    x_stack = np.vstack([x_stack, f])
    y_stack = np.vstack([y_stack, [0]])

li = listdir(test_1_path)
for file in li:
    f = misc.imread(test_1_path + file)
    f = np.reshape(f, [input_size])
    x_stack = np.vstack([x_stack, f])
    y_stack = np.vstack([y_stack, [1]])


feed_dict = {X: x_stack, Y: y_stack, keep_prob: 1.0}
acc = sess.run([accuracy], feed_dict=feed_dict)
print('Final Acc: {}'.format(acc))