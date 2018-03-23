import tensorflow as tf

W = tf.placeholder(tf.float32, shape=(10, 784))
x = tf.placeholder(tf.float32, shape=(784, 1))
y = tf.placeholder(tf.float32, shape=(10, 1))
b = tf.placeholder(tf.float32, shape=(10, 1))

softmax   = tf.exp(tf.matmul(W, x) + b) / tf.reduce_sum(tf.exp(tf.matmul(W, x) + b))
softmaxTF = tf.nn.softmax(tf.matmul(W, x) + b)

# tf.reduce_max vs tf.maximum? --> comparing two tensors or reducing the dimension of a tensor
p = tf.argmax(y)

with tf.Session() as sess:

  writer = tf.summary.FileWriter('./graphs/ex35', sess.graph)