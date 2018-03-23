""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = "data/birth_life_2010.txt"

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
# X = tf.placeholder(tf.float32, name='X')
# Y = tf.placeholder(tf.float32, name='Y')
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))

# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
X, Y     = iterator.get_next()  # X is birth rate, Y is life expectancy

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
# w = tf.get_variable('w', dtype=tf.float32, initializer=tf.constant(0.))
# b = tf.get_variable('b', dtype=tf.float32, initializer=tf.constant(0.))

# For scalar tensors: use () or [] as shape!
w = tf.get_variable('w', dtype=tf.float32, initializer=tf.random_normal(()))
b = tf.get_variable('b', dtype=tf.float32, initializer=tf.random_normal(()))

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
# loss = tf.square(Y - Y_predicted, name='loss')
loss = utils.huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate=1.1).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

start = time.time()

epochs = 100

learningCurve = np.zeros(epochs)

with tf.Session() as sess:

    # Create a filewriter to write the model's graph to TensorBoard
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 7: initialize the necessary variables, in this case, w and b
    # sess.run(tf.variables_initializer([w, b]))
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model for 100 epochs
    for i in range(epochs):
        total_loss = 0
        sess.run(iterator.initializer)
        # Execute train_op and get the value of loss.
        # Don't forget to feed in data for placeholders
        # without 'infinite' loop of while True: only once!
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l

        except tf.errors.OutOfRangeError:
            pass
        learningCurve[i] = total_loss/n_samples
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])

print('Took: %f seconds' %(time.time() - start))

# uncomment the following lines to see the plot
plt.figure()
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.figure()
plt.plot(learningCurve)
plt.figure()
plt.plot(data[:, 1] - (data[:, 0] * w_out + b_out))
plt.show()

"""
Exercise 4.1

(a) Done
(b) Done
(c) Done
(d) Done
(e) A very noisy graph, small numbers --> good!

Exercise 4.2

(a) Neglects outliers --> lower error
(b) tf.data is a bit faster, about 40%
(c) Differences in best step size, and thus speed of convergence
"""