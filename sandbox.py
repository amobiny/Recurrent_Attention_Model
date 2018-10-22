import tensorflow as tf
import numpy as np

w0 = tf.get_variable("w0", shape=[5, 1], initializer=tf.truncated_normal_initializer())
w1 = tf.get_variable("w1", shape=[1, 1], initializer=tf.truncated_normal_initializer())
w2 = tf.get_variable("w2", shape=[5, 1], initializer=tf.truncated_normal_initializer())
w3 = tf.get_variable("w3", shape=[5, 1], initializer=tf.truncated_normal_initializer())
b0 = tf.constant(np.random.rand(1,))
x = tf.placeholder(tf.float32, shape=[None, 5], name="x")

a0 = tf.matmul(x, w0)
a1 = tf.matmul(a0, w1)
a2 = tf.matmul(x, w2*w3)

a1 = a1 + a2
a1 = tf.stop_gradient(a1)
loss = tf.reduce_mean(tf.square(a1 - a2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
gradients = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(gradients)
tf.summary.scalar('w0', tf.reduce_mean(w0))
tf.summary.scalar('w1', tf.reduce_mean(w1))
tf.summary.scalar('w2', tf.reduce_mean(w2))
tf.summary.scalar('w3', tf.reduce_mean(w3))

tf.summary.histogram('w0', w0)
tf.summary.histogram('w1', w1)
tf.summary.histogram('w2', w2)
tf.summary.histogram('w3', w3)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter('./experiments/', sess.graph)
    for i in range(10000):
        _, summary = sess.run([train_op, merged], feed_dict={x: np.random.rand(4, 5)})
        train_writer.add_summary(summary, i)


