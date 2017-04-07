import tensorflow as tf
import time
import input

import input
import numpy as np

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
LOG_DIR = "/Users/Yuta/Python/Hiragana/Log2"

Data = input.Input()


def variable_summaries(var, name=None):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + 'mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + 'stddev', stddev)
    tf.summary.scalar(name + 'max', tf.reduce_max(var))
    tf.summary.scalar(name + 'min', tf.reduce_min(var))
    tf.summary.histogram(name + 'histogram', var)


def inference(features, keep_prob=1):
    """

    :param features:  Tensor
        imput featues. shape(batch_size, IMAGE_HIGHT * IMAGE * WIDTH)
    :return: logits
    """
    x = tf.reshape(features, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="input_image")

    # define some functions
    def weight_variable(name, shape, stddev=0.2):
        """
        return weight which has shape=shape
        :param shape: list
        :return: tf.Variable which is initialized
        """
        var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        return var

    def bias_variable(name, shape, const=0.1):
        """
        return bias which has shape
        :param shape: list
        :return: tf.Variable which is initialized
        """
        var = tf.get_variable(name, shape, initializer=tf.constant_initializer(const))
        return var

    def conv2d(x, W):
        """
        apply filters W to x
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """
        do pooling
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("conv1"):
        weight_1 = weight_variable("weight", shape=[5, 5, 1, 32])
        bias1 = bias_variable("bias", shape=[32])
        h1 = tf.nn.relu(conv2d(x, weight_1) + bias1)

        h1_pooled = max_pool_2x2(h1)

    with tf.variable_scope("conv2"):
        weight_2 = weight_variable("weight", shape=[3, 3, 32, 64])
        bias2 = bias_variable("bias", shape=[64])
        h2 = tf.nn.relu(conv2d(h1_pooled, weight_2) + bias2)

        h2_pooled = max_pool_2x2(h2)
        # h2_pooled has shape of (batch. 16, 16, 64)
        h2_flattend = tf.reshape(h2_pooled, [-1, 16 * 16 * 64])

    with tf.variable_scope("fc1"):
        weight_fc1 = weight_variable("weight", shape=[16 * 16 * 64, 1024])
        bias_fc1 = bias_variable("bias", shape=[1024])

        h_fc1 = tf.nn.relu(tf.matmul(h2_flattend, weight_fc1) + bias_fc1)

        # drop_out
        h_fc1_dropped = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    with tf.variable_scope("fc2"):
        weight_fc2 = weight_variable("weight", shape=[1024, 75])
        bias2 = bias_variable("bias", shape=[75])

        logits = tf.matmul(h_fc1, weight_fc2) + bias2

    return logits


def cross_entropy(labels, logits):
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar("cross_entropy", cross_entropy)

    return cross_entropy


def get_train_op(loss, global_step):
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)
    with tf.name_scope("Adam"):
        optimizer = tf.train.AdamOptimizer(1e-4)
        gradients = optimizer.compute_gradients(loss)
        # add summay to
        for grad, var in gradients:
                   # tf.summary.histogram(var.name, grad)
            variable_summaries(grad, name=var.name)

        train_step = optimizer.apply_gradients(gradients, global_step=global_step)
    return train_step


def get_accuracy(labels, logits):
    with tf.name_scope("accuracy"):
        is_correct = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        # add accuracy to summary
        tf.summary.scalar("accuracy", accuracy)
        return accuracy


def train(step=100, batch_size=100):
    """ train model"""

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])

        with tf.name_scope('label'):
            y_ = tf.placeholder(tf.float32, [None, 75])

        k = tf.placeholder(tf.float32, name="keep_prob")
        global_step = tf.Variable(0, trainable=False, name='global_step')

        logits = inference(x, k)
        var_list = tf.global_variables()

        loss = cross_entropy(y_, logits)

        train_op = get_train_op(loss, global_step=global_step)
        acc = get_accuracy(y_, logits)

        # merge summary
        merged = tf.summary.merge_all()
        # init saver
        saver = tf.train.Saver(var_list=var_list, max_to_keep=15)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            read_model(sess, saver=saver)

            test_writer = tf.summary.FileWriter(LOG_DIR + "/test", sess.graph)
            train_writer = tf.summary.FileWriter(LOG_DIR + "/train")
            start_time = time.time()

            for step in range(step):
                x_train, y_train = Data.train_batch(batch_size)
                # g step is global step
                g_step = global_step.eval(sess)

                if step % 50 == 0:
                    # record test data
                    x_test, y_test = Data.test_batch(1000)
                    summary, accuracy = sess.run([merged, acc], feed_dict={x: x_test, y_: y_test, k: 1})
                    test_writer.add_summary(summary, global_step=g_step)
                    time_passed = time.time() - start_time
                    print("step: %5d, accuracy: %4f, time passed: %4.2f" % (step, accuracy, time_passed))

                    summary, _ = sess.run([merged, train_op], feed_dict={x: x_train, y_: y_train, k: 0.5})
                    train_writer.add_summary(summary, global_step=g_step)
                else:
                    sess.run(train_op, feed_dict={x: x_train, y_: y_train, k: 0.5})

                if step % 300 == 299:
                    save_model(sess, saver=saver, global_step=g_step)


def read_model(session, saver):
    path_of_model = tf.train.latest_checkpoint(LOG_DIR + "/model")
    if path_of_model is None:
        print("Model not found. Failed to restore")
    else:
        saver.restore(session, path_of_model)
        print("Model restored from %s." % path_of_model)


def save_model(session, saver, global_step):
    save_path = saver.save(session, LOG_DIR + "/model/model.ckpt", global_step=global_step)
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    train(step=3000)
