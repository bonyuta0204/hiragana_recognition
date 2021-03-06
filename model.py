import tensorflow as tf
import time

import input
import numpy as np
import argparse

FLAGS = None

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
# LOG_DIR = "/Users/Yuta/Python/Hiragana/Log_bn_improved"
LOG_DIR = None

Data = input.Input()


def variable_summaries(var, name=""):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + 'mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + 'stddev', stddev)
    tf.summary.scalar(name + 'max', tf.reduce_max(var))
    tf.summary.scalar(name + 'min', tf.reduce_min(var))
    tf.summary.histogram(name + 'histogram', var)


def inference(features, is_training):
    """

    :param is_training:
    :param features:  Tensor
        imput featues. shape(batch_size, IMAGE_HIGHT * IMAGE * WIDTH)
    :return: logits
    """
    x = tf.reshape(features,
                   [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1],
                   name="input_image")

    k = tf.cond(is_training, lambda: tf.constant(0.5, tf.float32),
                lambda: tf.constant(1.0, tf.float32))
    tf.summary.scalar("keep_prob", k)

    # define some functions

    def weight_variable(name, shape, stddev=0.2):
        """
        return weight which has shape=shape
        :param shape: list
        :return: tf.Variable which is initialized
        """
        var = tf.get_variable(name,
                              shape,
                              initializer=
                              tf.truncated_normal_initializer(stddev=stddev))
        return var

    def bias_variable(name, shape, const=0.1):
        """
        return bias which has shape
        :param shape: list
        :return: tf.Variable which is initialized
        """
        var = tf.get_variable(name, shape,
                              initializer=tf.constant_initializer(const))
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
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME")

    # actually building model
    with tf.variable_scope("conv1"):
        weight_1 = weight_variable("weight", shape=[5, 5, 1, 32])
        bias1 = bias_variable("bias", shape=[32])
        h1 = conv2d(x, weight_1) + bias1

        h1 = tf.layers.batch_normalization(h1, training=is_training,
                                           scale=False,
                                           name="batch_normalization")
        h1 = tf.nn.relu(h1)
        h1_pooled = max_pool_2x2(h1)

    with tf.variable_scope("conv2"):
        weight_2 = weight_variable("weight", shape=[3, 3, 32, 64])
        bias2 = bias_variable("bias", shape=[64])

        h2 = conv2d(h1_pooled, weight_2) + bias2
        h2 = tf.layers.batch_normalization(h2, training=is_training,
                                           scale=False,
                                           name="batch_normalization")
        h2 = tf.nn.relu(h2)

        h2_pooled = max_pool_2x2(h2)
        # h2_pooled has shape of (batch. 16, 16, 64)
        h2_flattend = tf.reshape(h2_pooled,
                                 [-1, (int(IMAGE_HEIGHT / 4))
                                  * (int(IMAGE_WIDTH / 4)) * 64])
        h2_flattend = tf.nn.dropout(h2_flattend, keep_prob=k, name="dropout")

    with tf.variable_scope("fc1"):
        weight_fc1 = weight_variable("weight",
                                     shape=[(int(IMAGE_HEIGHT / 4))
                                            * (int(IMAGE_WIDTH / 4)) * 64,
                                            1024])

        bias_fc1 = bias_variable("bias", shape=[1024])

        h_fc1 = tf.matmul(h2_flattend, weight_fc1) + bias_fc1
        h_fc1_bn = tf.layers.batch_normalization(h_fc1, training=is_training,
                                                 scale=False,
                                                 name="batch_normalization")
        h_fc1_bn = tf.nn.relu(h_fc1_bn)

    with tf.name_scope("dropout"):
        # drop_out

        h_fc1_dropped = tf.nn.dropout(h_fc1_bn, keep_prob=k, name="dropout")

    with tf.variable_scope("fc2"):
        weight_fc2 = weight_variable("weight", shape=[1024, 75])
        bias2 = bias_variable("bias", shape=[75])

        logits = tf.matmul(h_fc1_dropped, weight_fc2) + bias2

    return logits


def cross_entropy(labels, logits):
    with tf.name_scope("softmax"):
        soft_max = tf.nn.softmax(logits, name="soft_max")
        variable_summaries(soft_max)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       (labels=labels, logits=logits))
    tf.summary.scalar("cross_entropy", cross_entropy)

    return cross_entropy


def get_train_op(loss, global_step):
    with tf.name_scope("Adam"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        lr = tf.cond(global_step < 5000,
                     lambda: tf.constant(0.001, dtype=tf.float32),
                     lambda: tf.constant(0.0001, dtype=tf.float32))

        tf.summary.scalar("learning rate", lr)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = optimizer.compute_gradients(loss)
            # add summay to
            for grad, var in gradients:
                # add summary for gradient
                variable_summaries(grad, name=var.name)
                # pass
            train_step = optimizer.apply_gradients(gradients,
                                                   global_step=global_step)
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

        # k = tf.placeholder(tf.float32, name="keep_prob")
        is_training = tf.placeholder(tf.bool, name="is_training")

        logits = inference(x, is_training)

        var_list = tf.global_variables()
        # add summary of variable
        with tf.name_scope("Summaries"):
            for var in var_list:
                variable_summaries(var, name=var.name)

        # define global_step
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # add global_step to var_list
        var_list.append(global_step)

        # define loss
        loss = cross_entropy(y_, logits)

        # define train op
        train_op = get_train_op(loss, global_step=global_step)
        acc = get_accuracy(y_, logits)

        # merge summary
        merged = tf.summary.merge_all()
        # init saver
        saver = tf.train.Saver(var_list=var_list, max_to_keep=15)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            read_model(sess, saver=saver, log=LOG_DIR)

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
                    summary, accuracy, c_entropy = sess.run([merged, acc, loss],
                                                            feed_dict={x:
                                                                           x_test,
                                                                       y_: y_test,
                                                                       is_training: False})
                    test_writer.add_summary(summary, global_step=g_step)
                    time_passed = time.time() - start_time

                    print(
                        "step: %5d, accuracy: %.4f, time passed: %7.2f,  Cross Entropy: %5.2f"
                        % (step, accuracy, time_passed, c_entropy))

                    summary, _ = sess.run([merged, train_op],
                                          feed_dict={x: x_train, y_: y_train,
                                                     is_training: True})
                    train_writer.add_summary(summary, global_step=g_step)
                else:
                    sess.run(train_op,
                             feed_dict={x: x_train, y_: y_train,
                                        is_training: True})

                if step % 300 == 299:
                    save_model(sess, saver=saver, global_step=g_step,
                               log=LOG_DIR)


def read_model(session, saver, log):
    print(log)
    path_of_model = tf.train.latest_checkpoint(log)
    if path_of_model is None:
        print("Model not found. Failed to restore")
    else:
        saver.restore(session, path_of_model)
        print("Model restored from %s." % path_of_model)


def save_model(session, saver, global_step, log):
    save_path = saver.save(session, log + "/model.ckpt",
                           global_step=global_step)
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("--dir", type=str, default="Log",
                        help="directory to save logs. default : Log")
    FLAGS = parser.parse_args()
    LOG_DIR = "/Users/Yuta/Python/Hiragana/Log/" + FLAGS.dir
    print(LOG_DIR)

    train(step=20000)
