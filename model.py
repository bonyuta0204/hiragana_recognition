import tensorflow as tf
import input
import numpy as np

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

def inference(features):
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

    with tf.variable_scope("conv1") :
        weight_1 = weight_variable("weight", shape=[5, 5, 1, 32])
        bias1 = bias_variable("bias", shape=[32])
        h1 = tf.nn.relu(conv2d(x, weight_1) + bias1)

        h1_pooled = max_pool_2x2(h1)

    with tf.variable_scope("conv2"):
        weight_2 = weight_variable("weight", shape=[3, 3, 32, 64])
        bias2 = bias_variable("bias", shape=[64])
        h2 = tf.nn.relu(conv2d(h1_pooled, weight_2) + bias2)

        h2_pooled = max_pool_2x2(h2)

