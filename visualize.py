import tensorflow as tf
from model import *
import matplotlib.pyplot as plt

LOG = None

def get_filter1():
    with tf.Graph().as_default():
        xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        logits = inference(xs, is_training=tf.constant(False, tf.bool))
        var_list = tf.global_variables()
        saver = tf.train.Saver(var_list=var_list)
        with tf.Session() as sess:
            read_model(sess, saver, log=LOG)
            for var in var_list:
                print(var.name)
            filter1 = sess.run("conv1/weight:0")

            print(filter1.shape)
            return filter1


def visulize_filter(var):
    shape = var.shape
    # visualize each filter
    for filter in range(shape[-1]):
        image = var[:, :, 0, filter]
        if filter <= 36:
            plt.subplot(6, 6, filter + 1)
            plt.imshow(image, cmap="gray")
    plt.show()


def visualize_activation(tensor):
    # shape = {batch, IM_HEIGHT, IM_WIDTH, Channel]
    shape = tensor.shape

    # visualize each filter
    for filter in range(shape[-1]):
        image = tensor[0, :, :, filter]
        if filter <= 35:
            plt.subplot(6, 6, filter + 1)
            plt.imshow(image)
    plt.show()


def get_activation1():
    with tf.Graph().as_default():
        xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        logits = inference(xs, is_training=tf.constant(False, tf.bool))
        var_list = tf.global_variables()
        saver = tf.train.Saver(var_list=var_list)
        with tf.Session() as sess:
            read_model(sess, saver, log=LOG)

            activation1 = sess.run("conv2/Relu:0",
                                   feed_dict={xs: Data.test_batch(n=1)[0]})

            return activation1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize filter")
    parser.add_argument("--dir", type=str, default="Log",
                        help="directory to save logs. default : Log")
    FLAGS = parser.parse_args()
    LOG = "/Users/Yuta/Python/Hiragana/" + FLAGS.dir

    # visulize_filter(get_filter1())
    visualize_activation(get_activation1())
