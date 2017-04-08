# coding: utf-8


import re
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import *

IMAGE_HEIGHT = 127
IMAGE_WIDTH = 128


def file_name_to_label(filename):
    # to get hex value of character code
    hex_label = re.findall("24\w{2}\.png", filename)
    char = hex_label[0][2:4]
    label = int(char, 16)
    # to make label start from 0
    return label - 34


def binary_to_tensor(binary):
    image = tf.image.decode_png(binary, channels=1)
    feature = tf.reshape(image, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    return feature


def file_name_to_binary(filename):
    with open(filename, "rb") as f:
        binary = f.read()
    return binary


def make_dataset(directory="data_hiragana/", shape=None):
    """
    make data. data format is [data_size, (image + label)].
    Resize data when shape is not None.
    
    Parrameter:
        directory: string
            directory where png files are located
        shape: list [height, width] or None
            resize the image to (height, width) when shape is given
    Return:
        tf.tensor: tensor. the value of the data is the value
    """
    files = os.listdir(directory)
    labels = [file_name_to_label(x) for x in files]
    # convert label to tensor
    labels = tf.constant(labels, dtype=tf.float32)
    labels = tf.reshape(labels, [-1, 1])

    # load png data as tensor
    binaries = [file_name_to_binary(directory + x) for x in files]
    features = [binary_to_tensor(x) for x in binaries]
    # concat features 
    features = tf.concat(features, axis=0)
    # resize the vector when it is not none.
    if shape is not None:
        features = tf.image.resize_images(features, size=shape)

    if shape is None:
        shape = [IMAGE_HEIGHT, IMAGE_WIDTH]

    features = tf.cast(features, tf.float32)
    # flatted image
    features = tf.reshape(features, [-1, shape[0] * shape[1]])

    # create data by concating features and labels

    data = tf.concat([features, labels], axis=1)

    return data


def make_csv(data, sprit=None):
    """
    make csv from tensor.
    
    parameter:
        data: tf.Tensor 
            tensor containing data
    sprit: float
        split the train and test data. sprit is ratio for test data.
        
    return: None
    """
    with tf.Session() as sess:
        # features = make_dataset()

        data = data.eval()
        print(data.shape)
    # read data as DataFrame
    Dataframe = pd.DataFrame(data)

    # change name of column for labels
    names = Dataframe.columns.tolist()
    names[-1] = "label"
    Dataframe.columns = names

    if sprit == None:
        # make csv
        Dataframe.to_csv("labeled_data.csv", index=False)
    else:
        # sprit the data
        train, test = train_test_split(Dataframe, test_size=sprit, random_state=0)
        # write to csv
        train.to_csv("train_data.csv", index=False)
        test.to_csv("test_data.csv", index=False)


if __name__ == "__main__":
    data = make_dataset(shape=(32, 32))
    print("finish loading data. now writing......")
    make_csv(data, sprit=0.2)
