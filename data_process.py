
# coding: utf-8

# In[107]:

import struct
import re
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[101]:

IMAGE_HEIGHT = 127
IMAGE_WIDTH = 128


# In[65]:

def file_name_to_label(filename):
    # to get hex value of character code
    hex_label = re.findall("24\w{2}", filename)
    char = hex_label[0][2:]
    label = int(char, 16)
    # to make label start from 0
    return label -34


# In[112]:

def binary_to_tensor(binary):
    image = tf.image.decode_png(binary, channels=1)
    feature = tf.reshape(image, [1, IMAGE_HEIGHT,IMAGE_WIDTH, 1 ])
    return feature

def file_name_to_binary(filename):
    with open(filename, "rb") as f:
        binary = f.read()
    return binary


# In[121]:

def make_dataset(directory="data_hiragana/"):
    files = os.listdir(directory)
    labels = [file_name_to_label(x) for x in files]
    # convert label to tensor
    labels = tf.constant(labels, dtype=tf.uint8)
    labels = tf.reshape(labels, [-1, 1])
    
    # load png data as tensor
    binaries = [file_name_to_binary(directory + x) for x in files]
    features = [binary_to_tensor(x) for x in binaries]
    
    # concat features 
    features = tf.concat(features, axis =0)
    features = tf.reshape(features, [-1, IMAGE_HEIGHT * IMAGE_WIDTH])
    
    # create data by concating features and labels
    
    data = tf.concat([features, labels], axis=1)
    

    return data


# In[148]:

def make_csv():
    with tf.Session() as sess:
        features = make_dataset()

        data = features.eval()
    
    # read data as DataFrame
    Dataframe = pd.DataFrame(data)
    
    # change name of column for labels
    names = Dataframe.columns.tolist()
    names[-1] = "label"
    Dataframe.columns = names
    
    # make csv
    Dataframe.to_csv("labeled_data.csv", index=False)


# In[ ]:

def sprit_data():

