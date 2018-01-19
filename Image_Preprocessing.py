import argparse
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf

def get_cwd():
    if os.getcwd() != '/home/fish/图片/shells':
        os.chdir('/home/fish/图片/shells')
    return os.getcwd()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_and_decode(filename):
    cwd = os.getcwd()
    if cwd != '/home/fish/图片/three_tfrecords':
        os.chdir('/home/fish/图片/three_tfrecords')

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename], num_epochs=100)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([15], tf.string),
            'img_raw': tf.FixedLenFeature([256, 256, 3], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.float32)
    img = tf.reshape(img, [256, 256, 3])
    label = tf.decode_raw(label, tf.float32)
    label = tf.reshape(label, [15])
    label = tf.cast(label, tf.int32)
    images, sparse_labels = tf.train.shuffle_batch(
        [img, label], batch_size=128, num_threads=2,
        capacity=1000 + 3 * 24,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
    return images, sparse_labels

def main():
    images, labels = read_and_decode(filename='train.tfrecords')
    print(images)
    print(labels)

if __name__ == '__main__':
    main()