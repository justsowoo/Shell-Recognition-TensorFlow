import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

def get_cwd():
    if os.getcwd() != '/home/fish/图片/shells':
        os.chdir('/home/fish/图片/shells')
    return os.getcwd()

def img_read():
    cwd = get_cwd()
    root_path =  '/home/fish/图片'
    dir = os.path.join(root_path, 'shells2')
    if not os.path.exists(dir):
        os.mkdir(dir)

    data_set = dict()
    is_first = True
    pic_num = 0
    species = 0
    for root,dirs,files in os.walk(cwd): 
        count = 0
        if dirs == []:
            a = open((root+'/a.txt'), 'r')
            num = int(a.read())
            file_images = np.empty((num, 256, 256, 3))
            #name = root.split('/')[-1]
            file_labels = [species for i in range(num)]
            species += 1
            pic_num += num

            for file in files:  
                img = cv2.imread(file)
                if img == None:
                    continue
                file_images[count] = img
                count += 1
                
            if (is_first):
                images = file_images
                labels = file_labels
                is_first = False
            else:
                images = np.concatenate((images, file_images))
                labels = labels + file_labels

            a.close()
            #if count%400==0:  
                #print count  
    data_set = {'images': images,
                'labels': labels,
                'pic_num': pic_num}
    return data_set

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    images = data_set['images']
    labels = data_set['labels']
    num_examples = data_set['pic_num']

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(get_cwd(), name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = cv2.imencode('.jpg', image[index])[1].tostring()
        #image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
     # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label

def main():
    data_set = img_read()
    convert_to(data_set, name = 'all_pic')

if __name__ == '__main__':
    main()