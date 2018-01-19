import os
import random

import cv2
import numpy as np 
import tensorflow as tf 

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def species_to_num(name):
    species = [0 for _ in range(15)]
    if name.split('_')[0] == '宝贝科':
        species[0] = 1
    elif name.split('_')[0] == '芋螺科':
        species[1] = 1
    elif name.split('_')[0] == '蛾螺科':
        species[2] = 1
    elif name.split('_')[0] == '榧螺科':
        species[3] = 1
    elif name.split('_')[0] == '凤螺科':
        species[4] = 1
    elif name.split('_')[0] == '蚶科':
        species[5] = 1
    elif name.split('_')[0] == '盔螺科':
        species[6] = 1
    elif name.split('_')[0] == '帘蛤科':
        species[7] = 1
    elif name.split('_')[0] == '马蹄螺科':
        species[8] = 1
    elif name.split('_')[0] == '鸟蛤科':
        species[9] = 1
    elif name.split('_')[0] == '细带螺科':
        species[10] = 1
    elif name.split('_')[0] == '玉螺科':
        species[11] = 1
    elif name.split('_')[0] == '贻贝科':
        species[12] = 1
    elif name.split('_')[0] == '砗磲科':
        species[13] = 1
    elif name.split('_')[0] == '扇贝科':
        species[14] = 1
    else:
        raise NameError('No species named %s' % str(file))
    species = bytes(str(species), encoding='utf-8')
    return species

def create_tfrecords():
    if os.getcwd() != '/home/fish/图片/all_in_one':
        os.chdir('/home/fish/图片/all_in_one')
    #dir = '/home/fish/图片/images_tfrecords'
    dir = '/home/fish/图片/three_tfrecords'
    if not os.path.exists(dir):
        os.mkdir(dir)
    #filename = os.path.join(dir, 'all_pic.tfrecords')
    #writer = tf.python_io.TFRecordWriter(filename)
    #count = 0
    dev_count = 0
    test_count = 0
    train_count = 0
    species = np.zeros((15))
    """
    for root, dirs, files in os.walk(os.getcwd()):
        if dirs == []:
            for file in files:
                img = cv2.imread(os.path.join(root, str(file)))
                if img.shape == None:
                    print('error')
                    continue
    """
    for root, dirs, files in os.walk(os.getcwd()):
        if dirs == []:
            num = len(files)
            #dev set
            filename = os.path.join(dir, 'dev.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(int(num*0.2)):
                index = random.randint(0, num-1)
                while (index > len(files)-1):
                    index = random.randint(0, num-1)
                img = cv2.imread(os.path.join(root, str(files[index])))
                image_raw = cv2.imencode('.jpg', img)[1].tostring()
                species = species_to_num(str(files[index]))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _bytes_feature(species),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
                files.remove(files[index])
                dev_count += 1
            writer.close()
            #test set
            filename = os.path.join(dir, 'test.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(int(num*0.2)):
                index = random.randint(0, int(num*0.8-1))
                while (index > len(files)-1):
                    index = random.randint(0, int(num*0.8-1))

                img = cv2.imread(os.path.join(root, str(files[index])))
                image_raw = cv2.imencode('.jpg', img)[1].tostring()
                species = species_to_num(str(files[index]))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _bytes_feature(species),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
                files.remove(files[index])
                test_count += 1
            writer.close()
            #train set
            filename = os.path.join(dir, 'train.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)
            for file in files:
            
                img = cv2.imread(os.path.join(root, str(file)))
                image_raw = cv2.imencode('.jpg', img)[1].tostring()
                species = species_to_num(str(file))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _bytes_feature(species),
                    'image_raw': _bytes_feature(image_raw)
                    }))
                writer.write(example.SerializeToString())
                train_count += 1
            writer.close()          
    """
                if img.shape != (256, 256, 3):
                    raise ValueError("Image size %d doesn't match 256,256,3 " % img.shape)
                species = species_to_num(str(file))
                image_raw = cv2.imencode('.jpg', img)[1].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(int(species)),
                    'image_raw': _bytes_feature(image_raw)
                    }))
                writer.write(example.SerializeToString())
                print('count==', count)
                count += 1
    writer.close()
    """
    print('dev = ', dev_count)
    print('test = ', test_count)
    print('train = ', train_count)

def main():
    create_tfrecords()

if __name__ == '__main__':
    main()
    