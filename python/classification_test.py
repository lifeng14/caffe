#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'lifeng'

import numpy as np
import caffe
import sys
import os
import matplotlib.pyplot as plt

caffe_root = '../'
sys.path.insert(0, os.path.join(caffe_root, 'python'))

def vis_square(data):
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.show()

# Setting caffe net
caffe.set_device(0)
caffe.set_mode_gpu()
model_def = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
mean_file = os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
net = caffe.Net(model_def, model_weights, caffe.TEST)

# Setting transformer
mu = np.load(mean_file)
mu = mu.mean(1).mean(1)
transformer = caffe.io.Transformer({'data': net.blobs['data'].shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# Process image
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)

# Perform net forward
net.blobs['data'].data[...] = transformed_image
out = net.forward()
out_prob = out['prob'][0]
print 'predicted class is:', out_prob.argmax()

labels_file = os.path.join(caffe_root, 'data/ilsvrc12/synset_words.txt')
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'predicted label is:', labels[out_prob.argmax()]

# Visualize conv1
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0,2,3,1))


