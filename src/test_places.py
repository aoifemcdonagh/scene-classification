# Script for running inference using models trained on Places dataset

import sys
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppressing caffe printouts of network initialisation
import caffe
import numpy as np
import matplotlib.pyplot as plt
import cv2

caffe.set_mode_gpu()

prototxt= 'deploy_googlenet_places365.prototxt'
caffemodel= 'googlenet_places365.caffemodel'
image = caffe.io.load_image(sys.argv[1])  # Load image (range [0, 1])
#image = cv2.imread(sys.argv[1])

# load the class label
file_name = 'categories_places365.txt'
classes = list()

# Get class names
with open(file_name) as class_file:
    [classes.append(line.strip().split(' ')[0][3:]) for line in class_file]

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

# Create transformer to process input image
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.array([0.485, 0.456, 0.406]))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
transformed_image = transformer.preprocess('data', image)

# Feed transformed image through network
out = net.forward_all(data=np.asarray([transformed_image]))

# zip results and classes, end up with sortable list of tuples
results = list(zip(classes, out['prob'][0]))

# sort results and get top 5 based on probability (i.e. i[1]
sorted_results = sorted(results, key=lambda i: i[1], reverse=True)[:5]

for result in sorted_results:
    print('{:.3f} -> {}'.format(result[1], result[0]))