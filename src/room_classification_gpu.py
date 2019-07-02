# Script which continuously classifies scene type from video/camera feed
# Uses GoogLeNet trained on Places365 dataset, available at: https://github.com/CSAILVision/places365

import sys
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppressing caffe printouts of network initialisation
import caffe
import numpy as np
import cv2


if __name__ == "__main__":
    caffe.set_mode_gpu()

    # load the class label
    file_name = 'categories_places365.txt'
    classes = list()

    # Get class names
    with open(file_name) as class_file:
        [classes.append(line.strip().split(' ')[0][3:]) for line in class_file]

    prototxt= 'deploy_googlenet_places365.prototxt'
    caffemodel= 'googlenet_places365.caffemodel'

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # Create transformer to process input image
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([0.485, 0.456, 0.406]))
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_channel_swap('data', (2, 1, 0))
    #transformer.set_raw_scale('data', 255.0)

    input_stream = sys.argv[1]
    if input_stream == 'cam':
        input_stream = 0
    else:
        assert os.path.isfile(input_stream), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    window_frame = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        for i in range(0, 5):
            ret, frame = cap.read()

        transformed_image = transformer.preprocess('data', frame)

        # Feed transformed frame through network
        out = net.forward_all(data=np.asarray([transformed_image]))

        # zip results and classes, end up with sortable list of tuples
        results = list(zip(classes, out['prob'][0]))

        # sort results and get top 5 based on probability (i.e. i[1]
        sorted_results = sorted(results, key=lambda i: i[1], reverse=True)[:5]

        for result in sorted_results:
            message = '{:.3f} -> {}'.format(result[1], result[0])
            print(message)
            #cv2.putText(frame, message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, fontScale=3, color=(0, 255, 0))
        message = message = '{:.3f} -> {}'.format(sorted_results[0][1], sorted_results[0][0])
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
