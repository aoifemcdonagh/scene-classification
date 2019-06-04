# Script for demoing room classification model using raspberry pi and Intel NCS

import sys
import os
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin

if __name__ == "__main__":
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging

    model_xml = "googlenet_places365.xml"  # assuming models in same dir
    model_bin = "googlenet_places365.bin"

    # Get class info
    # load the class label
    file_name = 'categories_places365.txt'
    classes = list()

    # Get class names
    with open(file_name) as class_file:
        [classes.append(line.strip().split(' ')[0][3:]) for line in class_file]

    # Plugin initialization for Movidius stick
    plugin = IEPlugin(device="MYRIAD")

    # Initialise network
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    n = 1
    c = 3
    h = 224
    w = 224

    net.reshape({input_blob: (n, c, h, w)})  # Reshape so that n=1

    # Load network to plugin once done initialisation
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net, num_requests=2)

    input_stream = sys.argv[1]
    if input_stream == 'cam':
        input_stream = 0
    else:
        assert os.path.isfile(input_stream), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)  # Start video capture from camera

    cur_request_id = 0
    next_request_id = 1

    render_time = 0
    ret, frame = cap.read()

    window_frame = cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    current_room_type = ''
    counter = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        in_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("frame", in_frame)
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.infer(inputs={input_blob: in_frame})

        # Parse detection results of the current request
        out = exec_net.requests[0].outputs

        # zip results and classes, end up with sortable list of tuples
        results = list(zip(classes, out['prob'][0]))

        # sort results and get top 5 based on probability (i.e. i[1]
        sorted_results = sorted(results, key=lambda i: i[1], reverse=True)[:5]

        message = '{:.3f} -> {}'.format(sorted_results[0][1], sorted_results[0][0])
        current_room_type = sorted_results[0][0]  # Room type with highest confidence

        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        if (time.time() - counter) > 1:  # Wait for 1s
            os.system('aplay wav_files/{}.wav'.format(current_room_type))  # assuming file in current directory
            counter = time.time()

    del net
    del exec_net
    del plugin
