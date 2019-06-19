# Script for demoing room classification model using raspberry pi and Intel NCS

import sys
import os
import cv2
import time
import logging as log
import argparse
import csv
import subprocess
from openvino.inference_engine import IENetwork, IEPlugin

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True, type=str)
    parser.add_argument("-d", "--delay", help="number of seconds delay between inference/clicks", type=int, default=1)
    parser.add_argument("--csv", help="csv file containing class groupings", type=str, default="groups.csv")
    parser.add_argument("-p", "--preset_file_dir", help="path to dir containing preset audio files",
                        default="wav_files")
    parser.add_argument("-v", "--voice", help="Use this option to play reverb presets with voice. Default is to use clicks.", 
			action='store_true', default=False)
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", type=str,
                        default="googlenet_places365.xml")
    return parser


def get_group(class_name, file_path):
    """
    Function for returning a grouping which corresponds to a reverb preset
    :param class_name: Highest confidence class from model, string
    :param file_path: path to csv file containing class groups
    :return: Group name corresponding to preset
    """
    print("classname: " + class_name)
    preset = "other"  # This will be output if no matches
    with open(file_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        headers = next(read_csv, None)  # Get headers in csv file

        for row in read_csv:
            if class_name == row[0]:
                preset = row[1]

    return preset


if __name__ == "__main__":
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging

    args = build_argparser().parse_args()

    # Set the directory path for loading preset files
    preset_file_dir = ''
    if args.voice is True:  # Play voice presets if specified
        preset_file_dir = args.preset_file_dir + '/voice'
    else:  # if voice presets aren't chosen play clicks
        preset_file_dir = args.preset_file_dir + '/clicks'

    model_xml = args.model  # assuming models in same dir
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

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

    input_stream = args.input
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
    message = ''
    counter = time.time()
    previous_group = 'office'


    while cap.isOpened():
        ret, frame = cap.read()

        # Perform inference every 2 seconds
        if (time.time() - counter) > args.delay:  # Wait for 2s
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

            group = get_group(current_room_type, args.csv)  # Get preset grouping which matches room class

            if group == 'other':  #i.e. not a room class we have a preset for
                subprocess.Popen(["aplay", "{}/{}.wav".format(preset_file_dir, previous_group)])  # Play preset
            else:  # we have a preset for this class. Play corresponding preset file
                subprocess.Popen(["aplay", "{}/{}.wav".format(preset_file_dir, group)])  # Play preset
                previous_group = group
            counter = time.time()

        # Show every frame
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    del net
    del exec_net
    del plugin
