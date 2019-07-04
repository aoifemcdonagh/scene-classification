# Scene Classification
This repo contains code written for a project completed as part of my master's thesis. The project aim was to classify a limited number of room types and 'sonify' an estimation of their reverberation. This was achieved by playing a test sound (either a click or a short voice recording) mixed with reverberation pre-set which best matched that room type according to an experienced audio engineer. 

The model used is a GoogLeNet model trained on the Places365 dataset. This model and other models trained on Places365 are available to download here: https://github.com/CSAILVision/places365

## Dependencies
Dependency | Install Guide/Notes
-----------|--------------
python 3.5+ |
CUDA | https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
CuDNN | https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
caffe 1.0.0 | https://github.com/adeelz92/Install-Caffe-on-Ubuntu-16.04-Python-3 Follow steps carefully since they depend on your CUDA, CuDNN and python versions
numpy 1.14.3+ |
OpenCV | Install using pip3, not during OpenVino install.
OpenVino 2019 R1.1| https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick Note untick OpenCV

## Audio files
The audio files are assumed to be in a directory labelled `wav_files/` in the top directory of this repository. In the `wav_files/` directory should be two directories, `clicks/` and `voice/`, containing the sample sounds mixed with reverberation presets. File names are assumed to be the room type label with a `.wav` extension. 


scene_classification
├──scene-classification
|   ├── ncs_classify.py
|   ├── gpu_classify.py
|   ├── test_places.py
|   └── demo_ncs.sh
├── models
|   ├── deploy_googlenet_places365.prototxt
|   ├── googlenet_places365.caffemodel
|   ├── googlenet_places365.xml
|   ├── googlenet_places365.bin
|   └── googlenet_places365.mapping
**├── wav_files**
**|   ├──voice**
**|   └──clicks**
├── categories_places365.txt
├── groups.csv
├── requirements.txt
└── README.md



## Usage

#### ncs_classify.py
Continuously classifies scene type from a video/camera feed. An Intel Neural Compute Stick is used to perform inference.
**Arguments**:
`-i` `--input` 'cam' or path to a video file
`-d` `--delay` number of seconds delay between inference passes and sound playback
`--csv` CSV file containing class groupings. Default is groups.txt
`-p` `--preset_file_dir` path to directory containing preset audio files. Default is `wav_files` directory
`-v` `--voice` Option to play reverb presets with voice. Default is to use clicks.
`-m` `--model` Path to an .xml file with a trained model. Default is `models/googlenet_places365.xml`

#### gpu_classify.py 
Continuously classifies scene type from a video/camera feed. A GPU is used to perform inference. Input is either a path to a video file or 'cam' to specify camera input.
Example execution: 
`python3 gpu_classify.py cam`

#### test_places.py
Runs inference on an input image. Prints top 5 results from inference.
Example execution: 
`python3 test_places.py sampleimage.jpg`

#### demo_ncs.sh
Shell script to run ncs_classify.py

