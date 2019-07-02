# Scene Classification
This repo contains code written as part of a project completed for my master's thesis. The project aim was to classify a limited number of room types and 'sonify' an estimation of their reverberation. This was achieved by playing a test sound (either a click or a short voice recording) mixed with reverberation pre-set which best matched that room type according to an experienced audio engineer. 


## Dependencies
Dependency | Install Guide/Notes
-----------|--------------
python 3.5+ |
CUDA | https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
CuDNN | https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
caffe 1.0.0 | https://github.com/adeelz92/Install-Caffe-on-Ubuntu-16.04-Python-3 Follow steps carefully since they depend on your CUDA, CuDNN and python versions
numpy 1.14.3+ |
OpenCV | Install using pip3, not during OpenVino install.
OpenVino | https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick Note untick OpenCV

## Audio files
The audio files are assumed to be in a directory labelled `wav_files/` in the top directory of this repository. In the `wav_files/` directory should be two directories, `clicks/` and `voice/`, containing the sample sounds mixed with reverberation presets. File names are assumed to be the room type label with a `.wav` extension. 

```
scene_classification
|--src
	|-- demo_room_classification.py
	|-- room_classification_gpu.py
	|-- test_places.py
	|-- room_demo.sh
|-- models
	|-- deploy_googlenet_places365.prototxt
	|-- googlenet_places365.caffemodel
	|-- googlenet_places365.xml
	|-- googlenet_places365.bin
	|-- googlenet_places365.mapping
|-- wav_files
	|--voice
	|--clicks
|-- categories_places365.txt
|-- groups.csv
|-- README.md

```
