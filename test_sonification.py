import sys
import os
import cv2
import time

current_room_type = ''
counter = time.time()

while True:
    if (time.time() - counter) > 1:  # Wait for 1s
        os.system('aplay ~/Downloads/RE__Calabasas_demo_spaces/HallClick.wav')
        counter = time.time()