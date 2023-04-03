import json
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import re
import logging

# list of json file paths
json_dir = "../action/annotation"
json_files = sorted(
    [
        os.path.join(json_dir, fname)
        for fname in os.listdir(json_dir)
        if fname.endswith(".json")
    ]
)


for json_path in json_files:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("Failed to load annotation file:{}".format(e))

    # Loop through each video
    for video_id, segments in data.items():
        # Loop through each segment and calculate the duration in frames
        for segment in segments:
            start, end = segment['segment']
            duration = end - start + 1
            segment['duration_frame'] = duration

    # Save the updated JSON file
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error("Failed to load annotation file:{}".format(e))



