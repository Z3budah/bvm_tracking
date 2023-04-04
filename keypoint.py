# -*- coding: utf-8 -*-
"""keypoint.py
 Using openpose to get keypoints from videos.
"""
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

import re
import logging
import json

from tqdm import tqdm

dir_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# current dir_path
# os.path.dirname(os.path.realpath(__file__))

# import openpose
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/openpose/build/python/openpose/Release');
        os.environ['PATH'] = os.environ[
                                 'PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' + dir_path + '/openpose/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    logging.error(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class Keypoint:
    def __init__(self, json_dir="../action/annotation", video_dir="../action/video",
                 keypoint_dir="../action/keypoint"):
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        # specified model
        params["model_folder"] = "../openpose/models/"
        params["model_pose"] = "COCO"
        params["net_resolution"] = "256x256"
        params["number_people_max"] = 1

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        self.json_dir = json_dir
        self.video_dir = video_dir
        self.keypoint_dir = keypoint_dir

    def get_annotation(self):
        json_files = sorted(
            [
                os.path.join(self.json_dir, fname)
                for fname in os.listdir(self.json_dir)
                if fname.endswith(".json")
            ]
        )
        return json_files

    def get_keypoint(self, json_path):

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.error("Failed to load annotation file:{}".format(e))

        # extract relevant information
        video_id = list(data.keys())[0]
        video_path = os.path.join(self.video_dir, video_id + ".mp4")
        print(video_path)

        cap = cv2.VideoCapture(video_path)

        for idx, annotation in enumerate(data.get(video_id, [])):

            start_frame = annotation["segment"][0]
            end_frame = annotation["segment"][1]
            # get the "label"/"keypoint" field if it exists, otherwise set to None
            label = annotation.get("label", None)
            duration_frame = annotation.get("duration_frame", None)
            keypoint = annotation.get("keypoint", None)
            # if this segment has been detected, skip the remaining code inside this loop
            if keypoint and os.path.exists(keypoint):
                continue


            # keypoint array
            keypoints = np.empty([1, 18, 3])

            # capture the video segment using cv2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Process Image
                datum = op.Datum()
                datum.cvInputData = frame
                self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                try:
                    # Display Imageb
                    # print("Body keypoints: \n" + str(datum.poseKeypoints) + str(datum.poseKeypoints.shape))
                    keypoints = np.append(keypoints, datum.poseKeypoints, axis=0)
                except Exception as e:
                    logging.info("Failed to get keypoints:{}".format(e))

                cv2.imshow(label, datum.cvOutputData)
                # cv2.imshow('frame', gray)
                if cv2.waitKey(1) == ord('q'):
                    break

            try:
                detect_rate = keypoints.shape[0] / duration_frame

                if detect_rate > 0.90:
                    # save keypoint file
                    print(keypoints.shape)
                    npy_path = os.path.join(self.keypoint_dir, video_id + "_" + str(idx) + ".npy")
                    print("Keypoint path:" + npy_path)
                    np.save(npy_path, keypoints)
                    annotation['keypoint'] = npy_path
                else:
                    logging.error(str(detect_rate) + "means too many frames can not detect keypoint, discard this "
                                                     "video segment")

            except Exception as e:
                logging.error("Can not save keypoints:{}".format(e))

        cap.release()
        cv2.destroyAllWindows()

        # Save the updated JSON file
        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error("Failed to load annotation file:{}".format(e))




kp = Keypoint()
annotations = kp.get_annotation();
for ann in tqdm(annotations):
    kp.get_keypoint(ann)
