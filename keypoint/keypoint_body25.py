# -*- coding: utf-8 -*-
"""keypointCOCO.py
 Using body-25 openpose to get keypoints from videos.
"""
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pandas as pd

import re
import logging
import json

from scipy.interpolate import Akima1DInterpolator

from tqdm import tqdm


# get valid point from adjacent frames
def fix_lost_keypoint(keypoints, width, height, c_threshold=0.1):
    # select frame with average confidence > 0.1, discard low confidence frames

    valid_frame = np.mean(keypoints, axis=1)[:, 2] > c_threshold

    new_keypoints = keypoints[valid_frame].copy()
    l = len(new_keypoints)
    # for each point check all the frame
    # body_25


    for p in range(14):
        logging.info("processsing point {}".format(p))

        #
        arr_p = np.array([new_keypoints[f][p] for f in range(0, l)])

        # valid point: 0<=x<=width, 0<=y<=height, 0<confidence<1.0
        valid = np.logical_and.reduce(
            [arr_p.T[0] <= width, arr_p.T[1] <= height, arr_p.T[2] > c_threshold, arr_p.T[2] < 1.0])
        if not valid.any():
            logging.error("all points are invalid")
            continue

        # Akima interpolation
        # extend the range at the beginning and end of the valid array, at frame -1 and l : (x,y) take the nearest valid value
        t = np.array(range(0, l))
        t = t[valid]
        t = np.insert(t, 0, -1)
        t = np.append(t, l)

        ctrl_points = arr_p.T[:2].T[valid]
        ctrl_points = np.insert(ctrl_points, 0, ctrl_points[0], axis=0)
        ctrl_points = np.append(ctrl_points, [ctrl_points[-1]], axis=0)

        inter_func = Akima1DInterpolator(t, ctrl_points)

        # get invalid frame index
        fail_f = np.where(valid == False)[0]
        inter_points = inter_func(fail_f)

        nan_f = []
        for i, f_idx in enumerate(fail_f):
            arr_p[f_idx][0] = inter_points[i][0]
            arr_p[f_idx][1] = inter_points[i][1]
            arr_p[f_idx][2] = c_threshold + 0.01
            # logging.info(f_idx, "new point: [{:.2f},{:.2f}]".format(inter_points[i][0], inter_points[i][1]))
            if np.isnan(inter_points[i][0]) or np.isnan(inter_points[i][1]):
                nan_f.append(f_idx)
                logging.error(p, f_idx, "new point: [{:.2f},{:.2f}]".format(inter_points[i][0], inter_points[i][1]))
            else:
                valid[f_idx] = True
                new_keypoints[f_idx][p] = arr_p[f_idx]

    return new_keypoints, valid_frame


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

params = dict()
# specified openpose model
params["model_folder"] = "../openpose/models/"
params["model_pose"] = "BODY_25"
params["net_resolution"] = "256x256"
params["number_people_max"] = 1

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def openpose_get_keypoint(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    origin_output = video_path.replace('.mp4', '_original.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(origin_output, fourcc, fps, (width, height))

    keypoints = np.empty([1, 25, 3])
    error_frame = [0]
    for frame_idx in tqdm(range(total_frame)):
        # each frame
        ret, frame = cap.read()
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break

        # Process Image
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        try:
            keypoints = np.append(keypoints, datum.poseKeypoints, axis=0)
            # print("frame{},{}".format(frame_idx, keypoints[frame_idx][15]))
        except Exception as e:
            logging.error("Failed to get keypoints:{}".format(e))
            error_frame.append(frame_idx)

        output_video.write(datum.cvOutputData)
    output_video.release()

    cv2.destroyAllWindows()
    logging.info("origin keypoint drawed.")
    print(keypoints[0][15])
    try:
        npy_path = video_path.replace('.mp4', '_original.npy')
        np.save(npy_path, keypoints[1:])
        error_path = video_path.replace('.mp4', '_error.npy')
        np.save(error_path, error_frame)
    except Exception as e:
        logging.error("Can not save keypoints:{}".format(e))

    return npy_path, error_path


def draw_fixed_keypoints(video_path, keypoint_path, error_path):

    try:
        keypoints = np.load(keypoint_path)
        error_frame = np.load(error_path)
        print(error_frame)
    except Exception as e:
        logging.error("Failed to load annotation file:{}".format(e))


    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    new_keypoints, valid_frame = fix_lost_keypoint(keypoints, width, height)
    # print(keypoints[0][15], new_keypoints[0][15])
    logging.info("new keypoints fixed.")
    print(new_keypoints.shape, valid_frame.shape, total_frame)
    # pose pair and color defined
    pose_pairs = [[0, 1], [0, 15], [0, 16], [15, 17], [16, 18], [1, 2], [1, 5],
                  [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12],
                  [9, 10], [10, 11], [11, 22], [11, 24], [22, 23], [12, 13],
                  [13, 14], [14, 21], [14, 19], [19, 20]]

    pose_colors = [
        (255., 0., 85.), (255., 0., 0.), (255., 85., 0.), (255., 170., 0.),
        (255., 255., 0.), (170., 255., 0.), (85., 255., 0.), (0., 255., 0.),
        (255., 0., 0.), (0., 255., 85.), (0., 255., 170.), (0., 255., 255.),
        (0., 170., 255.), (0., 85., 255.), (0., 0., 255.), (255., 0., 170.),
        (170., 0., 255.), (255., 0., 255.), (85., 0., 255.), (0., 0., 255.),
        (0., 0., 255.), (0., 0., 255.), (0., 255., 255.), (0., 255., 255.), (0., 255., 255.)
    ]

    fixed_output = video_path.replace('.mp4', '_fixed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(fixed_output, fourcc, fps, (width, height))

    valid_idx = 0
    for f in tqdm(range(total_frame)):
        ret, frame = cap.read()

        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break
        # valid_frame
        if f in error_frame:
            cv2.putText(frame, "not keypoint detected in frame {}.".format(f), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        elif valid_frame[valid_idx]:
            kp = new_keypoints[valid_idx]
            kp = kp.reshape((-1, 3))
            for i, pair in enumerate(pose_pairs):
                x1, y1, c1 = kp[pair[0]]
                x2, y2, c2 = kp[pair[1]]
                try:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), pose_colors[i], 3)
                except Exception as e:
                    logging.error("bad point frame {}: point{}:[{:.2f},{:.2f}],point {}:[{:.2f},{:.2f}]".format(
                        valid_idx, pair[0], x1, y1, pair[1], x2, y2))

            valid_idx = valid_idx + 1
        else:
            valid_idx = valid_idx + 1
            cv2.putText(frame, "frame {} is not valid.".format(f), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        print(f,valid_idx)
        output_video.write(frame)

    output_video.release()
    cv2.destroyAllWindows()


video_dir = "../../action/video"

video_files = sorted(
    [
        os.path.join(video_dir, fname)
        for fname in os.listdir(video_dir)
        if fname.endswith(".mp4") or fname.endswith(".avi")
    ]
)
print(video_files)

for vp in video_files:
    # keypoint_path, error_path = openpose_get_keypoint(vp)
    # print("keypoint_path:", keypoint_path)
    keypoint_path = '../action/video/normal general movement-preterm_0_original.npy'
    error_path = '../action/video/normal general movement-preterm_0_error.npy'
    draw_fixed_keypoints(vp, keypoint_path, error_path)



