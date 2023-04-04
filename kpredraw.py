# -*- coding: utf-8 -*-
import os
import re

import numpy as np
import cv2
import logging
import multiprocessing as mp
from tqdm import tqdm

"""kpredraw.py
Redraw the joint keypoints.  Fixed low confidence/out of image joint keypoint by interpolation between frames.
"""

# list of json file paths
keypoint_dir = "../action/keypoint"
keypoints = sorted(
    [
        os.path.join(keypoint_dir, fname)
        for fname in os.listdir(keypoint_dir)
        if fname.endswith(".npy")
    ]
)

# load the background  image
image = cv2.imread('black.jpg')
height, width, _ = image.shape

# joint keypoint confidence thereshold
c_threshold = 0.1


# get valid point from adjacent frames
def fix_lost_keypoint(keypoints):
    # select frame with average confidence > 0.1, discard low confidence frames
    valid_frame = np.mean(keypoints, axis=1)[:, 2] > c_threshold
    new_keypoints = keypoints[valid_frame]
    l = len(new_keypoints)
    # for each point check all the frame
    for p in range(0, 18):
        logging.info("processsing point {}".format(p))
        arr_p = np.array([new_keypoints[f][p] for f in range(0, l)])
        # valid point: 0<=x<=width, 0<=y<=height, 0<confidence<1.0
        valid = np.logical_and.reduce(
            [arr_p.T[0] <= width, arr_p.T[1] <= height, arr_p.T[2] > c_threshold, arr_p.T[2] < 1.0])
        # get invalid frame index
        fail_f = np.where(valid == False)[0]
        if len(fail_f) == 0:
            logging.info("all the frames are qualified")
            continue
        elif len(fail_f) == l:
            # all the frames are unqualified
            logging.error("all the frames are unqualified")
            continue
        else:
            logging.info("all the invalid frames:{}".format(fail_f))
            for f_idx in fail_f:
                logging.info("invalid: frame {} point {}".format(f_idx, p))
                # previous valid frame
                f_prev = f_idx - 1
                while f_prev >= 0:
                    if valid[f_prev]:
                        break
                    else:
                        f_prev = f_prev - 1

                # next valid frame
                f_next = f_idx + 1

                while f_next < l:
                    if valid[f_next]:
                        break
                    else:
                        f_next = f_next + 1

                logging.info(f_idx, "old point:", arr_p[f_idx])

                if f_prev < 0:
                    arr_p[f_idx] = arr_p[f_next]
                elif f_next >= l:
                    arr_p[f_idx] = arr_p[f_prev]
                else:
                    logging.info("prev frame {} and next frame {}:".format(f_prev, f_next), valid[f_prev],
                                 valid[f_next])

                    # interpolation
                    inter = f_next - f_prev + 1
                    arr_p[f_idx][0] = (f_idx - f_prev) / inter * arr_p[f_prev][0] + (f_next - f_idx) / inter * \
                                      arr_p[f_next][0]
                    arr_p[f_idx][1] = (f_idx - f_prev) / inter * arr_p[f_prev][1] + (f_next - f_idx) / inter * \
                                      arr_p[f_next][1]
                    arr_p[f_idx][2] = c_threshold + 0.01

                logging.info(f_idx, "new point: [{:.2f},{:.2f}]".format(arr_p[f_idx][0], arr_p[f_idx][1]))

                new_keypoints[f_idx][p] = arr_p[f_idx]

    return new_keypoints


# pose pair and color defined
pose_pairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

pose_colors = [(0, 100, 255), (0, 100, 255), (0, 255, 255), (0, 100, 255), (0, 255, 255), (0, 100, 255),
               (0, 255, 0), (255, 200, 100), (255, 0, 255), (0, 255, 0), (255, 200, 100), (255, 0, 255),
               (0, 0, 255), (255, 0, 0), (200, 200, 0), (255, 0, 0), (200, 200, 0)]


# redraw the joint keypoints
def redraw(pose_keypoints_2d, fixed=True):
    # Loop over each frame and draw the keypoints for each frame
    for idx, frame in enumerate(pose_keypoints_2d):

        # Reshape the keypoints array to a Nx3 array
        keypoints = frame.reshape((-1, 3))

        for i, pair in enumerate(pose_pairs):
            x1, y1, c1 = keypoints[pair[0]]
            x2, y2, c2 = keypoints[pair[1]]
            if c1 > c_threshold and c2 > c_threshold:
                try:
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), pose_colors[i], 3)
                except Exception as e:
                    logging.error("bad point frame {}: point{}:[{:.2f},{:.2f}],point {}:[{:.2f},{:.2f}]".format(
                        idx, pair[0], x1, y1, pair[1], x2, y2))
        # Display the resulting image
        title = "Fixed Pose Keypoints" if fixed else "Original Pose Keypoints"
        cv2.imshow(title, image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # cv2.waitKey(0)
        # Clear the image before drawing the next frame
        image[:] = 0

    cv2.destroyAllWindows()


def run_multiprocess(original, fixed):
    p1 = mp.Process(target=redraw, args=(fixed,))
    p2 = mp.Process(target=redraw, args=(original, False,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == '__main__':

    for keypoint_path in tqdm(keypoints):
        # load the pose keypoints from the numpy array
        pose_keypoints_2d = np.load(keypoint_path)
        new_keypoints = fix_lost_keypoint(pose_keypoints_2d)
        run_multiprocess(pose_keypoints_2d, new_keypoints)
        np.save(re.sub("keypoint", "new_keypoint", keypoint_path),new_keypoints)
