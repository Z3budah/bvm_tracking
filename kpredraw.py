# -*- coding: utf-8 -*-
"""kpredraw.py
Redraw the joint keypoints.  Fixed low confidence/out of image joint keypoint by Akima interpolation.
"""
import os
import re
import json
import numpy as np
import cv2
import logging
import multiprocessing as mp
from tqdm import tqdm

from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import interp1d



# list of json file paths
json_dir="../action/annotation"
json_files = sorted(
    [
        os.path.join(json_dir, fname)
        for fname in os.listdir(json_dir)
        if fname.endswith(".json")
    ]
)

# joint keypoint confidence thereshold
c_threshold = 0.1


# get valid point from adjacent frames
def fix_lost_keypoint(keypoints, image):
    # select frame with average confidence > 0.1, discard low confidence frames
    valid_frame = np.mean(keypoints, axis=1)[:, 2] > c_threshold
    new_keypoints = keypoints[valid_frame]
    l = len(new_keypoints)
    height, width, _= image.shape
    # for each point check all the frame
    for p in range(0, 14):
        logging.info("processsing point {}".format(p))
        arr_p = np.array([new_keypoints[f][p] for f in range(0, l)])

        # valid point: 0<=x<=width, 0<=y<=height, 0<confidence<1.0
        valid = np.logical_and.reduce([arr_p.T[0] <= width, arr_p.T[1] <= height, arr_p.T[2] > c_threshold, arr_p.T[2] < 1.0])
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
        ctrl_points = np.insert(ctrl_points, 0, ctrl_points[0],axis=0)
        ctrl_points = np.append(ctrl_points, [ctrl_points[-1]], axis=0)

        inter_func = Akima1DInterpolator(t,ctrl_points)

        # get invalid frame index
        fail_f = np.where(valid == False)[0]
        inter_points = inter_func(fail_f)

        nan_f = []
        for i, f_idx in enumerate(fail_f):
            arr_p[f_idx][0] = inter_points[i][0]
            arr_p[f_idx][1] = inter_points[i][1]
            arr_p[f_idx][2] = c_threshold + 0.01
            #logging.info(f_idx, "new point: [{:.2f},{:.2f}]".format(inter_points[i][0], inter_points[i][1]))
            if np.isnan(inter_points[i][0]) or np.isnan(inter_points[i][1]):
                nan_f.append(f_idx)
                logging.error(p,f_idx, "new point: [{:.2f},{:.2f}]".format(inter_points[i][0], inter_points[i][1]))
            else:
                valid[f_idx] = True
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
def redraw(pose_keypoints_2d, image, title, fixed=True):
    title = (title + "Fixed Pose Keypoints" if fixed else "Original Pose Keypoints")
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
        cv2.putText(image, "frame:{}".format(idx), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 3)
        cv2.imshow(title, image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # cv2.waitKey(0)
        # Clear the image before drawing the next frame
        image[:] = 0

    cv2.destroyAllWindows()


def run_multiprocess(original, fixed, img, title):
    p1 = mp.Process(target=redraw, args=(fixed,img,title))
    p2 = mp.Process(target=redraw, args=(original,img,title, False,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == '__main__':
    # for keypoint_path in tqdm(keypoints):
    #     # load the pose keypoints from the numpy array
    #     pose_keypoints_2d = np.load(keypoint_path)
    #     new_keypoints = fix_lost_keypoint(pose_keypoints_2d)
    #     run_multiprocess(pose_keypoints_2d, new_keypoints)
    #     np.save(re.sub("keypoint", "new_keypoint", keypoint_path),new_keypoints)
    try:
        with open(json_files[0], "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("Failed to load annotation file:{}".format(e))

    # extract relevant information
    width = data["width"]
    height = data["height"]
    fps = data["fps"]
    # create the background  image
    img = np.zeros((height, width, 3), np.uint8)

    video_id = list(data.keys())[0]
    for annotation in data.get(video_id, []):
        if annotation["keypoint"] is None:
            continue
        if os.path.exists(re.sub("keypoint", "new_keypoint", annotation["keypoint"])):
            logging.error("Keypoints already fixed.")
            continue
        print(annotation["keypoint"])
        keypoints = np.load(annotation["keypoint"])
        new_keypoints = fix_lost_keypoint(keypoints,img)
        print(new_keypoints[:,0:14,0:2].shape)
        # redraw(new_keypoints, img, annotation["label"])
        np.save(re.sub("keypoint", "new_keypoint", annotation["keypoint"]), new_keypoints[:,0:14,0:2])
        # run_multiprocess(keypoints, new_keypoints, img, annotation["label"])



