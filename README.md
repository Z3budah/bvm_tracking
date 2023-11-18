# Motion Tracking of baby video
## video_scrape: 
bilibili.py\youtube.py: scrape video from bilibili\Youtube.
## keypoint:
keypoint_COCO.py: Using pyopenpose to get keypoints from videos.
<br>kpredraw.py: Using Akima interpolation fixed invalid joint keypoint. Redraw the joint keypoints.
<br>keypoint_body25.py: Same as above, but save keypoint and fixed keypoint video. 

## feature cluster
**tot_cluster.ipynb**: A temporal segmentation clustering module. Not yet completed.
<br>The reference paper is as follows
<br>@article{Kumar2021UnsupervisedAS, title={Unsupervised Action Segmentation by Joint Representation Learning and Online Clustering}, author={Sateesh Kumar and Sanjay Haresh and Awais Ahmed and Andrey Konin and M. Zeeshan Zia and Quoc-Huy Tran}, journal={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2021}, pages={20142-20153} }
<br>
<br>**pred_cluster.ipynb**: try to use Fixed-weight(FW) encoder from PREDICT&CLUSTER to cluster infant movement features.
<br>The reference paper is as follows
<br>@inproceedings{su2020predict,
  title={Predict \& cluster: Unsupervised skeleton based action recognition},
  author={Su, Kun and Liu, Xiulong and Shlizerman, Eli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9631--9640},
  year={2020}
}