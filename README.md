# Motion Tracking of baby video
bilibili.py\youtube.py: scrape video from bilibili\Youtube.
keypoint.py: Using pyopenpose to get keypoints from videos.
kpredraw.py: Using Akima interpolation fixed invalid joint keypoint. Redraw the joint keypoints.
cluster.ipynb: A temporal segmentation clustering modeule. Not yet completed. The reference paper is as follows
@article{Kumar2021UnsupervisedAS,
  title={Unsupervised Action Segmentation by Joint Representation Learning and Online Clustering},
  author={Sateesh Kumar and Sanjay Haresh and Awais Ahmed and Andrey Konin and M. Zeeshan Zia and Quoc-Huy Tran},
  journal={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
  pages={20142-20153}
}
