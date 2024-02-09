# deep model challenges

This repository uses specific videos to evaluate the performance of state-of-the-art action recognition models (two-stream I3D convnet and R(2+1)D-18 RGB convnet) against visual challenges (as an example rapid display of targets!)

The I3D model used here (and its checkpoints) is inspired by [this repository](https://github.com/piergiaj/pytorch-i3d) with some modifications. It is pretrained on Kinetics and Charades datasets. The R(2+1)D-18 model is pretrained on Kinetics dataset.

[extract_frames.py](extract_frames.py) will extract rgb and optical flow (using TV-L1 algorithm) frames.
