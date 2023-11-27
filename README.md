# I3D

This repository uses specific videos to evaluate the performance of two-stream I3D against visual challenges (as an example rapid display of images!)

The I3D model used here is inspired by [this repository](https://github.com/piergiaj/pytorch-i3d) with some minor changes.

[feature_extraction.py](feature_extraction.py) will extract rgb and optical flow (using TV-L1 algorithm) frames and then these frames will be fed to the I3D model to extract rgb and flow features.
