# deep model challenges

This repository uses specific videos to evaluate the performance of state-of-the-art action recognition model (two-stream I3D convnet) against visual challenges (here rapid display of targets!)

The I3D model used here (and its checkpoints) is inspired by [this repository](https://github.com/piergiaj/pytorch-i3d) with some modifications. It is pretrained on Kinetics and Charades datasets. [model.py](model.py) contains the I3D architecture and the checkpoints containing the pretrained weights are in checkpoints folder. 

[RSVPGenerator.py](RSVPGenerator.py) will create an object which will generate RSVP videos by constructing masks from stimuli and putting stimuli and masks in a rapid presentation.

Here is a much slower example of the generated videos:

<img src='./assests/1.gif' width='500' height='500'>

<br />

[extract_frames.py](extract_frames.py) will extract rgb and optical flow (using TV-L1 algorithm) frames from the generated videos.

[extract_activations.py](extract_activations.py) will extract activations from the specified layers of the model.

