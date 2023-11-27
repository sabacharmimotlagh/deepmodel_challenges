import argparse
import cv2
import os
import numpy as np
import torch
from I3D import InceptionI3d

# function to extract rgb frames
def extract_rgb_frames(path):

    # Calculating frame numbers of the video
    video = cv2.VideoCapture(path)
    success = True
    NUM_OF_FRS = 0

    while(success):
        NUM_OF_FRS += 1
        success, frame = video.read()

    # Open the video file
    video = cv2.VideoCapture(path)
    rgb_frames = np.zeros((NUM_OF_FRS-1,224,224,3))

    success = True
    cframe = 0

    # read video frames and save them in rgb variable
    while(success):
        success, frame = video.read()
  
        if success:
            cframe += 1
            resized_frame = cv2.resize(frame, (224,224))
            rgb_frames[cframe-1,:,:,:] = resized_frame
            

    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

    return rgb_frames



# function to extract optical flow frames using TV-L1 algorithm
def extract_flow_frames(path):

    # Calculating frame numbers of the video
    video = cv2.VideoCapture(path)
    success = True
    NUM_OF_FRS = 0

    while(success):
        NUM_OF_FRS += 1
        success, frame = video.read()


    # Open the video file
    video = cv2.VideoCapture(path)

    # Read the first frame
    ret, prev_frame = video.read()
    resized_prev_frame = np.zeros((224,224,3))
    resized_prev_frame = cv2.resize(prev_frame, (224,224))

    prev_gray = cv2.cvtColor(resized_prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a zero-filled array to accumulate optical flow
    flow_frames = np.zeros((NUM_OF_FRS-1,224,224,2))

    ret = True
    cframe = 0
    while ret:
        ret, frame = video.read()

        if ret: 
            cframe += 1
            resized_frame = np.zeros((224,224,3))
            resized_frame = cv2.resize(frame, (224,224))
            
            # Convert the current frame to grayscale
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute TV-L1 Optical Flow
            flow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow_result = flow.calc(prev_gray, gray, None)
            
            # Accumulate optical flow
            flow_frames[cframe-1,:,:,:] = flow_result
            
            # Update previous frame
            prev_gray = gray
            
            # Visualization of the optical flow (optional)
            # magnitude, angle = cv2.cartToPolar(flow_result[..., 0], flow_result[..., 1])
            # flow_map = np.zeros_like(resized_frame)
            # flow_map[..., 0] = angle * 180 / np.pi / 2
            # flow_map[..., 1] = 255
            # flow_map[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # flow_map = cv2.cvtColor(flow_map, cv2.COLOR_HSV2BGR)
            
            # # Display the optical flow visualization
            # cv2.imshow(flow_map)


    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

    return flow_frames


def run(mode, root, load_model='', save_dir=''):

    # if you want rgb features
    if mode == 'rgb':

        # extracting RGB frames using the extract_rgb_frames function
        rgb_frames = extract_rgb_frames(root)

        # Convert RGB frames to PyTorch tensor
        rgb_tensor = torch.from_numpy(rgb_frames).permute(3, 0, 1, 2).float()

        # adding batch_size = 1 to the dimension as this is one video for evaluation
        rgb_tensor = torch.unsqueeze(rgb_tensor, 0)
        # The .permute(3, 0, 1, 2) rearranges dimensions assuming the input format is (frames, height, width, channels)
        # If your format is different, adjust the permutation accordingly

        # setup the model for rgb feature extraction
        i3d = InceptionI3d(400, in_channels=3)

        # load saved weights on the model
        i3d.load_state_dict(torch.load(load_model))
        
        # Set model to evaluate mode 
        i3d.eval()
        
        with torch.no_grad():
            # extract features and save them in save_dir directory
            rgb_features = i3d.extract_features(rgb_tensor)

        np.save(os.path.join(save_dir), rgb_features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


    # if you want flow features
    else:

        # extracting flow frames using the extract_flow_frames function
        flow_frames = extract_flow_frames(root)

        # Convert RGB frames to PyTorch tensor
        flow_tensor = torch.from_numpy(flow_frames).permute(3, 0, 1, 2).float()

        # adding batch_size = 1 to the dimension as this is one video for evaluation
        flow_tensor = torch.unsqueeze(flow_tensor, 0)
        # The .permute(3, 0, 1, 2) rearranges dimensions assuming the input format is (frames, height, width, channels)
        # If your format is different, adjust the permutation accordingly

        # setup the model for rgb feature extraction
        i3d = InceptionI3d(400, in_channels=2)

        # load saved weights on the model
        i3d.load_state_dict(torch.load(load_model))
        
        # Set model to evaluate mode 
        i3d.eval()

        with torch.no_grad():
            # extract features and save them
            flow_features = i3d.extract_features(flow_tensor)
        
        np.save(os.path.join(save_dir), flow_features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='rgb or flow')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()
    
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
