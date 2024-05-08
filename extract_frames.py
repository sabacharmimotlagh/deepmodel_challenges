import argparse
import cv2
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--root', type=str)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()


# extract rgb frames from videos
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



# extract optical flow frames using TV-L1 algorithm
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



def run(filename, mode, root, save_dir=''):

    name = filename.split('.')[0]

    # if you want rgb features
    if mode == 'rgb':

        # extracting RGB frames using the extract_rgb_frames function
        rgb_frames = extract_rgb_frames(root)
        # save rgb frames in save_dir directory
        np.save(os.path.join(save_dir, name+'_rgb'), rgb_frames)
        print('Successfully extracted rgb frames ...')


    # if you want flow features
    else:

        # extracting flow frames using the extract_flow_frames function
        flow_frames = extract_flow_frames(root)
        # save flow tensor in save_dir directory
        np.save(os.path.join(save_dir, name+'_flow'), flow_frames)
        print('Successfully extracted flow frames ...')


# run the code to extract frames
run(filename=args.filename, mode=args.mode, root=args.root, save_dir=args.save_dir)