from prepare_test_dataset.RSVPVideoGenerator import RSVPVideoGenerator
import os

# define the parameters for the video generation
speed = 1
image_dir = './prepare_test_dataset/stimuli/'
save_dir = './prepare_test_dataset/videos/'

# run the code to create and save videos
videos = RSVPVideoGenerator(speed, image_dir, save_dir, file_type='tif', patch_size=20, nmasks=10, all_target=False)
videos.video_design(fps=25)

# extract frames from the generated videos
for filename in os.listdir(save_dir):
    if filename.endswith('.mp4'):
        path = save_dir + '/' + filename

        # extract rgb frames
        command = f"python3 extract_frames.py --filename={filename} --mode='rgb' --root={path} --save_dir='./video_frames'"
        os.system(command)

        # extract flow frames
        command = f"python3 extract_frames.py --filename={filename} --mode='flow' --root={path} --save_dir='./video_frames'"
        os.system(command)