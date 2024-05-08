from RSVPGenerator import RSVPGenerator
import os

# # define the parameters for the video generation
# speed = 1
# image_dir = './data/stimuli/'
# save_dir = './data/videos/'

# # run the code to create and save videos
# videos = RSVPGenerator(speed, image_dir, save_dir, file_type='tif', patch_size=20, nmasks=10, all_target=False)
# videos.video_design(fps=25)

# # extract frames from the generated videos
# for filename in os.listdir(save_dir):
#     if filename.endswith('.mp4'):
#         path = save_dir + '/' + filename

#         # extract rgb frames
#         command = f"python3 extract_frames.py --filename={filename} --mode='rgb' --root={path} --save_dir='./data/video_frames'"
#         os.system(command)

#         # extract flow frames
#         command = f"python3 extract_frames.py --filename={filename} --mode='flow' --root={path} --save_dir='./data/video_frames'"
#         os.system(command)

activation_layers = ['Conv3d_1a_7x7', 'Mixed_4c', 'Mixed_5c']
for layer in activation_layers:
    # extract rgb activations
    command = f"python3 extract_activations.py --activation_layer={layer} --mode='rgb' --root='./data/video_frames' --save_dir='./data/layer_activations' "
    os.system(command)

    # extract flow activations
    command = f"python3 extract_activations.py --activation_layer={layer} --mode='flow' --root='./data/video_frames' --save_dir='./data/layer_activations'"
    os.system(command)