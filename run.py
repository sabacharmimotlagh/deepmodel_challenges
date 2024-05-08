from RSVPGenerator import RSVPGenerator
import numpy as np
import os

## define the parameters for the video generation
speed = 1
image_dir = './data/stimuli/'
save_dir = './data/videos/'

# create and save videos from stimuli
videos = RSVPGenerator(speed, image_dir, save_dir, file_type='tif', patch_size=20, nmasks=10, all_target=False)
videos.video_design(fps=25)


## extract frames from the generated videos
for filename in os.listdir(save_dir):
    if filename.endswith('.mp4'):
        path = save_dir + '/' + filename

        # extract rgb frames
        command = f"python3 extract_frames.py --filename={filename} --mode='rgb' --root={path} --save_dir='./data/video_frames'"
        os.system(command)

        # extract flow frames
        command = f"python3 extract_frames.py --filename={filename} --mode='flow' --root={path} --save_dir='./data/video_frames'"
        os.system(command)


## extract activations from the extracted frames
activation_layers = ['Conv3d_1a_7x7', 'Mixed_4c', 'Mixed_5c']

for layer in activation_layers:
    # extract rgb activations
    command = f"python3 extract_activations.py --activation_layer={layer} --mode='rgb' --root='./data/video_frames' --save_dir='./data/layer_activations' "
    os.system(command)

    # extract flow activations
    command = f"python3 extract_activations.py --activation_layer={layer} --mode='flow' --root='./data/video_frames' --save_dir='./data/layer_activations'"
    os.system(command)


## create and save RDMs from extracted activations using the Euclidean distance
layers = ['Conv3d_1a_7x7', 'Mixed_4c', 'Mixed_5c']

# initialize RDMs
RDM_rgb = np.zeros((3, 40, 40))      # layer x stimulus x stimulus
RDM_flow = np.zeros((3, 40, 40))

PATH = './data/layer_activations/'

# create RDMs using the Euclidean distance
for layer in layers:

    # rgb_activation
    name = 'rgb_activation_' + layer + '.npy'
    rgb_activation = np.load(PATH + name)

    # flow_activation
    name = 'flow_activation_' + layer + '.npy'
    flow_activation = np.load(PATH + name)

    for i in range(40):
        for j in range(40):
            RDM_rgb[layers.index(layer), i, j] = np.linalg.norm(rgb_activation[i, :, :, :, :].squeeze() - rgb_activation[j, :, :, :, :].squeeze())
            RDM_flow[layers.index(layer), i, j] = np.linalg.norm(flow_activation[i, :, :, :, :].squeeze() - flow_activation[j, :, :, :, :].squeeze())
# save RDMs
PATH = './data/rdms'

np.save(os.path.join(PATH, 'RDM_rgb'), RDM_rgb)
np.save(os.path.join(PATH, 'RDM_flow'), RDM_flow)