import argparse
import numpy as np
import os
import torch
from model import InceptionI3d

parser = argparse.ArgumentParser()
parser.add_argument('--activation_layer', type=str)
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--root', type=str)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()


def get_activation(name):
    activation = {}
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def extract_rgb_activation(path, save_dir, activation_layer):

    # load the video frames
    enteries = os.listdir(path)

    # setup the model for rgb prediction
    i3d = InceptionI3d(400, in_channels=3, activation_layer=activation_layer)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load('./checkpoints/rgb_charades.pt'))

    i3d._modules.get(activation_layer).register_forward_hook(get_activation(activation_layer))

    rgb_activations = np.array([])

    for filename in enteries:
        if filename.endswith('_rgb.npy'):
            num_of_video = int(filename.split('_')[0])
            PATH = path + '/' + filename
            rgb = np.load(PATH)

            # Convert RGB frames to PyTorch tensor
            rgb_tensor = torch.from_numpy(rgb).permute(3, 0, 1, 2).float()
            rgb_tensor = torch.unsqueeze(rgb_tensor, 0)

            activation = {}
            # Set model to evaluate mode
            i3d.eval()

            with torch.no_grad():
                rgb_pred = i3d(rgb_tensor)

        rgb_activations[num_of_video-1, :, :, :, :] = activation[activation_layer].numpy()


    np.save(os.path.join(save_dir, 'rgb_activation_', activation_layer), rgb_activations)


def extract_flow_activation(path, save_dir, activation_layer):

    # load the video frames
    enteries = os.listdir(path)

    # setup the model for flow prediction
    i3d = InceptionI3d(400, in_channels=2, activation_layer=activation_layer)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load('./checkpoints/flow_charades.pt'))

    i3d._modules.get(activation_layer).register_forward_hook(get_activation(activation_layer))

    flow_activations = np.array([])

    for filename in enteries:
        if filename.endswith('_flow.npy'):
            num_of_video = int(filename.split('_')[0])
            PATH = path + '/' + filename
            flow = np.load(PATH)

            # Convert flow frames to PyTorch tensor
            flow_tensor = torch.from_numpy(flow).permute(3, 0, 1, 2).float()
            flow_tensor = torch.unsqueeze(flow_tensor, 0)

            activation = {}
            # Set model to evaluate mode
            i3d.eval()

            with torch.no_grad():
                flow_pred = i3d(flow_tensor)

            flow_activations[num_of_video-1, :, :, :, :] = activation[activation_layer].numpy()


    np.save(os.path.join(save_dir, 'flow_activation_', activation_layer), flow_activations)

