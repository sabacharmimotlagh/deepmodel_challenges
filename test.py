import os
import torch
import numpy as np
from model import InceptionI3d

PATH = '/content/drive/MyDrive/videos/video_frames/'
enteries = os.listdir(PATH)

# using hook function to extract activations from different layers
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation_layer = 'Conv3d_1a_7x7'

# setup the model for rgb prediction
i3d = InceptionI3d(400, in_channels=3, activation_layer=activation_layer)
i3d.replace_logits(157)
i3d.load_state_dict(torch.load('rgb_charades.pt'))

i3d._modules.get(activation_layer).register_forward_hook(get_activation(activation_layer))

rgb_activations = np.zeros((40,64,17,112,112))

for filename in enteries:
    if filename.endswith("_rgb.npy"):
        num_of_video = int(filename.split('_')[0][-2:])
        path = PATH + '/' + filename
        rgb = np.load(path)

        # Convert RGB frames to PyTorch tensor
        rgb_tensor = torch.from_numpy(rgb).permute(3, 0, 1, 2).float()
        rgb_tensor = torch.unsqueeze(rgb_tensor, 0)

        activation = {}
        # Set model to evaluate mode
        i3d.eval()

        with torch.no_grad():
            rgb_pred = i3d(rgb_tensor)

        rgb_activations[num_of_video-1, :, :, :, :] = activation[activation_layer].numpy()


np.save(os.path.join(PATH, 'rgb_activation_Conv3d_1a_7x7'), rgb_activations)