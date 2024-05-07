import os
import cv2
import re
import random
import numpy as np
from tqdm import tqdm

class RSVPVideoGenerator:
    """
    A class for generating RSVP (Rapid Serial Visual Presentation) videos based on stimuli and masks.

    Args:
        video_speed (int): The speed of the video playback.
        stimuli_dir (str): The directory containing the stimuli images.
        save_dir (str): The directory to save the generated videos.
        file_type (str, optional): The file type of the stimuli images. Defaults to 'tif'.
        patch_size (int, optional): The size of each patch in the images. Defaults to 20.
        nmasks (int, optional): The number of masks to use in each video. Defaults to 10.
        all_target (bool, optional): Whether to use all stimuli as targets in each video. Defaults to False.
    """

    def __init__(self, video_speed, stimuli_dir, save_dir, file_type='tif', patch_size=20, nmasks=10, all_target=False):
        self.video_speed = video_speed
        self.stimuli_dir = stimuli_dir
        self.save_dir = save_dir
        self.file_type = file_type
        self.patch_size = patch_size
        self.nmasks = nmasks
        self.all_target = all_target

    def load_images(self):
        """
        Loads images from the specified directory.

        Returns:
            dict: A dictionary containing the loaded images.
        """
        im = {}
        files = os.listdir(self.stimuli_dir)
        
        for image_file in files:
            if self.file_type in image_file:
                image_name = int(image_file.split('.')[0])
                im[image_name] = cv2.imread(os.path.join(self.stimuli_dir, image_file))

        return im

    def create_stimuliset(self):
        """
        Creates a set of masks by dividing the images into patches and shuffling them.

        Returns:
            dict: A dictionary containing the original and scrambled images.
        """
        im = self.load_images()
        nstimuli = len(im)

        for stimulus in range(1, nstimuli+1):
            image = im[stimulus]
            num_rows, num_cols = image.shape[:2]
            num_patches_rows = num_rows // self.patch_size
            num_patches_cols = num_cols // self.patch_size
            patches = []

            for i in range(num_patches_rows):
                for j in range(num_patches_cols):
                    patch = image[i*self.patch_size: (i+1)*self.patch_size, j*self.patch_size: (j+1)*self.patch_size]
                    patches.append(patch)

            random.shuffle(patches)
            scrambled_image = np.zeros_like(image)

            for i, patch in enumerate(patches):
                x = i % num_patches_rows
                y = i // num_patches_cols
                scrambled_image[y*self.patch_size: (y+1)*self.patch_size, x*self.patch_size: (x+1)*self.patch_size] = patch

            im[stimulus+nstimuli] = scrambled_image

        return im

    def video_design(self, fps=25):
        """
        Generates RSVP videos based on the stimuli and masks and saving the videos to specified directory.

        Returns:
            None
        """
        im = self.create_stimuliset()
        nstimuli = len(im) // 2
        stimuli = list(range(1, len(im)//2+1))
        masks = list(range(len(im)//2+1, len(im)+1))
        F = np.zeros((nstimuli, (self.nmasks+1)*self.video_speed), dtype=np.uint8)

        if self.all_target:
            for i in range(nstimuli):
                F[i, :] = np.repeat(stimuli[i], (self.nmasks+1)*self.video_speed)
        else:
            for i in range(nstimuli):
                mask_indices = np.random.choice(masks, self.nmasks, replace=False)
                F[i, :(self.nmasks//2)*self.video_speed] = [item for item in mask_indices[:(self.nmasks//2)] for _ in range(self.video_speed)]
                F[i, (self.nmasks//2)*self.video_speed:(self.nmasks//2 + 1)*self.video_speed] = np.repeat(stimuli[i], self.video_speed)
                F[i, (self.nmasks//2 + 1)*self.video_speed:] = [item for item in mask_indices[(self.nmasks//2):] for _ in range(self.video_speed)]

        for i in tqdm(range(nstimuli), desc='Generating Videos'):
            video_name = os.path.join(self.save_dir, f'{i+1}.mp4')
            writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (im[1].shape[1], im[1].shape[0]))

            for j in range(F.shape[1]):
                frame = im[F[i, j]]
                writer.write(frame)

            writer.release()
