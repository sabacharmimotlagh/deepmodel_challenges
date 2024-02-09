import os
import cv2
import numpy as np
from tqdm import tqdm
from load_images import load_images

def video_design(speed, image_dir, save_dir, all_target=False):
    # Load images
    im = load_images(image_dir, 'tif')

    # Stimuli and masks indices
    stimuli = list(range(1, 41))
    masks = list(range(41, 81))

    # Form the matrix for all stimuli
    F = np.zeros((40, 11*speed), dtype=np.uint8)

    if all_target:
        F[i, :] = np.repeat(stimuli[i], 11*speed)

    else:
        for i in range(40):
            F[i, :5*speed] = np.tile(np.random.choice(masks, 5), speed)
            F[i, 5*speed:6*speed] = np.repeat(stimuli[i], speed)
            F[i, 6*speed:] = np.tile(np.random.choice(masks, 5), speed)

    # Create and save videos for each stimulus
    for i in tqdm(range(40), desc='Generating Videos'):
        video_name = os.path.join(save_dir, f'video{i + 1:02}.mp4')
        writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (im[0].shape[1], im[0].shape[0]))

        # Write the frames to the video
        for j in range(F.shape[1]):
            frame = im[F[i, j] - 1]  # Convert 1-based index to 0-based index
            writer.write(frame)

        writer.release()
