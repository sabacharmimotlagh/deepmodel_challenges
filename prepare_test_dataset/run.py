from RSVPVideoGenerator import RSVPVideoGenerator

speed = 5
image_dir = './prepare_test_dataset/stimuli/'
save_dir = './prepare_test_dataset/videos/'

# run the code to create and save videos
videos = RSVPVideoGenerator(speed, image_dir, save_dir, file_type='tif', patch_size=20, nmasks=10, all_target=False)
videos.video_design(fps=25)