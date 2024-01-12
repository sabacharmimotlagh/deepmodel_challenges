clc;
clear all;

%%
speed = 1;
image_dir = '.\prepare_test_dataset\120stimuli';
save_dir = '.\prepare_test_dataset\videos\';

% run the code to create and save videos
video_design(speed, save_dir, image_dir);