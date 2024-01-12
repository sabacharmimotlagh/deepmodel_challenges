function video_design(speed, save_dir, image_dir)

    % load images
    im = load_images(image_dir,'tif'); 
    
    % stimuli and masks indices
    stimuli = [1:40];
    masks = [61:100];
    
    % form the matrix for all stimuli
    for i=1:40
        F(i, 1:5*speed) = repelem(masks(randperm(length(masks),5)), speed);             % 5 different masks
        F(i, 5*speed+1:6*speed) = repelem(stimuli(i), speed);                            % one target
        F(i, 6*speed+1:11*speed) = repelem(masks(randperm(length(masks),5)), speed);    % 5 different masks
    end

    F = im(F);
   
    % create and save videos for each stimuli
    for i=1:40

        % create the video writer with 25 fps
        writerObj = VideoWriter([save_dir , 'video', num2str(i, '%02d')], 'MPEG-4');
        writerObj.FrameRate = 25;
        
        % open the video writer
        open(writerObj);
        
        % write the frames to the video
        for j=1:size(F,2)
            % convert the image to a frame
            frame = F{i,j} ;    
            writeVideo(writerObj, frame);
        end
        % close the writer object
        close(writerObj);
    end
end
