function video_design_all_target(speed, save_dir, image_dir)

    % load images
    im = load_images(image_dir,'tif'); 
    
    % stimuli indices
    stimuli = [1:40];
    
    % form the matrix for all stimuli
    for i=1:40
        F(i, 1:11*speed) = repelem(stimuli(i), 11*speed);
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
