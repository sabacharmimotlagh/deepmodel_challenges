function im = load_images(directory,filetype)

% Loads images from a given directory

files = dir(directory);

im = {};
for i = 1:length(files)
    if strfind(files(i).name,filetype)
        im{end+1} = imread(fullfile(directory, files(i).name));
    end
end

