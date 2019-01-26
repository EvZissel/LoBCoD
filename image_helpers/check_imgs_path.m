%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a helper function to check that there are valid image files or
% a .mat in the input path that can be used by CreateImages.m
%
% @file
% @author Matthew Zeiler
% @date Jun 28, 2011
%
% @image_file @copybrief check_imgs_path.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief check_imgs_path.m
%
% @param path the path to check
%
% @retval valid 1 if the path is valid and 0 if it is no good.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [valid] = check_imgs_path(path)


if(strcmp(path(end),'/')==0)
    path = [path '/'];
end

valid = 1;

% Make sure the path is a string to continue the checks, otherwise it's fine as a variable.
if(ischar(path))
    % As a string it has to be a file or a directory
    if(~exist(path,'file') || ~exist(path,'dir'))
        fprintf('Image Path is not a directory or a file\n');
        valid = 0;
        return;
    end
    
    
    % Cell array listing all files and paths.
    subdir = dir(path);
    [blah,files] = split_folders_files(subdir);
    % Empty directory
    if(exist(path,'dir') && length(files)==0)
        fprintf('No files in Image directory\n');
        valid = 0;
        return;
    end        
    
    if(length(files)==1)% Can be a single image in the directory or a .mat file.
        if(strcmp(files(1).name(end-3:end),'.mat'))
            valid = 1; % okay
            return;
        else % make sure it is an image
            try                
                imread([path files(1).name]);
            catch
                fprintf('There is no single .mat file in the directory or the single file is not images.\n');
                valid = 0; % Not an image
                return;
            end
        end
    end                    
else
    valid = 1;
end
