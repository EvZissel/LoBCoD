%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Return two struct arrays of just the folers and just the files of the input
% struct array.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief split_folders_files.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief split_folders_files.m
%
% @param input a struct array of both files and folders combined.
% @retval folders a struct array of the folders
% @retval files a struct array of the files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [folders,files] = split_folders_files(input)

% 
% folders = input(find(~cellfun(@iszero,{input(:).isdir})));
% 
% files = input(find(cellfun(@iszero,{input(:).isdir})));
% 
% 
%     function [result] = iszero(input)
%         
%         if(input==0)
%             result = 1;
%         else
%             result = 0;
%         end
%     end

B = struct2cell(input); 
dirs = cell2mat(B(4,:)); 
folders = input(logical(dirs)); 
files = input(~logical(dirs));

end