function [shapes] = load_folder(foldername, numObjects)
%
for id = 1 : numObjects
    filename = sprintf("%s/interp_%d_30.ply", foldername, id-1);
    fprintf('%d : load %s \n', id-1, filename);
    shapes{id} = read_ply(filename);
end



