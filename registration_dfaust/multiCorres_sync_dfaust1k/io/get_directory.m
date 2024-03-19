function [path] = get_directory(path)
    if not(isfolder(path))
        mkdir(path)
    end
end