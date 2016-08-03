%% Make .mat File Function
%
% A Brock, 9 June 2016
%
% This function crawls through the Modelnet40 OFF files, voxelizes them to
% a set resolution, then saves them all in one big-ass .mat file.
% Note that this requires the voxellizationelationation toolbox from the
% modelnet.cs.princeton.edu website.

%% WARNING: This function consumes a stupidly huge amount of memory  (~10GB RAM) and requires several hours to run.

%% Add more layers
clc
clear all
close all

%% Walk through, read in an OFF, convert it to voxel grid.
folders = ls;
folders(any(folders'=='.'),:) = []; % Get rid of non-folder files; consider using dir and the isdir struct element to do this more certainly.
% Voxel Grid Resolution
res = 32; % Ignore the "unused variable" warning on this one; we're using naughty "evals."
% number of rotations
num_rotations = 24;
% Prepare training struct
% train = struct();
% loop across folders

for class = 1:length(folders)
    classname = folders(class,1:find(folders(class,:)==' ')-1);
    train_files = dir(strcat(classname,'\train\*.off'));
    % Loop across train files
%     eval(strcat(classname,'=zeros(length(train_files),1,res,res,res);'));
    eval(strcat(classname,'=logical(zeros(length(train_files),num_rotations,res,res,res));'))
    for i = 1:length(train_files)
        fprintf('Reading training %s, on %i of %i.\n',classname,i,length(train_files))
        [vertices,faces] = read_off(strcat(classname,'\train\',train_files(i).name));
        vertices = vertices - repmat(mean(vertices,1),size(vertices,1),1);
        FV.faces = faces;
        
        for j = 1:num_rotations
            th = 2*pi*(j-1)/num_rotations;
            FV.vertices = [vertices(:,1)*cos(th) - vertices(:,2)*sin(th),vertices(:,1)*sin(th)+vertices(:,2)*cos(th),vertices(:,3)];
            eval(strcat(classname,'(i,j,:,:,:)=logical(polygon2voxel(FV,[res, res, res],''auto''));'));
        end
        %         train(i).(classname)=polygon2voxel(FV,[32, 32, 32],'auto');
    end
end
% Clean up and save
clear vertices faces FV train_files test_files classname folders res class i j num_rotations th
save('train24_32.mat')

%% Get testfiles
clear all
folders = ls;
folders(any(folders'=='.'),:) = []; % Get rid of non-folder files; consider using dir and the isdir struct element to do this more certainly.
num_rotations = 24;
% Voxel Grid Resolution
res = 32;
for class = 1:length(folders)
    classname = folders(class,1:find(folders(class,:)==' ')-1);
    test_files = dir(strcat(classname,'\test\*.off'));
    % Loop across train files
    eval(strcat(classname,'=logical(zeros(length(test_files),num_rotations,res,res,res));'))
    for i = 1:length(test_files)
        fprintf('Reading testing %s, on %i of %i.\n',classname,i,length(test_files))
        [vertices,faces] = read_off(strcat(classname,'\test\',test_files(i).name));
        vertices = vertices - repmat(mean(vertices,1),size(vertices,1),1);
        FV.faces = faces;
        for j = 1:num_rotations
            th = 2*pi*(j-1)/num_rotations;
            FV.vertices = [vertices(:,1)*cos(th) - vertices(:,2)*sin(th),vertices(:,1)*sin(th)+vertices(:,2)*cos(th),vertices(:,3)];
            eval(strcat(classname,'(i,j,:,:,:)=logical(polygon2voxel(FV,[res, res, res],''auto''));'));
        end
        %         train(i).(classname)=polygon2voxel(FV,[32, 32, 32],'auto');
    end
end
clear vertices faces FV train_files test_files classname folders res class i num_rotations th j
save('test24_32.mat');
