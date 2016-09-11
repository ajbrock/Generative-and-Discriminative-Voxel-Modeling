%% Ensembling Script
% A Brock, 2016
%
% Take the outputs of multiple models' predictions and combine them.
%% Excavate the Playing Field
clc
clear all
close all
tic
class_ids = {'airplane', 'bathtub', 'bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup', 'curtain', 'desk', 'door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person', 'piano', 'plant', 'radio','range_hood','sink','sofa', 'stairs', 'stool', 'table','tent','toilet','tv_stand', 'vase', 'wardrobe', 'xbox'};
%% Get Results and Models

% Targets
y = csvread('y.csv');
% Modify targets to account for the fact that MATLAB starts at 1
y = y(1:12:end)+1;

% Acquire model results
names = dir();
names={names.name};
names = names(3:end);
n = 1;
for i=1:length(names)
    if strcmp(names{i}(end-3:end),'.csv')&&(~strcmp(names{i},'y.csv'))
        models{n} = names{i}(1:end-4);
        n=n+1;
    end
end

% Get model accuracies
x = cell(length(models),1);
w = zeros(length(models),1);
% Get all data
for i = 1:length(models)
    x{i} = csvread(strcat(models{i},'.csv'));
    [~,yx] = max(x{i},[],2);
    w(i) = sum(y==yx)/length(y);
end

% Sort model accuracies
[~,order] = sort(w,'descend');

%% Get results!

z = zeros(size(x{1}));
for i = 1:length(x)
    z=z+x{i};
end
[~,predictions] = max(z,[],2);
accuracy = sum(y==predictions)/length(y);

fprintf('Accuracy is %8.8f, with %i correct examples out of a total of %i instances.\n',accuracy,int16(accuracy*length(y)),length(y))