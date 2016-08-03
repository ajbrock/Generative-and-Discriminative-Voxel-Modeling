##Script for creating Modelnet40  Datasets
# A Brock, 2016

# This code, in its current form, saves an NPZ file by reading in a .mat file
# (previously prepared using the make_mats.m code) containing arrays of 
# the voxellated modelnet40 models, with 24 different rotations for each model.
# Each 5D-array is organized in the format (instance-rotations-spatialdim1-spatialdim2-spatialdim3).
# This code currently separates the rotations out such that each rotation is a separate instance, but if
# you have GPU memory to spare and are feeling masochistic, you're welcome to treat the rotations as channels
# (a la RGB in 2D imagery). I found that this didn't really change performance and just made my models take
# up more memory.
#
# This file also has commented out code which is pretty close to being able to store this data in an
# HDF5 dataset accessible by fuel, but no guarantees as I always had enough RAM available to load the
# whole dataset and didn't really need to worry about the hard-disk format.

import numpy as np
import scipy.io
from collections import OrderedDict

# from fuel.datasets import  IndexableDataset
# from fuel.datasets.hdf5 import H5PYDataset
# import h5py


# Load data
train = scipy.io.loadmat('train24_32.mat')

# Delete extra .matfile stuff
del train['__globals__']
del train['__header__']
del train['__version__']

# Prepare data arrays
targets = np.asarray([],dtype=np.uint8);
features = np.zeros((1,1,32,32,32),dtype=np.uint8);

# Select which classes to read in
class_keys = sorted(train.keys())
# class_keys = ["bathtub","bed", "chair", "desk", "dresser", "monitor", "night_stand","sofa", "table","toilet"] # Keys for modelnet10

for i,key in enumerate(class_keys):
    targets = np.append(targets,i*np.ones(24*len(train[key]),dtype=np.uint8))
    features = np.append(features,np.reshape(train[key],(24*np.shape(train[key])[0],1,32,32,32)),axis=0)
    if i==0:
        features=np.delete(features,0,axis=0)
    del train[key]
del train

np.savez_compressed('modelnet40_rot24_train.npz',**{'features':features,'targets':targets})
# np.reshape(features,(12*np.shape(features)[0],1,32,32,32)
# writer = npytar.NpyTarWriter(fname)
# writer.add(targets,features)    
# features = np.delete(features,0,axis=0) # Delete first entry   
# f = h5py.File('dataset.hdf5', mode='w')
# featureset = f.create_dataset('features',np.shape(features),dtype='uint8')
# targetset = f.create_dataset('targets',np.shape(targets),dtype='uint8')
# featureset[...] = features
# targetset[...] = targets
# featureset.dims[0].label = 'batch'
# featureset.dims[0].label = 'rotation'
# featureset.dims[0].label = 'i'
# featureset.dims[0].label = 'j'
# featureset.dims[0].label = 'k'
# targetset.dims[0].label = 'batch'
# split_dict = {'train': {'features': (0, len(targets)),'targets': (0, len(targets))}}
# f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
# f.flush()
# f.close()
# dataset = IndexableDataset(indexables=OrderedDict([('features', features), ('targets', targets)]),
                           # axis_labels=OrderedDict([('features', ('batch', 'rotation', 'i','j','k')),
                           # ('targets', ('batch', 'index'))]))



# Failed targets attempts
# targets = np.zeros(sum([len(train[a_key]) for a_key in class_keys]),dtype=np.int)
# targets = np.append(targets,i*np.ones(len(train[a_key])) for i,a_key in enumerate(class_keys))
# targets[i] = i*np.ones(len(train[class_keys[i]]));