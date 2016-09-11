# Generative-and-Discriminative-Voxel-Modeling
Voxel-Based Variational Autoencoders, VAE GUI, and Convnets for Classification

![GUI](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modelling/blob/master/doc/GUI3.png)

This repository contains code for the paper ["Generative and Discriminative Voxel Modeling with Convolutional Neural Networks,"](https://arxiv.org/abs/1608.04236) and the [Voxel-Based Variational Autoencoders](https://www.youtube.com/watch?v=LtpU1yBStlU) and [Voxel-Based Deep Networks for Classification videos.](https://www.youtube.com/watch?v=OAgfUOg79wc)

## Installation
To run the VAE and GUI, you will need:

- [Theano](http://deeplearning.net/software/theano/) 
- [lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)
- [path.py](https://github.com/jaraco/path.py)
- [VTK](http://www.vtk.org/) and its python bindings
- [cuDNN](https://developer.nvidia.com/cudnn)

If you want to plot latent space mappings, you will need [matplotlib](http://matplotlib.org/).

To train and test classifier ConvNets, you will need:
- [Theano](http://deeplearning.net/software/theano/) 
- [lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)
- [cuDNN](https://developer.nvidia.com/cudnn)


Download the repository and add the main folder to your PYTHONPATH, or uncomment and modify the sys.path.insert lines in whatever script you want to run.

## Preparing the data
I've included several .tar versions of Modelnet10, which can be used to train the VAE and run the GUI. If you wish to write more .tar files (say, of Modelnet40) for use with the VAE and GUI, download  [the dataset](http://modelnet.cs.princeton.edu/) and then see [voxnet](https://github.com/dimatura/voxnet).

For the Discriminative model, I've included a MATLAB script in utils to convert raw Modelnet .off files into MATLAB arrays, then a python script to convert the MATLAB arrays into either .npz files or hdf5 files (for use with [fuel](https://github.com/mila-udem/fuel)). 

The _nr.tar files contain the unaugmented Modelnet10 train and test sets, while the other tar files have 12 copies of each model, rotated evenly about the vertical axis. 

## Running the GUI
I've included a pre-trained model (VAE.npz) trained on Modelnet10, which can be used to run the GUI:

```sh
python Generative/GUI.py Generative/VAE.py datasets/shapenet10_test_nr.tar Generative/VAE.npz
```

## Training the VAE
If you wish to train a model, the VAE.py file contains the model configuration, and the train_VAE.py file contains the training code, which can be run like this:

```sh
python Generative/train_VAE.py Generative/VAE.py datasets/shapenet10_train.tar Generative/shapenet10_test.tar
```
By default, this code will save (and overwrite!) the weights to a .npz file with the same name as the config.py file (i.e. "VAE.py -> VAE.npz"), and will output a jsonl log of the training with metrics recorded after every chunk (a chunk being a set of minibatches loaded into shared memory). The binary reconstruction accuracy is evaluated on the test set after every N epochs (defined in the config file), and evaluates both false positives and false negatives.

A good model will obtain a very low false negative rate, while most any model can get near-perfect false positives (and therefore very high overall reconstruction accuracy).

## Training a Classifier
The VRN.py file contains the model configuration and definitions for any custom layer types. The model can be trained with:

```sh
python Discriminative/train.py Discriminative/VRN.py datasets/modelnet40_rot_train.npz
```
Note that running the train function will start from scratch and overwrite any pre-trained model weights (.npz files with the same name as their corresponding config files). Use the --resume=True option to resume training from an earlier session or from one of the provided pre-trained models.

The first time you compile these functions may take a very long time, and may exceed the maximum recursion depth allowed by python.

## Testing a Classifier
#
You can evaluate a classifier's performance on the ModelNet40 dataset, averaging predictions over 12 rotations, with:

```sh
python Discriminative/test.py Discriminative/VRN.py datasets/modelnet40_rot_test.npz
```

## Evaluating an Ensemble
#
You can produce a simple ensemble by averaging multiple models' predictions on the test sets. I provide six pre-trained models for this purpose, along with .csv files containing their outputs on ModelNet40.
Use the test_ensemble.py script to produce a .csv file with the model's predictions, and use the ensemble.m MATLAB script to combine and evaluate all the results.

## Acknowledgments
This code was originally based on [voxnet](https://github.com/dimatura/voxnet) by D. Maturana.
