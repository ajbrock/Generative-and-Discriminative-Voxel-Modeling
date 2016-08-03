# Generative-and-Discriminative-Voxel-Modeling
Voxel-Based Variational Autoencoders, VAE GUI, and Convnets for Classification

![GUI](https://github.com/ajbrock/blob/master/doc/GUI3.png)

This repository contains code from the forthcoming paper, "Generative and Discriminative Voxel Modeling with Convolutional Neural Networks," and the [Voxel-Based Variational Autoencoders](https://www.youtube.com/watch?v=LtpU1yBStlU) and Voxel-Based Deep Networks for Classification videos.

## Installation
To run the VAE and GUI, you will need:

- [Theano](http://deeplearning.net/software/theano/) 
- [lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)
- [path.py](https://github.com/jaraco/path.py)
- [VTK](http://www.vtk.org/) and its python bindings

If you want to plot latent space mappings, you will need [matplotlib](http://matplotlib.org/)

Download the repository and add the main folder to your PYTHONPATH, or uncomment and modify the sys.path.insert lines in whatever script you want to run.

## Preparing the data
I've included several .tar versions of Modelnet10, along with a MATLAB script to convert raw Modelnet .off files into MATLAB arrays, then a python script to convert the MATLAB arrays into either .npz files or hdf5 files (for use with [https://github.com/mila-udem/fuel]). If you wish to write more .tar files (say, of Modelnet40) for use with the VAE and GUI, see [voxnet](https://github.com/dimatura/voxnet).

## Running the GUI
I've included a pre-trained model (VAE.npz) trained on modelnet10, which can be used to run the GUI:

```sh
python Generative/GUI.py Generative/VAE.py datasets/shapenet10_test_nr.tar Generative/VAE.npz
```

## Notes
Discriminative models coming soon!

## Acknowledgments
This code was originally based on [voxnet](https://github.com/dimatura/voxnet) by D. Maturana.
