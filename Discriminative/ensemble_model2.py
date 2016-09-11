## Ensemble model 2
import numpy as np

import lasagne
import lasagne.layers
import lasagne.layers.dnn

import voxnet

lr_schedule = { 0: 0.005}

cfg = {'batch_size' : 48,
       'learning_rate' : lr_schedule,
       'decay_rate' : 0.1,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 40,
       'batches_per_chunk': 64,
       'max_epochs' : 250,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 5,
       }
def InceptionLayer(incoming,param_dict,block_name):
    branch = [0]*len(param_dict)
    # Loop across branches
    for i,dict in enumerate(param_dict):
        for j,style in enumerate(dict['style']): # Loop up branch
            branch[i] = voxnet.layers.Conv3dDNNLayer(
                input_layer = incoming if j == 0 else branch[i],
                num_filters = dict['num_filters'][j],
                filter_size = dict['filter_size'][j],
                border_mode = dict['border_mode'][j] if 'border_mode' in dict else None,
                pad = dict['pad'][j] if 'pad' in dict else None,
                strides = dict['strides'][j],
                W = lasagne.init.GlorotNormal(),
                nonlinearity = dict['nonlinearity'][j],
                name = block_name+'_'+str(i)+'_'+str(j)) if style=='convolutional' else lasagne.layers.NonlinearityLayer(lasagne.layers.dnn.Pool3DDNNLayer(
                incoming=incoming if j == 0 else branch[i],
                pool_size = dict['filter_size'][j],
                mode = dict['mode'][j],
                stride = dict['strides'][j],
                pad = dict['pad'][j],
                name = block_name+'_'+str(i)+'_'+str(j)),
                nonlinearity = dict['nonlinearity'][j])
                # Apply Batchnorm    
            branch[i] = lasagne.layers.batch_norm(branch[i],name = block_name+'_bnorm_'+str(i)+'_'+str(j)) if dict['bnorm'][j] else branch[i]
        # Concatenate Sublayers        
            
    return lasagne.layers.ConcatLayer(incomings=branch,name=block_name)  
    
def get_model():
    lasagne.random.set_rng(np.random.RandomState(1234))
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims
    elu = lasagne.nonlinearities.elu
    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv0 = InceptionLayer(incoming = l_in, 
            param_dict = [{'num_filters':[16],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[16],
                'filter_size':[(1,1,1)],
                'border_mode':['same'],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]}],
            block_name = 'conv0')   
    l_conv1 = InceptionLayer(incoming = l_conv0, 
            param_dict = [{'num_filters':[16],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[16],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]}],
            block_name = 'conv1') 
    l_conv2 = InceptionLayer(incoming = l_conv1, 
            param_dict = [{'num_filters':[8],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[8],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[8,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[8,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv2') 
    l_conv3 = InceptionLayer(incoming = l_conv2, 
            param_dict = [{'num_filters':[32],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[32],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]}],
            block_name = 'conv3')
    l_conv4 = InceptionLayer(incoming = l_conv3, 
            param_dict = [{'num_filters':[16],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[16],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[16,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[16,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv4')
    l_conv5= InceptionLayer(incoming = l_conv4, 
            param_dict = [{'num_filters':[64],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[64],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(1,1,1)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]}],
            block_name = 'conv5')
    l_conv6 = InceptionLayer(incoming = l_conv5, 
            param_dict = [{'num_filters':[32],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[32],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[32,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[32,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv6')        
    l_fc1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        incoming = l_conv6,
        num_units = 512,
        W = lasagne.init.GlorotNormal(),
        name =  'fc1'
        ),name = 'bnorm7') 
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_fc1,
        num_units = n_classes,
        W = lasagne.init.GlorotNormal(),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
