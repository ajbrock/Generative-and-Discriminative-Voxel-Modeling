###
# 45-Layer Voxception-ResNet Config File
# A Brock
#
# This file 
import theano
import theano.tensor as T
import numpy as np

import lasagne
import lasagne.layers

# If you for some reason don't use cuDNN:
# 1. Use cuDNN.
# 2. If that fails, replace all the DNN layers with another 3D conv layer.
import lasagne.layers.dnn

# I use a lot of aliases to help keep my code more compact;
# consider changing these back to their full callouts if you like.
from lasagne.layers import ElemwiseSumLayer as ESL
from lasagne.layers import NonlinearityLayer as NL
from lasagne.init import Orthogonal as initmethod
from lasagne.nonlinearities import elu
from lasagne.layers import batch_norm as BN

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

lr_schedule = { 0: 0.002,12:0.0002}

cfg = {'batch_size' : 1,
       'learning_rate' : lr_schedule,
       'decay_rate' : 0,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 40,
       'batches_per_chunk': 1,
       'max_epochs' : 250,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 2,
       }
       
# Convenience function for creating inception blocks.  
def InceptionLayer(incoming,param_dict,block_name):
    branch = [0]*len(param_dict)
    # Loop across branches
    for i,dict in enumerate(param_dict):
        for j,style in enumerate(dict['style']): # Loop up branch
            branch[i] = lasagne.layers.dnn.Conv3DDNNLayer(
                incoming = incoming if j == 0 else branch[i],
                num_filters = dict['num_filters'][j],
                filter_size = dict['filter_size'][j],
                pad = dict['pad'][j] if 'pad' in dict else dict['border_mode'][j] if 'border_mode' in dict else None,
                stride = dict['strides'][j],
                W = initmethod('relu'),
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

# Vanilla Resnet layer with ELUs    
def ResLayer(incoming, IB):
    return NL(ESL([IB,incoming]),elu)
    
   
# If-else Drop Layer, adopted from Christopher Beckham's recipe:
#  https://github.com/Lasagne/Recipes/pull/67
class IfElseDropLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nonlinearity=elu, survival_p=0.5,
                 **kwargs):
        super(IfElseDropLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.p = 1-survival_p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            return theano.ifelse.ifelse(
                T.lt(self._srng.uniform( (1,), 0, 1)[0], self.p),
                input,
                T.zeros(input.shape)
            ) 

# Pre-activation stochastically-dropped ResNet wrapper         
def ResDrop(incoming, IB, p):
    return ESL([IfElseDropLayer(IB,survival_p=p),incoming])
    
# Non-preactivation stochastically-dropped Resnet Wrapper
def ResDropNoPre(incoming, IB, p):
    return NL(ESL([IfElseDropLayer(IB,survival_p=p),incoming]),elu)     

    
def get_model():
    lasagne.random.set_rng(np.random.RandomState(1234))
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims
    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv0 = lasagne.layers.dnn.Conv3DDNNLayer(
        incoming = l_in,
        num_filters = 32,
        filter_size = (3,3,3),
        stride = (1,1,1),
        pad = 'same',
        W = initmethod(),
        nonlinearity = None,
        name = 'l_conv0')        
    l_conv1 = ResDrop(incoming = l_conv0, 
        IB = InceptionLayer(incoming = NL(BN(l_conv0,name='bn_conv0'),elu), 
            param_dict = [{'num_filters':[8,8,16],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[8,16],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv1'),p=0.95)   
    l_conv2 = ResDrop(incoming = l_conv1, 
        IB = InceptionLayer(incoming = NL(BN(l_conv1,name='bn_conv1'),elu), 
            param_dict = [{'num_filters':[8,8,16],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[8,16],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv2'),p=0.9)
    l_conv3 = ResDrop(incoming = l_conv2, 
        IB = InceptionLayer(incoming = NL(BN(l_conv2,name='bn_conv2'),elu), 
            param_dict = [{'num_filters':[8,8,16],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[8,16],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv3'),p=0.8)             
    l_conv4 = InceptionLayer(incoming = NL(BN(l_conv3,name='bn_conv3'),elu), 
            param_dict = [{'num_filters':[16],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [None],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[16],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [None],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[16,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,None],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[16,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,None],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv4')       
    l_conv5 = ResDrop(incoming = l_conv4, 
        IB = InceptionLayer(incoming = NL(BN(l_conv4,name='bn_conv4'),elu), 
            param_dict = [{'num_filters':[16,16,32],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[16,32],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv5'),p=0.7)
    l_conv6 = ResDrop(incoming = l_conv5, 
        IB = InceptionLayer(incoming = NL(BN(l_conv5,name='bn_conv5'),elu), 
            param_dict = [{'num_filters':[16,16,32],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[16,32],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv6'),p=0.6)
    l_conv7 = ResDrop(incoming = l_conv6, 
        IB = InceptionLayer(incoming = NL(BN(l_conv6,name='bn_conv6'),elu), 
            param_dict = [{'num_filters':[16,16,32],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[16,32],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv7'),p=0.5)
    l_conv8 = InceptionLayer(incoming = NL(BN(l_conv7,name='bn_conv7'),elu), 
            param_dict = [{'num_filters':[32],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [None],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[32],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [None],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[32,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,None],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[32,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,None],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv8')        
    l_conv9 = ResDrop(incoming = l_conv8, 
        IB = InceptionLayer(incoming = NL(BN(l_conv8,name='bn_conv8'),elu), 
            param_dict = [{'num_filters':[32,32,64],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[32,64],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv9'),p=0.5)
    l_conv10 = ResDrop(incoming = l_conv9, 
        IB = InceptionLayer(incoming = NL(BN(l_conv9,name='bn_conv9'),elu), 
            param_dict = [{'num_filters':[32,32,64],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[32,64],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv10'),p=0.45)
    l_conv11 = ResDrop(incoming = l_conv8, 
        IB = InceptionLayer(incoming = NL(BN(l_conv10,name='bn_conv10'),elu), 
            param_dict = [{'num_filters':[32,32,64],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[32,64],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv11'),p=0.40)        
    l_conv12 = InceptionLayer(incoming = NL(BN(l_conv11,name='bn_conv11'),elu), 
            param_dict = [{'num_filters':[64],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [None],
                'style': ['convolutional'],
                'bnorm':[0]},
                {'num_filters':[64],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [None],
                'style': ['convolutional'],
                'bnorm':[0]},
                {'num_filters':[64,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,None],
                'style': ['convolutional','pool'],
                'bnorm':[0,0]},
                {'num_filters':[64,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,None],
                'style': ['convolutional','pool'],
                'bnorm':[0,0]}],
            block_name = 'conv12')
    l_conv13 = ResDrop(incoming = l_conv12, 
        IB = InceptionLayer(incoming = NL(BN(l_conv12,name='bn_conv12'),elu), 
            param_dict = [{'num_filters':[64,64,128],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[64,128],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv13'),p=0.35)
    l_conv14 = ResDrop(incoming = l_conv13, 
        IB = InceptionLayer(incoming = NL(BN(l_conv13,name='bn_conv13'),elu), 
            param_dict = [{'num_filters':[64,64,128],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[64,128],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv14'),p=0.30)
    l_conv15 = ResDrop(incoming = l_conv14, 
        IB = InceptionLayer(incoming = NL(BN(l_conv14,name='bn_conv14'),elu), 
            param_dict = [{'num_filters':[64,64,128],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1,1,0]},
                {'num_filters':[64,128],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1,0]}],
            block_name = 'conv15'),p=0.25)        
    l_conv16 = InceptionLayer(incoming = NL(BN(l_conv15,name='bn_conv15'),elu), 
            param_dict = [{'num_filters':[128],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[128],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[128,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[128,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv16')
    l_conv17 = ResDropNoPre(l_conv16,BN(lasagne.layers.dnn.Conv3DDNNLayer(
        incoming = l_conv16,
        num_filters = 512,
        filter_size = (3,3,3),
        pad = 'same',
        W = initmethod('relu'),
        nonlinearity = None,
        name = 'l_conv17'),name='bn_conv17'),0.5)
    l_pool = BN(lasagne.layers.GlobalPoolLayer(l_conv17),name='l_pool')
    l_fc1 = BN(lasagne.layers.DenseLayer(
        incoming = l_pool,
        num_units = 512,
        W = initmethod('relu'),
        nonlinearity = elu,
        name =  'fc1'
        ),name = 'bnorm_fc1') 
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_fc1,
        num_units = n_classes,
        W = initmethod(),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
