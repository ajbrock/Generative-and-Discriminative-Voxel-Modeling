## Ensemble model 1
import theano
import theano.tensor as T
import numpy as np

import lasagne
import lasagne.layers
import lasagne.layers.dnn
from lasagne.layers import ElemwiseSumLayer as ESL
from lasagne.layers import NonlinearityLayer as NL
from lasagne.init import Orthogonal as initmethod
from lasagne.nonlinearities import elu
from lasagne.layers import batch_norm as BN

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

lr_schedule = { 0: 0.0002,15:0.00002}

cfg = {'batch_size' : 48,
       'learning_rate' : lr_schedule,
       'decay_rate' : 0,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 40,
       'batches_per_chunk': 32,
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

def ResLayer(incoming, IB):
    return NL(ESL([IB,incoming]),elu)
    
   
    
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

def ResDrop(incoming, IB, p):
    return NL(ESL([IfElseDropLayer(IB,survival_p=p),incoming]),elu)             
def get_model():
    lasagne.random.set_rng(np.random.RandomState(1234))
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims
    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv0 = BN(lasagne.layers.dnn.Conv3DDNNLayer(
        incoming = l_in,
        num_filters = 32,
        filter_size = (3,3,3),
        pad = 'same',
        W = initmethod('relu'),
        nonlinearity = elu,
        name = 'l_conv0'),name='bn_conv0')        
    l_conv1 = ResDrop(incoming = l_conv0, 
            IB = InceptionLayer(incoming = l_conv0,param_dict = [{'num_filters':[8,8,16],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[8,16],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv1'),p=0.95)   
    l_conv2 = ResDrop(incoming = l_conv1, 
            IB = InceptionLayer(incoming = l_conv1, param_dict = [{'num_filters':[8,8,16],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[8,16],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv2'),p=0.9)       
    l_conv3 = InceptionLayer(incoming = l_conv2, 
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
            block_name = 'conv3')       
    l_conv4 = ResDrop(incoming = l_conv3, 
            IB = InceptionLayer(incoming = l_conv3, param_dict = [{'num_filters':[16,16,32],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[16,32],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv4'),p=0.8)
    l_conv5 = ResDrop(incoming = l_conv4, 
            IB = InceptionLayer(incoming = l_conv4, param_dict = [{'num_filters':[16,16,32],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[16,32],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv5'),p=0.7)
     
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
    l_conv7 = ResDrop(incoming = l_conv6, 
            IB = InceptionLayer(incoming = l_conv6, param_dict = [{'num_filters':[32,32,64],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[32,64],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv7'),p=0.6)
    l_conv8 = ResDrop(incoming = l_conv7, 
            IB = InceptionLayer(incoming = l_conv7, param_dict = [{'num_filters':[32,32,64],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[32,64],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv8'),p=0.5)
    l_conv9 = InceptionLayer(incoming = l_conv8, 
            param_dict = [{'num_filters':[64],
                'filter_size':[(3,3,3)],
                'border_mode':['same'],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[64],
                'filter_size':[(1,1,1)],
                'pad':[(0,0,0)],
                'strides':[(2,2,2)],
                'nonlinearity': [elu],
                'style': ['convolutional'],
                'bnorm':[1]},
                {'num_filters':[64,1],
                'mode': [0,'max'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]},
                {'num_filters':[64,1],
                'mode': [0,'average_inc_pad'],
                'filter_size':[(3,3,3),(3,3,3)],
                'pad':[(1,1,1),(1,1,1)],
                'strides':[(1,1,1),(2,2,2)],
                'nonlinearity': [None,elu],
                'style': ['convolutional','pool'],
                'bnorm':[0,1]}],
            block_name = 'conv9')
    l_conv10 = ResDrop(incoming = l_conv9, 
            IB = InceptionLayer(incoming = l_conv9, param_dict = [{'num_filters':[64,64,128],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[64,128],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv10'),p=0.4)
    l_conv11 = ResDrop(incoming = l_conv10, 
            IB = InceptionLayer(incoming = l_conv10, param_dict = [{'num_filters':[64,64,128],
                'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                'border_mode':['same']*3,
                'strides':[(1,1,1)]*3,
                'nonlinearity': [elu,elu,None],
                'style': ['convolutional']*3,
                'bnorm':[1]*3},
                {'num_filters':[64,128],
                'filter_size':[(3,3,3)]*2,
                'border_mode':['same']*2,
                'strides':[(1,1,1)]*2,
                'nonlinearity': [elu,None],
                'style': ['convolutional']*2,
                'bnorm':[1]*2}],
            block_name = 'conv11'),p=0.4)
    l_conv12 = InceptionLayer(incoming = l_conv11, 
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
            block_name = 'conv12')
    l_conv13 = ResDrop(l_conv12,BN(lasagne.layers.dnn.Conv3DDNNLayer(
        incoming = l_conv12,
        num_filters = 512,
        filter_size = (3,3,3),
        pad = 'same',
        W = initmethod('relu'),
        nonlinearity = None,
        name = 'l_conv13'),name='bn_conv13'),0.3)        
    l_fc1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        incoming = l_conv13,
        num_units = 1024,
        W = initmethod('relu'),
        nonlinearity = elu,
        name =  'fc1'
        ),name = 'bnorm_fc1') 
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_fc1,
        num_units = n_classes,
        W = initmethod('relu'),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
