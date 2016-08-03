#####################################################
# Introspective Variational Autoencoder Config File #
#####################################################
# A Brock, 2016
import numpy as np
import lasagne
import lasagne.layers
import theano.tensor as T
from layers import Conv3dDNNLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Start with a low learning rate then increase learning rate on second epoch
lr_schedule = { 0: 0.0001,1:0.005}

# Configuration Dictionary
cfg = {'batch_size' : 64,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 64,
       'max_epochs' : 150,
       'max_jitter_ij' : 4,
       'max_jitter_k' : 4,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 5,
       'num_latents': 100,
       'introspect' : True,
       'discriminative': False,
       'cc' : False,
       'kl_div': False,
       }
       
# Gaussian Sample Layer adopted from Tencia Lee's Lasagne VAE Recipe.    
class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)

# Get model
# The interp flag is for use with the GUI, and splits the graph in two.
# This isn't necessary, as lasagne supports assigning inputs to non-input layers).        
def get_model(interp=False):
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims
    
    l_in = lasagne.layers.InputLayer(shape=shape)
    l_enc_conv1 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_in,
        num_filters = 8,
        filter_size = [3,3,3],
        border_mode = 'valid',
        strides = [1,1,1],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name =  'enc_conv1'
        ),name = 'bnorm1')
    l_enc_conv2 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_enc_conv1,
        num_filters = 16,
        filter_size = [3,3,3],
        border_mode = 'same',
        strides = [2,2,2],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name = 'enc_conv2'
        ),name = 'bnorm2')    
    l_enc_conv3 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_enc_conv2,
        num_filters = 32,
        filter_size = [3,3,3],
        border_mode = 'valid',
        strides = [1,1,1],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name = 'enc_conv3'
        ),name = 'bnorm3')
    l_enc_conv4 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_enc_conv3,
        num_filters = 64,
        filter_size = [3,3,3],
        border_mode = 'same',
        strides = [2,2,2],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name = 'enc_conv4'
        ),name = 'bnorm4')

    l_enc_fc1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        incoming = l_enc_conv4,
        num_units = 343,
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name =  'enc_fc1'
        ),
        name = 'bnorm5')    
    l_enc_mu = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        incoming = l_enc_fc1,
        num_units=cfg['num_latents'],
        nonlinearity = None, 
        name='enc_mu'
        ),name='bnorm6')
    l_enc_logsigma = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        incoming = l_enc_fc1,
        num_units=cfg['num_latents'],
        nonlinearity = None,
        name='enc_logsigma'),name='bnorm7')
    l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
    l_Z_in = lasagne.layers.InputLayer(shape=(None, cfg['num_latents']),name = 'l_Z_in')
    l_class_conditional = lasagne.layers.InputLayer(shape=(None, 10),name = 'Class_conditional')   
    l_Z_cc = lasagne.layers.ConcatLayer(incomings = [l_Z_in,l_class_conditional] if interp else [l_Z,l_class_conditional],name = 'Zccmerge')
    l_dec_fc1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        incoming = l_Z_cc if cfg['cc'] else l_Z_in if interp else l_Z,
        num_units = 343,
        nonlinearity = lasagne.nonlinearities.elu,
        W=lasagne.init.GlorotNormal(),
        name='l_dec_fc1'),
        name = 'bnorm8') 
    l_unflatten = lasagne.layers.ReshapeLayer(
        incoming = l_dec_fc1,
        shape = ([0],1,7,7,7),
        )   
    l_dec_conv1 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_unflatten,
        num_filters = 64,
        filter_size = [3,3,3],
        border_mode = 'same',
        strides = [1,1,1],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name =  'l_dec_conv1'
        ),name = 'bnorm10')
    l_dec_conv2 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_dec_conv1,
        num_filters = 32,
        filter_size = [3,3,3],
        border_mode = 'valid',
        strides = [1.0/2, 1.0/2, 1.0/2],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name =  'l_dec_conv2'
        ),name = 'bnorm11')    
    l_dec_conv3 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_dec_conv2,
        num_filters = 16,
        filter_size = [3,3,3],
        border_mode = 'same',
        strides = [1,1,1],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name =  'l_dec_conv3'
        ),name = 'bnorm12')
    l_dec_conv4 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_dec_conv3,
        num_filters = 8,
        filter_size = [4,4,4],
        border_mode = 'valid',
        strides = [1.0/2, 1.0/2, 1.0/2],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = lasagne.nonlinearities.elu,
        name =  'l_dec_conv4'
        ),name = 'bnorm13')    
    l_dec_conv5 = lasagne.layers.batch_norm(Conv3dDNNLayer(
        input_layer = l_dec_conv4,
        num_filters = 1,
        filter_size = [3,3,3],
        border_mode = 'same',
        strides = [1,1,1],
        W = lasagne.init.GlorotNormal(),
        nonlinearity = None,
        name =  'l_dec_conv5'
        ),name = 'bnorm14')         
        
    
    l_classifier = lasagne.layers.DenseLayer(
        incoming = l_Z,
        num_units = n_classes,
        W = lasagne.init.GlorotNormal(),
        nonlinearity = None,
        name = 'classifier'
        )
        
    return {'l_in':l_in, 
            'l_out':l_dec_conv5,
            'l_mu':l_enc_mu,
            'l_ls':l_enc_logsigma,
            'l_latents':l_Z,
            'l_Z_in': l_Z_in,
            'l_cc': l_class_conditional,
            'l_classifier': l_classifier}
