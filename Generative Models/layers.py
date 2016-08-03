### Voxel-Based VAE Layers
# A Brock 2016
# Originally adopted from Voxnet



import numpy as np

import lasagne
from lasagne.layers import Layer


import theano
import theano.tensor as T

from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda.dnn import GpuDnnConvDesc,  GpuDnnConv3dGradI
from lasagne.utils import as_tuple




__all__ = [
        'Conv3dDNNLayer',
        'Upscale3DLayer'
        ]


        
# cuDNN based 3d conv layer with support for fractionally strided convolutions.       
class Conv3dDNNLayer(Layer):
    def __init__(self, input_layer, num_filters, filter_size,
            strides=(1,1,1),
            border_mode=None,
            W=lasagne.init.Normal(std=0.001), # usually 0.01
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad=None,
            flip_filters=True,
            **kwargs):
        """
        input_shape: (frames, height, width)
        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """
        super(Conv3dDNNLayer, self).__init__(input_layer, **kwargs)

        # TODO note that lasagne allows 'untied' biases, the same shape
        # as the input filters.
        self.num_filters = num_filters
        self.filter_size = filter_size
        if strides is None:
            self.strides = (1,1,1)
        else:
            self.strides = tuple(strides)
        self.flip_filters = flip_filters
        
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and 'pad'. To avoid ambiguity, please specify only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0, 0)
        elif border_mode is not None:
            self.border_mode = border_mode
            if border_mode == 'valid':
                self.pad = (0, 0, 0)
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] -1, self.filter_size[2] - 1)
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case is probably not worth supporting.
                self.pad = ((self.filter_size[0]) // 2,
                            (self.filter_size[1]) // 2,
                            (self.filter_size[2]) // 2)
            else:
                raise RuntimeError("Unsupported border_mode for Conv3dLayer: %s" % border_mode)
        else:
            self.pad = tuple(pad)

        self.W = self.add_param(W, self.get_W_shape(), name='W')
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_filters,), name='b', regularizable=False)
        

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0], self.filter_size[1], self.filter_size[2])

    def get_output_shape_for(self, input_shape):
        """ input is bct01
        """
        batch_size = input_shape[0]
        volume_shape = np.asarray(input_shape[-3:]).astype(np.float32)
        filter_size = np.asarray(self.filter_size).astype(np.float32)
        pad = np.asarray(self.pad).astype(np.float32)
        strides = np.asarray(self.strides).astype(np.float32)
        # TODO check this is right. also it depends on border_mode
        # this assumes strides = 1
        #out_dim = (video_shape-(2*np.floor(kernel_shape/2.))).astype(np.int32)
        #out_dim = ( (volume_shape-filter_size) + 1).astype(np.int32)
        
        if any([s<1.0 for s in self.strides]):
            strides = np.asarray([int(1.0/s) for s in self.strides]).astype(np.float32)
            out_dim = ( (volume_shape -1)*strides + filter_size - 2*pad).astype(np.int32)
        else:
            strides = np.asarray(self.strides).astype(np.float32)
            out_dim = ( (volume_shape + 2*pad - filter_size) // strides + 1 ).astype(np.int32)
        return (batch_size, self.num_filters, out_dim[0], out_dim[1], out_dim[2])

    def get_output_for(self, input, *args, **kwargs):
        
        conv_mode = 'conv' if self.flip_filters else 'cross'

        
        # Fractionally strided convolutions
        if any([s<1.0 for s in self.strides]):
            subsample=tuple([int(1.0/s) for s in self.strides])

            img_shape = list(self.output_shape)
            if img_shape[0] is None:
                img_shape[0] = input.shape[0]
            image = T.alloc(0.,*img_shape)
            base = dnn.dnn_conv3d(img = image,
                                    kerns = self.W.transpose(1,0,2,3,4),
                                    subsample = subsample,
                                    border_mode = self.pad,
                                    conv_mode = conv_mode
                                    )                      
            conved = T.grad(base.sum(), wrt = image, known_grads = {base: input})
                                    
        else:
            conved = dnn.dnn_conv3d(img = input,
                                    kerns = self.W,
                                    subsample = self.strides,
                                    border_mode = self.pad,
                                    conv_mode = conv_mode
                                    )

        
        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x', 'x')

        return self.nonlinearity(activation)

        
# Repeat upscale 3d layer       
class Upscale3DLayer(Layer):
    def __init__(self, incoming, scale_factor, **kwargs):
        super(Upscale3DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 3)

        if any([s<1 for s in self.scale_factor]):
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        if output_shape[4] is not None:
            output_shape[4] *= self.scale_factor[2]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b, c = self.scale_factor
        upscaled = input
        if c > 1:
            upscaled = T.extra_ops.repeat(upscaled, c, 4)
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
         
        return upscaled        


