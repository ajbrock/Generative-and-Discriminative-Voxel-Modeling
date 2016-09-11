## Voxel-Based ConvNet Ensembling Function
# A Brock
# This code evaluates a model's output on every example of the modelnet40 test set, using 24 rotations.
# You may need to modify the binary range of this function to fit the model you're currently evaluating.

import argparse
import imp
import time
import logging
import math
import os

import numpy as np
from path import Path
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as sbcuda
import lasagne

# import sys
# sys.path.insert(0, 'C:\Users\Andy\Generative-and-Discriminative-Voxel-Modeling')
from utils import checkpoints,metrics_logging

from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define the training functions
def make_training_functions(cfg,model):
    
    
    # Inputs
    X = T.TensorType('float32', [False]*5)('X')    

    # Y for test classification
    y = T.TensorType('int32', [False]*1)('y')    
    
    # Output layer
    l_out = model['l_out']

    # Batch Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    test_batch_slice = slice(batch_index*24, (batch_index+1)*24)
    test_batch_i = batch_index
    
    # Network Output
    y_hat_deterministic = lasagne.layers.get_output(l_out,X,deterministic=True)

    # Sum every 24 examples
    raw_pred = T.sum(y_hat_deterministic,axis=0)
    pred = T.argmax(raw_pred)
    
    # Error rate
    classifier_test_error_rate = T.cast( T.mean( T.neq(pred, T.mean(y,dtype='int32'))), 'float32' )

    # Shared Variables
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    # Test function
        
    test_error_fn = theano.function([batch_index], [classifier_test_error_rate,pred,raw_pred], givens={
            X: X_shared[test_batch_slice],
            y:  T.cast( y_shared[test_batch_slice], 'int32')      
        })    
    tfuncs = {
             'test_function':test_error_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index
            }
    return tfuncs, tvars, model

def main(args):
    
    # Load model
    lasagne.random.set_rng(np.random.RandomState(0))
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
   
    weights_fname =str(args.config_path)[:-3]+'.npz'

    model = config_module.get_model()
    print('Compiling theano functions...')
    tfuncs, tvars,model = make_training_functions(cfg,model)
    metadata = checkpoints.load_weights(weights_fname, model['l_out'])
    best_acc = metadata['best_acc'] if 'best_acc' in metadata else 0
    print('best acc = '+str(best_acc))

    print('Testing...')
    
    # Get Test Data 
    xt = np.asarray(np.load('modelnet40_rot24_test.npz')['features'],dtype=np.float32)
    yt = np.asarray(np.load('modelnet40_rot24_test.npz')['targets'],dtype=np.float32)
   
    n_rotations = 24  
    confusion_matrix = np.zeros((40,40),dtype=np.int)
    num_test_batches = int(math.ceil(float(len(xt))/float(n_rotations)))
    test_chunk_size = n_rotations*cfg['batches_per_chunk']
    num_test_chunks=int(math.ceil(float(len(xt))/test_chunk_size))
  
    test_class_error = []
    pred_array = []
    test_itr=0
    
    # Evaluate on test set
    for chunk_index in xrange(num_test_chunks):
        upper_range = min(len(yt),(chunk_index+1)*test_chunk_size) # 
        x_shared = np.asarray(xt[chunk_index*test_chunk_size:upper_range,:,:,:,:],dtype=np.float32)
        y_shared = np.asarray(yt[chunk_index*test_chunk_size:upper_range],dtype=np.float32)
        num_batches = int(math.ceil(float(len(x_shared))/n_rotations))
        tvars['X_shared'].set_value(6.0 * x_shared-1.0, borrow=True)
        tvars['y_shared'].set_value(y_shared, borrow=True)
        lvs, accs = [],[]        
        for bi in xrange(num_batches):
            test_itr+=1
            print(test_itr)
            [batch_test_class_error,confusion,raw_pred] = tfuncs['test_function'](bi) # Get the test       
            test_class_error.append(batch_test_class_error)
            pred_array.append(raw_pred)
            # print(confusion)
            # confusion_matrix+=confusion
            # confusion_matrix[confusion,int(yt[cfg['n_rotations']*test_itr])]+=1
    
    # print(confusion_matrix)
    
    # Save outputs to csv files.
    np.savetxt(str(args.config_path)[:-3]+'.csv', np.asarray(pred_array), delimiter=",")
    t_class_error = 1-float(np.mean(test_class_error))
    print('Test error is: ' + str(t_class_error))




### TODO: Clean this up and add the necessary arguments to enable all of the options we want.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    args = parser.parse_args()
    main(args)
