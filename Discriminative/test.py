###
# Discriminative Voxel-Based ConvNet Training Function
# A Brock, 2016


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
from utils import checkpoints, metrics_logging
from collections import OrderedDict
import matplotlib


# Define the testing functions
def make_testing_functions(cfg,model):
 
    # Input Array
    X = T.TensorType('float32', [False]*5)('X')  
    
    # Shared Variable for input array
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    # Class Vector
    y = T.TensorType('int32', [False]*1)('y')

    # Shared Variable for class vector
    y_shared = lasagne.utils.shared_empty(1, dtype='float32') 
    
    # Output layer
    l_out = model['l_out']

    # Batch Parameters
    batch_index = T.iscalar('batch_index')
    test_batch_slice = slice(batch_index*cfg['n_rotations'], (batch_index+1)*cfg['n_rotations'])
    test_batch_i = batch_index

    # Get output
    y_hat_deterministic = lasagne.layers.get_output(l_out,X,deterministic=True)
 
    # Average across rotation examples
    pred = T.argmax(T.sum(y_hat_deterministic,axis=0))

    # Get error rate
    classifier_test_error_rate = T.cast( T.mean( T.neq(pred, T.mean(y,dtype='int32'))), 'float32' )

    # Compile Functions 
    test_error_fn = theano.function([batch_index], [classifier_test_error_rate,pred], givens={
            X: X_shared[test_batch_slice],
            y:  T.cast( y_shared[test_batch_slice], 'int32')      
        })    
    tfuncs = {'test_function':test_error_fn}
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
            }
    return tfuncs, tvars, model

# Main Function
def main(args):

    # Load config module
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
   
    # Find weights file
    weights_fname =str(args.config_path)[:-3]+'.npz'

    # Get Model
    model = config_module.get_model()
    
    # Compile functions
    print('Compiling theano functions...')
    tfuncs, tvars,model = make_testing_functions(cfg,model)
    
    # Load weights
    metadata = checkpoints.load_weights(weights_fname, model['l_out'])
    
    # Check if previous best accuracy is in metadata from previous tests
    best_acc = metadata['best_acc'] if 'best_acc' in metadata else 0
    print('best accuracy = '+str(best_acc))


    print('Testing...')
    itr = 0
    
    # Load testing data into memory
    xt = np.asarray(np.load(args.data_path)['features'],dtype=np.float32)
    yt = np.asarray(np.load(args.data_path)['targets'],dtype=np.float32)

    # Get number of rotations to average across. If you want this to be different from
    # the number of rotations specified in the config file, make sure to change the 
    # indices of the test_batch_slice variable above, as well as which data file
    # you're reading from.
    n_rotations = cfg['n_rotations']
    
    # Confusion Matrix: Currently not implemented as it doesn't print prettily,
    # but easy to add in by uncommenting some of the commented lines in the loop.
    confusion_matrix = np.zeros((40,40),dtype=np.int)
    
    # Determine chunk size
    test_chunk_size = n_rotations*cfg['batches_per_chunk']
    
    # Determine number of chunks
    num_test_chunks=int(math.ceil(float(len(xt))/test_chunk_size))
    
    # Total number of test batches. Note that we're treating all the rotations
    # of a single instance as a single batch. There's definitely a more efficient
    # way to do this, and you'll want to be more efficient if you implement this in 
    # a validation loop, but testing should be infrequent enough that the extra few minutes
    # this costs is irrelevant.
    num_test_batches = int(math.ceil(float(len(xt))/float(n_rotations)))
    
    # Prepare test error  
    test_class_error = []
    
    # Initialize test iteration counter
    test_itr=0
    
    # Loop across chunks!
    for chunk_index in xrange(num_test_chunks):
    
        # Define upper index of chunk
        upper_range = min(len(yt),(chunk_index+1)*test_chunk_size)
        
        # Get chunks
        x_shared = np.asarray(xt[chunk_index*test_chunk_size:upper_range,:,:,:,:],dtype=np.float32)
        y_shared = np.asarray(yt[chunk_index*test_chunk_size:upper_range],dtype=np.float32)
        
        # Get number of batches for this chunk
        num_batches = len(x_shared)//n_rotations
        
        # Prepare data
        tvars['X_shared'].set_value(4.0 * x_shared-1.0, borrow=True)
        tvars['y_shared'].set_value(y_shared, borrow=True)
        
        # Loop across batches!
        for bi in xrange(num_batches):
        
            # Increment test iteration counter
            test_itr+=1
            
            # Test!
            [batch_test_class_error,pred] = tfuncs['test_function'](bi)
            
            # Record test results
            test_class_error.append(batch_test_class_error)
            
            # Optionally, update the confusion matrix
            # confusion_matrix[pred,int(yt[cfg['n_rotations']*test_itr])]+=1
    
    # Optionally, print confusion matrix
    # print(confusion_matrix)        
    
    # Get total accuracy
    t_class_error = 1-float(np.mean(test_class_error))
    print('Test accuracy is: ' + str(t_class_error))
    
    # Optionally save best accuracy
    # if t_class_error>best_acc:
            # best_acc = t_class_error
            # checkpoints.save_weights(weights_fname, model['l_out'],
                                            # {'best_acc':best_acc}) 




### TODO: Clean this up and add the necessary arguments to enable all of the options we want.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('data_path',type =Path, default = 'datasets/modelnet40_rot_test.npz')
    args = parser.parse_args()
    main(args)
