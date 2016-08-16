###
# Discriminative Voxel-Based ConvNet Training Function
# A Brock, 2016.


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

#####################
# Training Functions#
#####################
#
# This function compiles all theano functions and returns
# two dicts containing the functions and theano variables.
#
def make_training_functions(cfg,model):
    
    
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
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    
    #####################################
    # Step 1: Compute full forward pass #
    ######################################
    
    # Get outputs
    y_hat = lasagne.layers.get_output(l_out,X) 

    
    # Get deterministic outputs for validation
    y_hat_deterministic = lasagne.layers.get_output(l_out,X,deterministic=True)

    
    #################################
    # Step 2: Define loss functions #
    #################################
    
    # L2 regularization for all params
    l2_all = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)

    # Classifier loss function
    classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat), y)), 'float32')
    
    # Classifier Error Rate
    classifier_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat,axis=1), y)), 'float32' )            
            
    # Regularized Loss 
    reg_loss = cfg['reg']*l2_all + classifier_loss
    

    # Get all network params
    params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Handle annealing rate cases
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))
    
    
    ##########################
    # Step 3: Define Updates #
    ##########################

    updates=lasagne.updates.nesterov_momentum(reg_loss,params,learning_rate=learning_rate)

    update_iter = theano.function([batch_index], [classifier_loss,classifier_error_rate],
            updates=updates, givens={
            X: X_shared[batch_slice],
            y:  T.cast( y_shared[batch_slice], 'int32')
        })
  
    tfuncs = {'update_iter':update_iter,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars, model

## Data augmentation function from Voxnet, which randomly translates
## and/or horizontally flips a chunk of data. Note that
def jitter_chunk(src, cfg,chunk_index):
    np.random.seed(chunk_index)
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst



# Main Function
def main(args):

    # Set random seed to ensure identical network initializations.
    # Note that cuDNN's convolutions are nondeterministic, so this
    # does not guarantee that two networks will behave identically.
    lasagne.random.set_rng(np.random.RandomState(1234))
    
    # Load config file
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
   
    # Get weights and metrics filename
    weights_fname =str(args.config_path)[:-3]+'.npz'
    
    metrics_fname = weights_fname[:-4]+'METRICS.jsonl'
    
    # Prepare logs
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(metrics_fname))
    mlog = metrics_logging.MetricsLogger(metrics_fname, reinitialize=(not args.resume))
    
    # Get model
    model = config_module.get_model()
    
    # Compile functions
    logging.info('Compiling theano functions...')
    tfuncs, tvars,model = make_training_functions(cfg,model)
    
    # Resume training if file exists and you turn on the resume tag
    if os.path.isfile(weights_fname) and args.resume:
        print('loading weights')
        metadata = checkpoints.load_weights(weights_fname, model['l_out'])
   
    # GPU Memory Info; currently not implemented, but you can potentially
    # use this information to monitor GPU memory useage.
    baseGPUmem = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024/1024  
    
    # Training loop
    logging.info('Training...')
    itr = 0
    
    # Load data and shuffle training examples. 
    # Note that this loads the entire dataset into RAM! If you don't
    # have a lot of RAM, consider only loading chunks of this at a time.
    
    x = np.load(args.data_path)['features']
    
    # Seed the shuffle
    np.random.seed(42)
    
    # Define shuffle indices
    index = np.random.permutation(len(x))
    
    # Shuffle inputs
    x = x[index]
    
    # Shuffle targets to match inputs
    y = np.load(args.data_path)['targets'][index]

    # Define size of chunk to be loaded into GPU memory
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    
    # Determine number of chunks
    num_chunks = int(math.ceil(len(y)/float(chunk_size)))
    
    # Get current learning rate
    new_lr = np.float32(tvars['learning_rate'].get_value())
    
    # Loop across training epochs!
    for epoch in xrange(cfg['max_epochs']):
        
        # Tic
        epoch_start_time = time.time()
       
       # Update Learning Rate
        if isinstance(cfg['learning_rate'], dict) and epoch > 0:
            if any(x==epoch for x in cfg['learning_rate'].keys()):
                lr = np.float32(tvars['learning_rate'].get_value())
                new_lr = cfg['learning_rate'][epoch]
                logging.info('Changing learning rate from {} to {}'.format(lr, new_lr))
                tvars['learning_rate'].set_value(np.float32(new_lr))
        if cfg['decay_rate'] and epoch > 0:
            lr = np.float32(tvars['learning_rate'].get_value())
            new_lr = lr*(1-cfg['decay_rate'])
            logging.info('Changing learning rate from {} to {}'.format(lr, new_lr))
            tvars['learning_rate'].set_value(np.float32(new_lr))         
        
        # Loop across chunks!
        for chunk_index in xrange(num_chunks):
        
            # Define upper index of chunk to load
            # If you start doing complicated things with data loading, consider 
            # wrapping all of this into its own little function.
            upper_range = min(len(y),(chunk_index+1)*chunk_size)

            # Get current chunk
            x_shared = np.asarray(x[chunk_index*chunk_size:upper_range,:,:,:,:],dtype=np.float32)
            y_shared = np.asarray(y[chunk_index*chunk_size:upper_range],dtype=np.float32)
            
            # Get repeatable seed to shuffle jittered and unjittered instances within chunk.
            # Note that this seed varies between chunks, but will be constant across epochs.
            np.random.seed(chunk_index)
            
            # Get shuffled chunk indices for a second round of shuffling
            indices = np.random.permutation(2*len(x_shared))
            
            # Get number of batches in this chunk
            num_batches = 2*len(x_shared)//cfg['batch_size']
            
            # Combine data with jittered data, then shuffle and change binary range from {0,1} to {-1,3}, then load into GPU memory.
            tvars['X_shared'].set_value(4.0 * np.append(x_shared,jitter_chunk(x_shared, cfg,chunk_index),axis=0)[indices]-1.0, borrow=True)
            tvars['y_shared'].set_value(np.append(y_shared,y_shared,axis=0)[indices], borrow=True)
            
            # Prepare loss values
            lvs, accs = [],[]
            
            # Loop across batches!
            for bi in xrange(num_batches):
                
                # Train!
                [classifier_loss,class_acc] = tfuncs['update_iter'](bi)
                
                # Record batch loss and accuracy
                lvs.append(classifier_loss)
                accs.append(class_acc)
                
                # Update iteration counter
                itr += 1
            
            # Average losses and accuracies across chunk
            [closs,c_acc] = [float(np.mean(lvs)),1.0-float(np.mean(accs))]

            # Report and log losses and accuracies
            logging.info('epoch: {0:^3d}, itr: {1:d}, c_loss: {2:.6f}, class_acc: {3:.5f}'.format(epoch, itr, closs, c_acc))
            mlog.log(epoch=epoch, itr=itr, closs=closs,c_acc=c_acc)

        # Every Nth epoch, save weights
        if not (epoch%cfg['checkpoint_every_nth']):
            checkpoints.save_weights(weights_fname, model['l_out'],
                                                {'itr': itr, 'ts': time.time(),
                                                'learning_rate': new_lr}) 
        


    logging.info('training done')




### TODO: Clean this up and add the necessary arguments to enable all of the options we want.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('data_path',type =Path, default = 'Discriminative/modelnet40_rot_train.npz')
    parser.add_argument('--resume',type=bool,default=False)
    args = parser.parse_args()
    main(args)
