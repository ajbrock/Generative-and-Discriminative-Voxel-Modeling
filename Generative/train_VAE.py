###
# Introspective Autoencoder Main training Function
# A Brock, 2016


import argparse
import imp
import time
import logging
# import sys
# sys.path.insert(0, 'C:\Users\Andy\Generative-and-Discriminative-Voxel-Modeling')

import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne

from utils import checkpoints, npytar, metrics_logging

from collections import OrderedDict
import matplotlib
matplotlib.use('Agg') # Turn this off if you want to display plots on your own computer or have X11 forwarding set up.
import matplotlib.pyplot as plt


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
    
    # Class Vector, for classification or augmenting the latent space vector
    y = T.TensorType('float32', [False]*2)('y')
    
    # Shared variable for input array
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    
    # Shared variable for class vector
    y_shared = lasagne.utils.shared_empty(2, dtype='float32')

    
    # Input layer
    l_in = model['l_in']
    
    # Output layer
    l_out = model['l_out']
    
    # Latent Layer
    l_latents = model['l_latents']
    
    # Latent Means
    l_mu = model['l_mu']
    
    # Log-sigmas
    l_ls = model['l_ls']
    
    # Classifier
    l_classifier = model['l_classifier']
    
    # Class-conditional latents
    l_cc = model['l_cc']
    
    # Decoder Layers, including final output layer
    l_decoder = lasagne.layers.get_all_layers(l_out)[len(lasagne.layers.get_all_layers(l_latents)):]
    

    # Batch Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    
    #####################################
    # Step 1: Compute full forward pass #
    #####################################
    #
    # Note that calling get_output() builds a new graph each time.
    
    
    # Get outputs
    outputs = lasagne.layers.get_output([l_out]+[l_mu]+[l_ls]+[l_classifier]+lasagne.layers.get_all_layers(l_classifier),
                                        {l_in:X, model['l_cc']:y}) # Consider swapping l_classifier in for l_latents
    
    # Get the reconstruction
    X_hat = outputs[0]
    
    # Get latent means
    Z_mu = outputs[1]
    
    # Get latent logsigmas
    Z_ls = outputs[2]
    
    # Get classification guesses
    y_hat = outputs[3]
    
    # Get the outputs of the encoder layers, given the training input
    g_X = outputs[5:]
    
    # Get the outputs of the feature layers of the encoder given the reconstruction
    g_X_hat = lasagne.layers.get_output(lasagne.layers.get_all_layers(l_classifier)[1:],lasagne.nonlinearities.tanh(X_hat))
    
    # Get testing outputs
    [X_hat_deterministic,latent_values,y_hat_deterministic] = lasagne.layers.get_output([l_out,l_latents,l_classifier],
                                                                                        {l_in:X, model['l_cc']:y},deterministic=True)
    # Latent values at a given 
    # latent_values = lasagne.layers.get_output(l_latents,deterministic=True)
    # For classification
    # class_prediction = softmax_out = T.nnet.softmax(g_X[-1])
    
    #################################
    # Step 2: Define loss functions #
    #################################
    
    # L2 normalization for all params
    l2_all = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)
            
    # Weighted binary cross-entropy for use in voxel loss. Allows weighting of false positives relative to false negatives.
    # Nominally set to strongly penalize false negatives
    def weighted_binary_crossentropy(output,target):
        return -(98.0*target * T.log(output) + 2.0*(1.0 - target) * T.log(1.0 - output))/100.0
        
    # Voxel-Wise Reconstruction Loss 
    # Note that the output values are clipped to prevent the BCE from evaluating log(0).
    voxel_loss = T.cast(T.mean(weighted_binary_crossentropy(T.clip(lasagne.nonlinearities.sigmoid( X_hat ), 1e-7, 1.0 - 1e-7), X)),'float32')
   
    # KL Divergence from isotropic gaussian prior
    kl_div = -0.5 * T.mean(1 + 2*Z_ls - T.sqr(Z_mu) - T.exp(2 * Z_ls))

    
    # Compute classification loss if augmenting with a classification objective
    if cfg['discriminative']:
        print('discriminating')
        classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat), y)), 'float32')
        classifier_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat,axis=1), T.argmax(y,axis=1)) ), 'float32' )
        classifier_test_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat_deterministic,axis=1), T.argmax(y,axis=1))), 'float32' )
        
        # Sum the reconstruction loss, the regularization term, the KL divergence over the prior, and the classifier loss.
        # Optionally ignore the kl divergence term.
        reg_voxel_loss = voxel_loss + cfg['reg']*l2_all +classifier_loss+kl_div if cfg['kl_div'] else voxel_loss + cfg['reg']*l2_all +classifier_loss
    # If not, ignore classifier
    else:
        classifier_loss = None
        classifier_error_rate = None
        classifier_test_error_rate = None
        # Sum the reconstruction loss, the regularization term, and the KL divergence over the prior.
        # Optionally ignore the kl divergence term.
        reg_voxel_loss = voxel_loss + cfg['reg']*l2_all+kl_div if cfg['kl_div'] else voxel_loss + cfg['reg']*l2_all
    
    ##########################
    # Step 3: Define Updates #
    ##########################
    
    # Define learning rate in case of annealing or decay.
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))

    
    # All network params
    params = lasagne.layers.get_all_params(l_out,trainable=True)
    # Decoder params
    decoder_params = lasagne.layers.get_all_params(l_out,trainable=True)[len(lasagne.layers.get_all_params(l_latents,trainable=True)):]

    # Update dict
    updates = OrderedDict()
    
    # Reconstruction and Regularization SGD terms
    # Note that momentum (or a variant such as Adam) is added further down.
    voxel_grads = lasagne.updates.get_or_compute_grads(reg_voxel_loss,params)
    for param,grad in zip(params,voxel_grads):
        updates[param] = param - learning_rate * grad
    
    # Feature SGD Terms (AKA Introspective SGD Terms)
    # Note that momentum (or a variant such as Adam) is added further down. 
    # Optionally add scale term to weight deeper layers more heavily.
    if cfg['introspect']:
        # To scale weights differently, add /sum(xrange(1,len(g_X_hat)-1))
        # Also (i+1) to scale weights
        feature_loss = T.cast(T.mean([T.mean(lasagne.objectives.squared_error(g_X[i],g_X_hat[i])) for i in xrange(0,len(g_X_hat)-2)]),'float32')
        feature_grads = lasagne.updates.get_or_compute_grads(feature_loss,decoder_params)
        for param,grad in zip(decoder_params,feature_grads):
            updates[param] += - learning_rate * grad
    else:
        feature_loss = None
        
    # Apply nesterov momentum to all updates.
    updates = lasagne.updates.apply_nesterov_momentum(updates,momentum=cfg['momentum'])    
       
 
    # Reconstruction Accuracy Term
    error_rate = T.cast( T.mean( T.neq(T.ge(X_hat,0), T.ge(X,0))), 'float32' ) 
    
    # Test Reconstruction Accuracy
    test_error_rate = T.cast( T.mean( T.neq(T.ge(X_hat_deterministic,0), T.ge(X,0))), 'float32' )
    
    # Test Reconstruction True Positives
    true_positives = T.cast(T.mean(T.eq(T.ge(X_hat_deterministic,0), T.ge(X,0.5))*T.ge(X,0.5))/T.mean(T.ge(X,0.5)),'float32')
    
    # Test Reconstruction True Negatives
    true_negatives = T.cast(T.mean(T.eq(T.ge(X_hat_deterministic,0), T.ge(X,0.5))*T.lt(X,0.5))/T.mean(T.lt(X,0.5)),'float32')

    # List comprehension to define which outputs are available during training
    update_outs = [x for x in [voxel_loss, 
            feature_loss,
            classifier_loss,
            kl_div,
            classifier_error_rate,
            error_rate] if x is not None]
    
    # Training function
    update_iter = theano.function([batch_index],update_outs,
            updates=updates, givens={
            X: X_shared[batch_slice],
            y: y_shared[batch_slice]
        },on_unused_input='warn' )

    # List comprehension to define which outputs are available during testing
    test_outs =  [x for x in [test_error_rate,
                        classifier_test_error_rate,
                        latent_values,true_positives,true_negatives] if x is not None]
    # Test function
    test_error_fn = theano.function([batch_index], 
            test_outs, givens={
            X: X_shared[batch_slice],
            y: y_shared[batch_slice]           
        },on_unused_input='warn' )  
    
    # Dictionary of theano functions
    tfuncs = {'update_iter':update_iter,
             'test_function':test_error_fn,
            }
    # Dictionary of theano variables        
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars

## Data augmentation function from Voxnet, which randomly translates
## and/or horizontally flips a chunk of data.
def jitter_chunk(src, cfg):
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
    
    
## Data loading function, originally from VoxNet.
def data_loader(cfg, fname):

    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']//2
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = np.zeros((chunk_size,cfg['n_classes']),dtype = np.float32)
    counter = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc[cix,(int(name.split('.')[0])-1)] = 1
        counter.append(int(name.split('.')[0])-1)
        if len(counter) == chunk_size:
            indices = np.random.permutation(2*len(xc))
            yield (3.0 * np.append(xc,jitter_chunk(xc, cfg),axis=0)[indices] - 1.0, np.append(yc,yc,axis=0)[indices])
            counter = []
            yc.fill(0)
            xc.fill(0)
    if len(counter) > 0:
        # pad to nearest multiple of batch_size
        if len(counter)%cfg['batch_size'] != 0:
            new_size = int(np.ceil(len(counter)/float(cfg['batch_size'])))*cfg['batch_size']
            xc = xc[:new_size]
            xc[len(counter):] = xc[:(new_size-len(counter))]
            yc = yc[:new_size]
            yc[len(counter):] = yc[:(new_size-len(counter))]
            counter = counter + counter[:(new_size-len(counter))]        
        indices = np.random.permutation(2*len(xc))
        yield (3.0 * np.append(xc,jitter_chunk(xc, cfg),axis=0)[indices] - 1.0, np.append(yc,yc,axis=0)[indices])


 # Test data loading function, originally from VoxNet   
def test_data_loader(cfg,fname):
    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = np.zeros((chunk_size,cfg['n_classes']),dtype = np.float32)
    counter = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc[cix,(int(name.split('.')[0])-1)] = 1
        counter.append(int(name.split('.')[0])-1)
        if len(counter) == chunk_size:
            yield (3.0*xc-1.0, yc)
            counter = []
            yc.fill(0)
            xc.fill(0)
    if len(counter) > 0:
        # pad to nearest multiple of batch_size
        if len(counter)%cfg['batch_size'] != 0:
            new_size = int(np.ceil(len(counter)/float(cfg['batch_size'])))*cfg['batch_size']
            xc = xc[:new_size]
            xc[len(counter):] = xc[:(new_size-len(counter))]
            yc = yc[:new_size]
            yc[len(counter):] = yc[:(new_size-len(counter))]
            counter = counter + counter[:(new_size-len(counter))]
        yield (3.0*xc-1.0, yc)

# Main Function
def main(args):

    # Load config file
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
   
    # Define weights file name
    weights_fname = str(args.config_path)[:-3]+'.npz'
    
    # Define training metrics filename
    metrics_fname = weights_fname[:-4]+'METRICS.jsonl'
    
    # Prepare Logs
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(metrics_fname))
    mlog = metrics_logging.MetricsLogger(metrics_fname, reinitialize=True)
    
    # Get model and compile theano functions
    model = config_module.get_model()
    logging.info('Compiling theano functions...')
    tfuncs, tvars = make_training_functions(cfg,model)

    logging.info('Training...')
    
    # Iteration Counter. One iteration corresponds to one minibatch.
    itr = 0
    
    # Best true-positive rate
    best_tp = 0
    

    for epoch in xrange(cfg['max_epochs']):
        # Prepare data loader
        loader = (data_loader(cfg,args.train_file))
        
        # Update Learning Rate. Note that this version of the function does not support a decay rate;
        # See other training files in the discriminative section for this.
        if isinstance(cfg['learning_rate'], dict) and epoch > 0:
            if any(x==epoch for x in cfg['learning_rate'].keys()):
                lr = np.float32(tvars['learning_rate'].get_value())
                new_lr = cfg['learning_rate'][epoch]
                logging.info('Changing learning rate from {} to {}'.format(lr, new_lr))
                tvars['learning_rate'].set_value(np.float32(new_lr))
        
        # Initialize epoch-wise chunk counter
        iter_counter = 0;
        
        # Initialize Epoch-wise metrics
        vloss_e, floss_e, closs_e, d_kl_e, c_acc_e, acc_e = 0, 0, 0, 0, 0, 0 

        # Train!
        for x_shared, y_shared in loader: # Loop across chunks
            
            # Increment chunk counter
            iter_counter+=1
            
            # Determine number of batches in this chunk; this should only vary from
            # cfg['batches_per_chunk'] if we're at the end of the dataset.
            num_batches = len(x_shared)//cfg['batch_size']
            
            # Load chunk into memory
            tvars['X_shared'].set_value(x_shared, borrow=True)
            tvars['y_shared'].set_value(y_shared, borrow=True)
            
            # Initialize Chunk-wise metrics
            voxel_lvs,feature_lvs,class_lvs,kl_divs,class_accs,accs = [],[],[],[],[],[]            
            
            for bi in xrange(num_batches): # Loop across batches within chunk
                # Update!
                results = tfuncs['update_iter'](bi)
                
                # Assign results
                # This could definitely be done more cleanly with a list comprehension.
                voxel_loss = results[0]
                feature_loss = results[1] if cfg['introspect'] else 0 
                classifier_loss = results[1+cfg['introspect']] if cfg['discriminative'] else 0
                kl_div = results[1+cfg['introspect']+cfg['discriminative']]
                class_acc = results[2+cfg['introspect']+cfg['discriminative']] if cfg['discriminative'] else 0
                acc = results[2+cfg['introspect']+2*cfg['discriminative']]
               
                # Append results to chunk-wise result list; these will be averaged later.
                voxel_lvs.append(voxel_loss)
                feature_lvs.append(feature_loss)
                class_lvs.append(classifier_loss)
                kl_divs.append(kl_div)
                class_accs.append(class_acc)
                accs.append(acc)

                # Increment batch counter
                itr += 1
                
            # Average metrics across chunk
            [vloss, floss,closs, d_kl,c_acc,acc] = [float(np.mean(voxel_lvs)), float(np.mean(feature_lvs)),
                                                    float(np.mean(class_lvs)), float(np.mean(kl_divs)),
                                                    1.0-float(np.mean(class_accs)), 1.0-float(np.mean(accs))]
            
            # Update epoch-wise metrics                                                 
            vloss_e, floss_e, closs_e, d_kl_e, c_acc_e, acc_e = [vloss_e+vloss, floss_e+floss, closs_e+closs, d_kl_e+d_kl, c_acc_e+c_acc, acc_e+acc] 
            
            # Report and Log chunk-wise metrics  
            logging.info('epoch: {}, itr: {}, v_loss: {}, f_loss: {}, c_loss: {}, D_kl: {}, class_acc: {}, acc: {}'.format(epoch, itr, vloss, floss,
                                                                                                                           closs, d_kl, c_acc, acc))
            mlog.log(epoch=epoch, itr=itr, vloss=vloss,floss=floss, acc=acc,d_kl=d_kl,c_acc=c_acc)
        
        # Average  metrics across epoch
        vloss_e, floss_e, closs_e, d_kl_e, c_acc_e, acc_e = [vloss_e/iter_counter, floss_e/iter_counter, 
                                                             closs_e/iter_counter, d_kl_e/iter_counter,
                                                             c_acc_e/iter_counter, acc_e/iter_counter]
        #  Report and log epoch-wise metrics                                                    
        logging.info('Training metrics, Epoch {}, v_loss: {}, f_loss: {}, c_loss: {}, D_kl: {}, class_acc: {}, acc: {}'.format(epoch, vloss_e, floss_e,closs_e,d_kl_e,c_acc_e,acc_e))
        mlog.log(epoch=epoch, vloss_e=vloss_e, floss_e=floss_e, closs_e=closs_e, d_kl_e=d_kl_e, c_acc_e=c_acc_e, acc_e=acc_e)
        
        # Every Nth epoch, save weights
        if not (epoch%cfg['checkpoint_every_nth']):
            checkpoints.save_weights(weights_fname, model['l_out'],
                                            {'itr': itr, 'ts': time.time()})

    
    # When training is complete, check test performance
            test_loader = test_data_loader(cfg,'shapenet10_test_nr.tar')
            logging.info('Examining performance on test set')
            
            # Initialize test metrics
            test_error,test_class_error,latent_values,tp,tn = [],[],[],[],[]
             
            # Initialize true class array for 2D manifold plots
            true_class = np.array([],dtype=np.int)
            
            for x_shared,y_shared in test_loader: # Loop across test chunks
                
                # Calculate number of batches
                num_batches = len(x_shared)//cfg['batch_size']
                
                # Load test chunk into memory
                tvars['X_shared'].set_value(x_shared, borrow=True)
                tvars['y_shared'].set_value(y_shared, borrow=True)
                
                # Update true class array for 2D Manifold Plots
                true_class = np.append(true_class,np.argmax(y_shared,axis=1))
                
                for bi in xrange(num_batches): # Loop across minibatches
                
                    # Get test results
                    test_results = tfuncs['test_function'](bi)
                    
                    # Assign test results
                    # This could be done more cleanly with a list comprehension
                    batch_test_error=test_results[0]
                    batch_test_class_error = test_results[1] if cfg['discriminative'] else 0
                    latents = test_results[1+cfg['discriminative']]
                    batch_tp = test_results[2+cfg['discriminative']]
                    batch_tn = test_results[3+cfg['discriminative']]
                    test_error.append(batch_test_error)
                    test_class_error.append(batch_test_class_error)
                    latent_values.append(latents)
                    tp.append(batch_tp)
                    tn.append(batch_tn)
                    
            # Average results        
            t_error = 1-float(np.mean(test_error))
            true_positives = float(np.mean(tp))
            true_negatives = float(np.mean(tn))
            t_class_error = 1-float(np.mean(test_class_error))
            Zs = np.asarray(latent_values,np.float32)        
            
            # Report and log results
            logging.info('Test Accuracy: {}, Classification Test Accuracy: {}, True Positives: {}, True Negatives: {}'.format(t_error,t_class_error,true_positives,true_negatives))
            mlog.log(test_error=t_error,t_class_error = t_class_error,true_positives=true_positives,true_negatives=true_negatives)

            # Optionally plot and save 2D manifold if using only 2 latent variables.
            if np.shape(Zs)[2]==2:
                Zs = np.reshape(Zs,(np.shape(Zs)[0]*np.shape(Zs)[1],1,2))
                ygnd = np.asarray(true_class,np.int)
                plt.scatter(Zs[:,0,0],Zs[:,0,1],s = 30, c=ygnd,alpha = 0.5)
                plt.savefig('figs/'+weights_fname[:-4]+str(epoch)+'.png')
                plt.clf()

    
    logging.info('training done')
    checkpoints.save_weights(weights_fname, model['l_out'],
                                    {'itr': itr, 'ts': time.time()})

### TODO: Clean this up and add the necessary arguments to enable all of the options we want.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('train_file', type=Path,default='shapenet10_train_nr.tar')
    parser.add_argument('test_file', type=Path, default = 'shapenet10_test_nr.tar')
    args = parser.parse_args()
    main(args)
