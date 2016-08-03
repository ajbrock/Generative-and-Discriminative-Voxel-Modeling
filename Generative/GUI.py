###
# Voxel-Based Variational Autoencoder GUI
# A Brock, 2016


import imp
import argparse
import logging
from path import Path
# import sys
# sys.path.insert(0, 'C:\Users\Andy\Generative-and-Discriminative-Voxel-Modeling')
from utils import checkpoints, npytar

import numpy as np
import theano
import theano.tensor as T
import lasagne



import vtk
import numpy as np

#####################
# Training Functions#
#####################
#
# This function compiles all theano functions and returns
# two dicts containing the functions and theano variables.
#
def make_test_functions(cfg, model):

    # Latent variables (ENCODER)
    l_Z = model['l_latents']
    # Outputs (DECODER)
    l_out = model['l_out']
    # print([x.name for x in lasagne.layers.get_all_layers(l_out)])
    
    # Variable indicating current batch
    batch_index = T.iscalar('batch_index')
    
    # Input Array
    X = T.TensorType('float32', [False]*5)('X')
    
    # Latent Space Vector
    Z = T.TensorType('float32', [False]*2)('Z')
    
    # Class Vector, for classification or augmenting the latent space vector
    y = T.TensorType('float32', [False]*2)('y')

    # Slice of current batch
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    
    # Get latents
    [Zz,sigma] = lasagne.layers.get_output([l_Z,model['l_ls']],3*X-1.0,deterministic=True)

    # Sample from the decoder network
    if cfg['cc']: # If augmenting with class-conditional variable:
        out = lasagne.layers.get_output(l_out,{model['l_Z_in']:Z, model['l_cc']:y})
        dout = lasagne.layers.get_output(l_out,{model['l_Z_in']:Z, model['l_cc']:y},deterministic=True)
    else:
        out = lasagne.layers.get_output(l_out,Z)
        dout = lasagne.layers.get_output(l_out,Z, deterministic=True)
    
    # Shared
    params = lasagne.layers.get_all_params(l_out)

    # Binary array containing predictions
    pred = T.ge(dout,0) 
    
    # Shared Input Array
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    
    # Shared Latents
    Z_shared = lasagne.utils.shared_empty(2, dtype='float32')
    
    # Shared Class Vector
    y_shared = lasagne.utils.shared_empty(2, dtype='float32')
      
    # Inference functions
    Z_fn = theano.function([X],Zz)
    sigma_fn = theano.function([X],sigma)
    
    # Compile functions
    if cfg['cc']:
        out_fn = theano.function([Z,y], out,on_unused_input='warn')
        dout_fn = theano.function([Z,y], dout,on_unused_input='warn')
        pred_fn = theano.function([Z,y], pred,on_unused_input='warn')    
    else:
        out_fn = theano.function([Z], out,on_unused_input='warn')
        dout_fn = theano.function([Z], dout,on_unused_input='warn')
        pred_fn = theano.function([Z], pred,on_unused_input='warn')
		
    # Prepare Dicts
    tfuncs = {'out': out_fn,
             'dout' : dout_fn,
             'pred' : pred_fn,
             'Zfn': Z_fn,
			 'sigma_fn':sigma_fn,
             
            }
    tvars = {'X' : X,
             'y' : y,
             'Z' : Z,
             'X_shared' : X_shared,
             'Z_shared' : Z_shared,
             'y_shared' : y_shared,
            }
    return tfuncs, tvars
 
 # Convenience function to load data from tar files.
def data_loader(cfg, fname):
    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    # chunk_size = 1
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.uint8)
    reader = npytar.NpyTarReader(fname)
    yc = np.zeros((chunk_size,cfg['n_classes']),dtype = np.float32)
    counter = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.uint8)
        yc[cix,(int(name.split('.')[0])-1)] = 1
        counter.append(int(name.split('.')[0])-1)
        if len(counter) == chunk_size:
            yield (xc, yc)
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
        yield (xc, yc)  

def main(args):
    
    
    # Load config file
    config_module = imp.load_source('config', args.config_path)
    
    # Get cfg dict
    cfg = config_module.cfg
    
    # Get model
    model = config_module.get_model(interp=True)
    
    # Compile functions
    print('Compiling theano functions...')
    tfuncs, tvars = make_test_functions(cfg, model)
    
    # Load model weights
    print('Loading weights from {}'.format(args.weights_fname))
    checkpoints.load_weights(args.weights_fname, model['l_latents'])
    checkpoints.load_weights(args.weights_fname, model['l_out'])
    
    # prepare data loader
    loader = (data_loader(cfg, args.testing_fname))

    
    # Load test set into local memory and get inferred latent values for each
    # element of the test set. Consider not doing this if you have limited RAM.
    print('Evaluating on test set')
    ygnd = np.empty([1,1,32,32,32],dtype=np.uint8)
    cc = np.empty([1,10],dtype=np.float32)
    counter = 0
    for x_shared, y_shared in loader:
        ygnd = np.append(ygnd,x_shared,axis=0)
        cc = np.append(cc,y_shared,axis=0) 

    # Get rid of first entries. Yeah, you could do this better by starting with a regular ole' python list,
    # appending inside the for loop and then just calling np.asarray, but I didn't really know any python
    # when I wrote this. Sue me.*
    #
    # *Please don't sue me
    ygnd = np.delete(ygnd,0,axis=0)
    cc = np.delete(cc,0,axis=0)
    
    print('Test set evaluation complete, render time!')
    
    # Total number of models loaded and encoded
    num_instances = len(ygnd)-1;
    
    # Seed the rng for repeatable operation
    np.random.seed(1)
    
    # Index of shuffled data
    display_ix = np.random.choice(len(ygnd), num_instances,replace=False)

    # Resolution: Number of blocks per side on each voxel. Setting this to less than 3
    # results in some ugly renderings, though it will be faster. Setting it higher makes it
    # prettier but can slow it down. With this setting, I can run interpolation in real-time on
    # a laptop with 16GB RAM and a GT730M. A setup with a real graphics card should be able to do
    # much, much more.
    v_res = 3
    
    # Dimensionality of the voxel grid
    dim = 32

    # Function to produce the data matrix for rendering. 
    def make_data_matrix(x,intensity):  
        return intensity*np.repeat(np.repeat(np.repeat(x[0][0],v_res,axis=0),v_res,axis=1),v_res,axis=2)

    # VTK Image Importer
    dataImporter = vtk.vtkImageImport()

    # Make the initial data matrix
   
    # initial random latent vector
    z_1 = np.random.randn(1,cfg['num_latents']).astype(np.float32)
    
    if cfg['cc']: # If augmenting with class-conditional vector
        cc_1 = np.zeros((1,10),dtype=np.float32)
        cc_1[0,np.random.randint(10)] = 1
        data_matrix = make_data_matrix(np.asarray(tfuncs['pred'](z_1,cc_1),dtype=np.uint8),128)
    else:
        data_matrix = make_data_matrix(np.asarray(tfuncs['pred'](z_1),dtype=np.uint8),128)
        
    # VTK bookkeeping stuff to prepare the central model
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, int(dim*v_res)-1, 0, int(dim*v_res)-1, 0, int(dim*v_res)-1)
    dataImporter.SetWholeExtent(0, int(dim*v_res)-1, 0, int(dim*v_res)-1, 0, int(dim*v_res)-1)
    
    # Prepare the interpolant endpoints
    
    # endpoint data importer
    edi = [0,0,0,0]
    
    # Endpoint data matrices    
    dm = [0,0,0,0]
    
    # Endpoint Latent values
    eZ = [0,0,0,0]

    # Endpoint sigmas
    eS = [0,0,0,0]
    
    # Endpoint class-conditional values
    eCC = [0,0,0,0]
    
    # Endpoint intensity values for colormapping
    eIs = [64,128,192,255]
    
    # VTK Bookkeeping stuff for interpolant endpoints
    for i in xrange(4):
        eZ[i] = tfuncs['Zfn'](ygnd[None,display_ix[i]])[0]
        eS[i] = tfuncs['sigma_fn'](ygnd[None,display_ix[i]])[0]
        eCC[i] = cc[None,display_ix[i]] # This may need changing to preserve shape? to be a 1x10 matrix instead of a 10x1?
        dm[i] = make_data_matrix(ygnd[None,display_ix[i]],eIs[i]).tostring()
        edi[i] = vtk.vtkImageImport()
        edi[i].CopyImportVoidPointer(dm[i], len(dm[i]))
        edi[i].SetDataScalarTypeToUnsignedChar()
        edi[i].SetNumberOfScalarComponents(1)
        edi[i].SetDataExtent(0, int(dim*v_res)-1, 0, int(dim*v_res)-1, 0, int(dim*v_res)-1)
        edi[i].SetWholeExtent(0, int(dim*v_res)-1, 0, int(dim*v_res)-1, 0, int(dim*v_res)-1)
  
    
        
    # Prepare color and transparency values
    colorFunc = vtk.vtkColorTransferFunction()
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(64,1)
    alphaChannelFunc.AddPoint(128,1.0)
    alphaChannelFunc.AddPoint(192,1.0)
    alphaChannelFunc.AddPoint(255,1.0)
    
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(64, 0.0, 0.4, 0.8)
    colorFunc.AddRGBPoint(128,0.8,0.0,0.0)
    colorFunc.AddRGBPoint(192,0.8,0.0,0.7)
    colorFunc.AddRGBPoint(255,0.0,0.8,0.0)

    # Prepare volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    volumeProperty.ShadeOn() # Keep this on unless you want everything to look terrible
    volumeProperty.SetInterpolationTypeToNearest()
    
    # Optional settings
    # volumeProperty.SetSpecular(0.2)
    # volumeProperty.SetAmbient(0.4)
    # volumeProperty.SetDiffuse(0.6)
     
    # More VTK Bookkeeping stuff.
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    
    # Specify the data and raycast methods for the rendered volumes.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    
    # Endpoint volumeMappers
    evm = [0,0,0,0]
    for i in xrange(4):
        evm[i] = vtk.vtkVolumeRayCastMapper()
        evm[i].SetVolumeRayCastFunction(compositeFunction)
        evm[i].SetInputConnection(edi[i].GetOutputPort())

     
    # Prepare the volume for the draggable model.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    volume.SetPosition([0, 0, 0])
    
    # Endpoint volumes
    ev = [0,0,0,0]
    vps = [[0,-150.0,-150.0],[0,150.0,-150.0],[0,-150.0,150.0],[0,150.0,150.0]]
    for i in xrange(4):
        ev[i] = vtk.vtkVolume()
        ev[i].SetMapper(evm[i])
        ev[i].SetProperty(volumeProperty)
        ev[i].SetPosition(vps[i])
        ev[i].DragableOff()
        ev[i].PickableOff()
    
    # Simple linear 2D Interpolation function for interpolation between latent values.
    # Feel free to extend this to use more advanced interpolation methods.
    def interp2d(x,y,z_):
        x1 = vps[0][1]+97.5
        x2 = vps[1][1]
        y1 = vps[0][2]+97.5
        y2 = vps[2][2]

        return ( ( (z_[0] * (x2-x) * (y2-y)) + 
                   (z_[1] * (x-x1) * (y2-y)) + 
                   (z_[2] * (x2-x) * (y-y1)) + 
                   (z_[3] * (x-x1) * (y-y1)) ) /
                   ( (x2-x1) * (y2-y1) ) )
    
    
    ### Interactor Style
    # This class defines the user interface, and how the system reacts to different user inputs.
    class MyInteractorStyle(vtk.vtkInteractorStyleSwitch):
     
        def __init__(self,parent=None):
            

            # Index indicating which models are currently selected for endpoints
            self.ix = 0
            
            # Togglable flag indicating if the center model is being dragged or not
            self.drag = 0;
            
            # Picker
            self.picker = vtk.vtkCellPicker()
            self.picker.SetTolerance(0.001)
            
            # Set up observers. These functions watch for specific user actions.
            self.SetCurrentStyleToTrackballActor()
            self.GetCurrentStyle().AddObserver("MiddleButtonReleaseEvent",self.middleButtonReleaseEvent)
            self.GetCurrentStyle().AddObserver("MouseMoveEvent",self.mouseMoveEvent)
            self.SetCurrentStyleToTrackballCamera()
            self.GetCurrentStyle().AddObserver("MiddleButtonPressEvent",self.middleButtonPressEvent)
            self.GetCurrentStyle().AddObserver("KeyPressEvent",self.keyPress)
            

        #
        def mouseMoveEvent(self,obj,event):
            # Re-render every time the user moves the mouse while clicking.
            # If you have a slow computer, consider changing this to be thresholded so as to only have a certain resolution
            # (i.e. to only change if the mouse moves a certain distance, rather than any move)
           
           if self.drag and self.picker.GetProp3D():
                
                # Move object. This is a rewrite of the raw move object code;
                # Normally you would do this with the built-in version of this function and extend 
                # it pythonically in the normal way, but the python bindings for VTK don't expose
                # this function in that way (it's all hard-coded in C++) so this is a simple
                # re-implementation that gives more control over how the object is moved.
                # Specifically, this constrains the draggable object to only move in-plane
                # and prevents it going out of bounds.
                
                center = self.picker.GetProp3D().GetCenter()
                display_center = [0,0,0]
                new_point = [0,0,0,0]
                old_point = [0,0,0,0]
                motion_vector = [0,0]
                event_pos = self.GetCurrentStyle().GetInteractor().GetEventPosition()
                last_event_pos = self.GetCurrentStyle().GetInteractor().GetLastEventPosition()
                self.ComputeWorldToDisplay(self.GetDefaultRenderer(),
                                            center[0],
                                            center[1],
                                            center[2],
                                            display_center)
                
                self.ComputeDisplayToWorld(self.GetDefaultRenderer(),
                                            event_pos[0],
                                            event_pos[1],
                                            display_center[2],
                                            new_point)
                
                self.ComputeDisplayToWorld(self.GetDefaultRenderer(),
                                            last_event_pos[0],
                                            last_event_pos[1],
                                            display_center[2],
                                            old_point)
                
                # Calculate the position change, making sure to confine the object to within the boundaries.
                # Consider finding a way to do this so that it depends on position of center instead of mouse
                new_point[1] = max(min(vps[1][1], new_point[1]), vps[0][1]+97.5)
                new_point[2] = max(min(vps[2][2], new_point[2]), vps[0][2]+97.5)
                old_point[1] = max(min(vps[1][1], self.picker.GetProp3D().GetCenter()[1]), vps[0][1]+97.5)
                old_point[2] = max(min(vps[2][2], self.picker.GetProp3D().GetCenter()[2]), vps[0][2]+97.5)
                
                # Increment the position
                self.picker.GetProp3D().AddPosition(0,new_point[1]-old_point[1],new_point[2]-old_point[2])
                
                # Update Data
                if cfg['cc']:
                    data_string = make_data_matrix(np.asarray(tfuncs['pred']([interp2d(volume.GetCenter()[1],
                                                                                       volume.GetCenter()[2],eZ)],
                                                                              interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eCC)),
                                                                                        dtype=np.uint8),
                                                                                        int(interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eIs))).tostring()
                else:
                    data_string = make_data_matrix(np.asarray(tfuncs['pred']([interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eZ)]),
                                                                                        dtype=np.uint8),
                                                                                        int(interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eIs))).tostring()
                # Update the renderer's pointer to the data so that the GUI has the updated data.
                dataImporter.CopyImportVoidPointer(data_string, len(data_string))
                # Update the window.
                self.GetCurrentStyle().GetInteractor().Render()
            
                return
  
        # If the middle button is used to select the center object, change to
        # dragging mode. Else, use it to pan the view.
        def middleButtonPressEvent(self,obj,event):
            
            clickPos = self.GetCurrentStyle().GetInteractor().GetEventPosition()
            self.picker.Pick(clickPos[0],clickPos[1],0,self.GetDefaultRenderer())
            
            if self.picker.GetProp3D():
                self.SetCurrentStyleToTrackballActor()
                self.drag=1
                
                # Optional: Add the ability to modify interpolant endpoints.
                # else:
                    # If we click an interpolant endpoint, change that endpoint somehow. 
                    # self.drag = 0
            
            
            self.GetCurrentStyle().OnMiddleButtonDown()     
            # self.GetCurrentStyle().HighlightProp3D(volume)
            return
        
        # When we release, change style from TrackballActor to TrackballCamera.
        def middleButtonReleaseEvent(self,obj,event):
            self.SetCurrentStyleToTrackballCamera()
            self.drag=0
            self.GetCurrentStyle().OnMiddleButtonUp()
            return
        
        # If the user presses an arrow key, swap out interpolant endpoints and re-render.
        # If the user hits space, sample randomly in the latent space and re-render.
        # If class-conditional vectors are enabled and the user hits 1-9, render a random
        # class-conditional object.
        def keyPress(self,obj,event):
            key=self.GetCurrentStyle().GetInteractor().GetKeySym()
            if key == 'Right':
                # Increment index of which models we're using, re-render all endpoints
                self.ix+=1
                for i in xrange(4):
                    eZ[i] = tfuncs['Zfn'](ygnd[None,display_ix[self.ix+i]])[0]
                    eS[i] = tfuncs['sigma_fn'](ygnd[None,display_ix[self.ix+i]])[0]
                    eCC[i] = cc[None,display_ix[self.ix+i]]
                    dm[i] = make_data_matrix(ygnd[None,display_ix[self.ix+i]],eIs[i]).tostring()
                    edi[i].CopyImportVoidPointer(dm[i], len(dm[i]))
                if cfg['cc']:
                    data_string = make_data_matrix(np.asarray(tfuncs['pred']([interp2d(volume.GetCenter()[1],
                                                                                       volume.GetCenter()[2],eZ)],
                                                                              interp2d(volume.GetCenter()[1],
                                                                                       volume.GetCenter()[2],eCC)),
                                                                                        dtype=np.uint8),
                                                                                        int(interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eIs))).tostring()
                else:                                                                        
                    data_string = make_data_matrix(np.asarray(tfuncs['pred']([interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eZ)]),
                                                                                        dtype=np.uint8),
                                                                                        int(interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eIs))).tostring()
                dataImporter.CopyImportVoidPointer(data_string, len(data_string))
                
            
            elif key == 'Left' and (self.ix > 0):
                self.ix-=1
                for i in xrange(4):
                    eZ[i] = tfuncs['Zfn'](ygnd[None,display_ix[self.ix+i]])[0]
                    eS[i] = tfuncs['sigma_fn'](ygnd[None,display_ix[self.ix+i]])[0]
                    eCC[i] = cc[None,display_ix[self.ix+i]]
                    dm[i] = make_data_matrix(ygnd[None,display_ix[self.ix+i]],eIs[i]).tostring()
                    edi[i].CopyImportVoidPointer(dm[i], len(dm[i]))
                if cfg['cc']:
                    data_string = make_data_matrix(np.asarray(tfuncs['pred']([interp2d(volume.GetCenter()[1],
                                                                                       volume.GetCenter()[2],eZ)],
                                                                              interp2d(volume.GetCenter()[1],
                                                                                       volume.GetCenter()[2],eCC)),
                                                                                        dtype=np.uint8),
                                                                                        int(interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eIs))).tostring()

                else:        
                    data_string = make_data_matrix(np.asarray(tfuncs['pred']([interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eZ)]),
                                                                                        dtype=np.uint8),
                                                                                        int(interp2d(volume.GetCenter()[1],
                                                                                        volume.GetCenter()[2],eIs))).tostring()
                dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            elif key == 'space':
                # Random Z, with optional weighting.
                Z_rand = 0.5*np.random.randn(1,cfg['num_latents']).astype(np.float32)
                
                # Optionally, sample using the interpolated sigmas as well.
                # Z_rand = np.square(np.exp(interp2d(volume.GetCenter()[1],volume.GetCenter()[2],eS))*np.random.randn(1,cfg['num_latents']).astype(np.float32))
               
                if cfg['cc']: # if class-conditional, take the class vector into account
                    cc_rand = np.zeros((1,10),dtype=np.float32)
                    cc_rand[0,np.random.randint(10)] = 1
                    data_string = make_data_matrix(np.asarray(tfuncs['pred'](interp2d(volume.GetCenter()[1],
                                                                                       volume.GetCenter()[2],eZ)+Z_rand,cc_rand),
                                                                            dtype=np.uint8),
                                                                            int(interp2d(volume.GetCenter()[1],
                                                                            volume.GetCenter()[2],eIs))).tostring()
                else:
                    data_string = make_data_matrix(np.asarray(tfuncs['pred'](Z_rand),
                                                                            dtype=np.uint8),
                                                                            int(interp2d(volume.GetCenter()[1],
                                                                            volume.GetCenter()[2],eIs))).tostring()
                dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            
                # Generate random Class-conditional Z
            elif 0<= int(float(key))<=9 and cfg['cc']:
                cc_rand = np.zeros((1,10),dtype=np.float32)
                cc_rand[0,int(float(key))] = 5
                Z_rand = np.square(np.exp(interp2d(volume.GetCenter()[1],volume.GetCenter()[2],eS)))*np.random.randn(1,cfg['num_latents']).astype(np.float32)+interp2d(volume.GetCenter()[1],volume.GetCenter()[2],eZ)
                data_string = make_data_matrix(np.asarray(tfuncs['pred'](Z_rand,cc_rand),
                                                                            dtype=np.uint8),
                                                                            int(interp2d(volume.GetCenter()[1],
                                                                            volume.GetCenter()[2],eIs))).tostring()
                dataImporter.CopyImportVoidPointer(data_string, len(data_string))                                                            
                # print(key)
                
            # Render and pass event on
            self.GetCurrentStyle().GetInteractor().Render()   
            self.GetCurrentStyle().OnKeyPress()    
            return
            
    # Initialize the render window
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    # Initialize the render interactor
    renderInteractor = vtk.vtkRenderWindowInteractor()
    style = MyInteractorStyle()
    style.SetDefaultRenderer(renderer)
    renderInteractor.SetInteractorStyle(style)#volume_set=volume,Lvolume_set = Lvolume, Rvolume_set = Rvolume))
    renderInteractor.SetRenderWindow(renderWin)

    # Make boundary plane
    rgrid = vtk.vtkRectilinearGrid()
    rgrid.SetDimensions(1,2,2)
    xCoords = vtk.vtkFloatArray()
    xCoords.InsertNextValue(30)
    yCoords = vtk.vtkFloatArray()
    yCoords.InsertNextValue(vps[0][1]+97.5)
    yCoords.InsertNextValue(vps[1][1])
    rgrid.SetXCoordinates(xCoords)
    rgrid.SetYCoordinates(yCoords)
    rgrid.SetZCoordinates(yCoords)
    plane = vtk.vtkRectilinearGridGeometryFilter()
    plane.SetInputData(rgrid) 
    rgridMapper = vtk.vtkPolyDataMapper()
    rgridMapper.SetInputConnection(plane.GetOutputPort())
    wireActor = vtk.vtkActor()
    wireActor.SetMapper(rgridMapper)
    wireActor.GetProperty().SetRepresentationToWireframe()
    wireActor.GetProperty().SetColor(0, 0, 0)
    wireActor.PickableOff()
    wireActor.DragableOff()
    
    # Add model, endpoints, and boundary plane to renderer
    renderer.AddActor(wireActor)
    for i in xrange(4):
        renderer.AddVolume(ev[i])
    renderer.AddVolume(volume) 

    # set background to white. Optionally change it to a fun color, like "Lifeblood of the Untenderized."    
    renderer.SetBackground(1.0,1.0,1.0)

    # Set initial window size. You can drag the window to change size, but keep in mind that
    # the larger the window, the slower this thing runs. On my laptop, I get little spikes
    # of lag when I run this on full screen, though a more graphics-cardy setup should do fine.
    renderWin.SetSize(400, 400)
     
    # Exit function
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)
     
    # Add exit function
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
    
    # initialize interactor
    renderInteractor.Initialize()

    # Start application!
    renderer.ResetCamera()
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(20)
    renderer.GetActiveCamera().Dolly(2.8)
    renderer.ResetCameraClippingRange()    
    renderWin.Render()
    renderInteractor.Start()
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('testing_fname', type=Path, default='shapenet10_test_nr.tar')
    parser.add_argument('weights_fname', type=Path, default='VAE.npz')
    args = parser.parse_args()
    main(args)
    