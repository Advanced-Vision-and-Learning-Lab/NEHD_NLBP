# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
def Parameters(args,learn_hist=True,learn_edge_kernels=True,feature_init=True,
               learn_transform=True,dilation=1, mask_size=[3,3]):
    
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    #Location to store trained models
    #Always add slash (/) after folder name
    folder = args.folder
    
    #Mode for evaluation: 
    # 'config' (parameter and initialization settings)
    # 'kernel' (spatial resolution)
    # 'dilation' (spatial resolution without increasing number of params)
    mode = args.mode
    
    #Feature to be learned: EHD and LBP
    feature = args.feature
    
    #If LBP, select baseline for comparison
    LBP_method = args.LBP_method
    
    #Set mode to True for "Reconstruction" and False for "Classification"
    reconstruction = args.reconstruction
    
    #Flag to use histogram model or handcrafted features
    # Set to True to use histogram layer and False to use EHD/LBP + linear classifier model 
    histogram = args.histogram
    
    #Flags to learn histogram parameters (bins/centers) and spatial masks
    learn_hist = learn_hist
    learn_edge_kernels = learn_edge_kernels
    
    #Select aggregation type for layer: 'Local' or 'GAP'. 
    #Recommended is RBF (implements histogram function in paper)
    aggregation_type = 'Local'
    
    #Select dataset. Set to number of desired texture dataset
    data_selection = args.data_selection
    Dataset_names = {1: 'Fashion_MNIST',2: 'PRMI',
                     3:'BloodMNIST'} 
    
    #EHD and LBP parameters
    #mask_size - Convolution kernel size for edge responses
    #threshold - Threshold for no edge orientation
    #angle_res - angle resolution for masks rotations
    #Set whether to use sum (unnormalized count) or average pooling (normalized count)
    # (default: sum pooling)
    #normalize - normalize kernel values
    #window_size - Binning count window size
    #stride for count
    #R - radius of neighborhood for LBP
    #P - number of neighbors to consider for LBP
    #feature_init: Set to True if feature should be initialized to original handcrafted feature
    mask_size = args.kernel_size
    window_size = args.window_size
    angle_res = args.angle_res
    if 'EHD' in feature:
      angle_res = int(360/(mask_size[0]**2-1))
    normalize_count = True
    normalize_kernel = True #Need to be normalized for histogram layer (maybe b/c of hist initialization)
    threshold =  1/int(360/angle_res) #10e-3 #1/int(360/angle_res) #.9
    R = args.R
    P = args.P
    stride = args.stride
    dilation = args.dilation
    feature_init = feature_init
    learn_transform = learn_transform
    
    
    #Parallelize model (run on mulitple GPUs)
    Parallelize_model = args.parallelize_model
    
    #Number of bins for histogram layer. Recommended values are the number of 
    #different angle resolutions used (e.g., 3x3, 45 degrees, 8 orientations) or LBP (user choice).
    if feature == 'LBP':
        numBins = args.numBins
        out_channels = numBins
    else:
        numBins = int(360/angle_res)
        out_channels = numBins + 1
        
    parallel = False
    
    #Set learning rate for model
    #Recommended values are .001 and .01
    lr = args.lr
    
    #Set momentum for SGD optimizer. 
    #Recommended value is .9 (used in paper)
    alpha = .9
    
    #Parameters of Histogram Layer
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image 
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = 0
    
    #Apply rotation to test set (did not use in paper)
    #Set rotation to True to add rotation, False if no rotation (used in paper)
    #Recommend values are between 0 and 25 degrees
    #Can use to test robustness of model to rotation transformation
    rotation = False
    degrees = 25
    
    #Set whether to enforce sum to one constraint across bins (default: True)
    #Needed for EHD feature (softmax approximation of argmax)
    normalize_bins = True
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 1000
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs, 
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
   
    #if reconstruction, these settings are the original EHD/LBP feature, no need to 
    #Run multiple epochs
    if reconstruction and not(learn_hist) and not(learn_edge_kernels) and feature_init and not(learn_transform):
        num_epochs = 1
    else:
        num_epochs = args.num_epochs #30 for classification, 15 for reconsruction
        
    #Resize the image before center crop. Recommended values for resize is 256 (used in paper), 384,
    #and 512 (from http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)
    #Center crop size is recommended to be 256.
    #For MNIST and FashionMNIST, keep orginal 28 x 28
    resize_size = args.resize_size
    center_size = args.center_size
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = True
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = args.num_workers
    
    #Flag for TSNE visuals, set to True to create TSNE visual of features
    #Set to false to not generate TSNE visuals
    #Separate TSNE will visualize histogram and GAP features separately
    #If set to True, TSNE of histogram and GAP features will be created
    #Number of images to view for TSNE (defaults to all training imgs unless
    #value is less than total training images).
    TSNE_visual = True
    Separate_TSNE = False
    Num_TSNE_images = 100
    
    #Visualization parameters for figures
    fig_size = 16
    font_size = 30

    # Fusion
    fusion_method = args.fusion_method
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    Data_dirs = {'Fashion_MNIST': './Datasets/',   
                 'PRMI': './Datasets/PRMI',
                 'BloodMNIST': './Datasets/BloodMNIST'}
    
    Model_names = {'Fashion_MNIST': 'Neural_EHD',
                   'PRMI': 'Neural_EHD',
                   'BloodMNIST': 'Neural_EHD'}
    
    num_classes = {'Fashion_MNIST': 10,
                   'PRMI': 4,
                   'BloodMNIST': 8}
    
    Class_names = {'Fashion_MNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                                     'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag',
                                     'Ankle boot'],
                    'PRMI': ['cotton', 'papaya', 'sunflower', 'switchgrass'],
                    'BloodMNIST': [0, 1, 2, 3, 4, 5, 6, 7]}    
                 
    Splits = {'Fashion_MNIST': 3,
              'PRMI': 5,
              'BloodMNIST' : 3}
    
    Sizes = {'Fashion_MNIST': 28,
             'PRMI': center_size, 
             'BloodMNIST': center_size}    
    
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    
    Hist_model_name = 'N{}_Scale_{}_Dilate_{}'.format(feature,mask_size,dilation)
    base_model_name = 'Scale_{}_Dilate_{}'.format(mask_size,dilation)

# Adjust the number of channels depending on the dataset and fusion method  
    if (fusion_method is None and ("Derma_MNIST" in Dataset or "BloodMNIST" in Dataset)): 
        in_channels = 3
    elif ((fusion_method is None and ("MNIST" not in Dataset))): 
        in_channels = 3 
    else: 
        in_channels = 1

###############    
    if reconstruction:
        folder = folder + 'Reconstruction/'
    else:
        folder = folder + 'Classification/'
        
    if learn_hist and not(learn_edge_kernels): #Only update histogram layer
        params_settings = 'Hist'
    elif not(learn_hist) and learn_edge_kernels: #Only update spatial kernels
        params_settings = 'Kernels'
    elif learn_hist and learn_edge_kernels: #Update all params
        params_settings = 'L_All'
    else: #Base feature
        params_settings = 'F_All'

    
    #Return dictionary of parameters
    Network_parameters = {'save_results': save_results,'folder': folder, 
                          'histogram': histogram,'Dataset': Dataset, 'data_dir': data_dir,
                          'num_workers': num_workers,
                          'lr': lr,'momentum': alpha, 'step_size': step_size,
                          'gamma': gamma, 'batch_size' : batch_size, 
                          'num_epochs': num_epochs, 'resize_size': resize_size, 
                          'center_size': center_size, 'padding': padding, 
                          'mask_size': mask_size,'in_channels': in_channels, 
                          'out_channels': out_channels,'normalize_count': normalize_count, 
                          'normalize_bins': normalize_bins,'numBins': numBins,
                          'Model_names': Model_names, 'num_classes': num_classes, 
                          'Splits': Splits,'hist_model': Hist_model_name,
                          'pin_memory': pin_memory,'degrees': degrees, 
                          'rotation': rotation, 'aggregation_type': aggregation_type,
                          'window_size': window_size,'angle_res': angle_res, 
                          'threshold': threshold,'stride': stride,'reconstruction': reconstruction,
                          'feature_init': feature_init, 'learn_transform': learn_transform,
                          'parallel': parallel,'TSNE_visual': TSNE_visual,
                          'Separate_TSNE': Separate_TSNE, 'Parallelize_model': Parallelize_model,
                          'Num_TSNE_images': Num_TSNE_images,'fig_size': fig_size,
                          'font_size': font_size, 'normalize_kernel': normalize_kernel,
                          'feature': feature, 'Sizes': Sizes, 'learn_hist': learn_hist,
                          'learn_edge_kernels': learn_edge_kernels,
                          'params_settings': params_settings, 'Class_names': Class_names,
                          'dilation': dilation, 'base_model_name': base_model_name,
                          'mean': None, 'std': None, 'mode': mode, 'P': P, 'R': R,
                          'LBP_method': LBP_method, 'fusion_method': fusion_method}
    
    return Network_parameters

