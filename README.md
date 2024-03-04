# Histogram Layers for Neural Handcrafted Features:
**Histogram Layers for Neural Handcrafted Features**

_Joshua Peeples, Salim Al Kharsa, Luke Saleh, and Alina Zare_

![Fig1_Workflow](https://github.com/Advanced-Vision-and-Learning-Lab/NEHD_NLBP/blob/main/Images/NEHD.png)
![Fig2_Workflow](https://github.com/Advanced-Vision-and-Learning-Lab/NEHD_NLBP/blob/main/Images/NLBP.png)

Note: If this code is used, cite it: Joshua Peeples, Salim Al Kharsa, Luke Saleh, and Alina Zare. 
(2024, Month Day). Advanced Vision and Learning Lab/Neural Handcrafted Features: Initial Release (Version v1.0). 
[`Zendo`](https://zenodo.org/record/8023959). https://doi.org/10.5281/zenodo.8023959
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8023959.svg)](https://doi.org/10.5281/zenodo.8023959)

[`IEEE Xplore (IGARRS)`](https://ieeexplore.ieee.org/document/10281981)

[`arXiv`](https://arxiv.org/abs/2306.04037)

[`BibTeX`](https://github.com/Peeples-Lab/XAI_Analysis#citing-quantitative-analysis-of-primary-attribution-explainable-artificial-intelligence-methods-for-remote-sensing-image-classification)


In this repository, we provide the paper and code for the " Histogram Layers for Neural Handcrafted Features"

## Installation Prerequisites

This code uses python, pytorch, and associated libraries listed in req.txt
Please use [`Pytorch's website`](https://pytorch.org/get-started/locally/) to download necessary packages.
Download other dependencies from req.txt using
```pip install -r req.txt```

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. 

## Main Functions

The runs using the following functions. 

1. Intialize model  

```model = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```

5. Histogram Methods

```histogram_layer = NLBPLayer(**Parameters)```
```histogram_layer = NEHDLayer(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/Advanced-Vision-and-Learning-Lab/NEHD_NLBP

└── root dir
	├── demo.py   //Run this. Main demo file.
	├── Demo_Parameters.py // Parameters file for demo.
	├── Prepare_Data.py  // Load data for demo file.
	├── View_Results.py // Run this after demo to view saved results.
  	├── req.txt // Contains the requirements 
        ├── Datasets
        	├── PRMIDataset.py // Returns Index for PRMI dataset
        	├── Pytorch_Datasets.py // Return Index for Pytorch datasets
	└── Utils  //utility functions
		├── Base_Model.py // Returns the linear model passed through
		├── Compute_EHD.py // Returns the EHD algorithm layer
		├── Compute_FDR.py // Returns the FDR score
		├── Compute_LBP.py // Returns the LBP algorithm layer
		├── Compute_Sizes.py // Returns the output size
		├── Confusion_mats.py // Returns the confusion matrices visual
		├── Crisp_Histogram_visuals.py // Returns Histogram visuals
		├── Generate_Learning_Curves.py // Returns the learning curves both loss and accuracy
		├── Generate_Plots.py // Returns the plots used in reconstruction /* verify */
		├── Generate_TSNE_visual.py // Returns the TSNE plots for test data
		├── Histogram_Model.py // Returns the Histogram Network
		├── NEHD.py // Returns the NEHD layer implementation
		├── NLBP.py // Returns the NLBP layer implementation
		├── Network_functions.py // Contains the functions called in main
		├── RBFHistogramPooling.py // Returns the Histogram Layer
		├── Save_Results.py // Saves the results into the directory
		├── pytorchtools.py // Contains early stopping functionality
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2024 J. Peeples, Salim Al Kharsa, Luke Saleh, and Alina Zare. All rights reserved.

## <a name="CitingHistogramFeatures"></a>Citing Histogram Layers for Neural Handcrafted Features

If you use the code, please cite the following 
reference using the following entry.

**Plain Text:**

J.Peeples S.Al Kharsa L.Saleh and A.Zare, "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification,"  in 2023 IEEE International Geoscience and Remote Sensing Symposium IGARSS, pp. 950-953. IEEE, 2023

**BibTex:**
```
@inproceedings{peeples2024handcrafted,
  title={Histogram Layers for Neural Handcrafted Features},
  author={Peeples, Joshua Al Kharsa, Salim Saleh, Luke and Zare, Alina.},
  booktitle={TBD},
  pages={TBD},
  year={2024},
  organization={IEEE}
}

```
