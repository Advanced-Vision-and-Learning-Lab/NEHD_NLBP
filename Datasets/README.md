# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow the following instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

## FashionMNIST [[`BibTeX`](#CitingFashionMNIST)]


The FashionMNIST dataset is automatically downloaded to the "Datasets" folder within the root directory upon selecting the FashionMNIST dataset.

## <a name="CitingFashionMNIST"></a>Citing FashionMNIST

If you use the FashionMNIST dataset, please cite the following reference using the following entry.

**Plain Text:**

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Retrieved August 28, 2017, from arXiv preprint server (cs.LG/1708.07747).

**BibTex:**
```
@online{xiao2017/online,
  author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  date         = {2017-08-28},
  year         = {2017},
  eprintclass  = {cs.LG},
  eprinttype   = {arXiv},
  eprint       = {cs.LG/1708.07747},
}
```

##  Plant Root Minirhizotron Imagery (PRMI) [[`BibTeX`](#CitingPRMI)]

Please download the [`PRMI dataset`](https://gatorsense.github.io/PRMI/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `PRMI`
3. The structure of the `PRMI` folder is as follows:
```
└── root dir
    ├── val   // Validation data.
        ├── images   // Input images.
        ├── labels_image_gt   // Metadata for each image.
        ├── masks_pixel_gt   // Pixel label ground truth masks.
    ├── train // Training data.
        ├── images   // Input images.
        ├── labels_image_gt   // Metadata for each image.
        ├── masks_pixel_gt   // Pixel label ground truth masks.
    ├── test  // Test data. 
        ├── images   // Input images.
        ├── labels_image_gt   // Metadata for each image.
        ├── masks_pixel_gt   // Pixel label ground truth masks.
```
## <a name="CitingPRMI"></a>Citing PRMI

If you use the PRMI dataset, please cite the following reference using the following entry.

**Plain Text:**

W. Xu, G. Yu, Y. Cui, R. Gloaguen, A. Zare, J. Bonnette, J. Reyes-Cabrera, A. Rajurkar, D. Rowland, R. Matamala, 
J. Jastrow, T. Juenger, and F. Fritschi. “PRMI: A Dataset of Minirhizotron Images for Diverse Plant Root Study.” 
In AI for Agriculture and Food Systems (AIAFS) Workshops at the AAAI conference on artificial intelligence. 
February, 2022.

**BibTex:**
```
@misc{xu2022prmi,
      title={PRMI: A Dataset of Minirhizotron Images for Diverse Plant Root Study}, 
      author={Weihuang Xu and Guohao Yu and Yiming Cui and Romain Gloaguen and Alina Zare and Jason Bonnette 
      and Joel Reyes-Cabrera and Ashish Rajurkar and Diane Rowland and Roser Matamala and Julie D. Jastrow 
      and Thomas E. Juenger and Felix B. Fritschi},
      year={2022},
      eprint={2201.08002},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## BloodMNIST [[`BibTeX`](#CitingBloodMNIST)]


The BloodMNIST dataset is automatically downloaded to the "Datasets" folder within the root directory upon selecting the BloodMNIST dataset.

## <a name="CitingBloodMNIST"></a>Citing BloodMNIST

If you use the BloodMNIST dataset, please cite the following reference using the following entry.

**Plain Text:**

J. Yang, R. Shi, and B. Ni, “Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis,” in IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021, pp. 191–195.

Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., & Rodellar, J. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data in Brief, 30, 105474.

**BibTex:**
```
@inproceedings{medmnistv1,
    title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
    author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
    booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
    pages={191--195},
    year={2021}
}

@article{bloodmnist,
    title = {A dataset of microscopic peripheral blood cell images for development of automatic recognition systems},
    author = {Andrea Acevedo and Anna Merino and Santiago Alférez and Ángel Molina and Laura Boldú and José Rodellar},
    journal = {Data in Brief},
    volume = {30},
    pages = {105474},
    year = {2020},
}
```
