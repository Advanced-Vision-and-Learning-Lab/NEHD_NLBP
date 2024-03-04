# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow the following instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

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
/* question in the paper we used 4 of the classes should I add an instruction saying to make that edit? */ 
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
