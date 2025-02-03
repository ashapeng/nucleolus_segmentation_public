# nuclolus_segmentation
This repository is built based on the AllenCell segmenter(Classic Image Segmentation), is built with the affiliation with Weber Lab at McGill University.
The repository is built on Windows 10.

# Goals:
1. Segments the nucleolus with markers of granular component (GC).

2. Characterizes features of nucleoli: size, number, and shape.

3. Analyzes the correlation between different nucleolar proteins within the nucleolar mask.

# Installation
Packages are implemented in Python 3.9
## Step 1. Installation aicssegmenation

[aics-segmentation](https://github.com/AllenCell/aics-segmentation/tree/main)

[Install Instruction](https://github.com/AllenCell/aics-segmentation/blob/main/README.md)

## Step 2. Install nucleolus_segmentation_public
- Clone the repository from Github
  cd C:\Projects
  git clone https://github.com/ashapeng/nucleolus_segmentation_public.git
  
- Or you can download ZIP save it to the project folder

Note 1: Check the installer.txt file for essential packages to run this repository.

To install all packages, in anaconda prompt
```bash
cd C:\Projects\nucleolus_segmentation_public
pip install -r installer.txt
```

Note 2: Running view function (itkwidgets) may not work, follow [AllenCell's help]:
  * https://github.com/AllenCell/aics-segmentation/blob/main/README.md

  * You can use the [napari](https://napari.org/stable/tutorials/fundamentals/installation.html#napari-installation) as the alternative, this is more stable than itkwidgets in my hand.
    

## 3.Run this repository
- 3.1: Unzip test_image folder: this will be the data used for running this repository.

- 3.2. Run nucleolus_seg.ipynb: to test segmentation with one example image, then run batch mode to process folders

- 3-3. Run nucleolus_feature_descriptor.ipynb or intensity based analysis, after save nucleolar mask in 3.2
