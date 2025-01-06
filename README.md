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

aics-segmentation: https://github.com/AllenCell/aics-segmentation/tree/main

Instruction: https://github.com/AllenCell/aics-segmentation/blob/main/README.md

## Step 2. Install nucleolus_segmentation_public
- Clone the repository from Github
  cd C:\Projects
  git clone https://github.com/ashapeng/nucleolus_segmentation_public.git
- Or you can download ZIP save it to the project folder
- Unzip test_image folder

Note 1: You don't need additional packages to run this repository

Note 2: Running view function may not work, follow AllenCell's help:
  * https://github.com/AllenCell/aics-segmentation/blob/main/README.md
  * You can use the napari as the alternative, this is more stable than itkwidgets in my hand
    https://napari.org/stable/tutorials/fundamentals/installation.html#napari-installation

## 3.Run segmentation
### 3-1.  Run GC_seg.ipynb: to test segmentation with one example image, then batch mode
### 3-2. Run GC shape descriptor or intensity based analysis, after save GC mask in Step 3-1
