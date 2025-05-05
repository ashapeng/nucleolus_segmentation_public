# Nucleolus Segmentation

A specialized image segmentation tool for nucleolus analysis, developed during my time at Weber Lab at McGill University. This repository is built on the basis of AllenCell segmenter (Classic Image Segmentation).

# Goals:
1. Segments the nucleolus with markers of granular component (GC).

2. Characterizes features of nucleoli: size, number, and shape.

3. Analyzes the correlation between different nucleolar proteins within the nucleolar mask.

# Installation

## Prerequisites
- Python 3.9
- Anaconda (recommended)

## Step 1: Install aics-segmentation
1. Visit the [aics-segmentation repository](https://github.com/AllenCell/aics-segmentation/tree/main)
2. Follow the [official installation guide](https://github.com/AllenCell/aics-segmentation/blob/main/README.md)

## Step 2: Install nucleolus_segmentation_public
1. Clone the repository:
   ```bash
   cd C:\Projects
   git clone https://github.com/ashapeng/nucleolus_segmentation_public.git
   ```
   
   Alternatively, download the ZIP file and extract it to your project folder.

2. Install required packages:
   ```bash
   cd C:\Projects\nucleolus_segmentation_public
   pip install -r installer.txt
   ```

## Visualization Tools
For visualization, you have two options:
1. **itkwidgets**: Follow the [AllenCell's help guide](https://github.com/AllenCell/aics-segmentation/blob/main/README.md)
2. **napari** (recommended): More stable alternative. Installation instructions available at [napari.org](https://napari.org/stable/tutorials/fundamentals/installation.html#napari-installation)

## 3.Run this repository:
- 3.1: Unzip test_image folder: this will be the data used for running this repository.

- 3.2. Run nucleolus_seg.ipynb: to test segmentation with one example image, then run batch mode to process folders

- 3.3. Run nucleolus_feature_descriptor.ipynb or intensity based analysis, after save nucleolar mask in 3.2
