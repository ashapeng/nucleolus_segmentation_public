# C. elegans_larva_nuclolus_segmentation based on AllenCell segmenter
#### this algorithm is structured with Baljyot Parmar's input and help

## Step 1: Installation aicssegmenation

https://github.com/AllenCell/aics-segmentation/tree/main

follow this instruction: https://github.com/AllenCell/aics-segmentation/blob/main/README.md

## Step 2: download weberLab_workshop3_nucleolus_segmentation

Note 1: unzip test_image folder

Note 2: Running view function may not work, follow AllenCell's help:
  * https://github.com/AllenCell/aics-segmentation/blob/main/README.md
  * We use the packge itkwidgets for visualizaiotn within jupyter notebook. Currently, we find version 0.14.0 has slightly better performance in visualizing segmentation results. If you find this viwer keeps crashing in your browser, try pip uninstall itkwidgets and then pip install itkwidgets==0.12.2. For JupyterLab users, version >= 0.17.1 is needed.

## Step 3: Run segmentation
### Step 3.1: Run GC_seg.ipynb: to test segmentation with one example image, then batch mode
### Step 3.2: Run GC shape descriptor or intensity based analysis, after save GC mask in Step 3.1
