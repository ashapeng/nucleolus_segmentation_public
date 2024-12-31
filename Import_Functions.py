from numpy import load
import os
import numpy as np
import re
from skimage import io
from typing import List

#define a function to import .npy files
def import_npy(path):
    #make sure the path to the file exists
    if not os.path.exists(path):
        print('File does not exist')
        return None
    #load the data
    data = load(path)
    return data


def import_imgs(input_dir: str,image_name: str, is_mask: bool = False):
    
    """
    Parameters:
    ----------
    raw_dir: str
        directory to folders having folders of cells containing raw image, nucleus mask, background mask
    image_name: list
        need to include extension

    Returns: 
        img: nDarray
    """
    img = io.imread(os.path.join(input_dir,image_name))
    if is_mask:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32)
    return img
    


