# package for image read, write, work on images as ndim array
# import some basic functions
import os
import re
import math
import glob

import numpy as np
from skimage import io
from typing import Dict, List

import matplotlib.pyplot as plt

##############################################################
# import self_defined functions for segmentation based ALL ###
##############################################################

# pre-segmentation image process on raw image
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization, suggest_normalization_param, 
    image_smoothing_gaussian_3d,image_smoothing_gaussian_slice_by_slice)
from scipy.ndimage import gaussian_filter

# segmentation core function
from skimage.filters import threshold_otsu,threshold_triangle
# spot segmentation core function
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

# import functions for local threshold
from skimage.measure import regionprops,label,marching_cubes,mesh_surface_area

# import function for post-segmentation image process
from skimage.morphology import remove_small_objects, remove_small_holes,binary_erosion,ball,disk,binary_dilation,binary_closing,binary_opening

##############################################################
##              core functions end                         ###
##############################################################

# function to separate connected spots
from skimage.segmentation import clear_border,watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import nucleolus_processing_imgs as npi

# write image
import imageio

#### function related to canny
# from skimage.util.dtype import dtype_limits
# from skimage._shared.filters import gaussian
# from skimage._shared.utils import _supported_float_type, check_nD
# from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear


def image_2d_seg(raw_img: np.array, nucleus: np.array, sigma_2d: float) -> np.array:
    '''
     ----------
    Parameters:
    -----------
    raw_img: nD array
        The raw image array z-stack  multichannel image
    nucleus_mask: nD array
        The one slice nucleus mask array
    sigma_2d: float
        sigma of gaussian filter used for smoothing
    
    Returns:
    --------
    2d_seg: nD array after substracting background
    '''

    # maximal projection
    max_projection = np.stack([np.max(raw_img[...,channel],axis=0) for channel in range(raw_img.shape[-1])],axis=2)
    
    # normalize maximal projection by min_max normalization
    normalized_max_projection = np.zeros_like(max_projection)
    for channel in range(max_projection.shape[-1]):
        normalized_max_projection[...,channel] = (max_projection[...,channel]-np.min(max_projection[...,channel]))/(np.max(max_projection[...,channel])-np.min(max_projection[...,channel]))
    
    # smooth each channel
    smoothed_2d = np.stack([gaussian_filter(normalized_max_projection[...,channel],sigma=sigma_2d,mode="nearest",truncate=3) for channel in range(normalized_max_projection.shape[-1])],axis=2)

    # regional segmentation in the nucleus to separate forground(nucleolus) and background(nuclear plasma)
    thresholded = np.zeros_like(smoothed_2d)
    ## only keep signal within the nucleus mask use it to calculate the threshold
    smoothed_in_nucleus = np.stack([np.where(nucleus[0,...]>0,smoothed_2d[...,i],0) for i in range(smoothed_2d.shape[-1])],axis=2)
    for i in range(smoothed_2d.shape[-1]):
        cutoff1 = threshold_otsu(smoothed_in_nucleus[...,i][smoothed_in_nucleus[...,i]>0])
        thresholded[...,i] = smoothed_2d[...,i]>cutoff1 

    # post segmentation processing
    post_seg = np.zeros_like(thresholded)
    for i in range(thresholded.shape[-1]):
        each = thresholded[...,i]
        opened = binary_opening(each,footprint=np.ones((3,3))).astype(bool)
        holed_filled = remove_small_holes(opened)
        
        # only keep the largest object
        labeled = label(holed_filled,connectivity=2)
        props = regionprops(labeled)
        # Step 3: Find the label of the largest object
        largest_label = np.argmax([prop.area for prop in props]) + 1
        post_seg[...,i] = labeled == largest_label
        post_seg[...,i][post_seg[...,i]>0] = 1
    return post_seg

def bg_subtraction(raw_img: np.array, bg_mask: np.array, clip: bool = False) -> np.ndarray:
    '''
    The order of background subtraction and normalization depends on the specific application and the nature of the images being analyzed. However, in general, it is usually better to perform background subtraction before normalization.
    Background subtraction is typically used to enhance the contrast between the foreground objects and the background in an image, making it easier to segment the objects. 
    This process involves subtracting the background signal from the image, leaving only the foreground objects. 
    Normalization, on the other hand, is a process of scaling the pixel values of an image to a standard range or distribution to account for variations in illumination or exposure.
    
    '''

    """
    choose the proper method to handle outliers and other extrem values in the image data, when your data
    may contain significant noise or artifacts, which can skew the results of traditional normalization methods.
    If you have negative intensity values after background subtraction, there are several ways to handle them before normalizing your images:

    Clip the negative values to zero: This means that you simply set any negative values to zero, effectively removing any negative values from your image. This is a common approach in many image processing applications, particularly if negative values are not meaningful for your particular analysis. In my image analysis since negative pixel values represent regions outside cells or regions dimmer that the chosen background and do not provide useful information about the image content, and that they can safely be discarded.

    Shift the intensity values: This means that you add a constant value to all intensity values, such that the minimum intensity value becomes zero. By shifting the intensity values to be non-negative, you can ensure that all intensity values are valid and can be used for further processing or analysis.

    Apply a log transform: A log transform can be used to transform the intensity values to a more linear scale, while preserving the relative differences between the values. This can be particularly useful if your data has a large dynamic range or if you want to highlight differences between low-intensity values.

    Use a different normalization method: There are many different normalization methods available, each with their own strengths and weaknesses. For example, min-max normalization scales the intensity values to a fixed range of values, such as [0, 1], which can ensure that all values are positive. Alternatively, z-score normalization scales the intensity values to have a mean of zero and a standard deviation of one.

    Ultimately, the best approach for handling negative intensity values will depend on the specific characteristics of your image data and the downstream analysis you plan to perform. It may be useful to experiment with different normalization techniques and compare their results to determine which approach works best for your application.
    """

    """
    ----------
    Parameters:
    -----------
    raw_img: nD array
        The raw image array having 3 channels
    bg_mask: 3D array
        The mask array for calculating bg_mask intensity
    
    Returns:
    --------
    bg_substracted: nD array after substracting background
    """
    # check if image is 3D/2D
    img_dimension = raw_img.ndim

    # Initialize output array
    bg_substracted = np.zeros_like(raw_img,dtype=raw_img.dtype)
    if img_dimension == 4:
        # Loop over channels and z-planes
        for channel in range(raw_img.shape[-1]):
            for z in range(bg_mask.shape[0]):
                # Check if there are any pixels to process in this z-plane
                if np.count_nonzero(bg_mask[z,:,:])>0:
                    # Extract background pixels from raw_img using mask from bg_mask
                    bg_raw = np.where(np.logical_and(bg_mask[z,:,:],raw_img[z,:,:,channel]),raw_img[z,:,:,channel],0)
                    
                    # Calculate mean background intensity
                    mean_bg = np.mean(bg_raw[bg_raw>0])
                    
                    # Subtract background from raw_img and clip negative values to zero
                    bg_substracted[z,:,:,channel] = raw_img[z,:,:,channel] - mean_bg

                    if clip:
                        bg_substracted[z,:,:,channel] = np.clip(bg_substracted[z,:,:,channel],a_min=0,a_max=None)
    elif img_dimension == 3:
        bg_mask_2d = np.max(bg_mask,axis=0)
        for channel in range(raw_img.shape[-1]):
            # Extract background pixels from raw_img using mask from bg_mask
            bg_raw = np.where(bg_mask_2d,raw_img[...,channel],0)
            # Calculate mean background intensity
            mean_bg = np.mean(bg_raw[bg_raw>0])
            # Subtract background from raw_img and clip negative values to zero
            bg_substracted[...,channel] = raw_img[...,channel] - mean_bg
            if clip:
                bg_substracted[...,channel] = np.clip(bg_substracted[...,channel],a_min=0,a_max=None)
    return bg_substracted

def min_max_norm(raw_img:np.array, suggest_norm: bool=False)-> np.ndarray:
    """
    Min-max normalization is a common technique used in image processing and other data analysis applications to scale the values of a variable to a fixed range of values. The goal of this normalization technique is to transform the data so that it falls within a specific range of values, typically [0, 1] or [-1, 1].

    The process of min-max normalization involves two steps:

    Find the minimum and maximum values of the variable: This involves calculating the minimum and maximum values of the variable across all observations in your dataset. For image processing, this would involve finding the minimum and maximum intensity values across all pixels in your image.

    Scale the variable to the desired range: Once you have found the minimum and maximum values of the variable, you can scale the values to the desired range using the following formula:

    X_norm = (X - X_min) / (X_max - X_min)

    In this formula, X represents the original value of the variable, X_min represents the minimum value of the variable, X_max represents the maximum value of the variable, and X_norm represents the normalized value of the variable. By applying this formula, you can transform the original values of the variable so that they fall within the desired range of values.
    """
    '''
    Parameters:
    -----------
    raw_img: 3D array
        
    suggest_norm: bool
        whether use suggested scaling for min-max normalization image by Allen segmenter
    
    Returns:
    --------
    normalized_data: 3D array
        images after normalization
    '''
    # Make a copy of the data
    raw = raw_img.copy()

    if suggest_norm:
        # Get suggested scaling parameters
        low_ratio,up_ratio=suggest_normalization_param(raw)
        intensity_scaling_param = [low_ratio, up_ratio]
    else:
        intensity_scaling_param = [0]
    
    # Normalize the data
    normalized_data = intensity_normalization(raw, scaling_param=intensity_scaling_param)

    return normalized_data


def gaussian_smooth_stack(data1:np.array,sigma:list):
    '''
    The sigma parameter controls the standard deviation of the Gaussian distribution, which determines the amount of smoothing applied to the image. The larger the sigma, the more the image is smoothed. The truncate parameter controls the size of the kernel, and it specifies how many standard deviations of the Gaussian distribution should be included in the kernel. By default, truncate is set to 4.0.

    The size of the kernel can be calculated using the formula:

    kernel_size = ceil(truncate * sigma * 2 + 1)

    where ceil is the ceiling function that rounds up to the nearest integer.

    For example, if sigma is 1.5 and truncate is 4.0, the kernel size will be:

    kernel_size = ceil(4.0 * 1.5 * 2 + 1) = ceil(12.0 + 1) = 13

    Therefore, the size of the kernel for ndi.gaussian_filter in this case will be 13x13.

    Note that the kernel size is always an odd integer to ensure that the center of the kernel is located on a pixel.
    '''
    '''
    Parameters:
    -----------
    data1: 3D array
        The data to be nomalized and smoothed, as [plane,row,column],single channel
    sigma_by_channel: Dictionary 
        a dictionary of sigma for guassian smoothing ordered as each channel
    Returns:
    --------
    smoothed_data: 3D array
        images after gaussian smoothing
    '''
    if len(sigma)>1:
        # 3d smooth
        smoothed_data = image_smoothing_gaussian_3d(data1,sigma=sigma,truncate_range=3.0)
    else:
        # gaussian smoothing slice-by-slice
        smoothed_data = image_smoothing_gaussian_slice_by_slice(data1,sigma=sigma[0],truncate_range=3.0)
    return smoothed_data

def global_otsu(
        data1:np.ndarray, data2:np.ndarray,global_thresh_method: str, 
        mini_size:float, local_adjust:float=0.98, extra_criteria: bool=False, 
        return_object: bool = False, keep_largest: bool=False):
    '''
    Use Allen segmenter Implementation of "Masked Object Thresholding" algorithm. Specifically, the
    algorithm is a hybrid thresholding method combining two levels of thresholds.
    The steps are [1] a global threshold is calculated, [2] extract each individual
    connected componet after applying the global threshold, [3] remove small objects,
    [4] within each remaining object, a local Otsu threshold is calculated and applied
    with an optional local threshold adjustment ratio (to make the segmentation more
    and less conservative). An extra check can be used in step [4], which requires the
    local Otsu threshold larger than 1/3 of global Otsu threhsold and otherwise this
    connected component is discarded.
    '''
    '''
    Parameters:
    -----------
    data1: 3D array
        The data that has been smoothed, as [plane,row,column],single channel
    data2: 3D array
        The array within which otsu threshold for data 1 is processed, a binary image
    global_thresh_method: str
        which method to use for calculating global threshold. Options include:
        "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
        "ave" refers the average of "triangle" threshold and "mean" threshold.
    mini_size: float
        the size filter for excluding small object before applying local threshold
    local_adjust: float 
        a ratio to apply on local threshold, default is 0.98
    extra_criteria: bool
        whether to use the extra check when doing local thresholding, default is False
    return_object: bool
        whether return the low level threshold
    keep_largest: bool
       whether only keep the largest mask
    
    Returns:
    --------
    segmented_data: 3D array
        images after otsu segmentation

    '''
    # segment images
    if global_thresh_method == "tri" or global_thresh_method == "triangle":
        th_low_level = threshold_triangle(data1)
    elif global_thresh_method == "med" or global_thresh_method == "median":
        th_low_level = np.percentile(data1, 50)
    elif global_thresh_method == "ave" or global_thresh_method == "ave_tri_med":
        global_tri = threshold_triangle(data1)
        global_median = np.percentile(data1, 50)
        th_low_level = (global_tri + global_median) / 2

    bw_low_level = data1 > th_low_level
    bw_low_level = remove_small_objects(
        bw_low_level, min_size=mini_size, connectivity=1)
    bw_low_level = binary_dilation(bw_low_level,footprint=ball(1))
    # set top and bottom slice as zero
    bw_low_level[0,:,:]=0
    bw_low_level[-1,:,:]=0

    # local otsu
    bw_high_level = np.zeros_like(bw_low_level)
    lab_low, num_obj = label(bw_low_level, return_num=True, connectivity=1)
    if extra_criteria:
        local_cutoff = 0.333 * threshold_otsu(data1[data2>0])
        for idx in range(num_obj):
            single_obj = lab_low == (idx + 1)
            local_otsu = threshold_otsu(data1[single_obj > 0])
            final_otsu = npi.round_up(local_otsu, npi.decimal_num(local_otsu,two_digit=True))
            #final_otsu = math.ceil(local_otsu)
            print("otsu:{},rouned otsu:{},ajusted otsu:{}".format(local_otsu,final_otsu,final_otsu * local_adjust))
            if local_otsu > local_cutoff:
                bw_high_level[
                    np.logical_and(
                        data1 > final_otsu * local_adjust, single_obj
                    )
                ] = 1
    else:
        for idx in range(num_obj):
            single_obj = lab_low == (idx + 1)
            local_otsu = threshold_otsu(data1[single_obj > 0])
            final_otsu = npi.round_up(local_otsu, npi.decimal_num(local_otsu,two_digit=True))
            #final_otsu = math.ceil(local_otsu)
            print("otsu:{},rouned otsu:{},ajusted otsu:{}".format(local_otsu,final_otsu,final_otsu * local_adjust))
            bw_high_level[
                np.logical_and(
                    data1 > final_otsu * local_adjust, single_obj
                )
            ] = 1

    # post segmentation process: 
    # remove small dispersed obj and fill holes in each slice
    global_mask = np.zeros_like(bw_high_level)
    for z in range(bw_high_level.shape[0]):
        global_mask[z,:,:] = binary_closing(bw_high_level[z,:,:],footprint=disk(2))
        global_mask[z,:,:] = remove_small_holes(global_mask[z,:,:].astype(bool), area_threshold=np.count_nonzero(global_mask[z,:,:]),connectivity=2)
        global_mask[z,:,:] = remove_small_objects(global_mask[z,:,:].astype(bool), min_size=30,connectivity=2)
    # set the top and bottom as zero
    global_mask[0,:,:]=0
    global_mask[-1,:,:]=0

    # only keep the largest label
    if keep_largest:
        labeled_mask,label_mask_num = label(global_mask,return_num=True,connectivity=1)
        print("number of mask is: ", label_mask_num)
        mask_id_size = {}
        for id in range(1, label_mask_num+1,1):
            as_id = labeled_mask==id
            vol = np.count_nonzero(as_id)
            mask_id_size["{}".format(id)]=vol
            print("mask spot volumn",vol)        
        max_mask_vol_id = int(max(mask_id_size,key=mask_id_size.get))
        segmented_data = np.logical_or(
            np.zeros_like(global_mask),labeled_mask==max_mask_vol_id
            )
    else:
        segmented_data = global_mask
    
    # dilate
    dilated_mask = np.zeros_like(segmented_data)
    for z in range(segmented_data.shape[0]):
        dilated_mask[z,:,:] = binary_dilation(segmented_data[z,:,:].astype(bool),footprint=disk(1))
        #
        dilated_mask[z,:,:] = binary_closing(dilated_mask[z,:,:].astype(bool),footprint=ndi.generate_binary_structure(2,2))
        dilated_mask[z,:,:] = remove_small_holes(dilated_mask[z,:,:].astype(bool), area_threshold=np.count_nonzero(dilated_mask[z,:,:]),connectivity=2)
    # set the top and bottom as zero
    dilated_mask[0,:,:]=0
    dilated_mask[-1,:,:]=0
    
    if return_object:
        return dilated_mask > 0, bw_low_level
    else:
        return dilated_mask > 0
    
def intensity_guided_mask_dilation(mask_3d:np.ndarray, mask_2d:np.ndarray, raw_img: np.ndarray, max_projected_img: np.ndarray, upper_limit:float, dilate_radius:int, return_mask:bool=False):
    '''
    Parameters:
    -----------
    mask_3d: 3D array
        The mask needed to be dilated
    mask_2d: array of 2D array
        The mask generated from 2D image, multiple channels
    raw_img: 3D array
        Raw image with channels
    max_projected_img: str
        Maximal projected image, multiple channels, have the same shape as mask_2d
    upper_limit: float
        A ratio to apply on the maximal intensity of each channel to ensure dilated mask include all "true" signal
    dilate_radius: int
        The maximal radius to dilate
    return_mask: bool
        whether return the dilated mask
    
    Returns:
    --------
    raw_in_dilated_mask: 3D array
        Raw image signal in the dilated mask
    dilated_mask: 3D array
    '''
    # get intensity of each channel in 2d mask
    max_in_2d_seg = np.stack([np.where(mask_2d[...,i]>0, max_projected_img[...,i],0) for i in range(max_projected_img.shape[-1])],axis=2)
    
    # calculate the upper limit for each channel
    top_red = upper_limit*np.max(max_in_2d_seg[...,0])
    top_green = upper_limit*np.max(max_in_2d_seg[...,1])

    # dilate the mask
    dilated_mask = mask_3d.copy()
    for radius in np.arange(1,dilate_radius,2):
        
        # invert dilated mask
        inverted_dilated_mask = np.where(dilated_mask>0,0,1)
        
        # dilate GC mask
        dilated_mask = binary_dilation(mask_3d,footprint = ball(radius))
        
        # get the newly dilated mask
        increment_seg = np.logical_and(inverted_dilated_mask,dilated_mask)

        # compare the maximal value in each increment seg to the x% threshold, if it is higher than x% then dilate, else don't
        red_in_increment_seg = np.where(increment_seg>0, raw_img[...,0], 0)
        red_max = np.max(red_in_increment_seg)

        green_in_increment_seg = np.where(increment_seg>0, raw_img[...,1], 0)
        green_max = np.max(green_in_increment_seg)

        if (red_max < top_red) and (green_max < top_green):
            dilated_mask = binary_dilation(mask_3d,footprint=ball(radius))
            break

    # extract intensity value in the dilated mask
    raw_in_mask = np.stack([np.where(dilated_mask>0,raw_img[...,i],0) for i in range(raw_img.shape[-1])],axis=3)

    if return_mask:
        return raw_in_mask, dilated_mask
    else:
        return raw_in_mask

def segment_spot(
        raw_img:np.ndarray, nucleus_mask:np.ndarray, structure_mask:np.ndarray,
        LoG_sigma:list, mini_size: float, 
        invert_raw: bool=False, show_param: bool=False,arbitray_cutoff:bool=False):
    
    """
    * edge detect with ndi.gaussian_laplace(LoG filter): applies a Gaussain filter first to remove noise and then applies a Laplacian filter to detect edges

    * in practice, an input image is noisy, the high-frequency components of the image can dominate the LoG filter's output, leading to spurious edge detections and other artifacts. To mitigate this problem, the input image is typically smoothed with a Gaussian filter before applying the LoG filter.

    * It is common practice to apply a Gaussian smoothing filter to an image before applying the Laplacian of Gaussian operator to enhance the edges and features in the image.
    
    * A common approach is to use a value of sigma for the Laplacian of Gaussian filter that is larger than the sigma value used for the Gaussian smoothing filter. This is because the Gaussian smoothing filter removes high-frequency details from the image, which can make it difficult to distinguish features in the Laplacian of Gaussian filter output. By using a larger sigma value for the Laplacian of Gaussian filter, you can enhance larger features in the image without being affected by the smoothing effect of the Gaussian filter.

    However, the choice of sigma value for the Laplacian of Gaussian filter ultimately depends on the specific characteristics of the image and the desired level of feature enhancement. It's a good practice to experiment with different values of sigma for the Laplacian of Gaussian filter to find the best value for your specific image and application. 

    *  it is possible that smoothing the input image with a Gaussian filter before applying the Laplacian filter can cause over-smoothing, which may result in loss of detail or blurring of edges.

    The degree of smoothing depends on the size of the Gaussian kernel used in the filter. A larger kernel size corresponds to a stronger smoothing effect, while a smaller kernel size provides less smoothing. If the kernel size is too large, the filter may blur important features and reduce the contrast between different parts of the image.

    Therefore, it is important to choose an appropriate kernel size for the Gaussian filter based on the characteristics of the input image and the specific application requirements. If the input image is already relatively smooth, applying a Gaussian filter with a large kernel size may lead to over-smoothing and reduce the edge detection accuracy. On the other hand, if the input image is very noisy, a larger kernel size may be necessary to reduce the noise level effectively.

    the LoG kernel size is automatically calculated by:
    kernel_size = int(4 * sigma + 1)
    """
    """
    -----------
    raw_img: 3D array
        Raw image
    nucleus_mask: 3D array
        The binary mask within which LoG cutoff is kept
    structure_mask: 3D array
        The mask generated by global thresholding in which keep LoG
    LoG_sigma: list
        sigma range for LoG: based on estimated spot size range
    mini_size: float
        the minimum size of spot
    invert_raw: bool=False
        whether invert data1 or not, for vacuole True
    
    Returns:
    --------
    data1: 3D array
        binary spot mask
    """

    # check if inverting image
    if invert_raw:
        spot = np.max(raw_img) - raw_img
    else:
        spot = raw_img.copy()
    
    # mask region to apply LoG filter
    nucleus_mask_eroded = binary_erosion(nucleus_mask,footprint=ball(2))
    nucleus_mask_eroded[0,:,:]=0
    nucleus_mask_eroded[-1,:,:]=0

    # set LoG cutoff
    param_spot = []
    # transfer data into LoG form
    for LoG in LoG_sigma:
        spot_temp = np.zeros_like(spot)
        for z in range(spot.shape[0]):
            spot_temp[z, :, :] = -1 * (LoG**2) * ndi.filters.gaussian_laplace(spot[z, :, :], LoG)
        
        # only keep LoG value within data2 to avoide bright "noise"
        LoG_in_mask = []
        for id in np.argwhere(nucleus_mask_eroded):
            LoG_in_mask.append(spot_temp[id[0],id[1],id[2]])
        
        # calculate cut_off & check mean of LoG with different region size
        cut_off = npi.round_up(
            np.percentile(LoG_in_mask,96), 
            npi.decimal_num(np.percentile(LoG_in_mask,96),two_digit=True))
        param_spot.append([LoG,cut_off])
    
    # set cutoff as a mean
    """cutoff_mean = npi.round_up(
        np.mean(np.array(param_spot),axis=0)[1],
        npi.decimal_num(np.mean(np.array(param_spot),axis=0)[1],two_digit=True))"""
    cutoff_mean = np.mean(np.array(param_spot),axis=0)[1]
    print("Use the mean cutoff value",cutoff_mean)
    updated_param =[ [param_spot[i][0],cutoff_mean] for i in range(len(param_spot))]
    if show_param:
        print("spot seg parameter is:",param_spot)
        print("spot seg updated parameter is:",updated_param)
    spot_by_LoG = dot_2d_slice_by_slice_wrapper(spot, updated_param)
    
    ###############################
    #  remove objects slice-slice##
    ###############################
    # remove object smaller than a connectivity=2 region slice by slice
    spot_opened =np.zeros_like(spot_by_LoG)
    for z in range(spot_by_LoG.shape[0]):
        spot_opened[z,:,:] = binary_opening(
            spot_by_LoG[z,:,:],footprint=ndi.generate_binary_structure(2,2))
        spot_opened[z,:,:] = remove_small_objects(
            spot_opened[z,:,:],min_size=10,connectivity=2).astype(np.uint8)
        spot_opened[z,:,:] = binary_closing(
            spot_opened[z,:,:],footprint=ndi.generate_binary_structure(2,2))

    # only keep objects within structure_mask
    spot_in_structure = np.where(np.logical_and(spot_opened,structure_mask),1,0)
    
    # remove objects that only appears in one plane
    spot_on_multi_z_slices, vac_num = label(spot_in_structure,return_num=True,connectivity=2)
    for i in range(1,vac_num+1):
        p,r,c = np.where(spot_on_multi_z_slices==i)
        if len(set(p))<=2:
            spot_on_multi_z_slices=np.where(spot_on_multi_z_slices==i,0,spot_on_multi_z_slices)
    spot_on_multi_z_slices[spot_on_multi_z_slices>0]=1

    # size thresholding: remove object smaller than mini_size
    spot_size_threshold = remove_small_objects(spot_on_multi_z_slices.astype(bool),min_size=mini_size,connectivity=2)
    
    spot_size_threshold[spot_size_threshold>0]=255
    
    return spot_size_threshold

def final_gc_holes(spot_mask:np.ndarray, global_mask:np.ndarray):
    '''
    Parameters:
    -----------
    spot_mask: 3D array
        segmented spot mask
    global_mask: 3D array
        segmented correspond global mask
    Returns:
    --------
    final_merged_gc: 3D array
        mask after combined
    hole_filled_final: 3D array
        mask after filling holes
    holes: 3D array
        holes in the final gc mask
    '''

    # get final GC mask
    final_merged = global_mask.copy()
    final_merged[spot_mask>0]=0

    # fill holes in final mask
    hole_filled_final = np.zeros_like(final_merged)
    for z in range(final_merged.shape[0]):
        hole_filled_final[z,:,:] = remove_small_holes(final_merged[z,:,:].astype(bool), area_threshold=np.count_nonzero(global_mask[z,:,:]),connectivity=2)
    

    final_merged=final_merged.astype(np.uint8)
    final_merged[final_merged>0]=255

    hole_filled_final=hole_filled_final.astype(np.uint8)
    hole_filled_final[hole_filled_final>0]=255

    return final_merged, hole_filled_final

def dilate_gc(data1:np.ndarray, radius:float):
    '''
    Parameters:
    -----------
    data1: 3D array
        gc mask holes filled
    radius: float
        the dilation radius
    Returns:
    --------
    dilated_mask: 3D array
    '''
    dilated_mask = binary_dilation(data1,footprint=ball(radius))
    dilated_mask[0,:,:]=0
    dilated_mask[-1,:,:]=0
    dilated_mask = dilated_mask.astype(np.uint8)
    dilated_mask[dilated_mask>0]=255
    return dilated_mask

################
# integrate segmentation step into one function
def gc_segment( raw_image:np.ndarray, nucleus_mask:np.ndarray, background_mask:np.ndarray, sigma: float, local_adjust_for_GC: float):
    '''
    Parameters:
    -----------
    raw_image: nD array
        The raw data as [plane,row,column, channel]
    sigma: float
        the smooth sigma value
        if apply 2d smooth slice by slice input 1 value in each list, if apply 3d smooth input 3 value as the order of [plane,row,column]
    local_adjust_for_GC: float
    
    Returns:
    --------
    fina_gc: 3D array, holes: 3D array, hole_filled_gc: 3D array
    '''

    # Make copies of input data
    raw_img = raw_image.copy()
    nucleus_mask = nucleus_mask.copy()
    bg_mask = background_mask.copy()

    # normalize images based on min-max normalization
    normalized_img = np.stack([min_max_norm(raw_img[:,:,:,i]) for i in range(raw_img.shape[-1])],axis=3)

    # 3d smooth raw LPD7 image
    gc_smoothed_final = gaussian_filter(normalized_img[...,2],sigma=sigma,mode="nearest",truncate=3)

    # otsu segment each channel as for ground and background
    # adjust local_adjust parameter to make the segmentation more and less
    gc_otsu = global_otsu(data1=gc_smoothed_final,data2=nucleus_mask,global_thresh_method="ave",mini_size=1000,local_adjust=local_adjust_for_GC,extra_criteria=False,keep_largest=True)

    # guassian laplace edge detection vacules in GC
    gc_dark_spot = segment_spot(normalized_img[...,2],nucleus_mask,gc_otsu,LoG_sigma=list(np.arange(2.5,4,0.25,dtype=float)),mini_size=30,invert_raw=True)

    # merge dark spot and gc global mask and only keep holes in final final_gc
    final_gc, hole_filled_gc = final_gc_holes(gc_dark_spot, gc_otsu)

    # save mask
    final_gc = final_gc.astype(np.uint8)
    final_gc[final_gc>0]=255

    gc_dark_spot = gc_dark_spot.astype(np.uint8)
    gc_dark_spot[gc_dark_spot>0]=255

    hole_filled_gc = hole_filled_gc.astype(np.uint8)
    hole_filled_gc[hole_filled_gc>0]=255

    return final_gc, gc_dark_spot, hole_filled_gc
################

def ball_confocol(radius_xy,radius_z, dtype=np.uint8):
    # generate 3d ball as footprint
    # based on the confocol scope in Weber lab
    # confocol voxel dimension: Z, Y, X = 0.2, 0.0796631, 0.0796631
    # at 100x X 1.4 NA Z, Y, X = 1, 2.5, 2.5
    n_xy = 2*radius_xy+1
    n_z = 2*radius_z+1
    Z,Y,X = np.mgrid[-radius_z:radius_z:n_z*1j,
                    -radius_xy:radius_xy:n_xy*1j,
                    -radius_xy:radius_xy:n_xy*1j]
    s = X**2 + Y**2 + Z**2
    return np.array(s<=radius_xy*radius_z,dtype=dtype)
###############

def shift_img (img:np.ndarray):
    from skimage.measure import regionprops
    # move object into a larger frame to get convex and bounding box
    img2 = img.copy()
    img2[img2>0]=1
    feature = regionprops(img2)
    new_frame = np.empty(shape=tuple([i *2 for i in img2.shape]),dtype=img2.dtype)
    # get the coordinate of centroid
    centroid = feature[0]["centroid"]
    print("old image center: ",centroid)
    frame_center = tuple([i*0.5 for i in new_frame.shape])
    shifts = [int(axis[0]-axis[1]) for axis in list(zip(frame_center,centroid))]
    print("magnitude moves:", shifts)
    non_0_corr = np.nonzero(img2)
    new_z = [zz + shifts[0] for zz in non_0_corr[0]]
    new_y = [yy + shifts[1] for yy in non_0_corr[1]]
    new_x = [xx + shifts[2] for xx in non_0_corr[2]]
    coordinates = list(zip(new_z,new_y,new_x))
    for zyx in coordinates:
        new_frame[zyx[0],zyx[1],zyx[2]]=1
    return new_frame

def modified_canny(smoothed_image, low_threshold=None, high_threshold=None,
          mask=None, use_quantiles=False, *, mode='constant', cval=0.0):
    """Edge filter an smoothed_image using the Canny algorithm.
    
    Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
        Pattern Analysis and Machine Intelligence, 8:679-714, 1986

    Parameters
    ----------
    smoothed_image : 2D array
        Grayscale input smoothed_image to detect edges on; can be of any dtype.
    sigma : float, optional
        Standard deviation of the Gaussian filter.
    low_threshold : float, optional
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float, optional
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.
    use_quantiles : bool, optional
        If ``True`` then treat low_threshold and high_threshold as
        quantiles of the edge magnitude smoothed_image, rather than absolute
        edge magnitude values. If ``True`` then the thresholds must be
        in the range [0, 1].
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled during Gaussian filtering, where ``cval`` is the value when
        mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.

    Returns
    -------
    output : 2D array (smoothed_image)
        The binary edge map.

    See also
    --------
    skimage.filters.sobel

    
    """

    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the smoothed_image?

    if np.issubdtype(smoothed_image.dtype, np.int64) or np.issubdtype(smoothed_image.dtype, np.uint64):
        raise ValueError("64-bit integer images are not supported")

    check_nD(smoothed_image, 2)
    dtype_max = dtype_limits(smoothed_image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not(0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not(0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= dtype_max

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    # Image filtering
    #smoothed, eroded_mask = _preprocess(smoothed_image, mask, sigma, mode, cval)

    # Gradient magnitude estimation
    jsobel = ndi.sobel(smoothed_image, axis=1)
    isobel = ndi.sobel(smoothed_image, axis=0)
    magnitude = isobel * isobel
    magnitude += jsobel * jsobel
    np.sqrt(magnitude, out=magnitude)

    if use_quantiles:
        low_threshold, high_threshold = np.percentile(magnitude,
                                                      [100.0 * low_threshold,
                                                       100.0 * high_threshold])
    # erode mask
    s = ndi.generate_binary_structure(2, 2)
    eroded_mask = ndi.binary_erosion(mask, s, border_value=0)
    # Non-maximum suppression
    low_masked = _nonmaximum_suppression_bilinear(
        isobel, jsobel, magnitude, eroded_mask, low_threshold
    )

    # Double thresholding and edge tracking
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    low_mask = low_masked > 0
    strel = np.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask

    high_mask = low_mask & (low_masked >= high_threshold)
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]
    return output_mask