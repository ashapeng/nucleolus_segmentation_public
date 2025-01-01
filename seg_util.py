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

from scipy import ndimage as ndi

# segmentation core function
from skimage.filters import threshold_otsu,threshold_triangle

# spot segmentation core function
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
import round_numbers as rn

# import functions for local threshold
from skimage.measure import regionprops,label,marching_cubes,mesh_surface_area

# import function for post-segmentation image process
from skimage.morphology import remove_small_objects, remove_small_holes,binary_erosion,ball,disk,binary_dilation,binary_closing,binary_opening

##############################################################
##              core functions end                         ###
##############################################################

def image_2d_seg(raw_img: np.array, nucleus_mask: None, sigma_2d: float) -> np.array:
    '''
    This to segment the maximal porjection of z-stack images
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
    # check if image is multichannel, if not, add channel dimension
    img_dimension = raw_img.ndim
    if img_dimension == 3:
        raw_img = np.expand_dims(raw_img, axis=-1)
    
    # maximal projection
    max_projection = np.stack([np.max(raw_img[...,channel],axis=0) for channel in range(raw_img.shape[-1])],axis=2)

    # to avoid speckles in the image: set values greater than the 99.9th percentile to the 99.9th percentile
    max_projection_truncated = np.zeros_like(max_projection)
    for channel in range(max_projection.shape[-1]):
        max_projection_truncated[...,channel] = np.where(max_projection[...,channel]>np.percentile(max_projection[...,channel],99.9),np.percentile(max_projection[...,channel],99.9),max_projection[...,channel])

    # normalize by min_max normalization: (x-min)/(max-min)
    normalized_max_projection_truncated = np.zeros_like(max_projection_truncated)
    for channel in range(max_projection_truncated.shape[-1]):
        normalized_max_projection_truncated[...,channel] = (max_projection_truncated[...,channel]-np.min(max_projection_truncated[...,channel]))/(np.max(max_projection_truncated[...,channel])-np.min(max_projection_truncated[...,channel]))
    
    # smooth
    smoothed_2d = np.stack([ndi.gaussian_filter(normalized_max_projection_truncated[...,channel],sigma=sigma_2d,mode="nearest",truncate=3) for channel in range(normalized_max_projection_truncated.shape[-1])],axis=2)

    # segment with otsu
    thresholded = np.zeros_like(smoothed_2d)
    
    # regional segmentation in the nucleus, only keep signal within the nucleus mask use it to calculate the threshold
    if nucleus_mask is not None:
        smoothed_updated = np.stack([np.where(nucleus[0,...]>0,smoothed_2d[...,i],0) for i in range(smoothed_2d.shape[-1])],axis=2)
        
        for i in range(smoothed_2d.shape[-1]):
        cutoff1 = threshold_otsu(smoothed_updated[...,i][smoothed_updated[...,i]>0])
        thresholded[...,i] = np.where(smoothed_2d[...,i]>cutoff1,1,0)

    else:
        for i in range(smoothed_2d.shape[-1]):
            cutoff1 = threshold_otsu(smoothed_2d[...,i][smoothed_2d[...,i]>0])
            thresholded[...,i] = np.where(smoothed_2d[...,i]>cutoff1,1,0)

    # post segmentation processing: close holes and remove small objects
    post_seg = np.zeros_like(thresholded,dtype = np.uint8)
    for i in range(thresholded.shape[-1]):
        each = thresholded[...,i]
        opened = binary_opening(each,footprint=np.ones((3,3))).astype(bool)
        holed_filled = remove_small_holes(opened)
        
        # only keep the largest object
        labeled = label(holed_filled,connectivity=2)
        props = regionprops(labeled)
        
        # Find the label of the largest object
        largest_label = np.argmax([prop.area for prop in props]) + 1
        post_seg[...,i] = labeled == largest_label
        post_seg[...,i][post_seg[...,i]>0] = 255
    return post_seg

def bg_subtraction(raw_img: np.array, bg_mask: np.array, clip: bool = False) -> np.ndarray:
    """
        Subtract background from raw image with known background mask
    """

    """
        Background subtraction is typically used to enhance the contrast between the foreground objects and the background in an image, making it easier to segment the objects. 
        This process involves subtracting the background signal from the image, leaving only the foreground objects. 
    """
    
    """
        If you have negative intensity values after background subtraction, there are several ways to handle them before normalizing your images:

        Clip the negative values to zero: This means that you simply set any negative values to zero, effectively removing any negative values from your image. In my image analysis since negative pixel values represent regions outside cells or regions dimmer that the chosen background and do not provide useful information about the image content, and that they can safely be discarded.

        Shift the intensity values: This means that you add a constant value to all intensity values, such that the minimum intensity value becomes zero. By shifting the intensity values to be non-negative, you can ensure that all intensity values are valid and can be used for further processing or analysis.

        Apply a log transform: A log transform can be used to transform the intensity values to a more linear scale, while preserving the relative differences between the values. This can be particularly useful if your data has a large dynamic range or if you want to highlight differences between low-intensity values.

        Use a different normalization method: min-max normalization scales the intensity values to a fixed range of values, such as [0, 1], which can ensure that all values are positive. Alternatively, z-score normalization scales the intensity values to have a mean of zero and a standard deviation of one.
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

    # bg subtraction
    bg_substracted = np.zeros_like(raw_img,dtype=raw_img.dtype)
    if img_dimension == 4:
        for channel in range(raw_img.shape[-1]):
            for z in range(bg_mask.shape[0]):
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
        formula:

        X_norm = (X - X_min) / (X_max - X_min)

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

def gaussian_smooth_stack(img:np.array,sigma:list):
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
        The data to be smoothed, as [plane,row,column],single channel
    sigma_by_channel: Dictionary 
        a dictionary of sigma for guassian smoothing ordered as each channel
    Returns:
    --------
    smoothed_data: 3D array
        images after gaussian smoothing
    '''
    if len(sigma)>1:
        # 3d smooth
        smoothed_data = image_smoothing_gaussian_3d(img,sigma=sigma,truncate_range=3.0)
    else:
        # gaussian smoothing slice-by-slice
        smoothed_data = image_smoothing_gaussian_slice_by_slice(img,sigma=sigma[0],truncate_range=3.0)
    
    return smoothed_data

def global_otsu(
        img:np.ndarray, mask:np.ndarray,global_thresh_method: str, 
        mini_size:float, local_adjust:float=0.98, extra_criteria: bool=False, 
        return_object:bool = False, keep_largest: bool=False):
    '''
        Use Allen segmenter Implementation of "Masked Object Thresholding" algorithm. Specifically, the algorithm is a hybrid thresholding method combining two levels of thresholds.
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
        img: 3D array
            The image that has been smoothed, as [plane,row,column],single channel
        mask: 3D array
            The array within which otsu threshold for img is processed
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
        th_low_level = threshold_triangle(img)
    elif global_thresh_method == "med" or global_thresh_method == "median":
        th_low_level = np.percentile(img, 50)
    elif global_thresh_method == "ave" or global_thresh_method == "ave_tri_med":
        global_tri = threshold_triangle(img)
        global_median = np.percentile(img, 50)
        th_low_level = (global_tri + global_median) / 2

    bw_low_level = img > th_low_level
    bw_low_level = remove_small_objects(
        bw_low_level, min_size=mini_size, connectivity=1)
    bw_low_level = binary_dilation(bw_low_level, footprint=ball(1))
    # set top and bottom slice as zero
    bw_low_level[0,:,:]=0
    bw_low_level[-1,:,:]=0

    # local otsu
    bw_high_level = np.zeros_like(bw_low_level)
    lab_low, num_obj = label(bw_low_level, return_num=True, connectivity=1)
    if extra_criteria:
        local_cutoff = 0.333 * threshold_otsu(img[mask>0])
        for idx in range(num_obj):
            single_obj = lab_low == (idx + 1)
            local_otsu = threshold_otsu(img[single_obj > 0])
            final_otsu = rn.round_up(local_otsu, rn.decimal_num(local_otsu, two_digit=True))
            #final_otsu = math.ceil(local_otsu)
            print("otsu:{},rouned otsu:{},ajusted otsu:{}".format(local_otsu, final_otsu, final_otsu * local_adjust))
            if local_otsu > local_cutoff:
                bw_high_level[
                    np.logical_and(
                        img > final_otsu * local_adjust, single_obj
                    )
                ] = 1
    else:
        for idx in range(num_obj):
            single_obj = lab_low == (idx + 1)
            local_otsu = threshold_otsu(img[single_obj > 0])
            final_otsu = rn.round_up(local_otsu, rn.decimal_num(local_otsu, two_digit=True))
            print("otsu:{},rouned otsu:{},ajusted otsu:{}".format(local_otsu, final_otsu, final_otsu * local_adjust))
            bw_high_level[
                np.logical_and(
                    img > final_otsu * local_adjust, single_obj
                )
            ] = 1

    # post segmentation process: 
    # remove small dispersed obj and fill holes in each slice
    global_mask = np.zeros_like(bw_high_level)
    for z in range(bw_high_level.shape[0]):
        global_mask[z,:,:] = binary_closing(bw_high_level[z,:,:], footprint=disk(2))
        global_mask[z,:,:] = remove_small_holes(global_mask[z,:,:].astype(bool), area_threshold=np.count_nonzero(global_mask[z,:,:]),connectivity=2)
        global_mask[z,:,:] = remove_small_objects(global_mask[z,:,:].astype(bool), min_size=30,connectivity=2)
    # set the top and bottom as zero
    global_mask[0,:,:]=0
    global_mask[-1,:,:]=0

    # only keep the largest label
    if keep_largest:
        labeled_mask,label_mask_num = label(global_mask, return_num=True, connectivity=1)
        print("number of mask is: ", label_mask_num)
        mask_id_size = {}
        for id in range(1, label_mask_num+1,1):
            as_id = labeled_mask==id
            vol = np.count_nonzero(as_id)
            mask_id_size["{}".format(id)]=vol
            print("mask spot volumn",vol)   
        max_mask_vol_id = int(max(mask_id_size,key=mask_id_size.get))
        segmented_data = np.logical_or(
            np.zeros_like(global_mask),labeled_mask == max_mask_vol_id
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

def segment_spot(
        raw_img:np.ndarray, nucleus_mask:np.ndarray, nucleolus_mask:np.ndarray,
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
        nucleolus_mask: 3D array
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

        # only keep LoG value within nucleus_mask_eroded to avoide bright "noise"
        LoG_in_mask = []
        for id in np.argwhere(nucleus_mask_eroded):
            LoG_in_mask.append(spot_temp[id[0],id[1],id[2]])

        # calculate cut_off & check mean of LoG with different region size
        cut_off = rn.round_up(
            np.percentile(LoG_in_mask,96), 
            rn.decimal_num(np.percentile(LoG_in_mask,96),two_digit=True))
        param_spot.append([LoG,cut_off])

    # set cutoff as a mean
    cutoff_mean = np.mean(np.array(param_spot),axis=0)[1]
    print("Use the mean cutoff value",cutoff_mean)
    updated_param =[ [param_spot[i][0],cutoff_mean] for i in range(len(param_spot))]
    if show_param:
        print("spot seg parameter is:",param_spot)
        print("spot seg updated parameter is:",updated_param)
   
   # apply LoG filter to spot
    spot_by_LoG = dot_2d_slice_by_slice_wrapper(spot, updated_param)

    # remove object smaller than a connectivity=2 region slice by slice
    spot_opened =np.zeros_like(spot_by_LoG)
    for z in range(spot_by_LoG.shape[0]):
        spot_opened[z,:,:] = binary_opening(
            spot_by_LoG[z,:,:],footprint=ndi.generate_binary_structure(2,2))
        spot_opened[z,:,:] = remove_small_objects(
            spot_opened[z,:,:],min_size=10,connectivity=2).astype(np.uint8)
        spot_opened[z,:,:] = binary_closing(
            spot_opened[z,:,:],footprint=ndi.generate_binary_structure(2,2))

    # only keep objects within nucleolus_mask
    spot_in_structure = np.where(np.logical_and(spot_opened,nucleolus_mask),1,0)

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

def final_gc_holes(spot_mask:np.ndarray, nucleolus_mask:np.ndarray):
    """
        Parameters:
        -----------
        spot_mask: 3D array
            segmented spot mask with the segment_spot function
        nucleolus_mask: 3D array
            nucleolar mask with global_otsu function
        Returns:
        --------
        final_gc: 3D array
            mask after combined
        hole_filled_final: 3D array
            mask after filling holes
        holes: 3D array
            holes in the final gc mask
    """
    # get final GC mask
    final_gc = nucleolus_mask.copy()
    final_gc[spot_mask>0]=0

    # fill holes in final mask
    hole_filled_final = np.zeros_like(final_gc)
    for z in range(final_gc.shape[0]):
        hole_filled_final[z,:,:] = remove_small_holes(final_gc[z,:,:].astype(bool), area_threshold=np.count_nonzero(nucleolus_mask[z,:,:]),connectivity=2)
    

    final_gc=final_gc.astype(np.uint8)
    final_gc[final_gc>0]=255

    hole_filled_final=hole_filled_final.astype(np.uint8)
    hole_filled_final[hole_filled_final>0]=255

    return final_gc, hole_filled_final

################
# integrate GC segmentation step into one function
def gc_segment(raw_image:np.ndarray, nucleus_mask:np.ndarray, sigma: float, local_adjust_for_GC: float):
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

    # normalize images based on min-max normalization
    normalized_img = np.stack([min_max_norm(raw_img[:,:,:,i]) for i in range(raw_img.shape[-1])],axis=3)

    # 3d smooth raw LPD7 image
    gc_smoothed_final = ndi.gaussian_filter(normalized_img[...,2],sigma=sigma,mode="nearest",truncate=3)

    # otsu segment each channel as for ground and background
    # adjust local_adjust parameter to make the segmentation more and less
    gc_otsu = global_otsu(gc_smoothed_final, nucleus_mask, global_thresh_method="ave",mini_size=1000,local_adjust=local_adjust_for_GC,extra_criteria=False,keep_largest=True)

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
    # generate 3d ball as footprint, based on the confocol scope in Weber lab
    # confocol voxel dimension: Z, Y, X = 0.2, 0.0796631, 0.0796631
    # at 100x X 1.4 NA Z, Y, X = 1, 2.5, 2.5
    n_xy = 2*radius_xy+1
    n_z = 2*radius_z+1
    Z,Y,X = np.mgrid[-radius_z:radius_z:n_z*1j,
                    -radius_xy:radius_xy:n_xy*1j,
                    -radius_xy:radius_xy:n_xy*1j]
    s = X**2 + Y**2 + Z**2
    return np.array(s<=radius_xy*radius_z,dtype=dtype)