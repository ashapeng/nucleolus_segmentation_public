import numpy as np
import re
import os
import pandas as pd
import math
import glob
from typing import Dict, List, Any

import matplotlib.pyplot as plt
# import self_defined functions
import seg_util as su

from Import_Functions import import_imgs
from scipy import ndimage as ndi
from skimage import io, filters
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops,label,marching_cubes,mesh_surface_area, find_contours
from skimage.morphology import disk,ball,binary_dilation,binary_closing,binary_opening, remove_small_holes
################################################
#             mask based analysis              #
################################################

def shape_discriber(structure_mask: np.array, resolution:List, cell_id:str, measured_parameters:List):
    """
        ----------
        Parameters:
        -----------
        structure_mask: nD array
            segmented mask
        
        resolution: list
            The resolution of the image: as[z,y,x]
        
        cell_id: str
            the id of the cell: in the form of: 'experiment_set'_'cell folder'
        
        measured_parameters: list 
            the parameters of the measurements as strings
            should be:["cell_id", "obj_id", "surface_area", "volume" , "surface_to_volume_ratio", "sphericity", "aspect_ratio","solidity"]
            updated as needed. make sure also update the shape_discriber function
        
        Returns:
        --------
        df: pandas dataframe
            measurements stored as dataframe for each cell
    """

    #  expand the cavas to avoid the object touching the boundary
    structure_mask = np.pad(structure_mask,5,'constant',constant_values=0)

    # label object with distinct number and return object dataframe
    if np.count_nonzero(structure_mask) == 0:
        
        # create a pd_dataframe with the parameters as columns, and values as None
        df = pd.DataFrame(columns=measured_parameters,index=[0])
       
       # store measurements in dataframe: fill with 0 except cell_id
        for key in measured_parameters:
            if key == "cell_id":
                df.loc[0,key] = cell_id
            else:
                df.loc[0,key] = 0

    else:
        labeled_mask, num_objs = label(structure_mask, return_num=True,connectivity=2)
        df = pd.DataFrame(columns=measured_parameters, index=range(1,num_objs+1))
        
        if len(structure_mask.shape) == 3 and len(resolution) == 3:
            ##############################################################################
            # move object into a larger frame to calculate aspect_ratio and bbox,    #####
            # because major and minor axis will still be returned but not accurately,#####
            # if the bbox extend the image boundary                                  #####
            ##############################################################################
            labeled_mask = np.pad(labeled_mask,(int(labeled_mask.shape[0]/2),),'constant')
            resolution = [0.2, 0.126, 0.126]
            
            # meaure all objects in one image
            for i in range(1,num_objs+1):
                obj_seg = np.where(labeled_mask==i,labeled_mask,0)
                
                # measure size: volume, surface
                verts_surf,faces_surf, normals_surf,values_surf = marching_cubes(
                    obj_seg,spacing=tuple(resolution),allow_degenerate=True)
                surface_area = mesh_surface_area(verts_surf,faces_surf)
                voxel_vol = resolution[0]*resolution[1]*resolution[2]
                vol = np.count_nonzero(obj_seg)*voxel_vol
                
                # surface-to-volume ratio
                sv_ratio = surface_area/vol
                
                # sphericity: calculate spheracity based on the definition by Wadell 1935
                equiv_sphere_surf = ((np.pi)**(1/3))*((6*vol)**(2/3))
                sphericity = equiv_sphere_surf/surface_area

                aspect_ratio = regionprops(obj_seg)[0]["axis_minor_length"]/regionprops(obj_seg)[0]["axis_major_length"]
                solidity = regionprops(obj_seg)[0]["solidity"]
                # store measurements in dataframe for each object
                df.loc[i,df.columns] = pd.Series(
                    [cell_id, i, surface_area, vol, sv_ratio, sphericity, aspect_ratio, solidity], index=df.columns
                    )
                
        if len(structure_mask.shape) == 2 and len(resolution) == 2:
            labeled_mask = np.pad(labeled_mask,(int(labeled_mask.shape[0]/2),int(labeled_mask.shape[1]/2)),'constant')
            resolution = [0.126, 0.126]
            for i in range(1,num_objs+1):
                obj_seg = np.where(labeled_mask==i,labeled_mask,0)
                print(np.count_nonzero(obj_seg))
                # measure size: area, perimeter
                area = np.count_nonzero(obj_seg)*resolution[0]*resolution[1]
                perimeter = regionprops(obj_seg)[0]["perimeter"]*resolution[0]
                
                # surface-to-volume ratio
                pa_ratio = perimeter/area
                # circularity
                sphericity = 4*np.pi*area/(perimeter**2)
                
                aspect_ratio = regionprops(obj_seg)[0]["axis_minor_length"]/regionprops(obj_seg)[0]["axis_major_length"]
                
                # solidity: ratio of object area to convex hull area
                solidity = regionprops(obj_seg)[0]["solidity"]
                
                # store measurements in dataframe
                df.loc[i,df.columns] = pd.Series(
                    [cell_id, i, perimeter, area, pa_ratio, sphericity, aspect_ratio, solidity], index=df.columns
                    )
    return df

def batch_measure_shape(master_folder:str,mask_name:str, shape_parameters: List):
    """
        ----------
        Parameters:
        -----------
        master_folder: str
            direcotory to the folder containing all experiment set
        mask_name: str
            the name of the mask file
        shape_parameters: list 
            the parameters of the measurements as strings
            should be:["cell_id", "obj_id", "surface_area", "volume" , "surface_to_volume_ratio", "sphericity", "aspect_ratio","solidity"]
            updated as needed. make sure also update the shape_discriber function
        Returns:
        --------
        all_dfs: list of dataframes measured by each cell
    """
    # create a list to store measurements of all cells
    all_dfs = []

    # import sub-folders from the main folder
    for item in os.listdir(master_folder):
        # read experiment set folder
        if os.path.isdir(os.path.join(master_folder,item)):
            experiment_set_dir = os.path.join(master_folder,item)
            # extract experiment set name
            date = os.path.basename(experiment_set_dir)
            
            # read individual cell folder
            for cell in os.listdir(experiment_set_dir):
                cell_seg_dir = os.path.join(experiment_set_dir,cell)
                cell_id = date + "\\" + os.path.basename(cell_seg_dir)
                print(cell_seg_dir)

                # read the images
                mask = import_imgs(cell_seg_dir, mask_name)

                # measure GC mask for each cell and return data frame
                each_cell_df = shape_discriber(
                mask, resolution=[0.2, 0.08, 0.08],cell_id=cell_id,
                measured_parameters=shape_parameters)
                all_dfs.append(each_cell_df)
    
    return all_dfs

def group_gc_measure_df(measurement_dfs:List,number_parameters:List, size_parameters:List):
    """
    this is to group measurements in different ways: by each object, by each cell, 
    and save different features
    check if you have prarameters as follows:["cell_id", "obj_id", "surface_area", "volume" , "surface_to_volume_ratio", "void_ratio", "sphericity", "aspect_ratio","solidity"]
    -----------
    Parameters:
    -----------
    measurement_dfs: List
        a list of measurement dfs, should have the same columns
    number_parameters: List
        list as ["cell_id",measurement_to_plot,"stage"]
    size_parameters: List
        list as ["cell_id", "surface_area", "volume", "surface_to_volume_ratio","stage":str]
    grouped_by: str
        specify either "cell_id" or "obj_id"
    Returns:
    --------
    df: pandas dataframe
        
    """
    df = pd.concat(measurement_dfs,axis=0,ignore_index=True)

    #########################
    # group data by cell_id #
    #########################

    # get unique cell_id
    by_cell_df = df[["cell_id"]].drop_duplicates()

    # create number dataframe
    nb_variables: Dict[str, Any] = {}
    for key in number_parameters:
        nb_variables[key] = None
    number_by_cell_df = pd.DataFrame(columns = nb_variables,index = range(len(by_cell_df)))

    # create size dataframe
    size_variables: Dict[str, Any] = {}
    for key in size_parameters:
        size_variables[key] = None
    size_by_cell_df = pd.DataFrame(columns = size_variables,index = range(len(by_cell_df)))

    # sort values from raw dataframe in different feature dataframes
    for index, row in by_cell_df.iterrows():
        # extract larval stage from cell_id
        stage = row["cell_id"][row["cell_id"].find("L"):row["cell_id"].find("L")+2]

        # count number of objects
        obj_count = float(df.loc[df["cell_id"]==row["cell_id"],["obj_id"]].max().iloc[0])

        # store values in number dataframe
        number_by_cell_df.loc[index,number_by_cell_df.columns] = pd.Series(
            [row["cell_id"],obj_count,stage],index=number_by_cell_df.columns)


        # store values in size dataframe
        volume_sum = float(df.loc[df["cell_id"]==row["cell_id"],["volume"]].sum().iloc[0])
        surface_sum = float(df.loc[df["cell_id"]==row["cell_id"],["surface_area"]].sum().iloc[0])
        surf_vol_ratio = surface_sum/volume_sum

        size_by_cell_df.loc[index,size_by_cell_df.columns] = pd.Series([row["cell_id"],surface_sum,volume_sum,surf_vol_ratio,stage],index=size_by_cell_df.columns)
        

    # sort dataframes
    number_by_cell_df = number_by_cell_df.sort_values(by="stage",ascending=True)
    number_by_cell_df = number_by_cell_df.reset_index(drop=True)
    size_by_cell_df = size_by_cell_df.sort_values(by="stage",ascending=True)
    size_by_cell_df = size_by_cell_df.reset_index(drop=True)
    ###########################
    # group data by object_id #
    ###########################
    # store values in size dataframe
    size_by_obj_df = df[["cell_id", "obj_id", "surface_area", "volume", "surface_to_volume_ratio"]].copy()

    for index, row in size_by_obj_df.iterrows():
        size_by_obj_df.loc[index,"stage"] = row["cell_id"][row["cell_id"].find("L"):row["cell_id"].find("L")+2]

    size_by_obj_df = size_by_obj_df.sort_values("stage",ascending=True)
    size_by_obj_df = size_by_obj_df.reset_index(drop=True)

    # store values in morphology dataframe
    morphology_by_obj_df = df[["cell_id","obj_id","sphericity", "aspect_ratio","solidity"]].copy()

    for index, row in morphology_by_obj_df.iterrows():
        morphology_by_obj_df.loc[index,"stage"] = row["cell_id"][row["cell_id"].find("L"):row["cell_id"].find("L")+2]

    morphology_by_obj_df = morphology_by_obj_df.sort_values("stage",ascending=True)
    morphology_by_obj_df = morphology_by_obj_df.reset_index(drop=True)

    # return all dataframes
    return number_by_cell_df, size_by_cell_df, morphology_by_obj_df, size_by_obj_df

def box_plot(df: pd.DataFrame, measurement_to_plot:str, y_axis_label: str, show_mean:bool=False, add_title: str = None):
    """
    -----------
    Parameters:
    df: pandas dataframe
        the grouped larval dataframe to be plotted
    measurement_to_plot: str
        specify the measurement to be plotted based on the column name
    object_type: str
        specify the object type: "GC" or "hole"
    
    Returns:
    plot: matplotlib figure

    """
    # extracting unique larval stages and excluding missing values
    stages = df.stage.dropna().unique()
    # Sorting the stages in the desired order: ascending, [L1,L2,L3,L4]
    sorted_stages = pd.Series(stages).sort_values(ascending=True).tolist()

    # extracting number of objects in each larval stages
    L1 = df.loc[df["stage"]=="L1",measurement_to_plot]
    L2 = df.loc[df["stage"]=="L2",measurement_to_plot]
    L3 = df.loc[df["stage"]=="L3",measurement_to_plot]
    L4 = df.loc[df["stage"]=="L4",measurement_to_plot]
    vals = [L1,L2,L3,L4]

    # create a box plot
    fig, ax =plt.subplots(ncols=1,nrows=1,figsize=(4,4))
    ax.boxplot(vals,notch=None,whis=(5,95),showmeans=True, showfliers=False,
                meanprops={"marker":"^","markersize":10,"markerfacecolor":"white", "markeredgecolor":"b"},
                medianprops={"linestyle":'-', "color":'red', "linewidth":2})
    
    # Add legend
    if show_mean:
        mean_marker = plt.Line2D([0], [0], marker='^', color='b', markersize=10, label='Mean')
        median_line = plt.Line2D([0], [0], linestyle='-', color='red', linewidth=2, label='Median')
        ax.legend(handles=[mean_marker, median_line], loc='best')

    # add labels to the x-axis
    ax.set_xticks(range(1, len(vals) + 1), sorted_stages)

    # add individual data points to the box plot
    for i, lst in enumerate(vals):
        # make data points spreaded by normal distribution
        xs = np.random.normal(i + 1, 0.04, lst.shape[0])
        ax.scatter(xs, lst, color="k",alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Larval stages post synchronization",fontsize=12)
    ax.set_ylabel(y_axis_label,fontsize=12)
    if add_title != None:
        ax.set_title(add_title,fontsize=12)
    # ax.set_title('{} meaurement of channel'.format(object_type),fontsize=12)
    return fig

################################################
#        intensity based analysis              #
################################################
# intensity based analysis
def dilated_mask(mask:np.array, radius: int, dilated_3d: bool = False, dilate_slice_by_slice: bool = False):
    """
    Parameters:
    -----------
    mask: 3D array
        the gc mask has been filled holes
    radius: int
        the dilation radius
    dilated_3d: bool
        whether to dilate mask in 3d
    dilate_slice_by_slice: bool
        whether to dilate mask slice-by-slice

    Returns:
    --------
    dilated_mask: 3D array
    """
    # dilate mask in 3d
    if if_dilated:
        dilated_mask = binary_dilation(mask, footprint=ball(radius))

    # dilate mask slice-by-slice
    if dilate_slice_by_slice:
        dilated_mask = np.zeros_like(mask)
        for z in range(mask.shape[0]):
            if np.count_nonzero(mask[z, :, :]) > 0:
                dilated_mask[z, :, :] = binary_dilation(mask[z, :, :], footprint=disk(radius))
    
    # set top and bottom of dilated mask to 0
    dilated_mask[0, :, :] = 0
    dilated_mask[-1, :, :] = 0
    
    return dilated_mask

def coefficient_of_variances(gc_mask: np.ndarray, img: np.ndarray, cell_id: str, measured_parameters: List):
    """
    Parameters:
    -----------
    gc_mask: 3D array
        the gc mask has been filled holes
    img: nD array
        3 channel stacked image
    radius: int
        the dilation radius
    cell_id: str
        the id of the cell: in the form of: 'experiment_set'_'cell folder'
    measured_parameters: List
        list of parameters: ["cell_id", "cv_r", "cv_g", "cv_b", "qcd_r", "qcd_g", "qcd_b"]
    
    Returns:
    --------
    cv: list
        list of cv of each channel as the order: r,g,b
    qcd: list
        list of qcd of each channel as the order: r,g,b
    """
    ################################
    # process mask                 #
    ################################
    

    # extract gray value of each channel from dilated mask
    raw_red_in_mask = img[...,0][gc_mask>0]
    raw_green_in_mask = img[...,1][gc_mask>0]
    raw_blue_i_mask = img[...,2][gc_mask>0]

    ################################
    # process mask   done          #
    ################################
    # create a pd_dataframe to store measurements
    variables: Dict[str, Any] = {}
    for key in measured_parameters:
        variables[key] = None
    df = pd.DataFrame(variables, index=[0]) 

    # measure coefficient of variance
    cv_r = round(np.std(raw_red_in_mask)/np.mean(raw_red_in_mask),3)
    cv_g = round(np.std(raw_green_in_mask)/np.mean(raw_green_in_mask),3)
    cv_b = round(np.std(raw_blue_i_mask)/np.mean(raw_blue_i_mask),3)

    # measure quartile coefficient of dispersion, Q3-Q1/Q3+Q1, descriptive measurement of dispersion,less sensitive to outliers
    qcd_r = round((np.quantile(raw_red_in_mask,0.75)-np.quantile(raw_red_in_mask,0.25))/(np.quantile(raw_red_in_mask,0.75)+np.quantile(raw_red_in_mask,0.25)),3)
    qcd_g = round((np.quantile(raw_green_in_mask,0.75)-np.quantile(raw_green_in_mask,0.25))/(np.quantile(raw_green_in_mask,0.75)+np.quantile(raw_green_in_mask,0.25)),3)
    qcd_b = round((np.quantile(raw_blue_i_mask,0.75)-np.quantile(raw_blue_i_mask,0.25))/(np.quantile(raw_blue_i_mask,0.75)+np.quantile(raw_blue_i_mask,0.25)),3)

    # store values in the pd_dataframe
    df.loc[0,df.columns] = pd.Series([cell_id,cv_r,cv_g,cv_b,qcd_r,qcd_g,qcd_b],index=df.columns)


    return df

################################################
# function realted to get colocalization       #
################################################
def relative_intensity(raw_img:np.array, dilated_mask: np.array, upper_limit: float):
    """
    ----------
    Parameters:
    -----------
    raw_img: nD array
        3 channel stacked image
    dilated_mask: 3D array
        the gc mask has been filled holes and dilated
    upper_limit: float
        the upper limit of intensity
    Returns:
    --------
    img_top: nD array
        binary image with intenisty value above upper_limit
    """

    # get the intensity of each channel in the dilated_mask
    raw_in_mask = np.stack([np.where(dilated_mask>0,raw_img[...,i],0) for i in range(raw_img.shape[-1])],axis=3)

    # get the top intensity of each channel
    top_ = np.zeros_like(raw_in_mask)

    for i in range(raw_in_mask.shape[-1]):
        each_channel = raw_in_mask[...,i]
        top_[...,i] = np.where(each_channel>=upper_limit*each_channel.max(),each_channel,0)
    top_ = top_.astype(np.uint8)
    top_[top_ > 0] = 255
    return top_

def relative_number_of_pixels(raw_img:np.array, dilated_mask: np.array, upper_limit: float):
    """
    ----------
    Parameters:
    -----------
    raw_img: nD array
        3 channel stacked image
    dilated_mask: 3D array
        the gc mask has been filled holes and dilated
    upper_limit: float
        the upper limit of intensity
    Returns:
    --------
    img_top: nD array
        binary image with intenisty value above upper_limit
    """

    # get the intensity of each channel in the dilated_mask
    raw_in_mask = np.stack([np.where(dilated_mask>0,raw_img[...,i],0) for i in range(raw_img.shape[-1])],axis=3)

    # get the top intensity of each channel
    top_ = np.zeros_like(raw_in_mask)

    for i in range(raw_in_mask.shape[-1]):
        each_channel = raw_in_mask[...,i]

        # Flatten the image to a 1-dimensional array
        flattened_each = each_channel.flatten()

        # Sort in a descending order and returns the indices that correspond to the sorted array
        sorted_values_indices = np.argsort(flattened_each)[::-1]
        sorted_values = flattened_each[sorted_values_indices]
        values_greater_zero = sorted_values[sorted_values>0]
       

        # Calculate the threshold for the top 10% values
        threshold_index = int(len(values_greater_zero) * upper_limit)
        threshold_value = sorted_values[threshold_index]

        # Extract the top 10% values and their coordinates
        # top_percent_values.append(sorted_values[:threshold_index])
        
        coordinates = np.unravel_index(sorted_values_indices[:threshold_index], each_channel.shape)
        
        # store pixels of top 10% values
        zip_coordinates = list(zip(*coordinates))
        for coordinate in zip_coordinates:
            top_[coordinate[0],coordinate[1],coordinate[2],i] = each_channel[coordinate[0],coordinate[1],coordinate[2]]
        
    top_ = top_.astype(np.uint8)
    top_[top_ > 0] = 255
    return top_

def overlap_3channel (top_img: np.array, cell_id: str):
    """
    ----------
    Parameters:
    -----------
    top_img: nD array
        3 channel binary image after screened for top 10% intensity
    Returns:
    --------
    colocal_df: pd.DataFrame
        dataframe of colocalization between each channel
    """
    # read image
    img_r = top_img[...,0]
    img_g = top_img[...,1]
    img_b = top_img[...,2]

    # overlap binary image
    overlap_rg = np.logical_and(img_r,img_g)
    overlap_rb = np.logical_and(img_r,img_b)
    overlap_gb = np.logical_and(img_g,img_b)
    overlap = np.logical_and(overlap_rg,overlap_rb)

    # caluclate overlap ratio of each channel>0
    rg_over_r = np.count_nonzero(overlap_rg)/np.count_nonzero(img_r)
    rg_over_g = np.count_nonzero(overlap_rg)/np.count_nonzero(img_g)

    rb_over_r = np.count_nonzero(overlap_rb)/np.count_nonzero(img_r)
    rb_over_b = np.count_nonzero(overlap_rb)/np.count_nonzero(img_b)

    gb_over_g = np.count_nonzero(overlap_gb)/np.count_nonzero(img_g)
    gb_over_b = np.count_nonzero(overlap_gb)/np.count_nonzero(img_b)

    rgb_over_r = np.count_nonzero(overlap)/np.count_nonzero(img_r)
    rgb_over_g = np.count_nonzero(overlap)/np.count_nonzero(img_g)
    rgb_over_b = np.count_nonzero(overlap)/np.count_nonzero(img_b)

    # store values in the pd_dataframe
    colocal_df = pd.DataFrame(columns=['cell_id', 'rg_over_r','rg_over_g','rb_over_r','rb_over_b','gb_over_g','gb_over_b','rgb_over_r','rgb_over_g','rgb_over_b'],index=[0])

    colocal_df.loc[0,colocal_df.columns] = pd.Series([cell_id,rg_over_r,rg_over_g,rb_over_r,rb_over_b,gb_over_g,gb_over_b,rgb_over_r,rgb_over_g,rgb_over_b],index=colocal_df.columns)

    colocal_df["stage"] = cell_id[cell_id.find("L"):cell_id.find("L")+2]

    return colocal_df

def overlap_heatmap(mean_df: pd.DataFrame, stage: str):
    """
    """
    # create a heatmap of overlap ratio
    channels = ["Red","Green","Blue"]
    # store overlap in a 3x3 matrix
    overlap = np.zeros((3,3),dtype=float)

    # Red channel overlaps with channels
    overlap[0,0] = "{:.3f}".format(1)
    overlap[0,1] = "{:.3f}".format(mean_df.loc[0,"rg_over_r"])
    overlap[0,2] = "{:.3f}".format(mean_df.loc[0,"rb_over_r"])

    # Green channel overlaps with channels
    overlap[1,0] = "{:.3f}".format(mean_df.loc[0,"rg_over_g"])
    overlap[1,1] = "{:.3f}".format(1)
    overlap[1,2] = "{:.3f}".format(mean_df.loc[0,"gb_over_g"])

    # Blue channel overlaps with channels
    overlap[2,0] = "{:.3f}".format(mean_df.loc[0,"rb_over_b"])
    overlap[2,1] = "{:.3f}".format(mean_df.loc[0,"gb_over_b"])
    overlap[2,2] = "{:.3f}".format(1)

    # plot heatmap with ratios
    fig,axs = plt.subplots(1,1,figsize=(4,4))
    im = axs.imshow(overlap,cmap="viridis")
    # show ticks and labels with the repective list channels
    axs.set_xticks(np.arange(len(channels)),labels=channels,fontsize=15)
    axs.set_yticks(np.arange(len(channels)),labels=channels,fontsize=15)

    # Loop over data dimensions and create text annotations.
    for i in range(len(channels)):
        for j in range(len(channels)):
            text = axs.text(j, i, overlap[i, j],
                        ha="center", va="center", color="red", fontsize=20)
    axs.set_title(f"Ratio of overlapped pixels at {stage} (top 10%)",fontsize=15)
    return fig

################################################
# function realted to measure concentration    #
################################################
def get_largetest_slice(seg_hole_filled_mask: np.array):
    """
    ----------
    Parameters:
    -----------
    seg_hole_filled_mask: 3D array
        seg_mask filled holes
    Returns:
    --------
    z_slice: largest slice of seg_hole_filled_mask
    """
    # get the largest gc_mask_hole_filled slice
    gc_size = [np.count_nonzero(seg_hole_filled_mask[z,:,:]) for z in range(seg_hole_filled_mask.shape[0])]
    z_slice = gc_size.index(max(gc_size))
    return z_slice

def find_box_in_binary_region(fluorescent_image:np.array, nucleus_mask: np.array, seg_hole_filled_mask: np.array, box_size:int):
    """
    ----------
    Parameters:
    -----------
    fluorescent_image: nD array
        raw image
    binary_mask: 2D array
        nucleus mask
    box_size: int
        the size of rectangle, use even number
    Returns:
    --------
    box_mask: 3D array
    """
    # Get the region between two masks
    binary_mask = nucleus_mask - seg_hole_filled_mask
    binary_mask[binary_mask>0] = 1
    # Find the coordinates of positive regions in the binary image
    positive_regions = np.where(binary_mask > 0)

    # Initialize variables to store the minimum mean intensity and the corresponding box
    min_mean_intensity = float('inf')
    box_radius = box_size // 2

    # Iterate over the positive regions
    for i in range(len(positive_regions[0])):
        y = positive_regions[0][i]
        x = positive_regions[1][i]

        # Check if the 7x7 box can be extracted around the given coordinates
        if y >= box_radius and y < binary_mask.shape[0] - box_radius and \
           x >= box_radius and x < binary_mask.shape[1] - box_radius:
           
            # Extract the 7x7 box from the binary image
            binary_box = binary_mask[y - box_radius:y + box_radius + 1, x - box_radius:x + box_radius + 1]
            
            # Check in the binary image if the 7x7 box has any zeros(background pixels)
            if not np.any(binary_box == 0):
                # print(f"in the 7x7 box by {(y,x)} there is not zeros background pixels")
                # Extract the corresponding 7x7 box from the fluorescent image
                fluorescent_box = fluorescent_image[y - box_radius:y + box_radius + 1, x - box_radius:x + box_radius + 1]
                
                # Calculate the mean intensity of the 7x7 box
                mean_intensity = np.mean(fluorescent_box)

                # Update the minimum mean intensity and the corresponding box
                if mean_intensity < min_mean_intensity:
                    min_mean_intensity = mean_intensity
                    top_left_coord = (y, x)
    
    # Create the box mask the the corresponding coordinates
    box_mask = np.zeros_like(binary_mask)
    box_mask[top_left_coord[0]-box_radius:top_left_coord[0]+box_radius, top_left_coord[1]-box_radius:top_left_coord[1]+box_radius] = 1
    return box_mask

def concentration_gc(raw_img:np.array, raw_bg_subt:np.array, background_mask:np.array, nucleoplasm_mask:np.array, seg_mask:np.array, seg_hole_filled_mask: np.array, nucleus_mask:np.array,cell_id:str, measured_parameters:List):
    """
    ----------
    Parameters:
    -----------
    raw_img: 2D array
        raw image
    raw_bg_subt: 2D array
        raw image background subtracted
    background_mask: 2D array
        background_mask mask
    nucleoplasm_mask: 2D array
        nucleoplasm mask
    seg_mask: 2D array
        segmented mask
    seg_hole_filled_mask: 2D array
        seg_mask filled holes
    nucleus_mask: 2D array
        nucleus mask
    cell_id: str
        the id of the cell: in the form of: 'experiment_set'_'cell folder'
    measured_parameters: list 
        the parameters of the measurements as strings
        should be:["cell_id", "C_bg", "C_dilute", "C_dense", "pc", "nuclear area"]
    Returns:
    --------
    df: pandas dataframe
        measurements stored as dataframe for each cell
    """
    # copy raw image for background subtraction
    raw_gc = raw_img.copy()
    img = raw_bg_subt.copy()
    
    # create dataframe
    variables: Dict[str, Any] = {}
    for key in measured_parameters:
        variables[key] = None
    df = pd.DataFrame(variables, index=[0])

    # measure intensity in ROI after background subtraction
    bg_value = round(float(np.mean(img[background_mask>0])), 1)
    dilute_value = round(float(np.mean(img[nucleoplasm_mask>0])), 1)
    gc_value = round(float(np.mean(img[seg_mask>0])), 1)
    total_value =round(float(np.mean(img[nucleus_mask>0])),1)

    # partition coefficient based on raw image
    gc_value_raw = float(np.mean(raw_gc[seg_mask>0]))
    dilute_value_raw = float(np.mean(raw_gc[nucleoplasm_mask>0]))
    pc_value = round(gc_value_raw/dilute_value_raw, 3)

    # measure nuclear area
    # nuclear_area = round((np.count_nonzero(nucleus_mask) *(0.08*0.08)), 3)

    # add to dataframe
    df.loc[0,df.columns] = pd.Series([cell_id,bg_value,dilute_value,gc_value,total_value,pc_value],index=df.columns)

    df["stage"] = cell_id[cell_id.find("L"):cell_id.find("L")+2]

    return df

def group_sort_larval_df(df_list: List):
    """
    ----------
    Parameters:
    -----------
    df_list: List
        a list of measurement dfs, should have the same columns
    Returns:
    grouped_df: pandas dataframe
        the grouped df that has been sorted based on larval stages
    """
    grouped_df = pd.concat(df_list, axis=0, ignore_index=True)
    grouped_df = grouped_df.sort_values(by="stage", ascending=True)
    grouped_df = grouped_df.reset_index(drop=True)
    return grouped_df

################################################
# function realted to get extreme CV values    #
# and plot image with gc contours              #
################################################
def replace_at_index(input_string, character, new_value):
    # get the index of the character
    index = input_string.find(character)
    # Convert the input string to a list of characters
    char_list = list(input_string)

    # Replace the character at the specified index with the new value
    char_list[index] = new_value

    # Convert the modified list back to a string
    modified_string = ''.join(char_list)

    return modified_string

def seg_hole_filled_raw(master_seg_dir:str, master_raw_dir:str, cell_id:str, channel: int):
    # read images
    seg_hole_filled = import_imgs(os.path.join(master_seg_dir, cell_id), "hole_filled.tif")
    raw_img = import_imgs(os.path.join(master_raw_dir, cell_id), "Composite_stack.tif")

    # maximal projection mask
    max_proj_mask = np.max(seg_hole_filled, axis=0)
    # find contours of seg mask
    contours_seg = find_contours(max_proj_mask, 0.1)
    # corresponding raw image
    max_raw = np.max(raw_img[...,channel], axis=0)

    return max_raw, contours_seg

def plot_raw_contour(raw: np.array, contours: np.array, cv:float, cell_id:str, channel: int, min_or_max_cv: str, save_dir = None, save_fig: bool = False):
    new_cell_id = replace_at_index(cell_id, "\\", "_")
    fig, axs = plt.subplots(1,1,figsize=(4,4))
    axs.imshow(raw, cmap="gray")
    for contour in contours:
        axs.plot(contour[:,1], contour[:,0], linewidth=2, color="red", linestyle="dashed")
    axs.set_title(f"{new_cell_id} channel {channel} cv = {cv:.2f}", fontsize=10)
    axs.axis("off")
    plt.tight_layout()
    if save_dir != None:
        if save_fig:
            plt.savefig(os.path.join(save_dir, f"{new_cell_id}_channel_{channel}_{min_or_max_cv}_cv_{cv:.2f}.svg"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()
