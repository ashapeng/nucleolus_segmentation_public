from skimage import measure
import re
import math

def decimal_num(n,two_digit:bool=False):
    """
    step1: return the factional and integer parts of a number as a two-item tuple, both have the same sign as the number
    step2: convert float in the exponential format
    Step3: search for "-", return a tuple with three elements: 1. everything before "match";2. the "match"; 3. everything after the"match";
    based on book biology by number: use two significant digits in biology
    significant digits: are all digits that are not zero, plus zeros that are to the right of the first non-zero digit
    """
    
    """
    ----
    n: float
        the number to find decimal numbers
    two_digit: bool, Default is False
        whether to keep two significant digits or not
    -----
    return: decimal number
    """
    num_tuple = math.modf(abs(n))
    f = num_tuple[0] # the fractional part of a number
    f_str='%e' % f # convert float in the exponential format
    if two_digit:
        rouned_decimal = int(f_str.partition("-")[2]) + 1 # 
    else:
        rouned_decimal = int(f_str.partition("-")[2])
    
    return rouned_decimal    

# round numbers
def round_up(n, decimals):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def plot_volume(volume,voxel,ax,fluorescence,alp):
    #Find the mesh for a segmentation and plot it as a surface
    level = 0.5*(volume.max() + volume.min())
    verts, faces, normals, values = measure.marching_cubes(
        volume, spacing=voxel, level=level,allow_degenerate=True, method="lewiner",step_size=1)
    z,y,x = verts.T
    if re.match(".*rfp.*",fluorescence):
        ax.plot_trisurf(x,y,faces,z,linewidth=0,antialiased=True,color='red',alpha=alp,edgecolor='k')
    elif re.match(".*cfp.*",fluorescence):
        ax.plot_trisurf(x,y,faces,z, linewidth=0,antialiased=True,color='blue',alpha=alp,edgecolor='k')
    elif re.match(".*gfp.*",fluorescence):
        ax.plot_trisurf(x,y,faces,z, linewidth=0,antialiased=True,color='green',alpha=alp,edgecolor='k') 
    ax.dist = 0.5  # zoom in or zoom out
