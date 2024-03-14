# community_detection.py

import subprocess, os
import numpy as np
import pandas as pd

from network_utils import template_match
from rsfc_tools import save_nii
from visualization import expand_to333

PARCEL_TEMPLATE_PATH = 'data/gordon2016_parcels/Gordon333_TEMPLATE.pscalar.nii'

def detect_communities_infomap(sub_thresholded, outname, outdir, perform_template_match=True):

    """
    Runs the Infomap community detection algorithm on a given thresholded connectivity matrix,
    processes the output to assign communities, and saves the results in nifti format.
    Parameters:
        - sub_thresholded (numpy.ndarray): A 2D array representing the thresholded connectivity matrix.
        - outname (str): The base name for the output files (without extension).
        - outdir (str): The directory path where the output files will be saved.
        - perform_template_match (bool): Default True. Template matches the infomap communities to gordon networks
    Assumes:
        - Infomap is installed and accessible from the command line.
    Outputs:
        - Saves two nifti files in the specified output directory: one for the community assignments
        from Infomap, and another for the matched community assignments based on a predefined template.
    """

    # Extract upper triangle indices where connections exist
    inds = np.where(np.triu(sub_thresholded)>0)
    x_indices, y_indices = inds

    # Prepare the edge list for writing to the Infomap input file
    to_write = np.array((x_indices+1,y_indices+1,sub_thresholded[inds])).T

    infomap_input_path = os.path.join(outdir,outname)  

    nodes = sub_thresholded.shape[0] 
    length_x = len(x_indices) 
    
    # Writing the Infomap input file
    with open(infomap_input_path, 'w') as fid:
        fid.write('*Vertices %d\n' % nodes)
        for node in list(range(1, nodes + 1)):
            fid.write('%d "%d"\n' % (node, node))
        fid.write('*Edges %d\n' % length_x)
        for edge in to_write:
            fid.write('%d %d %f\n' % tuple(edge))

    # Running Infomap
    subprocess.call(f'infomap --clu -2 -s 1 -v -N 100 --out-name {outname} {infomap_input_path} {outdir}', shell=True) 

    # Cleaning up the input file
    os.remove(infomap_input_path)
    
    # Processing Infomap output
    subpath = os.path.join(outdir,f'{outname}.clu')
    clu_data = pd.read_csv(subpath, skiprows=10, delimiter=' ', header=None, names=['node','module','flow'])
    clu_data = clu_data.sort_values('node')

    # Save communties onto nifti. Currently hard coded for gordon parcels
    if perform_template_match == True:
        matched_modules = template_match(clu_data.module)
        save_nii(expand_to333(clu_data.module), f'{outname}', outdir, wb_required_template_path=PARCEL_TEMPLATE_PATH)
        save_nii(expand_to333(matched_modules), f'{outname}_matched', outdir, wb_required_template_path=PARCEL_TEMPLATE_PATH)
    else:
        save_nii(expand_to333(clu_data.module), f'{outname}', outdir, wb_required_template_path=PARCEL_TEMPLATE_PATH)