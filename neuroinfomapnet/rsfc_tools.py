import numpy as np
import subprocess, os
import matplotlib.pyplot as plt

path_to_wb = '/sphere/greene-lab/Shared_Tools/workbench/bin_linux64/wb_command'

def load_nii(nii_path, purge=True):
    '''
    Load in a .nii file to work with.
    This function converts the .nii to a text file using workbench, and then
    uses numpy to load in the text file.
    '''
    end = nii_path.find('.')
    outname = nii_path[:end] + '.txt'
    wb_comm = ' '.join([path_to_wb, '-cifti-convert -to-text', nii_path, outname])
    subprocess.call(wb_comm, shell=True)
    nii_data = np.loadtxt(outname)
    if purge:
        os.remove(outname)
    return nii_data

def save_nii(array, output_name, output_dir_path, wb_required_template_path, purge = True):
    '''
    Save a numpy array as a .nii file, utilizing workbench as an intermediary.
    Arguments:
        array: the numpy array to save
        output_name: the name to use for the output file
        output_dir_path: the directory path to save the output file in
        wb_required_template_path: a dtseries of the same dimension used as a template to write over
        purge: if True, delete the intermediate text file after creating the .nii file
    '''
    if not os.path.isdir(output_dir_path):
        raise Exception(f"The output folder {output_dir_path} does not exist")
    template_base = os.path.basename(wb_required_template_path)
    end = template_base.find('.')
    file_end = template_base[end:]
    out_path = os.path.join(output_dir_path, output_name)
    outnamecifti = out_path+file_end
    if os.path.isfile(outnamecifti):
        print('-WARNING: Overwriting')
    np.savetxt(out_path, array)
    wb_comm = ' '.join([path_to_wb, '-cifti-convert -from-text', out_path, wb_required_template_path, outnamecifti])
    subprocess.call(wb_comm, shell=True)
    if purge:
        os.remove(out_path)

def create_corrmat(ts):
    corrmat = np.corrcoef(ts)
    np.fill_diagonal(corrmat,0)
    z_trans_mat = np.arctanh(corrmat)
    np.fill_diagonal(z_trans_mat,1)
    return z_trans_mat

# Assumes parcels are ordered by reorder indicies
def make_corrfig(z_trans_mat, weights = False):
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patheffects as PathEffects

    ## Get all the visuals info for this parcellation#
    # label_names = ['Auditory','CinguloOperc','CinguloParietal','Default','DorsalAttn','FrontoParietal','None',
    #             'RetrosplenialTemporal','Salience','SMhand','SMmouth','VentralAttn','Visual']
    names_abbrev = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None',
                    'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual','Subcort']
    ## COLORS AND THINGS FOR CORR MATRIX IMAGE ##
    color_label_list = ['pink','purple','mediumorchid','red','lime','yellow','white',
                        'bisque','black','cyan','orange','teal','blue','brown']
    # Range for color bars
    range_list = ['0-24','24-64','64-69','69-110','110-142','142-166','166-213',
                '213-221','221-225','225-263','263-271','271-294','294-333','333-353']
    # CREATING numpy array of ranges #
    formatted_range_list = []
    for rl in range_list:
        rl = rl.split('-')
        rl =  map(int, rl)
        formatted_range_list.append(list(rl))
    labels = np.array(formatted_range_list)
    # Index list for lines
    line_list = [24,64,69,110,142,166,213,221,225,263,271,294,333]
#     fig, ax = plt.subplots(figsize=(20, 12))
    fig, ax = plt.subplots()
    if weights == True:
        im = ax.imshow(z_trans_mat, aspect='equal')
    else:
        cmap = cm.get_cmap('jet')
        low_thresh = -0.4
        high_thresh = 1.0
        im = ax.imshow(z_trans_mat, aspect='equal',cmap=cmap,vmin=low_thresh,vmax=high_thresh)
        ax.set_title('Z-Transformed Connectivity Matrix', fontsize=30)
        # TITLE AND COLORBAR ADJUSTMENTS #
        cbar = fig.colorbar(im, pad=0.0009)
        cbar.set_ticks([.8,.6,.4,.2,0,-0.2,-0.4,-0.6,-0.8])
        cbar.ax.set_ylabel('arctanh (z-transformed) values',rotation=270, labelpad=12, weight='bold')
        cbar.ax.tick_params(labelsize=5, pad=3)
        cbar.update_ticks()
        cbar.ax.yaxis.set_ticks_position('left')
    #DRAWING LINES
    for the_line in line_list:
        ax.axhline(y=the_line - .5, linewidth=1.5, color='white')
        ax.axvline(x=the_line - .5, linewidth=1.5, color='white')

    # CREATE AXES NEXT TO PLOT
    divider = make_axes_locatable(ax)
    axb = divider.append_axes("bottom", "10%", pad=0.02, sharex=ax)
    axl = divider.append_axes("left", "10%", pad=0.02, sharey=ax)
    axb.invert_yaxis()
    axl.invert_xaxis()
    axb.axis("off")
    axl.axis("off")
    # PLOT COLORED BARS TO THE AXES
    barkw = dict( color=color_label_list, linewidth=0.50, ec="k", clip_on=False, align='edge',)
    # bottom bar #
    axb.bar(labels[:,0]-.5,np.ones(len(labels)),
            width=np.diff(labels, axis=1).flatten(), **barkw)
    # side bar #
    axl.barh(labels[:,0]-.5,np.ones(len(labels)),
            height=np.diff(labels, axis=1).flatten(), **barkw)
    # SET MARGINS TO ZERO AGAIN
    ax.margins(0)
    ax.tick_params(axis="both", bottom=0, left=0, labelbottom=0,labelleft=0)
    # ADD TEXT IN THE COLOR BARS #
    for idx,x in enumerate(labels):
        align = (x[0] + x[1])/2
        axb.text(align,.5,names_abbrev[idx], fontsize=9, rotation=90, horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[PathEffects.withStroke(linewidth=.5, foreground="w")])
        axl.text(.5,align,names_abbrev[idx], fontsize=9, horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[PathEffects.withStroke(linewidth=.5, foreground="w")])
    fig.set_size_inches(30,18)

    # plt.savefig(mat_path.replace('.csv','.png'), dpi=1200, format='png', bbox_inches='tight')
    # CLEAR FIGURE #
    # plt.clf()
    # plt.cla()
    # plt.close()

def remove_diagonal(correlation_matrix): return correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)].reshape(correlation_matrix.shape[0], correlation_matrix.shape[0]-1)

def flatten_upper_triangle(arr): return arr[np.triu_indices_from(arr,1)]

def flatten_upper_triangle_3D(arr):
    size = arr.shape[2]
    conn_flat = []
    for i in range(size):
        conn_flat.append(flatten_upper_triangle(arr[:, :, i]))
    return np.stack(conn_flat, axis=0)

def reconstruct_symmetric_array(flattened):
    """
    Reconstructs a symmetric 333 square array from its flattened upper triangle (which does not include its diag).

    Args:
        flattened (list): Flattened upper triangle of the array (excluding diagonal elements).

    Returns:
        list: Symmetric square array.
    """

    arr = np.zeros((333,333))

    idx = 0
    for i in range(333):
        for j in range(i + 1, 333):
            arr[i,j] = flattened[idx]
            arr[j,i] = flattened[idx]
            idx += 1

    return arr

# Assumes parcels are ordered by reorder indicies
def get_flat_inds_for_net(net, within=False):
    range_list = [(0,24),(24,64), (64,69), (69,110),(110,142),(142,166),(166,213),
        (213, 221),(221, 225),(225, 263),(263, 271),(271, 294),(294, 333)]
    names_abbrev = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None',
        'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual']
    net_ind = names_abbrev.index(net)
    net_start, net_end= range_list[net_ind]

    mask = np.ones((333,333))
    mask[:, net_start:net_end+1] = 2
    mask[net_start:net_end+1, :] = 2

    if within == True:
        mask = np.ones((333,333))
        mask[net_start:net_end+1, net_start:net_end+1] = 2

    flatmask = np.triu(mask,1).flatten()[np.triu(mask,1).flatten().nonzero()]
    return np.where(flatmask == 2)[0]

# Assumes parcels are ordered by reorder indicies
def get_flat_inds_for_block(net1,net2):
    range_list = [(0,24),(24,64), (64,69), (69,110),(110,142),(142,166),(166,213),
        (213, 221),(221, 225),(225, 263),(263, 271),(271, 294),(294, 333)]
    names_abbrev = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None',
        'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual']

    net1_ind = names_abbrev.index(net1)
    net2_ind = names_abbrev.index(net2)

    net1_start, net1_end = range_list[net1_ind]
    net2_start, net2_end = range_list[net2_ind]

    mask = np.ones((333,333))
    mask[net1_start:net1_end, net2_start:net2_end+1] = 2
    mask[net2_start:net2_end, net1_start:net1_end+1] = 2

    flatmask = np.triu(mask,1).flatten()[np.triu(mask,1).flatten().nonzero()]
    return np.where(flatmask == 2)[0]


