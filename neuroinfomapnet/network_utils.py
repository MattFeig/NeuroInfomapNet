import numpy as np
from rsfc_utils import load_nii
from data.gordon2016_parcels.parcel_utils import none_inds

def sparsity_inds_adaptive(corrmat, threshold):
    """
    Generates a sparsity matrix based on adaptive thresholding of correlation matrix.

    Parameters:
    - corrmat (numpy.ndarray): The correlation matrix from which to generate the sparsity matrix.
    - threshold (float): The threshold as a proportion of the total number of nodes to determine the sparsity.

    Returns:
    - connectionmat (numpy.ndarray): A boolean matrix indicating connections between nodes after sparsification.
    """
    # Create a copy of the correlation matrix and set the diagonal to 0 to ignore self-connections
    sub = np.copy(corrmat)
    np.fill_diagonal(sub, 0)

    num_nodes = corrmat.shape[0]
    # Calculate the number of connections to retain per node, based on the threshold
    num_top_bythresh = np.ceil(num_nodes * threshold).astype(int)

    # Initialize a matrix to hold the connections post-sparsification
    connectionmat = np.zeros((num_nodes, num_nodes), dtype=bool)

    # Iterate through each node to determine its connections based on the threshold
    for i in range(num_nodes):
        # Check if there are any non-zero connections for the node
        if any(sub[:, i]):
            # Sort the connections in descending order by their correlation value
            sortinds = np.argsort(sub[:, i])[::-1]
            # Select the top connections based on the threshold and mark them as True in the connection matrix
            connectionmat[sortinds[:num_top_bythresh], i] = True
            connectionmat[i, sortinds[:num_top_bythresh]] = True  

    return connectionmat

def sparsity_inds(corrmat,threshold):
    """
    Returns a boolean array of the same size as the input 2D numpy array.
    The boolean array is True where the original array's value in the upper triangle
    is in the top 2.5% of values in the upper triangle.

    :param arr: 2D numpy array (symmetric)
    :return: Boolean 2D numpy array
    """
    arr = np.copy(corrmat)

    # Ensure the array is square and symmetric
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("The input array must be square.")

    # Extract the upper triangle values (excluding the diagonal)
    upper_triangle_values = arr[np.triu_indices(arr.shape[0], k=1)]

    # Calculate the 97.5 percentile value for the upper triangle
    threshold_value = np.percentile(upper_triangle_values, 100-(100*threshold))

    # Create a boolean array, initially all False
    bool_arr = np.zeros_like(arr, dtype=bool)

    # Set True in the boolean array where the original array's value exceeds the threshold
    bool_arr[np.triu_indices(arr.shape[0], k=1)] = upper_triangle_values > threshold_value

    # Apply identically to the lower triangle, to return the array in symettric values
    lower_triangle_values = arr[np.tril_indices(arr.shape[0], k=-1)]
    bool_arr[np.tril_indices(arr.shape[0], k=-1)] = lower_triangle_values > threshold_value

    return bool_arr

def apply_sparsity(corrmat, connectionmat):
    """
    Applies a boolean matrix to a correlation matrix, zeroing out all elements not indicated as connections.

    Parameters:
    - corrmat (numpy.ndarray): The original correlation matrix.
    - connectionmat (numpy.ndarray): A boolean matrix indicating connections between elements.

    Returns:
    - sub_thresholded (numpy.ndarray): The thresholded correlation matrix with non-connections zeroed out.
    """
    sub_thresholded = np.copy(corrmat)

    # Find and zero indices where the connection matrix is False (i.e., no connection)
    x_inverse_indices, y_inverse_indices = np.where(connectionmat == False)
    sub_thresholded[x_inverse_indices, y_inverse_indices] = 0
    
    return sub_thresholded


def template_match(community_assignments, template_cortex_path, delete_none_inds = False):
    """
    Matches community assignments to a cortical template, assigning network colors based on overlap.
    Currently hardcoded for 333 gordon parcels.
    
    Parameters:
    - community_assignments (numpy.ndarray): Array of community assignments for each cortical area.
    - delete_none_inds (bool, optional): If True, delete indices corresponding to cortical areas with no network assignment.

    Returns:
    - out_map_colored_single (numpy.ndarray): Array of network colors for each cortical area based on the highest template overlap.
    """

    template_cortex = load_nii(template_cortex_path)    # Load template
    dthr = .1   # Threshold for determining significant overlap
    
    if delete_none_inds == True:
        template_cortex = np.delete(template_cortex, none_inds)
        out_map_colored_single = np.repeat(0, 286)
    else:
        out_map_colored_single = np.repeat(0, 333)

    for p in np.arange(-1,max(community_assignments)+1): # Loop through community assignments in the current threshold
        if p in community_assignments:
            A = (community_assignments == p)
            Idx = np.where(community_assignments == p)[0]
            D_list = []

            net_colors = range(1,18)
            for templatenet in net_colors: # Loop through all the networks in a given template and calculate overlap
                B = (template_cortex.astype(int) == templatenet)
                D = np.logical_and(A,B).sum()/np.logical_or(A,B).sum()
                D_list.append(D)

            potential_matches = np.where(np.array(D_list)>dthr)[0]
            if np.max(D_list) > dthr:
                out_map_colored_single[Idx] = net_colors[np.argmax(D_list)]

    return out_map_colored_single
    
def consensus_networks(thresholds):
    """
    Determines the consensus network assignment for each node based on the last non-zero threshold encountered.
    
    Parameters:
    - thresholds (numpy.ndarray): A 2D array where each row represents a node and each column a different threshold level (sparest threshold as column 1).
    
    Returns:
    - final_network_assignments (numpy.ndarray): An array of consensus network assignments for each node.
    """
    # Initialize the output array to store the final network assignment for each node
    final_network_assignments = np.zeros(thresholds.shape[0])
    
    # Iterate over thresholds in reverse order
    for threshold_index in reversed(range(thresholds.shape[1])):
        # Iterate over all nodes
        for node_index in range(thresholds.shape[0]):
            # If a non-zero value is found for the current threshold, update the final network assignment
            if thresholds[node_index, threshold_index] != 0:
                final_network_assignments[node_index] = thresholds[node_index, threshold_index]
    
    return final_network_assignments
