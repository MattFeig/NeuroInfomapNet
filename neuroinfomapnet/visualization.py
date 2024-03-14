import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def convert_net_number_to_color(network_list):
    """
    Converts a list of gordon network labels to their corresponding color strings.

    Parameters:
    - network_list (list of int): List of  network labels.

    Returns:
    - List of strings representing the color of each community.
    """
    color_list_parcel = np.array(['white', 'red', 'darkblue', 'yellow', 'grey', 'lime', 'grey',
                                  'teal', 'black', 'purple', 'cyan', 'orange', 'darkviolet', 'grey',
                                  'grey', 'blue', 'wheat'])
    
    # Use list comprehension for concise and efficient mapping
    color_mapping = [color_list_parcel[int(comm)] if int(comm) != -1 else 'white' for comm in network_list]
    
    return color_mapping

def expand_to333(arr, none_inds):
    """
    Expands an array to a larger size by inserting a specific value (-1) at predefined indices.

    Parameters:
    - arr (numpy.ndarray): The original array to expand.
    - none_inds (list of int): Indices at which to insert the value -1.

    Returns:
    - expanded (numpy.ndarray): The expanded array.
    """
    expanded = np.copy(arr)
    for ind in none_inds:
        expanded = np.insert(expanded, ind, -1)
    return expanded

def plot_network_communities(network, communities):
    """
    Plots a network with its communities, coloring nodes based on their community.

    Parameters:
    - network (networkx.Graph): The network to plot.
    - communities (dict): A dictionary mapping node identifiers to their community.
    """
    # Create a color map from communities
    community_colors = {node: communities[node] for node in network.nodes()}
    
    # Draw the network
    pos = nx.spring_layout(network)  # Use spring layout for positioning nodes
    nx.draw_networkx_edges(network, pos)
    nx.draw_networkx_nodes(network, pos, node_color=list(community_colors.values()), 
                           cmap=plt.cm.jet, node_size=50)
    plt.show()
