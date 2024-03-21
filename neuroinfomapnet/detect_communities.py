import argparse, os
import numpy as np
from rsfc_utils import create_corrmat
from network_utils import sparsity_inds, apply_sparsity
from community_detection import detect_communities_infomap

def main():
    parser = argparse.ArgumentParser(description="Detect communities present in parcellated resting state timeseries")
    parser.add_argument("timseries_path" , help="Path to the parcellated timeseries")
    parser.add_argument("sub_id", type=str, help="Sub id for file naming purposes")
    parser.add_argument("out_dir", help="Path to output results")
    parser.add_argument("--threshold", type=float, default=.05, help="Sparsity threshold to use")

    # parse args
    args = parser.parse_args()

    # load data
    example_timeseries = np.genfromtxt(args.timseries_path)

    # create correlation matrix
    corr_mat = create_corrmat(example_timeseries)

    # Apply sparsity threshold
    strongest_connections = sparsity_inds(corr_mat,args.threshold)
    sub_thresholded = apply_sparsity(corr_mat, strongest_connections)

    # Run infomap
    outname = f'pajek_{args.sub_id}_thr{args.threshold}'
    detect_communities_infomap(sub_thresholded, outname, args.out_dir)

if __name__ == "__main__":
    main()