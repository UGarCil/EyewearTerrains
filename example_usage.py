'''
A main control module to read and handle Bruker/Veeco spm data files.
'''
import numpy as np
from bvspm.utils import process_and_visualize_spm, save_raw_tiff, calculate_sa

# path to your image file
path_file = r"D:\Garcilazo\Python\00Exercises\CHEMAI_projects\00_Norah\AFM\PolyCarb_10um.0_00008.spm"
# Get the heighmap image and metadata from the BVspm file
heightmap, metadata = process_and_visualize_spm(path_file)

# If heighmap ecists, print some basic statistics
# save the stats and heightmap information
if heightmap is not None:
    # Print basic statistics about the heightmap
    print(f"\nSuccessfully processed {path_file}")
    print(f"Image dimensions: {heightmap.shape[1]} x {heightmap.shape[0]} pixels")
    print(f"Height range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f} nm")
    print(f"Average roughness (Sa): {calculate_sa(heightmap):.3f} nm")
    save_raw_tiff(heightmap, path_file, metadata)