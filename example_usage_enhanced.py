'''
A main control module to read and handle Bruker/Veeco SPM data files with comprehensive analysis.
'''
import numpy as np
from bvspm.utils import process_and_visualize_spm, calculate_sa, calculate_ra


if __name__ == "__main__":
    
    # Path to your image file
    spm_path = r"D:\Garcilazo\Python\00Exercises\CHEMAI_projects\00_Norah\AFM\Results_May29\1to20.0_00004\1to20.0_00004.spm"
        
    print(f"Processing SPM file: {spm_path}")
    # Process and visualize the SPM file with comprehensive error analysis
    heightmap, metadata, error_stats_sa, error_stats_ra = process_and_visualize_spm(spm_path)
    
    if heightmap is not None:
        print(f"\nSuccessfully processed {spm_path}")
        print(f"Image dimensions: {heightmap.shape[1]} x {heightmap.shape[0]} pixels")
        print(f"Height range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f} nm")
        print(f"Sa roughness: {calculate_sa(heightmap):.3f} ± {error_stats_sa['roughness_uncertainty']:.3f} nm")
        print(f"Ra roughness: {calculate_ra(heightmap):.3f} ± {error_stats_ra['roughness_uncertainty']:.3f} nm")
        print("Check the 'results' folder for all generated files and analysis plots.")