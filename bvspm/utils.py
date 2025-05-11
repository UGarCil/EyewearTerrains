import numpy as np
import struct
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def save_raw_tiff(heightmap, file_path, metadata=None):
    """
    Save the raw heightmap data as a TIFF file with metadata.
    
    Args:
        heightmap: Numpy array containing the height data
        file_path: Original SPM file path, used to create output filename
        metadata: Dictionary of metadata (optional)
    
    Returns:
        Path to the saved TIFF file
    """
    import os
    from skimage import io
    import numpy as np
    
    # Create output filename for TIFF (replacing .spm with _raw.tif)
    base_filename = os.path.splitext(file_path)[0]
    tif_filename = f"{base_filename}_raw.tif"
    print(f"Saving raw heightmap data as: {tif_filename}")
    
    # Normalize the data for proper TIFF visualization (0-65535 for 16-bit depth)
    # This preserves the relative height differences
    heightmap_norm = heightmap.copy()
    # Handle NaN values
    heightmap_norm = np.nan_to_num(heightmap_norm)
    
    # Normalize to full 16-bit range
    min_val = np.min(heightmap_norm)
    max_val = np.max(heightmap_norm)
    range_val = max_val - min_val
    
    if range_val > 0:  # Avoid division by zero
        heightmap_norm = ((heightmap_norm - min_val) / range_val * 65535).astype(np.uint16)
    else:
        heightmap_norm = np.zeros_like(heightmap_norm, dtype=np.uint16)
    
    # Save as TIFF
    io.imsave(tif_filename, heightmap_norm)
    
    # Save metadata to a companion JSON file if provided
    if metadata:
        import json
        meta_filename = f"{base_filename}_metadata.json"
        
        # Select essential metadata to save
        essential_metadata = {
            "Scan Size": metadata.get("Scan Size", "N/A"),
            "Samps/line": metadata.get("Samps/line", "N/A"),
            "Lines": metadata.get("Lines", "N/A"),
            "Z sensitivity": metadata.get("@Sens. Zsens", "N/A"),
            "Z scale": metadata.get("@2:Z scale", "N/A"),
            "Original min height (nm)": float(np.min(heightmap)),
            "Original max height (nm)": float(np.max(heightmap)),
            "Original mean height (nm)": float(np.mean(heightmap)),
            "Sa roughness (nm)": float(calculate_sa(heightmap))
        }
        
        with open(meta_filename, 'w') as f:
            json.dump(essential_metadata, f, indent=2)
        
        print(f"Saved metadata to: {meta_filename}")
    
    return tif_filename


def parse_spm_header(file_path):
    """Parse the ASCII header of an SPM file and extract relevant metadata."""
    metadata = {}
    with open(file_path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if "*File list end" in line:
                break
            
            if line.startswith("\\"):
                key_value = line[1:].split(":", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key.strip()] = value.strip()
    
    return metadata

def get_calibration_factors(metadata):
    """Extract calibration factors from metadata."""
    # Get Z sensitivity (nm/V) from metadata
    z_sens = None
    for key in ["@Sens. Zsens", "@Sens. ZsensSens"]:
        if key in metadata:
            match = re.search(r"([\d.]+)\s*nm/V", metadata[key])
            if match:
                z_sens = float(match.group(1))
                break
    
    # Get Z scale (V/LSB) from metadata
    z_scale = None
    z_scale_line = None
    for key in ["@2:Z scale"]:
        if key in metadata:
            z_scale_line = metadata[key]
            match = re.search(r"\(([\d.]+)\s*V/LSB\)", metadata[key])
            if match:
                z_scale = float(match.group(1))
            else:
                # Try alternative format that might be present in the metadata
                match = re.search(r"([\d.e-]+)\s*V/LSB", metadata[key])
                if match:
                    z_scale = float(match.group(1))
    
    # If the above method didn't work, try to extract full scale data
    if z_scale is None and z_scale_line is not None:
        # Try to extract the floating point number directly
        numbers = re.findall(r"([\d.e-]+)", z_scale_line)
        if numbers and len(numbers) > 0:
            # The first number is typically the scale
            try:
                z_scale = float(numbers[0])
            except ValueError:
                pass
    
    # Get Z offset (V) from metadata
    z_offset = 0.0
    for key in ["@2:Z offset"]:
        if key in metadata:
            match = re.search(r"([-\d.]+)\s*V", metadata[key])
            if match:
                z_offset = float(match.group(1))
            else:
                # Alternative format extraction
                numbers = re.findall(r"([-\d.e-]+)", metadata[key])
                if numbers and len(numbers) > 0:
                    try:
                        z_offset = float(numbers[0])
                    except ValueError:
                        pass
    
    # Get Z magnify factor if present
    z_magnify = 1.0
    for key in ["@Z magnify"]:
        if key in metadata:
            match = re.search(r"([\d.e-]+)", metadata[key])
            if match:
                try:
                    z_magnify = float(match.group(1))
                except ValueError:
                    pass
    
    return z_sens, z_scale, z_offset, z_magnify

def read_binary_data_with_different_formats(file_path, data_offset, samps_line, num_lines):
    """Try different binary formats to read the data correctly."""
    with open(file_path, 'rb') as f:
        f.seek(data_offset)
        
        # Try different data formats
        formats_to_try = [
            ('int16', np.dtype('<i2')),  # Little-endian 16-bit int
            ('float32', np.dtype('<f4')), # Little-endian 32-bit float
            ('int32', np.dtype('<i4')),   # Little-endian 32-bit int
            ('uint16', np.dtype('<u2')),  # Little-endian 16-bit unsigned int
        ]
        
        best_heightmap = None
        best_format = None
        
        for format_name, dtype in formats_to_try:
            try:
                f.seek(data_offset)
                data_size = samps_line * num_lines * dtype.itemsize
                binary_data = f.read(data_size)
                
                if len(binary_data) < data_size:
                    continue  # Not enough data for this format
                
                heightmap = np.frombuffer(binary_data, dtype=dtype).reshape(num_lines, samps_line)
                
                # Check if data looks reasonable (not all zeros or NaNs)
                if np.all(heightmap == 0) or np.isnan(heightmap).any():
                    continue
                
                # This is our candidate
                best_heightmap = heightmap.copy()
                best_format = format_name
                break
                
            except Exception as e:
                print(f"Failed to read with format {format_name}: {e}")
                continue
        
        if best_heightmap is not None:
            print(f"Successfully read data with format: {best_format}")
            return best_heightmap
        else:
            # Fallback to the original method
            f.seek(data_offset)
            return np.fromfile(f, dtype='<f4', count=samps_line*num_lines).reshape(num_lines, samps_line)

def read_calibrated_spm(file_path):
    """
    Read SPM file with proper calibration using metadata.
    Returns calibrated heightmap in nanometers and metadata.
    """
    metadata = parse_spm_header(file_path)
    
    # Get image dimensions from metadata
    samps_line = int(metadata.get("Samps/line", 256))
    num_lines = int(metadata.get("Lines", 256))
    
    # Get calibration factors (with z_magnify)
    z_sens, z_scale, z_offset, z_magnify = get_calibration_factors(metadata)
    
    # Print calibration factors for debugging
    print(f"Calibration factors: z_sens={z_sens} nm/V, z_scale={z_scale} V/LSB, z_offset={z_offset} V, z_magnify={z_magnify}")
    
    # Check for data offset from metadata if available
    data_offset = None
    if "Data offset" in metadata:
        try:
            data_offset = int(metadata["Data offset"])
            print(f"Using data offset from metadata: {data_offset}")
        except (ValueError, TypeError):
            pass
    
    # If not found in metadata, locate it in the file
    if data_offset is None:
        with open(file_path, 'rb') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if b"*File list end" in line:
                    data_offset = f.tell()
                    break
                if pos > 100000:  # Safety check
                    raise ValueError("Could not find binary data offset")
    
        if data_offset is None:
            raise ValueError("Could not locate binary data in SPM file")
    
    print(f"Binary data starts at offset: {data_offset}")
    
    # Determine data type from metadata
    bytes_per_pixel = 4  # Default to 4 bytes (32-bit float)
    if "Bytes/pixel" in metadata:
        try:
            bytes_per_pixel = int(metadata["Bytes/pixel"])
            print(f"Bytes per pixel from metadata: {bytes_per_pixel}")
        except (ValueError, TypeError):
            pass
    
    # Read binary data with format appropriate for bytes_per_pixel
    if bytes_per_pixel == 4:
        dtype = np.dtype('<f4')  # 32-bit float
    elif bytes_per_pixel == 2:
        dtype = np.dtype('<i2')  # 16-bit int
    else:
        # If unusual format, try different formats
        heightmap = read_binary_data_with_different_formats(file_path, data_offset, samps_line, num_lines)
        dtype = None
    
    # If dtype was determined, read data directly
    if dtype is not None:
        with open(file_path, 'rb') as f:
            f.seek(data_offset)
            heightmap = np.fromfile(f, dtype=dtype, count=samps_line*num_lines).reshape(num_lines, samps_line)
    
    # Apply calibration if we have all factors
    if all(v is not None for v in [z_sens, z_scale]):
        # Convert from digital units to volts
        heightmap_volts = heightmap * z_scale
        # Apply Z magnify factor if present
        heightmap_volts = heightmap_volts * z_magnify
        # Convert from volts to nanometers using sensitivity
        heightmap_nm = heightmap_volts * z_sens
        # Apply offset
        if z_offset is not None:
            heightmap_nm = heightmap_nm + (z_offset * z_sens)
    else:
        # Fallback to raw data with a generic scaling if calibration not available
        print("Warning: Could not find complete calibration data, using generic scaling")
        # Apply a generic scaling to make the data visible
        heightmap_nm = heightmap * 1.0  # Use raw values but with float type
    
    # Handle any NaN or inf values
    heightmap_nm = np.nan_to_num(heightmap_nm, nan=0.0, posinf=0.0, neginf=0.0)
    
    return heightmap_nm, metadata

def calculate_sa(heightmap):
    """Calculate average roughness (Sa) from heightmap."""
    # Remove any potential NaN values before calculation
    heightmap_clean = heightmap[~np.isnan(heightmap)]
    
    if len(heightmap_clean) == 0:
        return 0.0  # Return 0 if all values are NaN
    
    mean_val = np.mean(heightmap_clean)
    Sa = np.mean(np.abs(heightmap_clean - mean_val))
    return Sa

def process_and_visualize_spm(file_path):
    """Process and visualize SPM file with proper calibration."""
    try:
        heightmap, metadata = read_calibrated_spm(file_path)
        
        # Ensure heightmap has no NaN values for visualization
        heightmap = np.nan_to_num(heightmap)
        
        # Calculate statistics
        Sa = calculate_sa(heightmap)
        
        print("=== Surface Roughness Analysis ===")
        print(f"Min height: {np.min(heightmap):.3f} nm")
        print(f"Max height: {np.max(heightmap):.3f} nm")
        print(f"Mean height: {np.mean(heightmap):.3f} nm")
        print(f"Sa roughness: {Sa:.3f} nm")
        
        # Get scan size from metadata for plot labels
        scan_size = metadata.get("Scan Size", "N/A")
        
        # Add preprocessing to fix the dots issue:
        # Check if the image seems to have isolated dots (may be due to specific data format)
        unique_values = np.unique(heightmap)
        if len(unique_values) < 1000 or np.count_nonzero(heightmap) < 0.1 * heightmap.size:
            print("Detected sparse/dotted data, applying preprocessing...")
            
            # Option 1: Apply Gaussian filter to connect the dots (if they're actually signal points)
            from scipy.ndimage import gaussian_filter
            heightmap = gaussian_filter(heightmap, sigma=1.0)
            
            # Option 2: If there are many zeros with some non-zero points, try interpolation
            if np.count_nonzero(heightmap == 0) > 0.8 * heightmap.size:
                print("Applying interpolation to sparse data...")
                # Create a mask of valid (non-zero) points
                mask = heightmap != 0
                # Get coordinates of valid points
                coords = np.array(np.nonzero(mask)).T
                # Get values of valid points
                values = heightmap[mask]
                
                # Create grid for interpolation
                from scipy.interpolate import griddata
                xi, yi = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]
                # Interpolate
                heightmap = griddata(coords, values, (xi, yi), method='cubic', fill_value=0)
        
        # For better visualization, use percentile-based clipping to handle outliers
        p1, p99 = np.percentile(heightmap, [1, 99])
        heightmap_clipped = np.clip(heightmap, p1, p99)
        
        
        # Extract scan size for proper scaling
        scan_size_x, scan_size_y = 500, 500  # Default if parsing fails
        if isinstance(scan_size, str):
            parts = scan_size.split()
            if len(parts) >= 2:
                try:
                    scan_size_x = float(parts[0])
                    scan_size_y = float(parts[1])
                except ValueError:
                    pass
                
        # # Create more meaningful colormap
        # fig, ax = plt.subplots(figsize=(10, 8))
        
        # # Use 'viridis' colormap for better detail visibility, similar to Gwyddion
        # img = ax.imshow(heightmap_clipped, cmap='viridis', 
        #                extent=[0, scan_size_x, 0, scan_size_y],
        #                interpolation='none')  # Use 'none' for raw pixel display
        
        # # Add colorbar with appropriate scaling
        # cbar = plt.colorbar(img, ax=ax, label='Height (nm)')
        
        # # Add title and labels
        # ax.set_title(f"Surface Topography\nScan Size: {scan_size_x} × {scan_size_y} nm, Sa = {Sa:.2f} nm")
        # ax.set_xlabel("X (nm)")
        # ax.set_ylabel("Y (nm)")
        
        # # Save the image
        # plt.savefig('spm_heightmap.png', dpi=300, bbox_inches='tight')
        
        # Create a second visualization that more closely matches Gwyddion's appearance
        _, ax2 = plt.subplots(figsize=(10, 8))
        
        # Use a grayscale colormap more similar to the Gwyddion example
        img2 = ax2.imshow(heightmap_clipped, cmap='gray', 
                        extent=[0, scan_size_x, 0, scan_size_y],
                        interpolation='none')
        
        cbar2 = plt.colorbar(img2, ax=ax2, label='Height (nm)')
        ax2.set_title(f"Surface Topography (Gwyddion Style)\nScan Size: {scan_size_x} × {scan_size_y} nm, Sa = {Sa:.2f} nm")
        ax2.set_xlabel("X (nm)")
        ax2.set_ylabel("Y (nm)")
        
        plt.savefig('spm_heightmap_gwyddion_style.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return heightmap, metadata
        
    except Exception as e:
        print(f"Error processing SPM file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Example usage
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1:
#         spm_path = sys.argv[1]
#     else:
#         spm_path = "./images/test.spm"  # Default name if not provided
        
#     print(f"Processing SPM file: {spm_path}")
#     # Process and visualize the SPM file
#     heightmap, metadata = process_and_visualize_spm(spm_path)
    
#     if heightmap is not None:
#         # Print basic statistics about the heightmap
#         print(f"\nSuccessfully processed {spm_path}")
#         print(f"Image dimensions: {heightmap.shape[1]} x {heightmap.shape[0]} pixels")
#         print(f"Height range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f} nm")
#         print(f"Average roughness (Sa): {calculate_sa(heightmap):.3f} nm")
#         save_raw_tiff(heightmap, spm_path, metadata)