import numpy as np
import struct
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import seaborn as sns

def create_results_directory(base_filename):
    """Create a results directory for output files."""
    results_dir = os.path.join(os.path.dirname(base_filename), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def evaluating_error_stats(heightmap, calculation_type="Sa", confidence_level=0.95):
    """
    Calculate comprehensive error statistics and uncertainty measures for surface roughness data.
    
    Args:
        heightmap: 2D numpy array of height values
        calculation_type: "Sa" or "Ra" to specify the type of calculation
        confidence_level: Confidence level for confidence intervals (default 0.95)
    
    Returns:
        dict: Dictionary containing various statistical measures
    """
    # Clean the data
    heightmap_clean = heightmap[~np.isnan(heightmap)]
    
    if len(heightmap_clean) == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'variance': 0.0, 'std_error': 0.0,
            'confidence_interval': (0.0, 0.0), 'coefficient_variation': 0.0,
            'skewness': 0.0, 'kurtosis': 0.0, 'sample_size': 0,
            'min_value': 0.0, 'max_value': 0.0, 'range': 0.0,
            'percentile_25': 0.0, 'percentile_75': 0.0, 'iqr': 0.0
        }
    
    # Basic statistics
    mean_val = np.mean(heightmap_clean)
    std_val = np.std(heightmap_clean, ddof=1)  # Sample standard deviation
    variance = np.var(heightmap_clean, ddof=1)  # Sample variance
    sample_size = len(heightmap_clean)
    std_error = std_val / np.sqrt(sample_size)  # Standard error of the mean
    
    # Confidence interval for the mean
    alpha = 1 - confidence_level
    dof = sample_size - 1
    t_critical = stats.t.ppf(1 - alpha/2, dof)
    margin_error = t_critical * std_error
    ci_lower = mean_val - margin_error
    ci_upper = mean_val + margin_error
    
    # Additional statistics
    coefficient_variation = (std_val / mean_val) * 100 if mean_val != 0 else np.inf
    skewness = stats.skew(heightmap_clean)
    kurt = stats.kurtosis(heightmap_clean)
    
    # Range and percentiles
    min_val = np.min(heightmap_clean)
    max_val = np.max(heightmap_clean)
    range_val = max_val - min_val
    percentile_25 = np.percentile(heightmap_clean, 25)
    percentile_75 = np.percentile(heightmap_clean, 75)
    iqr = percentile_75 - percentile_25
    
    # Calculate roughness-specific statistics
    if calculation_type.lower() == "sa":
        # For Sa: deviation from mean
        deviations = np.abs(heightmap_clean - mean_val)
        roughness_val = np.mean(deviations)
        roughness_std = np.std(deviations, ddof=1)
    elif calculation_type.lower() == "ra":
        # For Ra: calculate profile-wise and then aggregate
        ra_values = []
        # Calculate Ra for each row
        for row in range(heightmap.shape[0]):
            profile = heightmap[row, :]
            profile_clean = profile[~np.isnan(profile)]
            if len(profile_clean) > 0:
                profile_mean = np.mean(profile_clean)
                ra_row = np.mean(np.abs(profile_clean - profile_mean))
                ra_values.append(ra_row)
        
        # Calculate Ra for each column
        for col in range(heightmap.shape[1]):
            profile = heightmap[:, col]
            profile_clean = profile[~np.isnan(profile)]
            if len(profile_clean) > 0:
                profile_mean = np.mean(profile_clean)
                ra_col = np.mean(np.abs(profile_clean - profile_mean))
                ra_values.append(ra_col)
        
        if ra_values:
            roughness_val = np.mean(ra_values)
            roughness_std = np.std(ra_values, ddof=1) if len(ra_values) > 1 else 0.0
        else:
            roughness_val = 0.0
            roughness_std = 0.0
    
    return {
        'calculation_type': calculation_type,
        'mean': float(mean_val),
        'std': float(std_val),
        'variance': float(variance),
        'std_error': float(std_error),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'confidence_level': confidence_level,
        'coefficient_variation': float(coefficient_variation),
        'skewness': float(skewness),
        'kurtosis': float(kurt),
        'sample_size': int(sample_size),
        'min_value': float(min_val),
        'max_value': float(max_val),
        'range': float(range_val),
        'percentile_25': float(percentile_25),
        'percentile_75': float(percentile_75),
        'iqr': float(iqr),
        'roughness_value': float(roughness_val),
        'roughness_std': float(roughness_std),
        'roughness_uncertainty': float(roughness_std / np.sqrt(len(ra_values)) if calculation_type.lower() == "ra" and ra_values else std_error)
    }

def plot_error_statistics(heightmap, error_stats_sa, error_stats_ra, output_dir, base_filename):
    """
    Create comprehensive plots for error analysis and statistical visualization.
    
    Args:
        heightmap: 2D numpy array of height values
        error_stats_sa: Dictionary from evaluating_error_stats() for Sa
        error_stats_ra: Dictionary from evaluating_error_stats() for Ra
        output_dir: Directory to save plots
        base_filename: Base filename for saving plots
    """
    # Clean data for plotting
    heightmap_clean = heightmap[~np.isnan(heightmap)]
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Height distribution histogram
    ax1 = plt.subplot(2, 3, 1)
    n, bins, patches = ax1.hist(heightmap_clean, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(error_stats_sa['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {error_stats_sa['mean']:.3f} nm")
    ax1.axvline(error_stats_sa['mean'] + error_stats_sa['std'], color='orange', linestyle='--', alpha=0.8, label=f"+1σ: {error_stats_sa['std']:.3f} nm")
    ax1.axvline(error_stats_sa['mean'] - error_stats_sa['std'], color='orange', linestyle='--', alpha=0.8, label=f"-1σ")
    ax1.set_xlabel('Height (nm)')
    ax1.set_ylabel('Density')
    ax1.set_title('Height Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot for quartile analysis
    ax2 = plt.subplot(2, 3, 2)
    box_data = [heightmap_clean]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=['Height Data'])
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Height (nm)')
    ax2.set_title('Height Data Distribution\n(Quartiles & Outliers)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Q1: {error_stats_sa['percentile_25']:.3f} nm\nMedian: {np.median(heightmap_clean):.3f} nm\nQ3: {error_stats_sa['percentile_75']:.3f} nm\nIQR: {error_stats_sa['iqr']:.3f} nm"
    ax2.text(1.1, np.median(heightmap_clean), stats_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Roughness comparison with error bars
    ax3 = plt.subplot(2, 3, 3)
    roughness_types = ['Sa', 'Ra']
    roughness_values = [error_stats_sa['roughness_value'], error_stats_ra['roughness_value']]
    roughness_errors = [error_stats_sa['roughness_uncertainty'], error_stats_ra['roughness_uncertainty']]
    
    bars = ax3.bar(roughness_types, roughness_values, yerr=roughness_errors, capsize=10, 
                   color=['lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Roughness (nm)')
    ax3.set_title('Surface Roughness Comparison\n(with Uncertainty)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, roughness_values, roughness_errors)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.001,
                f'{val:.3f} ± {err:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Q-Q plot for normality assessment
    ax4 = plt.subplot(2, 3, 4)
    stats.probplot(heightmap_clean, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Test)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 2D heightmap with statistical overlay
    ax5 = plt.subplot(2, 3, 5)
    im = ax5.imshow(heightmap, cmap='viridis', interpolation='none')
    plt.colorbar(im, ax=ax5, label='Height (nm)')
    ax5.set_title('2D Height Map')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    
    # 6. Statistical summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create statistical summary
    summary_data = [
        ['Statistic', 'Sa Analysis', 'Ra Analysis'],
        ['Mean (nm)', f"{error_stats_sa['mean']:.4f}", f"{error_stats_ra['mean']:.4f}"],
        ['Std Dev (nm)', f"{error_stats_sa['std']:.4f}", f"{error_stats_ra['std']:.4f}"],
        ['Std Error (nm)', f"{error_stats_sa['std_error']:.4f}", f"{error_stats_ra['std_error']:.4f}"],
        ['CV (%)', f"{error_stats_sa['coefficient_variation']:.2f}", f"{error_stats_ra['coefficient_variation']:.2f}"],
        ['Skewness', f"{error_stats_sa['skewness']:.4f}", f"{error_stats_ra['skewness']:.4f}"],
        ['Kurtosis', f"{error_stats_sa['kurtosis']:.4f}", f"{error_stats_ra['kurtosis']:.4f}"],
        ['Roughness (nm)', f"{error_stats_sa['roughness_value']:.4f}", f"{error_stats_ra['roughness_value']:.4f}"],
        ['Uncertainty (nm)', f"{error_stats_sa['roughness_uncertainty']:.4f}", f"{error_stats_ra['roughness_uncertainty']:.4f}"],
    ]
    
    table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Statistical Summary', pad=20, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(base_filename)}_error_analysis.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Error analysis plot saved: {plot_filename}")
    
    plt.show()
    
    # Create a separate confidence interval plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Plot confidence intervals for both Sa and Ra
    metrics = ['Sa', 'Ra']
    values = [error_stats_sa['roughness_value'], error_stats_ra['roughness_value']]
    ci_sa = error_stats_sa['confidence_interval']
    ci_ra = error_stats_ra['confidence_interval']
    
    # For roughness confidence intervals, we'll use the roughness uncertainty
    ci_lower = [val - stats['roughness_uncertainty']*1.96 for val, stats in zip(values, [error_stats_sa, error_stats_ra])]
    ci_upper = [val + stats['roughness_uncertainty']*1.96 for val, stats in zip(values, [error_stats_sa, error_stats_ra])]
    
    x_pos = np.arange(len(metrics))
    ax.errorbar(x_pos, values, yerr=[np.array(values) - np.array(ci_lower), 
                                     np.array(ci_upper) - np.array(values)], 
                fmt='o', capsize=10, capthick=2, markersize=10, linewidth=2)
    
    ax.set_xlabel('Roughness Metric')
    ax.set_ylabel('Roughness Value (nm)')
    ax.set_title(f'Roughness Measurements with {error_stats_sa["confidence_level"]*100}% Confidence Intervals')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (val, lower, upper) in enumerate(zip(values, ci_lower, ci_upper)):
        ax.text(i, upper + 0.01, f'{val:.3f}\n[{lower:.3f}, {upper:.3f}]', 
                ha='center', va='bottom', fontweight='bold')
    
    ci_plot_filename = os.path.join(output_dir, f"{os.path.basename(base_filename)}_confidence_intervals.png")
    plt.savefig(ci_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Confidence interval plot saved: {ci_plot_filename}")
    
    plt.show()

def save_raw_tiff(heightmap, file_path, metadata=None, output_dir=None):
    """
    Save the raw heightmap data as a TIFF file with metadata.
    
    Args:
        heightmap: Numpy array containing the height data
        file_path: Original SPM file path, used to create output filename
        metadata: Dictionary of metadata (optional)
        output_dir: Directory to save files (if None, uses same directory as input)
    
    Returns:
        Path to the saved TIFF file
    """
    from skimage import io
    
    # Create output filename for TIFF
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    tif_filename = os.path.join(output_dir, f"{base_filename}_raw.tif")
    print(f"Saving raw heightmap data as: {tif_filename}")
    
    # Normalize the data for proper TIFF visualization (0-65535 for 16-bit depth)
    heightmap_norm = heightmap.copy()
    heightmap_norm = np.nan_to_num(heightmap_norm)
    
    # Normalize to full 16-bit range
    min_val = np.min(heightmap_norm)
    max_val = np.max(heightmap_norm)
    range_val = max_val - min_val
    
    if range_val > 0:
        heightmap_norm = ((heightmap_norm - min_val) / range_val * 65535).astype(np.uint16)
    else:
        heightmap_norm = np.zeros_like(heightmap_norm, dtype=np.uint16)
    
    # Save as TIFF
    io.imsave(tif_filename, heightmap_norm)
    
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
                match = re.search(r"([\d.e-]+)\s*V/LSB", metadata[key])
                if match:
                    z_scale = float(match.group(1))
    
    if z_scale is None and z_scale_line is not None:
        numbers = re.findall(r"([\d.e-]+)", z_scale_line)
        if numbers and len(numbers) > 0:
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
        
        formats_to_try = [
            ('int16', np.dtype('<i2')),
            ('float32', np.dtype('<f4')),
            ('int32', np.dtype('<i4')),
            ('uint16', np.dtype('<u2')),
        ]
        
        best_heightmap = None
        best_format = None
        
        for format_name, dtype in formats_to_try:
            try:
                f.seek(data_offset)
                data_size = samps_line * num_lines * dtype.itemsize
                binary_data = f.read(data_size)
                
                if len(binary_data) < data_size:
                    continue
                
                heightmap = np.frombuffer(binary_data, dtype=dtype).reshape(num_lines, samps_line)
                
                if np.all(heightmap == 0) or np.isnan(heightmap).any():
                    continue
                
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
            f.seek(data_offset)
            return np.fromfile(f, dtype='<f4', count=samps_line*num_lines).reshape(num_lines, samps_line)

def read_calibrated_spm(file_path):
    """
    Read SPM file with proper calibration using metadata.
    Returns calibrated heightmap in nanometers and metadata.
    """
    metadata = parse_spm_header(file_path)
    
    samps_line = int(metadata.get("Samps/line", 256))
    num_lines = int(metadata.get("Lines", 256))
    
    z_sens, z_scale, z_offset, z_magnify = get_calibration_factors(metadata)
    
    print(f"Calibration factors: z_sens={z_sens} nm/V, z_scale={z_scale} V/LSB, z_offset={z_offset} V, z_magnify={z_magnify}")
    
    data_offset = None
    if "Data offset" in metadata:
        try:
            data_offset = int(metadata["Data offset"])
            print(f"Using data offset from metadata: {data_offset}")
        except (ValueError, TypeError):
            pass
    
    if data_offset is None:
        with open(file_path, 'rb') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if b"*File list end" in line:
                    data_offset = f.tell()
                    break
                if pos > 100000:
                    raise ValueError("Could not find binary data offset")
    
        if data_offset is None:
            raise ValueError("Could not locate binary data in SPM file")
    
    print(f"Binary data starts at offset: {data_offset}")
    
    bytes_per_pixel = 4
    if "Bytes/pixel" in metadata:
        try:
            bytes_per_pixel = int(metadata["Bytes/pixel"])
            print(f"Bytes per pixel from metadata: {bytes_per_pixel}")
        except (ValueError, TypeError):
            pass
    
    if bytes_per_pixel == 4:
        dtype = np.dtype('<f4')
    elif bytes_per_pixel == 2:
        dtype = np.dtype('<i2')
    else:
        heightmap = read_binary_data_with_different_formats(file_path, data_offset, samps_line, num_lines)
        dtype = None
    
    if dtype is not None:
        with open(file_path, 'rb') as f:
            f.seek(data_offset)
            heightmap = np.fromfile(f, dtype=dtype, count=samps_line*num_lines).reshape(num_lines, samps_line)
    
    if all(v is not None for v in [z_sens, z_scale]):
        heightmap_volts = heightmap * z_scale
        heightmap_volts = heightmap_volts * z_magnify
        heightmap_nm = heightmap_volts * z_sens
        if z_offset is not None:
            heightmap_nm = heightmap_nm + (z_offset * z_sens)
    else:
        print("Warning: Could not find complete calibration data, using generic scaling")
        heightmap_nm = heightmap * 1.0
    
    heightmap_nm = np.nan_to_num(heightmap_nm, nan=0.0, posinf=0.0, neginf=0.0)
    
    return heightmap_nm, metadata

def calculate_sa(heightmap):
    """Calculate average roughness (Sa) from heightmap."""
    heightmap_clean = heightmap[~np.isnan(heightmap)]
    
    if len(heightmap_clean) == 0:
        return 0.0
    
    mean_val = np.mean(heightmap_clean)
    Sa = np.mean(np.abs(heightmap_clean - mean_val))
    return Sa

def calculate_ra(heightmap):
    """Calculate average roughness (Ra) from heightmap."""
    if np.any(np.isnan(heightmap)):
        mask = ~np.isnan(heightmap)
        if not np.any(mask):
            return 0.0
        
        heightmap_clean = heightmap.copy()
        mean_val = np.mean(heightmap[mask])
        heightmap_clean[~mask] = mean_val
    else:
        heightmap_clean = heightmap
    
    ra_rows = []
    for row in range(heightmap_clean.shape[0]):
        profile = heightmap_clean[row, :]
        mean_profile = np.mean(profile)
        ra_row = np.mean(np.abs(profile - mean_profile))
        ra_rows.append(ra_row)
    
    ra_cols = []
    for col in range(heightmap_clean.shape[1]):
        profile = heightmap_clean[:, col]
        mean_profile = np.mean(profile)
        ra_col = np.mean(np.abs(profile - mean_profile))
        ra_cols.append(ra_col)
    
    ra_all = np.mean(ra_rows + ra_cols)
    return ra_all

def process_and_visualize_spm(file_path):
    """Process and visualize SPM file with proper calibration and comprehensive error analysis."""
    try:
        heightmap, metadata = read_calibrated_spm(file_path)
        
        # Create results directory
        base_filename = os.path.splitext(file_path)[0]
        results_dir = create_results_directory(base_filename)
        
        heightmap = np.nan_to_num(heightmap)
        
        # Calculate basic roughness values
        Sa = calculate_sa(heightmap)
        Ra = calculate_ra(heightmap)
        
        # Calculate comprehensive error statistics
        print("=== Calculating Error Statistics ===")
        error_stats_sa = evaluating_error_stats(heightmap, "Sa")
        error_stats_ra = evaluating_error_stats(heightmap, "Ra")
        
        print("=== Surface Roughness Analysis with Uncertainty ===")
        print(f"Min height: {np.min(heightmap):.3f} nm")
        print(f"Max height: {np.max(heightmap):.3f} nm")
        print(f"Mean height: {np.mean(heightmap):.3f} nm ± {error_stats_sa['std_error']:.3f} nm")
        print(f"Sa roughness: {Sa:.3f} nm ± {error_stats_sa['roughness_uncertainty']:.3f} nm")
        print(f"Ra roughness: {Ra:.3f} nm ± {error_stats_ra['roughness_uncertainty']:.3f} nm")
        print(f"Coefficient of Variation (Sa): {error_stats_sa['coefficient_variation']:.2f}%")
        print(f"Coefficient of Variation (Ra): {error_stats_ra['coefficient_variation']:.2f}%")
        
        # Generate error statistics plots
        plot_error_statistics(heightmap, error_stats_sa, error_stats_ra, results_dir, base_filename)
        
        # Get scan size from metadata
        scan_size = metadata.get("Scan Size", "N/A")
        scan_size_x, scan_size_y = 500, 500
        if isinstance(scan_size, str):
            parts = scan_size.split()
            if len(parts) >= 2:
                try:
                    scan_size_x = float(parts[0])
                    scan_size_y = float(parts[1])
                except ValueError:
                    pass
        
        # Handle sparse/dotted data
        unique_values = np.unique(heightmap)
        if len(unique_values) < 1000 or np.count_nonzero(heightmap) < 0.1 * heightmap.size:
            print("Detected sparse/dotted data, applying preprocessing...")
            heightmap = gaussian_filter(heightmap, sigma=1.0)
            
            if np.count_nonzero(heightmap == 0) > 0.8 * heightmap.size:
                print("Applying interpolation to sparse data...")
                mask = heightmap != 0
                coords = np.array(np.nonzero(mask)).T
                values = heightmap[mask]
                
                xi, yi = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]
                heightmap = griddata(coords, values, (xi, yi), method='cubic', fill_value=0)
        
        # Create visualization with percentile clipping
        p1, p99 = np.percentile(heightmap, [1, 99])
        heightmap_clipped = np.clip(heightmap, p1, p99)
        
        # Create Gwyddion-style visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        img = ax.imshow(heightmap_clipped, cmap='gray', 
                       extent=[0, scan_size_x, 0, scan_size_y],
                       interpolation='none')
        
        cbar = plt.colorbar(img, ax=ax, label='Height (nm)')
        ax.set_title(f"Surface Topography\nScan Size: {scan_size_x} × {scan_size_y} nm\nSa = {Sa:.2f} ± {error_stats_sa['roughness_uncertainty']:.2f} nm, Ra = {Ra:.2f} ± {error_stats_ra['roughness_uncertainty']:.2f} nm")
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        
        # Save main visualization
        main_plot_filename = os.path.join(results_dir, f"{os.path.basename(base_filename)}_heightmap.png")
        plt.savefig(main_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Main heightmap saved: {main_plot_filename}")
        plt.show()
        
        # Save raw TIFF file
        tiff_filename = save_raw_tiff(heightmap, file_path, metadata, results_dir)
        
        # Save comprehensive metadata including error statistics
        import json
        meta_filename = os.path.join(results_dir, f"{os.path.basename(base_filename)}_metadata.json")
        
        comprehensive_metadata = {
            "original_metadata": {
                "Scan Size": metadata.get("Scan Size", "N/A"),
                "Samps/line": metadata.get("Samps/line", "N/A"),
                "Lines": metadata.get("Lines", "N/A"),
                "Z sensitivity": metadata.get("@Sens. Zsens", "N/A"),
                "Z scale": metadata.get("@2:Z scale", "N/A"),
            },
            "basic_statistics": {
                "Original min height (nm)": float(np.min(heightmap)),
                "Original max height (nm)": float(np.max(heightmap)),
                "Original mean height (nm)": float(np.mean(heightmap)),
                "Sa roughness (nm)": float(Sa),
                "Ra roughness (nm)": float(Ra)
            },
            "error_analysis_sa": error_stats_sa,
            "error_analysis_ra": error_stats_ra,
            "files_generated": {
                "tiff_file": os.path.basename(tiff_filename),
                "main_plot": os.path.basename(main_plot_filename),
                "error_analysis_plot": f"{os.path.basename(base_filename)}_error_analysis.png",
                "confidence_intervals_plot": f"{os.path.basename(base_filename)}_confidence_intervals.png"
            }
        }
        
        with open(meta_filename, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2)
        
        print(f"Comprehensive metadata saved to: {meta_filename}")
        print(f"All results saved in directory: {results_dir}")
        
        return heightmap, metadata, error_stats_sa, error_stats_ra
        
    except Exception as e:
        print(f"Error processing SPM file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Example usage
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1:
#         spm_path = sys.argv[1]
#     else:
#         spm_path = "./images/test.spm"  # Default name if not provided
        
#     print(f"Processing SPM file: {spm_path}")
#     # Process and visualize the SPM file with comprehensive error analysis
#     heightmap, metadata, error_stats_sa, error_stats_ra = process_and_visualize_spm(spm_path)
    
#     if heightmap is not None:
#         print(f"\nSuccessfully processed {spm_path}")
#         print(f"Image dimensions: {heightmap.shape[1]} x {heightmap.shape[0]} pixels")
#         print(f"Height range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f} nm")
#         print(f"Sa roughness: {calculate_sa(heightmap):.3f} ± {error_stats_sa['roughness_uncertainty']:.3f} nm")
#         print(f"Ra roughness: {calculate_ra(heightmap):.3f} ± {error_stats_ra['roughness_uncertainty']:.3f} nm")
#         print("Check the 'results' folder for all generated files and analysis plots.")