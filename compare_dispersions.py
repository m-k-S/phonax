import numpy as np
from scipy.interpolate import interp1d

def normalize_path_coordinates(qpoints):
    """
    Convert q-points to normalized path coordinates [0,1].
    
    Parameters:
    -----------
    qpoints : ndarray
        Array of shape (n_points, 3) containing q-points
        
    Returns:
    --------
    ndarray
        Normalized path coordinates
    """
    diffs = np.diff(qpoints, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    return cumulative / cumulative[-1]

def compare_frequency_segment(qpoints1, freq1, qpoints2, freq2, n_points=100):
    """
    Compare frequencies along one segment of the path.
    
    Parameters:
    -----------
    qpoints1, qpoints2 : ndarray
        Q-points for each calculation, shapes (n1, 3) and (n2, 3)
    freq1, freq2 : ndarray
        Frequencies for each calculation, shapes (n1, n_modes) and (n2, n_modes)
    n_points : int
        Number of points to use for interpolation
        
    Returns:
    --------
    dict
        Comparison metrics for this segment
    """
    # Get normalized coordinates
    coords1 = normalize_path_coordinates(qpoints1)
    coords2 = normalize_path_coordinates(qpoints2)
    
    # Create common coordinate system
    common_coords = np.linspace(0, 1, n_points)
    
    # Initialize arrays for interpolated frequencies
    n_modes = freq1.shape[1]
    freq1_interp = np.zeros((n_points, n_modes))
    freq2_interp = np.zeros((n_points, n_modes))
    
    # Interpolate each mode separately
    for mode in range(n_modes):
        interpolator1 = interp1d(coords1, freq1[:, mode], kind='linear')
        interpolator2 = interp1d(coords2, freq2[:, mode], kind='linear')
        
        freq1_interp[:, mode] = interpolator1(common_coords)
        freq2_interp[:, mode] = interpolator2(common_coords)
    
    # Calculate differences
    abs_diff = np.abs(freq1_interp - freq2_interp)
    rel_diff = abs_diff / (np.abs(freq2_interp) + 1e-10)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(np.mean(abs_diff**2)),
        'mae': np.mean(abs_diff),
        'max_diff': np.max(abs_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'mode_wise_rmse': np.sqrt(np.mean(abs_diff**2, axis=0)),  # RMSE per mode
        'interpolated_freqs1': freq1_interp,
        'interpolated_freqs2': freq2_interp,
        'common_coords': common_coords
    }
    
    return metrics

def compare_all_segments(qpoints1_list, freq1_list, qpoints2_list, freq2_list):
    """
    Compare frequencies across all segments.
    
    Parameters:
    -----------
    qpoints1_list, qpoints2_list : list of ndarray
        Lists of q-point arrays for each segment
    freq1_list, freq2_list : list of ndarray
        Lists of frequency arrays for each segment
        
    Returns:
    --------
    dict
        Comprehensive comparison metrics
    """
    n_segments = len(qpoints1_list)
    segment_metrics = []
    
    for i in range(n_segments):
        metrics = compare_frequency_segment(
            qpoints1_list[i], freq1_list[i],
            qpoints2_list[i], freq2_list[i]
        )
        segment_metrics.append(metrics)
    
    # Calculate overall metrics
    n_modes = freq1_list[0].shape[1]
    mode_wise_rmse = np.zeros(n_modes)
    for mode in range(n_modes):
        # Combine all segments for this mode
        all_diffs = []
        for seg_metrics in segment_metrics:
            freq1 = seg_metrics['interpolated_freqs1'][:, mode]
            freq2 = seg_metrics['interpolated_freqs2'][:, mode]
            all_diffs.extend(np.abs(freq1 - freq2))
        mode_wise_rmse[mode] = np.sqrt(np.mean(np.array(all_diffs)**2))
    
    overall_metrics = {
        'mean_rmse': np.mean([m['rmse'] for m in segment_metrics]),
        'max_rmse': np.max([m['rmse'] for m in segment_metrics]),
        'overall_mae': np.mean([m['mae'] for m in segment_metrics]),
        'mean_rel_diff': np.mean([m['mean_rel_diff'] for m in segment_metrics]),
        'mode_wise_rmse': mode_wise_rmse,
        'segment_wise': segment_metrics,
        'sampling_points': {
            'calc1': [q.shape[0] for q in qpoints1_list],
            'calc2': [q.shape[0] for q in qpoints2_list]
        }
    }
    
    return overall_metrics

def print_comparison_report(metrics):
    """
    Print a formatted report of the comparison metrics.
    """
    print("\nOverall Frequency Comparison Metrics:")
    print(f"Mean RMSE across segments: {metrics['mean_rmse']:.6f}")
    print(f"Maximum RMSE in any segment: {metrics['max_rmse']:.6f}")
    print(f"Overall MAE: {metrics['overall_mae']:.6f}")
    print(f"Mean relative difference: {metrics['mean_rel_diff']:.6f}")
    
    print("\nSegment-wise sampling points:")
    for i, (n1, n2) in enumerate(zip(
            metrics['sampling_points']['calc1'],
            metrics['sampling_points']['calc2'])):
        print(f"Segment {i+1}: {n1} points vs {n2} points")
    
    print("\nSegment-wise Metrics:")
    for i, seg_metrics in enumerate(metrics['segment_wise']):
        print(f"\nSegment {i+1}:")
        print(f"  RMSE: {seg_metrics['rmse']:.6f}")
        print(f"  MAE: {seg_metrics['mae']:.6f}")
        print(f"  Maximum difference: {seg_metrics['max_diff']:.6f}")
        print(f"  Mean relative difference: {seg_metrics['mean_rel_diff']:.6f}")
    
    print("\nMode-wise RMSE:")
    for i, rmse in enumerate(metrics['mode_wise_rmse']):
        print(f"Mode {i+1}: {rmse:.6f}")