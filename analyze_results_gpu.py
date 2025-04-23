#!/usr/bin/env python3
# Import libraries and check availability
try:
    import cudf
    has_cudf = True
except ImportError:
    has_cudf = False
    print("CUDA GPU support not available. Using CPU only.")

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr
import glob
import argparse
# multiprocessing is no longer needed – we always run serially
import datetime
import re
import time

parser = argparse.ArgumentParser(description="Analyze benchmark results.")
parser.add_argument('--input_dir', type=str, default="./bench-venkat/analysis/advanced", help='Directory containing CSV results to analyze')
parser.add_argument('--output_dir', type=str, help='Directory to store analysis results (default: input_dir/analysis_TIMESTAMP)')
# Parallel-execution options kept so existing CLI doesn't break,
# but they are ignored to force serial execution.
parser.add_argument('--parallel', action='store_true',
                    help='(ignored) Always runs serially now')
parser.add_argument('--max_processes', type=int, default=1,
                    help='(ignored) Always runs serially now')
# -----------------------------------------------------------------
# Always run serially, regardless of the CLI flags ----------------
# -----------------------------------------------------------------
args = parser.parse_args()
args.parallel = False        # hard-override
args.max_processes = 1       # hard-override

# Create output directory structure
input_dir = args.input_dir
if args.output_dir:
    base_output_dir = args.output_dir
else:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("bench-venkat", "analysis", "output", f"analysis_{timestamp}")

# Create output directory structure
os.makedirs(base_output_dir, exist_ok=True)

# Create directory for comparison data (for text files only)
comparison_dir = os.path.join(base_output_dir, "comparison_data")
os.makedirs(comparison_dir, exist_ok=True)

# Create a main log file
log_file = os.path.join(base_output_dir, "analysis_log.txt")
with open(log_file, 'w') as f:
    f.write(f"Analysis started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input directory: {input_dir}\n")
    f.write(f"Output directory: {base_output_dir}\n")
    f.write(f"Parallel processing: DISABLED (always runs serially)\n")
    f.write(f"Max processes: 1\n")
    f.write("\n" + "="*80 + "\n\n")

# Function to write to the main log file
def write_to_log(content):
    with open(log_file, 'a') as f:
        f.write(content + '\n')
        
# Function to write to a CSV-specific analysis file
def write_to_csv_analysis(csv_dir, content):
    analysis_file = os.path.join(csv_dir, "analysis_results.txt")
    with open(analysis_file, 'a') as f:
        f.write(content + '\n')

# Function to get a clean name for a file
def get_clean_name(filename):
    # Extract the base filename without path and extension
    base_name = os.path.basename(filename)
    clean_name = os.path.splitext(base_name)[0]
    # Remove any invalid characters for directory names
    clean_name = re.sub(r'[^\w\-_]', '_', clean_name)
    return clean_name

# Function to analyze a single CSV file
def analyze_csv_file(filename):
    # Get CSV base name for better logging
    base_name = os.path.basename(filename)
    
    # Create output directory for this CSV file
    csv_name = get_clean_name(filename)
    csv_output_dir = os.path.join(base_output_dir, csv_name)
    
    # Create directory structure for this CSV file
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_dir = os.path.join(csv_output_dir, "csv_data")
    # os.makedirs(csv_dir, exist_ok=True)
    
    # Create analysis file for this CSV
    analysis_file = os.path.join(csv_output_dir, "analysis_results.txt")
    with open(analysis_file, 'w') as f:
        f.write(f"Analysis of {filename}\n")
        f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Log to main log file
    write_to_log(f"\nAnalyzing: {csv_name}")
    write_to_log(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"LOADING FILE: {base_name}")
    write_to_log(f"LOADING FILE: {base_name}") 
    
    try:
        # Determine file size for logging
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        msg = f"File size: {file_size_mb:.2f} MB"
        write_to_log(msg)
        write_to_csv_analysis(csv_output_dir, msg)
        
        # Load the CSV file with cuDF for GPU acceleration if available
        start_time = time.time()
        try:
            if not has_cudf:
                raise ImportError("cuDF not available")

            start_time = time.time()
            df = cudf.read_csv(filename)
            load_time = time.time() - start_time
            msg = f"Loaded with cuDF in {load_time:.4f} seconds"
            write_to_csv_analysis(csv_output_dir, msg)
            write_to_log(msg)
        except Exception as e:
            # cuDF failed – if it looks like an OOM, log & skip.  No CPU fallback.
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or "cudaErrorMemoryAllocation" in err_msg:
                oom_msg = f"SKIPPED {base_name}: GPU out of memory ({err_msg})"
                write_to_csv_analysis(csv_output_dir, oom_msg)
                write_to_log(oom_msg)
                return None

            # Any other cuDF error: log & skip
            error_msg = f"ERROR: cuDF failed on {base_name}: {err_msg}"
            write_to_csv_analysis(csv_output_dir, error_msg)
            write_to_log(error_msg)
            return None
        
        # Try to extract bucket size from filename if not in data
        if 'bucket_size' not in df.columns:
            try:
                b_size = int(filename.split('_b')[-1].split('.csv')[0])
                df['bucket_size'] = b_size
                write_to_csv_analysis(csv_output_dir, f"Inferring bucket_size {b_size} from filename")
            except Exception as e:
                write_to_csv_analysis(csv_output_dir, f"Warning: Could not infer bucket_size from filename: {e}")
        
        # Return the dataframe and output directories
        return {
            'filename': filename,
            'csv_name': csv_name,
            'dataframe': df,
            'output_dir': csv_output_dir,
            'csv_dir': csv_dir,
            'analysis_file': analysis_file,
            'is_cudf': isinstance(df, cudf.DataFrame)
        }
    
    except Exception as e:
        error_msg = f"Error loading CSV file: {e}"
        write_to_csv_analysis(csv_output_dir, error_msg)
        write_to_log(f"Error processing {filename}: {e}")
        return None

# Main analysis functions
def calculate_basic_stats(df, algo_name, bucket_size):
    """Calculate basic statistics for a given algorithm and bucket size"""
    start_time = time.time()
    
    # Check if we're using cuDF or pandas
    is_cudf = isinstance(df, cudf.DataFrame)
    
    # Filter data for this algorithm and bucket size
    subset = df[(df['algorithm'] == algo_name) & (df['bucket_size'] == bucket_size)]
    
    # Basic stats
    count = len(subset)
    if count == 0:
        return {}
        
    # Normalized position stats
    if 'normalized' in subset.columns:
        norm_min = subset['normalized'].min()
        norm_max = subset['normalized'].max()
        norm_mean = subset['normalized'].mean()
        norm_std = subset['normalized'].std()
    else:
        norm_min = norm_max = norm_mean = norm_std = np.nan
    
    # Fairness correlation (token count vs normalized position)
    if 'user_token_sum' in subset.columns and 'normalized' in subset.columns:
        # For cuDF, we need to convert to numpy arrays for pearsonr
        if is_cudf:
            token_array = subset['user_token_sum'].to_numpy()
            norm_array = subset['normalized'].to_numpy()
            corr, p_value = pearsonr(token_array, norm_array)
        else:
            corr, p_value = pearsonr(subset['user_token_sum'], subset['normalized'])
    else:
        corr = p_value = np.nan
    
    # Format results
    result = {
        'algorithm': algo_name,
        'bucket_size': bucket_size,
        'count': count,
        'norm_min': float(norm_min) if not np.isnan(norm_min) else np.nan,
        'norm_max': float(norm_max) if not np.isnan(norm_max) else np.nan,
        'norm_mean': float(norm_mean) if not np.isnan(norm_mean) else np.nan,
        'norm_std': float(norm_std) if not np.isnan(norm_std) else np.nan,
        'fairness_correlation': float(corr) if not np.isnan(corr) else np.nan,
        'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
        'compute_time': time.time() - start_time,
        'gpu_accelerated': is_cudf
    }
    
    return result

def check_monotonicity(df, algo_name, bucket_size, max_users=100): 
    """Check if token count monotonically increases with normalized position"""
    start_time = time.time()
    
    # Check if we're using cuDF or pandas
    is_cudf = isinstance(df, cudf.DataFrame)
    
    # Check if necessary columns exist
    if 'user_id' not in df.columns or 'user_token_sum' not in df.columns or 'normalized' not in df.columns:
        return {
            'algorithm': algo_name,
            'bucket_size': bucket_size,
            'monotonic_pairs': 0,
            'non_monotonic_pairs': 0,
            'total_pairs': 0,
            'non_monotonic_pct': 0,
            'compute_time': 0,
            'gpu_accelerated': is_cudf
        }

    # Filter data for this algorithm and bucket size
    user_data = df[(df['algorithm'] == algo_name) & (df['bucket_size'] == bucket_size)]
    
    # Check if we have any data after filtering
    if len(user_data) == 0:
        return {
            'algorithm': algo_name,
            'bucket_size': bucket_size,
            'monotonic_pairs': 0,
            'non_monotonic_pairs': 0,
            'total_pairs': 0,
            'non_monotonic_pct': 0,
            'compute_time': 0,
            'gpu_accelerated': is_cudf
        }

    # Get all users for calculation (no sampling)
    unique_users = user_data['user_id'].unique()
    
    # Convert cuDF Series to NumPy array if needed
    if isinstance(unique_users, cudf.Series):
        unique_users = unique_users.to_numpy()
    
    print(f"  Using all {len(unique_users)} users for monotonicity check")

    # Initialize counters
    total_monotonic_pairs = 0
    total_non_monotonic_pairs = 0

    for user in unique_users:
        # Get data for this user
        subset = user_data[user_data['user_id'] == user].sort_values('timestamp')

        # Ensure we have token sums and normalized positions to compare
        if len(subset) <= 1:
            continue
            
        # Get token sums and normalized positions as arrays for faster processing
        if is_cudf:
            # For cuDF, convert to numpy for vectorized operations
            tokens = subset['user_token_sum'].to_numpy()
            norms = subset['normalized'].to_numpy()
        else:
            tokens = subset['user_token_sum'].values
            norms = subset['normalized'].values
        
        # Use vectorized operations for better performance
        # Calculate differences between consecutive elements
        token_diff = np.diff(tokens)
        norm_diff = np.diff(norms)
        
        # Count non-monotonic pairs (token increases but norm decreases, or vice versa)
        non_monotonic_count1 = np.sum((token_diff > 0) & (norm_diff < 0))
        non_monotonic_count2 = np.sum((token_diff < 0) & (norm_diff > 0))
        non_monotonic_count = non_monotonic_count1 + non_monotonic_count2
        
        # Total pairs is just the length of the diff array
        total_count = len(token_diff)
        monotonic_count = total_count - non_monotonic_count
        
        # Add to totals
        total_monotonic_pairs += monotonic_count
        total_non_monotonic_pairs += non_monotonic_count
    
    # Calculate final metrics
    total_pairs = total_monotonic_pairs + total_non_monotonic_pairs
    
    if total_pairs > 0:
        non_monotonic_pct = (total_non_monotonic_pairs / total_pairs) * 100
    else:
        non_monotonic_pct = 0
    
    compute_time = time.time() - start_time
        
    return {
        'algorithm': algo_name,
        'bucket_size': bucket_size,
        'monotonic_pairs': int(total_monotonic_pairs),
        'non_monotonic_pairs': int(total_non_monotonic_pairs),
        'total_pairs': int(total_pairs),
        'non_monotonic_pct': float(non_monotonic_pct),
        'compute_time': compute_time,
        'gpu_accelerated': is_cudf
    }

def calculate_load_balance(df, algo_name, bucket_size): 
    """Calculate load balance metrics based on pod assignments"""
    start_time = time.time()
    
    # Check if we're using cuDF or pandas
    is_cudf = isinstance(df, cudf.DataFrame)
    
    if 'best_pod' not in df.columns:
        return {
            'algorithm': algo_name,
            'bucket_size': bucket_size, 
            'load_std_dev': np.nan,
            'load_max_min_ratio': np.nan,
            'compute_time': 0,
            'gpu_accelerated': is_cudf
        }

    # Filter data for this algorithm and bucket size
    subset = df[(df['algorithm'] == algo_name) & (df['bucket_size'] == bucket_size)]
    
    # Count requests per pod
    pod_counts = subset['best_pod'].value_counts()

    # Calculate metrics
    std_dev = pod_counts.std()
    max_load = pod_counts.max()
    min_load = pod_counts.min()
    max_min_ratio = max_load / min_load if min_load > 0 else np.inf
    
    # Convert to Python types if using cuDF
    if is_cudf:
        std_dev = float(std_dev)
        max_load = float(max_load)
        min_load = float(min_load)
        max_min_ratio = float(max_min_ratio) if not np.isinf(max_min_ratio) else np.inf

    # Create result
    result = {
        'algorithm': algo_name,
        'bucket_size': bucket_size, 
        'load_std_dev': std_dev,
        'load_max_min_ratio': max_min_ratio,
        'compute_time': time.time() - start_time,
        'gpu_accelerated': is_cudf
    }
    
    # Add user_distribution if it exists in the dataframe
    if 'user_distribution' in subset.columns and len(subset['user_distribution'].unique()) == 1:
        user_dist = subset['user_distribution'].iloc[0]
        if is_cudf:
            user_dist = str(user_dist)
        result['user_distribution'] = user_dist
        
    return result

# Function to log performance metrics
def log_performance_metrics(results, operation_name, csv_output_dir=None):
    """Log performance metrics for an operation, comparing GPU vs CPU if available"""
    # Extract timing information
    if isinstance(results, list):
        # For lists of results (e.g., from multiple algorithms)
        gpu_results = [r for r in results if r.get('gpu_accelerated', False)]
        cpu_results = [r for r in results if not r.get('gpu_accelerated', False)]
        
        if gpu_results and cpu_results:
            gpu_time = sum(r.get('compute_time', 0) for r in gpu_results)
            cpu_time = sum(r.get('compute_time', 0) for r in cpu_results)
            gpu_count = len(gpu_results)
            cpu_count = len(cpu_results)
            
            avg_gpu_time = gpu_time / gpu_count if gpu_count > 0 else 0
            avg_cpu_time = cpu_time / cpu_count if cpu_count > 0 else 0
            
            if avg_cpu_time > 0:
                speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
                message = (f"Performance comparison for {operation_name}:\n"
                          f"  GPU: {avg_gpu_time:.4f}s avg ({gpu_time:.4f}s total for {gpu_count} operations)\n"
                          f"  CPU: {avg_cpu_time:.4f}s avg ({cpu_time:.4f}s total for {cpu_count} operations)\n"
                          f"  Speedup: {speedup:.2f}x\n")
                
                # Log to main log and CSV-specific log if provided
                write_to_log(message)
                if csv_output_dir:
                    write_to_csv_analysis(csv_output_dir, message)
                print(message)
    else:
        # For single result
        if 'compute_time' in results:
            is_gpu = results.get('gpu_accelerated', False)
            platform = "GPU" if is_gpu else "CPU"
            message = f"{operation_name} completed in {results['compute_time']:.4f}s using {platform}\n"
            
            # Log to main log and CSV-specific log if provided
            write_to_log(message)
            if csv_output_dir:
                write_to_csv_analysis(csv_output_dir, message)
            print(message)

# Main analysis workflow
print(f"Loading benchmark data from: {input_dir}")
file_pattern = os.path.join(input_dir, "*.csv")
all_files = glob.glob(file_pattern)
if not all_files:
    print(f"No CSV files found in {input_dir}")
    exit(1)

print(f"Found {len(all_files)} files to analyze.")
write_to_log(f"Found {len(all_files)} files to analyze.")

# Import RMM for memory management
try:
    import rmm
    has_rmm = True
except ImportError:
    has_rmm = False
    print("RMM not available. Will not clear GPU memory between files.")

# Process all CSV files one by one with full analysis per file
print(f"\nTotal files to process: {len(all_files)}\n")
write_to_log(f"Total files to process: {len(all_files)}\n")

csv_results = []
for i, file in enumerate(all_files):
    file_name = os.path.basename(file)
    print(f"\n[{i+1}/{len(all_files)}] === STARTING FILE {i+1}/{len(all_files)}: {file_name} ===")
    write_to_log(f"\n[{i+1}/{len(all_files)}] === STARTING FILE {i+1}/{len(all_files)}: {file_name} ===")
    
    try:
        # Step 1: Load the CSV file
        print(f"[{i+1}/{len(all_files)}] LOADING: {file_name}")
        
        # Create output directory for this CSV file
        csv_name = get_clean_name(file)
        csv_output_dir = os.path.join(base_output_dir, csv_name)
        
        # Create directory structure
        os.makedirs(csv_output_dir, exist_ok=True)
        csv_dir = os.path.join(csv_output_dir, "csv_data")
        # os.makedirs(csv_dir, exist_ok=True)
        
        # Create analysis file
        analysis_file = os.path.join(csv_output_dir, "analysis_results.txt")
        with open(analysis_file, 'w') as f:
            f.write(f"Analysis of {file}\n")
            f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
        # Log file size
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        write_to_csv_analysis(csv_output_dir, f"File size: {file_size_mb:.2f} MB")
        
        # Load file with appropriate method
        # ---------------------------
        # GPU-only loading block
        # ---------------------------
        try:
            if not has_cudf:
                raise ImportError("cuDF not available")

            start_time = time.time()
            df = cudf.read_csv(file)
            is_cudf = True
            load_time = time.time() - start_time
            msg = f"Loaded with cuDF in {load_time:.4f} seconds"
        except Exception as e:
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or "cudaErrorMemoryAllocation" in err_msg:
                skip_msg = f"[{i+1}/{len(all_files)}] SKIPPED (GPU OOM): {file_name}"
            else:
                skip_msg = f"[{i+1}/{len(all_files)}] SKIPPED (cuDF error): {file_name} – {err_msg}"

            write_to_csv_analysis(csv_output_dir, skip_msg)
            write_to_log(skip_msg)
            print(skip_msg)
            continue     # move on to next file
        
        write_to_csv_analysis(csv_output_dir, msg)
        write_to_log(msg)
        print(f"[{i+1}/{len(all_files)}] {msg}")
        
        # Step 2: Process basic statistics
        print(f"[{i+1}/{len(all_files)}] ANALYZING: Computing statistics for {file_name}")
        
        # Extract bucket size if needed
        if 'bucket_size' not in df.columns:
            try:
                b_size = int(file_name.split('_b')[-1].split('_')[0])
                df['bucket_size'] = b_size
                write_to_csv_analysis(csv_output_dir, f"Inferred bucket_size {b_size} from filename")
            except Exception as e:
                write_to_csv_analysis(csv_output_dir, f"Warning: Could not infer bucket_size from filename: {e}")
                print(f"[{i+1}/{len(all_files)}] Warning: Could not infer bucket_size from filename: {e}")
        
        # Basic statistics
        algo_names = df['algorithm'].unique().to_pandas() if is_cudf else df['algorithm'].unique()
        bucket_sizes = df['bucket_size'].unique().to_pandas() if is_cudf else df['bucket_size'].unique()
        
        write_to_csv_analysis(csv_output_dir, "\n=== Basic Statistics ===\n")
        
        for algo_name in algo_names:
            for b_size in sorted(bucket_sizes):
                # Filter data for this algorithm and bucket size
                algo_data = df[(df['algorithm'] == algo_name) & (df['bucket_size'] == b_size)]
                if len(algo_data) == 0:
                    continue
                    
                # Calculate statistics
                stats = calculate_basic_stats(df, algo_name, b_size)
                if stats:
                    # Write to analysis file
                    write_to_csv_analysis(csv_output_dir, f"Algorithm: {algo_name}, Bucket Size: {b_size}:")
                    write_to_csv_analysis(csv_output_dir, f"  Data points: {stats['count']}")
                    write_to_csv_analysis(csv_output_dir, f"  Normalized position range: {stats['norm_min']:.2f} to {stats['norm_max']:.2f}")
                    write_to_csv_analysis(csv_output_dir, f"  Mean normalized position: {stats['norm_mean']:.2f}")
                    write_to_csv_analysis(csv_output_dir, f"  Std dev of normalized position: {stats['norm_std']:.2f}")
                    write_to_csv_analysis(csv_output_dir, f"  Fairness (Token-Pod Correlation): {stats['fairness_correlation']:.4f} (p={stats['p_value']:.3f})\n")
                    
                    # Print to console
                    print(f"[{i+1}/{len(all_files)}] Algorithm: {algo_name}, Bucket Size: {b_size}:")
                    print(f"[{i+1}/{len(all_files)}]   Data points: {stats['count']}")
                    print(f"[{i+1}/{len(all_files)}]   Fairness (correlation): {stats['fairness_correlation']:.4f}")
                    
        # Step 3: Check monotonicity
        print(f"[{i+1}/{len(all_files)}] ANALYZING: Checking monotonicity for {file_name}")
        write_to_csv_analysis(csv_output_dir, "\n=== Monotonicity Analysis ===\n")
        
        required_mono_cols = ['user_id', 'user_token_sum', 'normalized', 'algorithm', 'bucket_size']
        if all(col in df.columns for col in required_mono_cols):
            for algo_name in algo_names:
                for b_size in sorted(bucket_sizes):
                    # Compute monotonicity
                    mono_result = check_monotonicity(df, algo_name, b_size)
                    if isinstance(mono_result, dict) and 'non_monotonic_pct' in mono_result:
                        msg = f"  {mono_result['algorithm']} (Bucket: {mono_result['bucket_size']}): {mono_result['non_monotonic_pct']:.2f}% non-monotonic ({mono_result['non_monotonic_pairs']}/{mono_result['total_pairs']} pairs)"
                        write_to_csv_analysis(csv_output_dir, msg)
                        print(f"[{i+1}/{len(all_files)}] {msg}")
                        
                        # Also log to main log file
                        write_to_log(msg)
        else:
            msg = f"Skipping Monotonicity analysis: Required columns not all found."
            write_to_csv_analysis(csv_output_dir, msg)
            print(f"[{i+1}/{len(all_files)}] {msg}")
        
        # Save result info
        result = {
            'csv_name': csv_name,
            'dataframe': df,
            'output_dir': csv_output_dir,
            'csv_dir': csv_dir,
            'analysis_file': analysis_file,
            'is_cudf': is_cudf
        }
        csv_results.append(result)
        
        # Mark completion
        print(f"[{i+1}/{len(all_files)}] FILE ANALYZED: {file_name}")
        write_to_log(f"[{i+1}/{len(all_files)}] FILE ANALYZED: {file_name}")
        write_to_csv_analysis(csv_output_dir, "\n=== Analysis Complete ===\n")
        write_to_csv_analysis(csv_output_dir, f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    except Exception as e:
        print(f"[{i+1}/{len(all_files)}] ERROR: Failed to process {file_name}: {str(e)}")
        write_to_log(f"[{i+1}/{len(all_files)}] ERROR: Failed to process {file_name}: {str(e)}")
    
    finally:    
        # Clear GPU memory after each file is fully analyzed
        if has_rmm and has_cudf:
            try:
                rmm.reinitialize()
                print(f"[{i+1}/{len(all_files)}] MEMORY FREED: After full analysis of {file_name}")
                write_to_log(f"[{i+1}/{len(all_files)}] MEMORY FREED: After full analysis of {file_name}")
            except Exception as e:
                print(f"[{i+1}/{len(all_files)}] WARNING: Could not clear GPU memory: {e}")
                write_to_log(f"[{i+1}/{len(all_files)}] WARNING: Could not clear GPU memory: {e}")

# Combine all dataframes for comparison analysis
start_time = time.time()
combined_dfs = []
has_cudf = False

# Check if any dataframes are cuDF DataFrames
for result in csv_results:
    df = result['dataframe']
    if isinstance(df, cudf.DataFrame):
        has_cudf = True
        break

# Convert all dataframes to the same type (all cuDF or all pandas)
for result in csv_results:
    df = result['dataframe']
    if has_cudf and not isinstance(df, cudf.DataFrame):
        # Convert pandas DataFrame to cuDF
        try:
            result['dataframe'] = cudf.DataFrame.from_pandas(df)
            result['is_cudf'] = True
            write_to_csv_analysis(result['output_dir'], "Converted pandas DataFrame to cuDF for combined analysis")
        except Exception as e:
            write_to_csv_analysis(result['output_dir'], f"Failed to convert to cuDF: {e}. Using pandas version.")
    elif not has_cudf and isinstance(df, cudf.DataFrame):
        # Convert cuDF DataFrame to pandas
        result['dataframe'] = df.to_pandas()
        result['is_cudf'] = False
        write_to_csv_analysis(result['output_dir'], "Converted cuDF DataFrame to pandas for combined analysis")
    
    combined_dfs.append(result['dataframe'])

# Concatenate all dataframes
if combined_dfs:
    try:
        if has_cudf:
            # Use cuDF concat for GPU acceleration
            combined_df = cudf.concat(combined_dfs, ignore_index=True)
            write_to_log("Using cuDF for combined analysis")
        else:
            # Use pandas concat
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            write_to_log("Using pandas for combined analysis")
            
        # Skip saving combined data to CSV to save disk space
        print(f"Skipping CSV output to save disk space")
        write_to_log(f"Skipping CSV output to save disk space")
        
        # Log performance for data combination
        combine_time = time.time() - start_time
        platform = "GPU (cuDF)" if has_cudf else "CPU (pandas)"
        message = f"Combined {len(combined_dfs)} dataframes in {combine_time:.4f}s using {platform}"
        print(message)
        write_to_log(message)
    except Exception as e:
        print(f"Error combining dataframes: {e}. Falling back to pandas.")
        write_to_log(f"Error combining dataframes: {e}. Falling back to pandas.")
        
        # Fall back to pandas if cuDF concat fails
        if has_cudf:
            pandas_dfs = [df.to_pandas() if isinstance(df, cudf.DataFrame) else df for df in combined_dfs]
            combined_df = pd.concat(pandas_dfs, ignore_index=True)
            write_to_log("Fell back to pandas for combined analysis due to error")
        else:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        combined_csv_path = os.path.join(comparison_dir, "combined_data.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined data to {combined_csv_path} (using pandas fallback)")
        write_to_log(f"Saved combined data to {combined_csv_path} (using pandas fallback)")
else:
    print("No valid data to analyze.")
    write_to_log("No valid data to analyze.")
    exit(1)

write_to_log("Data loaded. Performing analysis...\n")
print("Data loaded. Performing analysis...")

# Basic statistics - Group by algorithm AND bucket size
write_to_log("\n=== Basic Statistics (Grouped by Algo & Bucket Size) ===\n")
print("\n=== Basic Statistics (Grouped by Algo & Bucket Size) ===\n")

# Calculate basic stats for each algorithm and bucket size
basic_stats_results = []
for name, group in combined_df.groupby(['algorithm', 'bucket_size']):
    algo_name, b_size = name
    stats = calculate_basic_stats(combined_df, algo_name, b_size)
    if stats:
        basic_stats_results.append(stats)
        
        # Find which CSV files contain this algorithm and bucket size
        for result in csv_results:
            csv_df = result['dataframe']
            if algo_name in csv_df['algorithm'].unique() and b_size in csv_df['bucket_size'].unique():
                # Write stats to this CSV's analysis file
                write_to_csv_analysis(result['output_dir'], "\n=== Basic Statistics ===\n")
                write_to_csv_analysis(result['output_dir'], f"Algorithm: {algo_name}, Bucket Size: {b_size}:")
                write_to_csv_analysis(result['output_dir'], f"  Data points: {stats['count']}")

# Log performance metrics for basic statistics
log_performance_metrics(basic_stats_results, "Basic Statistics Calculation")

# Write detailed stats to CSV-specific files
for name, group in combined_df.groupby(['algorithm', 'bucket_size']):
    algo_name, b_size = name
    stats = next((s for s in basic_stats_results if s['algorithm'] == algo_name and s['bucket_size'] == b_size), None)
    if stats:
        for result in csv_results:
            csv_df = result['dataframe']
            if algo_name in csv_df['algorithm'].unique() and b_size in csv_df['bucket_size'].unique():
                write_to_csv_analysis(result['output_dir'], f"  Normalized position range: {stats['norm_min']:.2f} to {stats['norm_max']:.2f}")
                write_to_csv_analysis(result['output_dir'], f"  Mean normalized position: {stats['norm_mean']:.2f}")
                write_to_csv_analysis(result['output_dir'], f"  Std dev of normalized position: {stats['norm_std']:.2f}")
                write_to_csv_analysis(result['output_dir'], f"  Fairness (Token-Pod Correlation): {stats['fairness_correlation']:.4f} (p={stats['p_value']:.3f})\n")
                
                # Save stats to CSV file in this CSV's directory
                stats_df = pd.DataFrame([stats])
                stats_df.to_csv(os.path.join(result['csv_dir'], "basic_stats.csv"), index=False)
        
        # Also write to main log
        write_to_log(f"Algorithm: {algo_name}, Bucket Size: {b_size}:")
        write_to_log(f"  Data points: {stats['count']}")
        write_to_log(f"  Normalized position range: {stats['norm_min']:.2f} to {stats['norm_max']:.2f}")
        write_to_log(f"  Mean normalized position: {stats['norm_mean']:.2f}")
        write_to_log(f"  Std dev of normalized position: {stats['norm_std']:.2f}")
        write_to_log(f"  Fairness (Token-Pod Correlation): {stats['fairness_correlation']:.4f} (p={stats['p_value']:.3f})\n")
        
        # Print stats to console
        print(f"Algorithm: {algo_name}, Bucket Size: {b_size}:")
        print(f"  Data points: {stats['count']}")
        print(f"  Normalized position range: {stats['norm_min']:.2f} to {stats['norm_max']:.2f}")
        print(f"  Mean normalized position: {stats['norm_mean']:.2f}")
        print(f"  Std dev of normalized position: {stats['norm_std']:.2f}")
        print(f"  Fairness (Token-Pod Correlation): {stats['fairness_correlation']:.4f} (p={stats['p_value']:.3f})\n")

# Save basic stats to CSV in the comparison directory
if basic_stats_results:
    basic_stats_df = pd.DataFrame(basic_stats_results)
    basic_stats_df.to_csv(os.path.join(comparison_dir, "basic_statistics.csv"), index=False)

# Analyzing normalized position distributions
write_to_log("\nAnalyzing normalized position distributions...")
print("\nAnalyzing normalized position distributions...")

# Calculate distribution stats for each algorithm
# Convert cuDF Series to a format that can be iterated over
algo_names = combined_df['algorithm'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else combined_df['algorithm'].unique()
for algo_name in algo_names:
    algo_data = combined_df[combined_df['algorithm'] == algo_name]
    if 'normalized' in algo_data.columns:
        norm_mean = algo_data['normalized'].mean()
        norm_median = algo_data['normalized'].median()
        norm_std = algo_data['normalized'].std()
        norm_min = algo_data['normalized'].min()
        norm_max = algo_data['normalized'].max()
        
        # Find which CSV files contain this algorithm
        for result in csv_results:
            csv_df = result['dataframe']
            if algo_name in csv_df['algorithm'].unique():
                # Write stats to this CSV's analysis file
                write_to_csv_analysis(result['output_dir'], "\n=== Normalized Position Distribution ===\n")
                write_to_csv_analysis(result['output_dir'], f"Normalized position distribution for {algo_name}:")
                write_to_csv_analysis(result['output_dir'], f"  Mean: {norm_mean:.4f}")
                write_to_csv_analysis(result['output_dir'], f"  Median: {norm_median:.4f}")
                write_to_csv_analysis(result['output_dir'], f"  Std Dev: {norm_std:.4f}")
                write_to_csv_analysis(result['output_dir'], f"  Min: {norm_min:.4f}")
                write_to_csv_analysis(result['output_dir'], f"  Max: {norm_max:.4f}")
                
                # Save distribution stats to CSV file in this CSV's directory
                dist_stats = {
                    'algorithm': [algo_name],
                    'mean': [norm_mean],
                    'median': [norm_median],
                    'std_dev': [norm_std],
                    'min': [norm_min],
                    'max': [norm_max]
                }
                dist_df = pd.DataFrame(dist_stats)
                dist_df.to_csv(os.path.join(result['csv_dir'], "normalized_distribution.csv"), index=False)
        
        # Also write to main log
        write_to_log(f"\nNormalized position distribution for {algo_name}:")
        write_to_log(f"  Mean: {norm_mean:.4f}")
        write_to_log(f"  Median: {norm_median:.4f}")
        write_to_log(f"  Std Dev: {norm_std:.4f}")
        write_to_log(f"  Min: {norm_min:.4f}")
        write_to_log(f"  Max: {norm_max:.4f}")
        
        # Print to console
        print(f"\nNormalized position distribution for {algo_name}:")
        print(f"  Mean: {norm_mean:.4f}")
        print(f"  Median: {norm_median:.4f}")
        print(f"  Std Dev: {norm_std:.4f}")
        print(f"  Min: {norm_min:.4f}")
        print(f"  Max: {norm_max:.4f}")

# Check monotonicity of fairness score
write_to_log("\nChecking monotonicity of fairness score...")
print("\nChecking monotonicity of fairness score...")

# Run monotonicity check only if columns exist
all_monotonicity_list = []
required_mono_cols = ['user_id', 'user_token_sum', 'normalized', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_mono_cols):
    write_to_log("\nChecking monotonicity...")
    print("\nChecking monotonicity...")
    
    # Create a list of tasks for monotonicity checking
    monotonicity_tasks = []
    
    # Convert cuDF Series to a format that can be iterated over
    algo_names = combined_df['algorithm'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else combined_df['algorithm'].unique()
    bucket_sizes = combined_df['bucket_size'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else sorted(combined_df['bucket_size'].unique())
    
    for algo_name in algo_names:
        for bucket_size in sorted(bucket_sizes):
            monotonicity_tasks.append((combined_df, algo_name, bucket_size))
    
    # Run monotonicity checks
    monotonicity_start_time = time.time()
    # Always run sequentially
    for task in monotonicity_tasks:
        result = check_monotonicity(*task)
        all_monotonicity_list.append(result)
        
        if isinstance(result, dict) and 'non_monotonic_pct' in result:
            # Log to main log file
            write_to_log(f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
            print(f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
        else:
            write_to_log(f"  Monotonicity check: No valid data available")
            print(f"  Monotonicity check: No valid data available")
    
    # Log performance metrics for monotonicity checks
    log_performance_metrics(all_monotonicity_list, "Monotonicity Checking")
    
    # Write results to individual CSV files
    for result in all_monotonicity_list:
        if isinstance(result, dict) and 'non_monotonic_pct' in result:
            # Find which CSV files contain this algorithm and bucket size
            for csv_result in csv_results:
                csv_df = csv_result['dataframe']
                if result['algorithm'] in csv_df['algorithm'].unique() and result['bucket_size'] in csv_df['bucket_size'].unique():
                    # Write to this CSV's analysis file
                    write_to_csv_analysis(csv_result['output_dir'], "\n=== Monotonicity Analysis ===\n")
                    write_to_csv_analysis(csv_result['output_dir'], f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
                    
                    # Skip saving monotonicity results to CSV file to save disk space
                    # mono_df = pd.DataFrame([result])
                    # mono_df.to_csv(os.path.join(csv_result['csv_dir'], "monotonicity.csv"), index=False)
    
    # Check if we have valid results before creating DataFrame
    if all_monotonicity_list and all(isinstance(item, dict) for item in all_monotonicity_list):
        # Convert to DataFrame for easier analysis
        all_monotonicity_df = pd.DataFrame(all_monotonicity_list)
        
        # Skip saving monotonicity results to CSV to save disk space
        # all_monotonicity_df.to_csv(os.path.join(comparison_dir, "monotonicity_results.csv"), index=False)
    else:
        write_to_log("  No valid monotonicity data to analyze")
        print("  No valid monotonicity data to analyze")
else:
    write_to_log(f"\nSkipping Monotonicity analysis: Required columns ({', '.join(required_mono_cols)}) not all found.")
    print(f"\nSkipping Monotonicity analysis: Required columns ({', '.join(required_mono_cols)}) not all found.")

# Calculate load balance from actual assignments
write_to_log("\nCalculating load balance from actual assignments...")
print("\nCalculating load balance from actual assignments...")

# Run load balance check only if columns exist
required_lb_cols = ['best_pod', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_lb_cols):
    load_balance_results = []
    load_balance_start_time = time.time()
    
    # Group by algorithm and bucket size for load balance calculation
    if isinstance(combined_df, cudf.DataFrame):
        # For cuDF, we need to handle groupby differently
        algo_bucket_pairs = [(algo, bucket) for algo in combined_df['algorithm'].unique().to_pandas() 
                            for bucket in combined_df[combined_df['algorithm'] == algo]['bucket_size'].unique().to_pandas()]
        for algo_name, b_size in algo_bucket_pairs:
            group = combined_df[(combined_df['algorithm'] == algo_name) & (combined_df['bucket_size'] == b_size)]
            # Calculate load balance with GPU acceleration
            result = calculate_load_balance(group, algo_name, b_size)
            load_balance_results.append(result)
    else:
        # For pandas, we can use groupby directly
        for name, group in combined_df.groupby(['algorithm', 'bucket_size']):
            algo_name, b_size = name
            # Calculate load balance using pandas
            result = calculate_load_balance(group, algo_name, b_size)
            load_balance_results.append(result)
    
    # Log performance metrics for load balance calculations
    log_performance_metrics(load_balance_results, "Load Balance Calculation")
    
    # Write results to individual CSV files
    for result in load_balance_results:
        algo_name = result['algorithm']
        b_size = result['bucket_size']
        
        # Find which CSV files contain this algorithm and bucket size
        for csv_result in csv_results:
            csv_df = csv_result['dataframe']
            if algo_name in csv_df['algorithm'].unique() and b_size in csv_df['bucket_size'].unique():
                # Write to this CSV's analysis file
                write_to_csv_analysis(csv_result['output_dir'], "\n=== Load Balance Analysis ===\n")
                write_to_csv_analysis(csv_result['output_dir'], f"Load balance for {algo_name} (Bucket: {b_size}):")
                write_to_csv_analysis(csv_result['output_dir'], f"  Min load: {result.get('min_load', 0):.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Max load: {result.get('max_load', 0):.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Mean load: {result.get('mean_load', 0):.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Std dev: {result.get('std_load', 0):.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Load balance ratio (max/min): {result.get('load_balance_ratio', 0):.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Coefficient of variation (std/mean): {result.get('coefficient_of_variation', 0):.4f}")
                
                # Add GPU acceleration info if available
                if 'gpu_accelerated' in result:
                    platform = "GPU" if result['gpu_accelerated'] else "CPU"
                    write_to_csv_analysis(csv_result['output_dir'], f"  Computed using: {platform}")
                    if 'compute_time' in result:
                        write_to_csv_analysis(csv_result['output_dir'], f"  Computation time: {result['compute_time']:.4f}s")
                
                # Save load balance results to CSV file in this CSV's directory
                lb_single_df = pd.DataFrame([result])
                lb_single_df.to_csv(os.path.join(csv_result['csv_dir'], "load_balance.csv"), index=False)
        
        # Write to main log file
        write_to_log(f"\nLoad balance for {algo_name} (Bucket: {b_size}):")
        write_to_log(f"  Min load: {min_load:.2f}")
        write_to_log(f"  Max load: {max_load:.2f}")
        write_to_log(f"  Mean load: {mean_load:.2f}")
        write_to_log(f"  Std dev: {std_load:.2f}")
        write_to_log(f"  Load balance ratio (max/min): {load_balance_ratio:.2f}")
        write_to_log(f"  Coefficient of variation (std/mean): {cv:.4f}")
        
        # Print to console
        print(f"\nLoad balance for {algo_name} (Bucket: {b_size}):")
        print(f"  Min load: {min_load:.2f}")
        print(f"  Max load: {max_load:.2f}")
        print(f"  Mean load: {mean_load:.2f}")
        print(f"  Std dev: {std_load:.2f}")
        print(f"  Load balance ratio (max/min): {load_balance_ratio:.2f}")
        print(f"  Coefficient of variation (std/mean): {cv:.4f}")
    
    # Skip saving load balance results to CSV to save disk space
    # if load_balance_results:
    #     lb_df = pd.DataFrame(load_balance_results)
    #     lb_df.to_csv(os.path.join(comparison_dir, "load_balance.csv"), index=False)
else:
    write_to_log(f"\nSkipping Load Balance analysis: Required columns ({', '.join(required_lb_cols)}) not all found.")
    print(f"\nSkipping Load Balance analysis: Required columns ({', '.join(required_lb_cols)}) not all found.")

# Write summary section
write_to_log("\n=== Analysis Summary ===")
print("\n=== Analysis Summary ===")

# Add summary of analyzed files
write_to_log(f"\nAnalyzed {len(csv_results)} CSV files:")
for result in csv_results:
    write_to_log(f"  - {result['csv_name']} (Output directory: {result['output_dir']})")
    
    # Write completion timestamp to each CSV's analysis file
    write_to_csv_analysis(result['output_dir'], "\n" + "="*80)
    write_to_csv_analysis(result['output_dir'], "ANALYSIS COMPLETE")
    write_to_csv_analysis(result['output_dir'], "="*80)
    write_to_csv_analysis(result['output_dir'], f"\nAnalysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Get list of adaptive variants
algo_names = combined_df['algorithm'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else combined_df['algorithm'].unique()
adaptive_variants = [algo for algo in algo_names if 'adaptive' in algo]
if adaptive_variants:
    write_to_log(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    print(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    
    # Compare fairness correlation across variants
    write_to_log("\nFairness Correlation Comparison (higher is better):")
    print("\nFairness Correlation Comparison (higher is better):")
    
    for algo in adaptive_variants:
        # Convert bucket sizes to a format that can be iterated over if using cuDF
        bucket_sizes = combined_df['bucket_size'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else combined_df['bucket_size'].unique()
        for b_size in sorted(bucket_sizes):
            # Get weight type (balanced, fairness, utilization)
            if 'balanced' in algo:
                weight_type = "Balanced (0.5/0.5)"
            elif 'fairness' in algo:
                weight_type = "Fairness-focused"
            elif 'utilization' in algo:
                weight_type = "Utilization-focused"
            else:
                weight_type = algo
                
            # Get correlation for this algorithm and bucket size
            stats = [s for s in basic_stats_results if s['algorithm'] == algo and s['bucket_size'] == b_size]
            if stats:
                corr = stats[0]['fairness_correlation']
                p_val = stats[0]['p_value']
                write_to_log(f"  {weight_type} (Bucket Size {b_size}): {corr:.4f} (p={p_val:.3f})")
                print(f"  {weight_type} (Bucket Size {b_size}): {corr:.4f} (p={p_val:.3f})")
    
    # Compare monotonicity across variants
    write_to_log("\nMonotonicity Comparison:")
    print("\nMonotonicity Comparison:")
    
    for algo in adaptive_variants:
        # Convert bucket sizes to a format that can be iterated over if using cuDF
        bucket_sizes = combined_df['bucket_size'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else combined_df['bucket_size'].unique()
        for b_size in sorted(bucket_sizes):
            # Get weight type
            if 'balanced' in algo:
                weight_type = "Balanced (0.5/0.5)"
            elif 'fairness' in algo:
                weight_type = "Fairness-focused"
            elif 'utilization' in algo:
                weight_type = "Utilization-focused"
            else:
                weight_type = algo
                
            # Get monotonicity for this algorithm and bucket size
            mono_data = [m for m in all_monotonicity_list if m['algorithm'] == algo and m['bucket_size'] == b_size]
            if mono_data:
                non_mono_pct = mono_data[0]['non_monotonic_pct']
                write_to_log(f"  {weight_type} (Bucket Size {b_size}): {non_mono_pct:.2f}% non-monotonic")
                print(f"  {weight_type} (Bucket Size {b_size}): {non_mono_pct:.2f}% non-monotonic")
    
    # Compare load balance across variants
    write_to_log("\nLoad Balance Comparison:")
    print("\nLoad Balance Comparison:")
    
    # Skip detailed load balance comparison if we don't have the required columns
    if not all(col in combined_df.columns for col in required_lb_cols):
        write_to_log("  Load balance data not available for comparison.")
        print("  Load balance data not available for comparison.")
    else:    
        for algo in adaptive_variants:
            for b_size in sorted(combined_df['bucket_size'].unique()):
                # Get weight type
                if 'balanced' in algo:
                    weight_type = "Balanced (0.5/0.5)"
                elif 'fairness' in algo:
                    weight_type = "Fairness-focused"
                elif 'utilization' in algo:
                    weight_type = "Utilization-focused"
                else:
                    weight_type = algo
                    
                # Get load balance for this algorithm and bucket size
                lb_data = [lb for lb in load_balance_results if lb['algorithm'] == algo and lb['bucket_size'] == b_size]
                if lb_data:
                    std_dev = lb_data[0]['std_load']
                    max_min = lb_data[0]['load_balance_ratio']
                    write_to_log(f"  {weight_type} (Bucket Size {b_size}): StdDev={std_dev:.2f}, Max/Min={max_min:.2f}")
                    print(f"  {weight_type} (Bucket Size {b_size}): StdDev={std_dev:.2f}, Max/Min={max_min:.2f}")

# Write completion message
write_to_log("\nAnalysis complete. Results saved to:")
write_to_log(f"  {base_output_dir} (Main output directory)")
write_to_log(f"  {log_file} (Main log file)")
for result in csv_results:
    write_to_log(f"  {result['output_dir']} (Results for {result['csv_name']})")

print("\nAnalysis complete. Results saved to:")
print(f"  {base_output_dir} (Main output directory)")
print(f"  {log_file} (Main log file)")
for result in csv_results:
    print(f"  {result['output_dir']} (Results for {result['csv_name']})")
