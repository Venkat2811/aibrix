#!/usr/bin/env python3
# Import libraries and check availability
try:
    import cudf
    has_cudf = True
    # Also try to import nvidia-smi for GPU memory monitoring
    try:
        import nvidia_smi
        has_nvidia_smi = True
    except ImportError:
        has_nvidia_smi = False
        print("nvidia-smi not available. GPU memory monitoring disabled.")
except ImportError:
    has_cudf = False
    has_nvidia_smi = False
    print("CUDA GPU support not available. Using CPU only.")

# Import gc for memory management
import gc

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
parser.add_argument('--input_dir', type=str, default="./bench-venkat/analysis/adaptive-variants1", help='Directory containing CSV results to analyze')
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

            # First try to load the entire file
            try:
                start_time = time.time()
                df = cudf.read_csv(filename)
                load_time = time.time() - start_time
                msg = f"Loaded with cuDF in {load_time:.4f} seconds"
                write_to_csv_analysis(csv_output_dir, msg)
                write_to_log(msg)
                
                # Log memory usage after successful load
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = info.used / 1024 / 1024
                mem_total_mb = info.total / 1024 / 1024
                mem_msg = f"GPU Memory: Used {mem_used_mb:.2f}MB / Total {mem_total_mb:.2f}MB ({(info.used/info.total)*100:.2f}%)"
                write_to_csv_analysis(csv_output_dir, mem_msg)
                write_to_log(mem_msg)
                nvidia_smi.nvmlShutdown()
                
            except Exception as e:
                # If OOM error, try chunking on GPU
                err_msg = str(e)
                if "out of memory" in err_msg.lower() or "cudaErrorMemoryAllocation" in err_msg:
                    oom_msg = f"GPU out of memory ({err_msg}). Attempting GPU chunking..."
                    write_to_csv_analysis(csv_output_dir, oom_msg)
                    write_to_log(oom_msg)
                    print(oom_msg)
                    
                    # First determine number of rows to estimate chunk size
                    row_count = sum(1 for _ in open(filename, 'r'))
                    # Cast to pure-Python int for cudf.read_csv - Force Python int
                    chunk_size = int(max(1, (row_count - 1) // 2))
                    rows_to_read = int(chunk_size)  # Ensure it's a Python int
                    
                    # Debug output to check type
                    type_info = f"Debug: chunk_size={chunk_size} (type: {type(chunk_size).__name__}), rows_to_read={rows_to_read} (type: {type(rows_to_read).__name__})"
                    write_to_log(type_info)
                    print(type_info)
                    
                    msg = f"File has approximately {row_count} rows. Processing in chunks of {rows_to_read} rows on GPU."
                    write_to_csv_analysis(csv_output_dir, msg)
                    write_to_log(msg)
                    print(msg)
                    
                    # Process first chunk on GPU
                    start_time = time.time()
                    first_chunk_start = time.time()
                    write_to_log(f"Loading first chunk ({rows_to_read} rows) into GPU...")
                    # Use a simple Python primitive int
                    first_chunk = cudf.read_csv(
                        filename,
                        nrows=rows_to_read
                    )
                    first_chunk_time = time.time() - first_chunk_start
                    write_to_log(f"First chunk loaded in {first_chunk_time:.4f} seconds")
                    
                    # Process first chunk and store results
                    first_results = process_chunk(first_chunk, csv_output_dir, "first chunk")
                    
                    # Clear GPU memory before loading second chunk
                    del first_chunk
                    gc.collect()
                    
                    # Log memory cleanup
                    write_to_log("Cleared first chunk from GPU memory")
                    
                    # Process second chunk on GPU
                    second_chunk_start = time.time()
                    write_to_log(f"Loading second chunk (remaining rows) into GPU...")
                    
                    # Create skiprows list with explicit int conversion
                    skip_rows_list = [int(i) for i in range(1, chunk_size+1)]
                    write_to_log(f"Debug: skiprows list type: {type(skip_rows_list).__name__}, first element type: {type(skip_rows_list[0]).__name__}")
                    
                    second_chunk = cudf.read_csv(
                        filename,
                        skiprows=skip_rows_list
                    )
                    second_chunk_time = time.time() - second_chunk_start
                    write_to_log(f"Second chunk loaded in {second_chunk_time:.4f} seconds")
                    
                    # Process second chunk and store results
                    second_results = process_chunk(second_chunk, csv_output_dir, "second chunk")
                    
                    # Combine results (not the dataframes - we're keeping everything in GPU memory)
                    df = cudf.concat([first_results, second_chunk], ignore_index=True)
                    
                    # Clear second chunk from memory
                    del second_chunk
                    gc.collect()
                    
                    load_time = time.time() - start_time
                    msg = f"Loaded and processed with cuDF in chunks: {load_time:.4f} seconds"
                    write_to_csv_analysis(csv_output_dir, msg)
                    write_to_log(msg)
                    print(msg)
                else:
                    # Any other cuDF error: log & skip
                    error_msg = f"ERROR: cuDF failed on {base_name}: {err_msg}"
                    write_to_csv_analysis(csv_output_dir, error_msg)
                    write_to_log(error_msg)
                    print(error_msg)
                    return None
        except Exception as e:
            # Any other error: log & skip
            error_msg = f"ERROR: Failed to process {base_name}: {str(e)}"
            write_to_csv_analysis(csv_output_dir, error_msg)
            write_to_log(error_msg)
            print(error_msg)
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

# Function to process a chunk of data and return results
def process_chunk(chunk_df, csv_output_dir, chunk_name):
    """Process a chunk of data and return the processed dataframe"""
    # Log the chunk processing
    chunk_size = len(chunk_df)
    msg = f"Processing {chunk_name} with {chunk_size} rows"
    write_to_log(msg)
    write_to_csv_analysis(csv_output_dir, msg)
    print(msg)
    
    # Here you would do any preprocessing needed on the chunk
    # For now, we're just returning the chunk as-is
    return chunk_df

# Main analysis workflow
print(f"Loading benchmark data from: {input_dir}")
file_pattern = os.path.join(input_dir, "*.csv")
all_files = glob.glob(file_pattern)
if not all_files:
    print(f"No CSV files found in {input_dir}")
    exit(1)

print(f"Found {len(all_files)} files to analyze.")
write_to_log(f"Found {len(all_files)} files to analyze.")

# Sort CSV files for consistent order
all_files.sort()

# Skip the first 25 files and process the remaining ones
if len(all_files) > 25:
    print(f"Skipping first 25 files, processing remaining {len(all_files) - 25} out of {len(all_files)} total files")
    all_files = all_files[25:]

# Import RMM for memory management
try:
    import rmm
    has_rmm = True
except ImportError:
    has_rmm = False
    print("RMM not available. Will not clear GPU memory between files.")

# Initialize result containers
basic_stats_results = []
all_monotonicity_list = []

# Process all CSV files one by one with full analysis per file
print(f"\nTotal files to process: {len(all_files)}\n")
write_to_log(f"Total files to process: {len(all_files)}\n")

csv_results = []
for i, file in enumerate(all_files):
    file_name = os.path.basename(file)
    print(f"\n[{i+1}/{len(all_files)}] === STARTING FILE {i+1}/{len(all_files)}: {file_name} ===")
    write_to_log(f"\n[{i+1}/{len(all_files)}] === STARTING FILE {i+1}/{len(all_files)}: {file_name} ===")
    
    # If this isn't the first file, sleep for 2 seconds to allow memory cleanup
    if i > 0:
        sleep_msg = "Sleeping for 2 seconds before processing next file to allow memory cleanup..."
        print(sleep_msg)
        write_to_log(sleep_msg)
        time.sleep(2)
        
        # Force garbage collection
        gc.collect()
        
        # Log GPU memory status if nvidia-smi is available
        if has_nvidia_smi:
            try:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = info.used / 1024 / 1024
                mem_total_mb = info.total / 1024 / 1024
                mem_msg = f"GPU Memory after cleanup: Used {mem_used_mb:.2f}MB / Total {mem_total_mb:.2f}MB ({(info.used/info.total)*100:.2f}%)"
                print(mem_msg)
                write_to_log(mem_msg)
                nvidia_smi.nvmlShutdown()
            except Exception as e:
                write_to_log(f"Error getting GPU memory info: {str(e)}")
    
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
        # ---------------------------  GPU-only loading block  ---------------------------
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
            # ---------------- GPU OOM  →  split the file in 2 chunks ----------------
            if "out of memory" in err_msg.lower() or "cudaerrormemoryallocation" in err_msg.lower():
                oom_msg = f"[{i+1}/{len(all_files)}] GPU OOM on {file_name}. " \
                          "Retrying with two half-size chunks."
                print(oom_msg)
                write_to_log(oom_msg)
                write_to_csv_analysis(csv_output_dir, oom_msg)

                # New approach: Use pandas to read first, then convert to cudf
                oom_msg = f"[{i+1}/{len(all_files)}] GPU OOM on {file_name}. Trying pandas+cudf approach."
                print(oom_msg)
                write_to_log(oom_msg)
                write_to_csv_analysis(csv_output_dir, oom_msg)
                
                # Determine number of rows and calculate chunk size
                with open(file, "r") as fh:
                    row_cnt = sum(1 for _ in fh) - 1
                half_rows = int(max(1, row_cnt // 2))
                write_to_log(f"File rows (excl. header): {row_cnt} -> chunk size {half_rows}")
                
                # Container for partial results
                partial_results = []
                
                for part_idx, chunk_start in enumerate([0, half_rows], start=1):
                    part_label = f"chunk {part_idx}/2"
                    rows_to_read = half_rows if part_idx == 1 else (row_cnt - half_rows)
                    
                    part_msg = f"→ Loading {part_label} with pandas (start={chunk_start}, rows={rows_to_read})"
                    print(part_msg)
                    write_to_log(part_msg)
                    write_to_csv_analysis(csv_output_dir, part_msg)
                    
                    part_start = time.time()
                    try:
                        # Step 1: Read with pandas to avoid cudf.read_csv issue
                        if part_idx == 1:
                            # First chunk: read from beginning with nrows
                            pandas_chunk = pd.read_csv(file, nrows=rows_to_read)
                        else:
                            # Second chunk: skip first chunk and read rest
                            pandas_chunk = pd.read_csv(file, skiprows=range(1, chunk_start+1))
                        
                        # Step 2: Convert pandas DataFrame to cudf
                        conversion_start = time.time()
                        part_df = cudf.DataFrame.from_pandas(pandas_chunk)
                        conversion_time = time.time() - conversion_start
                        
                        # Log success
                        load_time = time.time() - part_start
                        msg = f"{part_label} loaded in {load_time:.2f}s (pandas) + {conversion_time:.2f}s (to cudf), rows={len(part_df)}"
                        write_to_log(msg)
                        write_to_csv_analysis(csv_output_dir, msg)
                        print(msg)
                    except Exception as part_e:
                        # give up on this file
                        err = f"❌ Failed to load {part_label}: {part_e}"
                        write_to_log(err)
                        write_to_csv_analysis(csv_output_dir, err)
                        print(err)
                        raise   # handled by outer try/except

                    load_sec = time.time() - part_start
                    write_to_log(f"{part_label} loaded in {load_sec:.2f}s, rows={len(part_df)}")

                    # ---------------- run per-file analysis on this chunk ----------------
                    process_start = time.time()
                    # reuse the existing analysis code paths with the *same* local
                    # variables that expect df
                    df = part_df
                    # Write chunk header to analysis file
                    chunk_header = f"\n=== {part_label.upper()} ANALYSIS ===\n"
                    write_to_csv_analysis(csv_output_dir, chunk_header)
                    
                    # call the basic-stats / monotonicity helpers directly
                    # (they are pure and don't mutate df)
                    part_result = {}
                    
                    # Write basic stats for this chunk
                    write_to_csv_analysis(csv_output_dir, "\n=== Basic Statistics ===\n")
                    for algo in df['algorithm'].unique().to_pandas():
                        for bs in df['bucket_size'].unique().to_pandas():
                            # Calculate basic stats
                            stats = calculate_basic_stats(df, algo, bs)
                            part_result.setdefault('basic', []).append(stats)
                            
                            # Write chunk-specific basic stats to analysis file
                            if stats:
                                write_to_csv_analysis(csv_output_dir, f"Algorithm: {algo}, Bucket Size: {bs} ({part_label}):")
                                write_to_csv_analysis(csv_output_dir, f"  Data points: {stats['count']}")
                                write_to_csv_analysis(csv_output_dir, f"  Normalized position range: {stats['norm_min']:.2f} to {stats['norm_max']:.2f}")
                                write_to_csv_analysis(csv_output_dir, f"  Mean normalized position: {stats['norm_mean']:.2f}")
                                write_to_csv_analysis(csv_output_dir, f"  Std dev of normalized position: {stats['norm_std']:.2f}")
                                write_to_csv_analysis(csv_output_dir, f"  Fairness (Token-Pod Correlation): {stats['fairness_correlation']:.4f} (p={stats['p_value']:.3f})\n")
                    
                    # Write monotonicity results for this chunk
                    write_to_csv_analysis(csv_output_dir, "\n=== Monotonicity Analysis ===\n")
                    for algo in df['algorithm'].unique().to_pandas():
                        for bs in df['bucket_size'].unique().to_pandas():
                            # Calculate monotonicity
                            mono_result = check_monotonicity(df, algo, bs)
                            part_result.setdefault('mono', []).append(mono_result)
                            
                            # Write chunk-specific monotonicity to analysis file
                            if isinstance(mono_result, dict) and 'non_monotonic_pct' in mono_result:
                                msg = f"  {mono_result['algorithm']} (Bucket: {mono_result['bucket_size']}, {part_label}): {mono_result['non_monotonic_pct']:.2f}% non-monotonic ({mono_result['non_monotonic_pairs']}/{mono_result['total_pairs']} pairs)"
                                write_to_csv_analysis(csv_output_dir, msg)
                                
                    partial_results.append(part_result)
                    proc_sec = time.time() - process_start
                    write_to_log(f"{part_label} analysed in {proc_sec:.2f}s")
                    print(f"{part_label} analysed in {proc_sec:.2f}s")

                    # ----------- free GPU RAM for next chunk -----------
                    del part_df
                    gc.collect()
                    if has_rmm:
                        try:
                            rmm.reinitialize()
                        except Exception:
                            pass

                # ---------------- merge partial_results ----------------
                # Write combined results header
                write_to_csv_analysis(csv_output_dir, "\n=== COMBINED RESULTS FROM ALL CHUNKS ===\n")
                
                # Get all unique algorithm/bucket size combinations
                algo_bucket_combos = set()
                for pr in partial_results:
                    for stat in pr['basic']:
                        algo_bucket_combos.add((stat['algorithm'], stat['bucket_size']))
                
                # Properly aggregate statistics across chunks
                agg_basic_stats = {}
                agg_mono_stats = {}
                
                for algo, bucket in algo_bucket_combos:
                    # Collect all stats for this algo/bucket pair
                    all_basic_stats = []
                    all_mono_stats = []
                    
                    for pr in partial_results:
                        basic_matches = [s for s in pr['basic'] if s['algorithm'] == algo and s['bucket_size'] == bucket]
                        mono_matches = [m for m in pr['mono'] if m['algorithm'] == algo and m['bucket_size'] == bucket]
                        
                        if basic_matches:
                            all_basic_stats.append(basic_matches[0])
                        if mono_matches:
                            all_mono_stats.append(mono_matches[0])
                    
                    # Aggregate basic stats
                    if all_basic_stats:
                        # Start with a copy of the first stats object
                        agg_stat = all_basic_stats[0].copy()
                        
                        # Sum up counts
                        total_count = sum(s['count'] for s in all_basic_stats)
                        agg_stat['count'] = total_count
                        
                        # Weighted average for means
                        if total_count > 0:
                            agg_stat['norm_mean'] = sum(s['norm_mean'] * s['count'] for s in all_basic_stats) / total_count
                            agg_stat['norm_std'] = sum(s['norm_std'] * s['count'] for s in all_basic_stats) / total_count
                        
                        # Get global min/max
                        agg_stat['norm_min'] = min(s['norm_min'] for s in all_basic_stats)
                        agg_stat['norm_max'] = max(s['norm_max'] for s in all_basic_stats)
                        
                        # Use weighted fairness correlation
                        total_corr = sum(s['fairness_correlation'] * s['count'] for s in all_basic_stats)
                        agg_stat['fairness_correlation'] = total_corr / total_count if total_count > 0 else 0
                        
                        # Store aggregated stats
                        agg_basic_stats[(algo, bucket)] = agg_stat
                    
                    # Aggregate monotonicity stats
                    if all_mono_stats:
                        # Start with a copy of the first stats object
                        agg_mono = all_mono_stats[0].copy()
                        
                        # Sum up counts
                        agg_mono['monotonic_pairs'] = sum(m['monotonic_pairs'] for m in all_mono_stats)
                        agg_mono['non_monotonic_pairs'] = sum(m['non_monotonic_pairs'] for m in all_mono_stats)
                        agg_mono['total_pairs'] = sum(m['total_pairs'] for m in all_mono_stats)
                        
                        # Recalculate percentage
                        if agg_mono['total_pairs'] > 0:
                            agg_mono['non_monotonic_pct'] = (agg_mono['non_monotonic_pairs'] / agg_mono['total_pairs']) * 100
                        else:
                            agg_mono['non_monotonic_pct'] = 0
                        
                        # Store aggregated stats
                        agg_mono_stats[(algo, bucket)] = agg_mono
                
                # Write combined basic stats
                write_to_csv_analysis(csv_output_dir, "\n=== Combined Basic Statistics ===\n")
                for key, stats in agg_basic_stats.items():
                    algo, bucket = key
                    write_to_csv_analysis(csv_output_dir, f"Algorithm: {algo}, Bucket Size: {bucket} (COMBINED):")
                    write_to_csv_analysis(csv_output_dir, f"  Data points: {stats['count']}")
                    write_to_csv_analysis(csv_output_dir, f"  Normalized position range: {stats['norm_min']:.2f} to {stats['norm_max']:.2f}")
                    write_to_csv_analysis(csv_output_dir, f"  Mean normalized position: {stats['norm_mean']:.2f}")
                    write_to_csv_analysis(csv_output_dir, f"  Std dev of normalized position: {stats['norm_std']:.2f}")
                    write_to_csv_analysis(csv_output_dir, f"  Fairness (Token-Pod Correlation): {stats['fairness_correlation']:.4f} (p={stats['p_value']:.3f})\n")
                
                # Write combined monotonicity
                write_to_csv_analysis(csv_output_dir, "\n=== Combined Monotonicity Analysis ===\n")
                for key, mono in agg_mono_stats.items():
                    algo, bucket = key
                    msg = f"  {algo} (Bucket: {bucket}, COMBINED): {mono['non_monotonic_pct']:.2f}% non-monotonic ({mono['non_monotonic_pairs']}/{mono['total_pairs']} pairs)"
                    write_to_csv_analysis(csv_output_dir, msg)
                    
                # Convert aggregated stats back to lists for global aggregation
                merged_basic = list(agg_basic_stats.values())
                merged_mono = list(agg_mono_stats.values())

                # save for later global aggregation
                basic_stats_results.extend(merged_basic)
                all_monotonicity_list.extend(merged_mono)
                
                # Log success message
                success_msg = f"[{i+1}/{len(all_files)}] Successfully analyzed {file_name} in chunks"
                print(success_msg)
                write_to_log(success_msg)
                write_to_csv_analysis(csv_output_dir, "\n" + success_msg)

                # skip the normal single-df path
                continue

            # ---------- any other cuDF error : skip file ----------
            skip_msg = f"[{i+1}/{len(all_files)}] SKIPPED (cuDF error): {file_name} – {err_msg}"
            write_to_csv_analysis(csv_output_dir, skip_msg)
            write_to_log(skip_msg)
            print(skip_msg)
            continue
        
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
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Log GPU memory status after file completion if nvidia-smi is available
        if has_nvidia_smi:
            try:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = info.used / 1024 / 1024
                mem_total_mb = info.total / 1024 / 1024
                mem_msg = f"GPU Memory after file completion: Used {mem_used_mb:.2f}MB / Total {mem_total_mb:.2f}MB ({(info.used/info.total)*100:.2f}%)"
                print(mem_msg)
                write_to_log(mem_msg)
                nvidia_smi.nvmlShutdown()
            except Exception as e:
                write_to_log(f"Error getting GPU memory info: {str(e)}")

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
        
        # Always convert to pandas for final analysis to avoid OOM errors
        # The individual file processing leverages GPU, but combined analysis uses CPU
        if has_cudf and isinstance(combined_df, cudf.DataFrame):
            write_to_log("Converting to pandas for final cross-file analysis to avoid memory issues")
            print("Converting to pandas for final cross-file analysis to avoid memory issues")
            combined_df = combined_df.to_pandas()
            # Keep has_cudf flag true since we want to continue using GPU for individual files
            # But we'll check for pandas dataframe type when needed
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

# Free memory before heavy analysis
gc.collect()  # Collect Python garbage
if has_cudf and has_rmm:
    try:
        rmm.reinitialize()
        print("Reinitialized GPU memory before global analysis")
        write_to_log("Reinitialized GPU memory before global analysis")
    except Exception as e:
        print(f"WARNING: Could not reinitialize GPU memory: {e}")
        write_to_log(f"WARNING: Could not reinitialize GPU memory: {e}")

# Basic statistics - Group by algorithm AND bucket size
write_to_log("\n=== Basic Statistics (Grouped by Algo & Bucket Size) ===\n")
print("\n=== Basic Statistics (Grouped by Algo & Bucket Size) ===\n")

# Calculate basic stats for each algorithm and bucket size - but using chunking
# to avoid memory issues, similar to the per-file approach

# Get all unique algorithm/bucket size combinations
algo_bucket_pairs = []
for result in csv_results:
    csv_df = result['dataframe']
    algos = csv_df['algorithm'].unique()
    buckets = csv_df['bucket_size'].unique()
    
    # Handle both cuDF and pandas Series
    if hasattr(algos, 'to_pandas'):
        algos = algos.to_pandas()
    if hasattr(buckets, 'to_pandas'):
        buckets = buckets.to_pandas()
        
    for algo in algos:
        for bucket in buckets:
            algo_bucket_pairs.append((algo, bucket))

# Remove duplicates
algo_bucket_pairs = list(set(algo_bucket_pairs))
write_to_log(f"Found {len(algo_bucket_pairs)} unique algorithm/bucket size combinations for analysis")
print(f"Found {len(algo_bucket_pairs)} unique algorithm/bucket size combinations for analysis")

# Process each algorithm/bucket combination separately
for algo_name, b_size in algo_bucket_pairs:
    print(f"Processing global stats for {algo_name}, bucket size {b_size}...")
    write_to_log(f"Processing global stats for {algo_name}, bucket size {b_size}...")
    
    # Filter each dataframe before combining them to reduce memory usage
    filtered_dfs = []
    for result in csv_results:
        csv_df = result['dataframe']
        
        try:
            # Check if this CSV contains this algo/bucket combo
            if algo_name in csv_df['algorithm'].unique() and b_size in csv_df['bucket_size'].unique():
                # Filter the dataframe first to only include this algo/bucket
                if isinstance(csv_df, cudf.DataFrame):
                    # Using cuDF
                    subset = csv_df[(csv_df['algorithm'] == algo_name) & (csv_df['bucket_size'] == b_size)]
                else:
                    # Using pandas
                    subset = csv_df[(csv_df['algorithm'] == algo_name) & (csv_df['bucket_size'] == b_size)]
                
                if len(subset) > 0:
                    filtered_dfs.append(subset)
                    result_msg = f"  {result['csv_name']}: Algorithm {algo_name}, Bucket Size {b_size}, Rows: {len(subset)}"
                    print(result_msg)
                    write_to_log(result_msg)
        except Exception as e:
            print(f"Error filtering {result['csv_name']} for {algo_name}/{b_size}: {e}")
            write_to_log(f"Error filtering {result['csv_name']} for {algo_name}/{b_size}: {e}")
    
    # Now process this specific algo/bucket combo
    if filtered_dfs:
        try:
            # Combine filtered dataframes
            if has_cudf and all(isinstance(df, cudf.DataFrame) for df in filtered_dfs):
                # Use cuDF concat for GPU acceleration if all are cuDF
                filtered_df = cudf.concat(filtered_dfs, ignore_index=True)
            else:
                # Convert any cuDF dataframes to pandas first
                pandas_filtered = [df.to_pandas() if isinstance(df, cudf.DataFrame) else df for df in filtered_dfs]
                filtered_df = pd.concat(pandas_filtered, ignore_index=True)
            
            # Calculate statistics just for this specific algo/bucket
            stats = calculate_basic_stats(filtered_df, algo_name, b_size)
            
            # Free memory immediately
            del filtered_df
            del filtered_dfs
            gc.collect()
            if has_rmm and has_cudf:
                try:
                    rmm.reinitialize()
                except Exception:
                    pass
            
            if stats:
                basic_stats_results.append(stats)
        except Exception as e:
            print(f"Error processing combined stats for {algo_name}/{b_size}: {e}")
            write_to_log(f"Error processing combined stats for {algo_name}/{b_size}: {e}")

# Find which CSV files contain this algorithm and bucket size
for name, group in combined_df.groupby(['algorithm', 'bucket_size']):
    algo_name, b_size = name
    stats = next((s for s in basic_stats_results if s['algorithm'] == algo_name and s['bucket_size'] == b_size), None)
    if stats:
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

# Get unique algorithm names across all CSV files
unique_algos = set()
for result in csv_results:
    csv_df = result['dataframe']
    if 'algorithm' in csv_df.columns and 'normalized' in csv_df.columns:
        if isinstance(csv_df, cudf.DataFrame):
            algos = csv_df['algorithm'].unique().to_pandas()
        else:
            algos = csv_df['algorithm'].unique()
        unique_algos.update(algos)

# Process each algorithm separately
for algo_name in unique_algos:
    print(f"Processing normalized position distribution for {algo_name}...")
    write_to_log(f"Processing normalized position distribution for {algo_name}...")
    
    # Filter each dataframe and collect statistics
    filtered_dfs = []
    for result in csv_results:
        csv_df = result['dataframe']
        
        try:
            # Check if this CSV contains this algorithm and normalized column
            if 'algorithm' in csv_df.columns and 'normalized' in csv_df.columns and algo_name in csv_df['algorithm'].unique():
                # Filter the dataframe
                if isinstance(csv_df, cudf.DataFrame):
                    subset = csv_df[csv_df['algorithm'] == algo_name]
                else:
                    subset = csv_df[csv_df['algorithm'] == algo_name]
                
                if len(subset) > 0:
                    filtered_dfs.append(subset)
        except Exception as e:
            print(f"Error filtering normalized data for {result['csv_name']} ({algo_name}): {e}")
            write_to_log(f"Error filtering normalized data for {result['csv_name']} ({algo_name}): {e}")
    
    # Process this algorithm's distribution if we have data
    if filtered_dfs:
        try:
            # Combine filtered dataframes
            if has_cudf and all(isinstance(df, cudf.DataFrame) for df in filtered_dfs):
                filtered_df = cudf.concat(filtered_dfs, ignore_index=True)
            else:
                pandas_filtered = [df.to_pandas() if isinstance(df, cudf.DataFrame) else df for df in filtered_dfs]
                filtered_df = pd.concat(pandas_filtered, ignore_index=True)
            
            # Calculate distribution statistics
            norm_mean = filtered_df['normalized'].mean()
            norm_median = filtered_df['normalized'].median()
            norm_std = filtered_df['normalized'].std()
            norm_min = filtered_df['normalized'].min()
            norm_max = filtered_df['normalized'].max()
            
            # Find which CSV files contain this algorithm and write results
            for result in csv_results:
                csv_df = result['dataframe']
                if 'algorithm' in csv_df.columns and algo_name in csv_df['algorithm'].unique():
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
            
            # Free memory
            del filtered_df
            del filtered_dfs
            gc.collect()
            if has_rmm and has_cudf:
                try:
                    rmm.reinitialize()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Error processing normalized position distribution for {algo_name}: {e}")
            write_to_log(f"Error processing normalized position distribution for {algo_name}: {e}")

# Check monotonicity of fairness score
write_to_log("\nChecking monotonicity of fairness score...")
print("\nChecking monotonicity of fairness score...")

# Run monotonicity check only if columns exist
# all_monotonicity_list already initialized earlier
required_mono_cols = ['user_id', 'user_token_sum', 'normalized', 'algorithm', 'bucket_size']

# Check if any dataframe has the required columns
has_required_cols = False
for result in csv_results:
    csv_df = result['dataframe']
    if all(col in csv_df.columns for col in required_mono_cols):
        has_required_cols = True
        break

if has_required_cols:
    write_to_log("\nChecking monotonicity...")
    print("\nChecking monotonicity...")
    
    # Process each algorithm/bucket combination separately (using the same pairs from earlier)
    for algo_name, b_size in algo_bucket_pairs:
        print(f"Processing monotonicity for {algo_name}, bucket size {b_size}...")
        write_to_log(f"Processing monotonicity for {algo_name}, bucket size {b_size}...")
        
        # Filter each dataframe before combining them to reduce memory usage
        filtered_dfs = []
        for result in csv_results:
            csv_df = result['dataframe']
            
            try:
                # Check if this CSV contains this algo/bucket combo and required columns
                if (algo_name in csv_df['algorithm'].unique() and 
                    b_size in csv_df['bucket_size'].unique() and
                    all(col in csv_df.columns for col in required_mono_cols)):
                    
                    # Filter the dataframe first to only include this algo/bucket
                    if isinstance(csv_df, cudf.DataFrame):
                        # Using cuDF
                        subset = csv_df[(csv_df['algorithm'] == algo_name) & 
                                       (csv_df['bucket_size'] == b_size)]
                    else:
                        # Using pandas
                        subset = csv_df[(csv_df['algorithm'] == algo_name) & 
                                       (csv_df['bucket_size'] == b_size)]
                    
                    if len(subset) > 0:
                        filtered_dfs.append(subset)
                        result_msg = f"  Checking monotonicity in {result['csv_name']} for {algo_name}/{b_size}"
                        write_to_log(result_msg)
            except Exception as e:
                print(f"Error filtering for monotonicity {result['csv_name']} for {algo_name}/{b_size}: {e}")
                write_to_log(f"Error filtering for monotonicity {result['csv_name']} for {algo_name}/{b_size}: {e}")
        
        # Now process this specific algo/bucket combo
        if filtered_dfs:
            try:
                # Combine filtered dataframes
                if has_cudf and all(isinstance(df, cudf.DataFrame) for df in filtered_dfs):
                    # Use cuDF concat for GPU acceleration if all are cuDF
                    filtered_df = cudf.concat(filtered_dfs, ignore_index=True)
                else:
                    # Convert any cuDF dataframes to pandas first
                    pandas_filtered = [df.to_pandas() if isinstance(df, cudf.DataFrame) else df for df in filtered_dfs]
                    filtered_df = pd.concat(pandas_filtered, ignore_index=True)
                
                # Calculate monotonicity just for this specific algo/bucket
                mono_result = check_monotonicity(filtered_df, algo_name, b_size)
                all_monotonicity_list.append(mono_result)
                
                if isinstance(mono_result, dict) and 'non_monotonic_pct' in mono_result:
                    # Log to main log file
                    msg = f"  {mono_result['algorithm']} (Bucket: {mono_result['bucket_size']}): {mono_result['non_monotonic_pct']:.2f}% non-monotonic ({mono_result['non_monotonic_pairs']}/{mono_result['total_pairs']} pairs)"
                    write_to_log(msg)
                    print(msg)
                else:
                    write_to_log(f"  Monotonicity check: No valid data available for {algo_name}/{b_size}")
                    print(f"  Monotonicity check: No valid data available for {algo_name}/{b_size}")
                
                # Free memory immediately
                del filtered_df
                del filtered_dfs
                gc.collect()
                if has_rmm and has_cudf:
                    try:
                        rmm.reinitialize()
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error processing monotonicity for {algo_name}/{b_size}: {e}")
                write_to_log(f"Error processing monotonicity for {algo_name}/{b_size}: {e}")
    
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
        write_to_log(f"  Min load: {result.get('min_load', 0):.2f}")
        write_to_log(f"  Max load: {result.get('max_load', 0):.2f}")
        write_to_log(f"  Mean load: {result.get('mean_load', 0):.2f}")
        write_to_log(f"  Std dev: {result.get('std_load', 0):.2f}")
        write_to_log(f"  Load balance ratio (max/min): {result.get('load_balance_ratio', 0):.2f}")
        write_to_log(f"  Coefficient of variation (std/mean): {result.get('coefficient_of_variation', 0):.4f}")
        
        # Print to console
        print(f"\nLoad balance for {algo_name} (Bucket: {b_size}):")
        print(f"  Min load: {result.get('min_load', 0):.2f}")
        print(f"  Max load: {result.get('max_load', 0):.2f}")
        print(f"  Mean load: {result.get('mean_load', 0):.2f}")
        print(f"  Std dev: {result.get('std_load', 0):.2f}")
        print(f"  Load balance ratio (max/min): {result.get('load_balance_ratio', 0):.2f}")
        print(f"  Coefficient of variation (std/mean): {result.get('coefficient_of_variation', 0):.4f}")
    
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

# Define known distributions for better reporting
known_distributions = ["balanced", "high_usage", "bursty"]

# Get list of adaptive variants
algo_names = combined_df['algorithm'].unique().to_pandas() if isinstance(combined_df, cudf.DataFrame) else combined_df['algorithm'].unique()
adaptive_variants = [algo for algo in algo_names if 'adaptive' in algo]
if adaptive_variants:
    write_to_log(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    print(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    
    # Check for distributions in the data
    found_distributions = []
    for dist in known_distributions:
        if any(dist in file for file in all_files):
            found_distributions.append(dist)
    
    if found_distributions:
        write_to_log(f"Found {len(found_distributions)} distributions: {', '.join(found_distributions)}")
        print(f"Found {len(found_distributions)} distributions: {', '.join(found_distributions)}")
    
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

    # Print out all collected monotonicity data
    if all_monotonicity_list and any(isinstance(item, dict) and 'non_monotonic_pct' in item for item in all_monotonicity_list):
        # Create a dictionary to organize results by algorithm and bucket size
        mono_by_algo_bucket = {}
        
        # Organize all valid monotonicity results
        for result in all_monotonicity_list:
            if isinstance(result, dict) and 'algorithm' in result and 'bucket_size' in result and 'non_monotonic_pct' in result:
                algo = result['algorithm']
                bucket = result['bucket_size']
                key = (algo, bucket)
                mono_by_algo_bucket[key] = result
        
        # Print results for each algorithm/bucket pair that we found
        for (algo, bucket), result in sorted(mono_by_algo_bucket.items()):
            # Determine the descriptive weight type
            if 'balanced' in algo:
                weight_type = "Balanced (0.5/0.5)"
            elif 'fairness' in algo:
                weight_type = "Fairness-focused"
            elif 'utilization' in algo:
                weight_type = "Utilization-focused"
            else:
                weight_type = algo
            
            # Print result with detailed information
            non_mono_pct = result['non_monotonic_pct']
            non_mono_pairs = result['non_monotonic_pairs']
            total_pairs = result['total_pairs']
            
            msg = f"  {weight_type} (Bucket Size {bucket}): {non_mono_pct:.2f}% non-monotonic ({non_mono_pairs}/{total_pairs} pairs)"
            write_to_log(msg)
            print(msg)
    else:
        write_to_log("  No valid monotonicity data collected for comparison.")
        print("  No valid monotonicity data collected for comparison.")
    
    # Compare load balance across variants
    write_to_log("\nLoad Balance Comparison:")
    print("\nLoad Balance Comparison:")
    
    # Skip detailed load balance comparison if we don't have the required columns
    if not all(col in combined_df.columns for col in required_lb_cols):
        write_to_log("  Load balance data not available for comparison.")
        print("  Load balance data not available for comparison.")
    else:    
        for algo in adaptive_variants:
            # Convert to pandas Series if using cuDF
            bucket_sizes = combined_df['bucket_size'].unique()
            if isinstance(bucket_sizes, cudf.Series):
                bucket_sizes = bucket_sizes.to_pandas()
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
                    
                # Get load balance for this algorithm and bucket size
                lb_data = [lb for lb in load_balance_results if lb['algorithm'] == algo and lb['bucket_size'] == b_size]
                if lb_data:
                    std_dev = lb_data[0].get('load_std_dev', 0)
                    max_min = lb_data[0].get('load_max_min_ratio', 0)
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
