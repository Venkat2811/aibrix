#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr
import glob
import argparse
import multiprocessing as mp
import datetime
import re

parser = argparse.ArgumentParser(description="Analyze benchmark results.")
parser.add_argument('--input_dir', type=str, default="./bench-venkat/analysis/advanced", help='Directory containing CSV results to analyze')
parser.add_argument('--output_dir', type=str, help='Directory to store analysis results (default: input_dir/analysis_TIMESTAMP)')
parser.add_argument('--parallel', action='store_true', help='Run analysis in parallel (default: False)')
parser.add_argument('--max_processes', type=int, default=4, help='Maximum number of parallel processes (default: 4)')
args = parser.parse_args()

# Create output directory structure
input_dir = args.input_dir
if args.output_dir:
    base_output_dir = args.output_dir
else:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("bench-venkat", "analysis", "output", f"analysis_{timestamp}")

# Create output directory structure
os.makedirs(base_output_dir, exist_ok=True)

# Create subdirectory for comparison data
comparison_dir = os.path.join(base_output_dir, "comparison_data")
os.makedirs(comparison_dir, exist_ok=True)

# Create a main log file
log_file = os.path.join(base_output_dir, "analysis_log.txt")
with open(log_file, 'w') as f:
    f.write(f"Analysis started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input directory: {input_dir}\n")
    f.write(f"Output directory: {base_output_dir}\n")
    f.write(f"Parallel processing: {args.parallel}\n")
    if args.parallel:
        f.write(f"Max processes: {args.max_processes}\n")
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
    print(f"\nAnalyzing {filename}...")
    
    # Create output directory for this CSV file
    csv_name = get_clean_name(filename)
    csv_output_dir = os.path.join(base_output_dir, csv_name)
    
    # Create directory structure for this CSV file
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_csv_dir = os.path.join(csv_output_dir, "csv_data")
    os.makedirs(csv_csv_dir, exist_ok=True)
    
    # Create analysis file for this CSV
    analysis_file = os.path.join(csv_output_dir, "analysis_results.txt")
    with open(analysis_file, 'w') as f:
        f.write(f"Analysis of {filename}\n")
        f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Log to main log file
    write_to_log(f"\nAnalyzing: {csv_name}")
    write_to_log(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(filename)
        
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
            'csv_dir': csv_csv_dir,
            'analysis_file': analysis_file
        }
    
    except Exception as e:
        error_msg = f"Error loading CSV file: {e}"
        write_to_csv_analysis(csv_output_dir, error_msg)
        write_to_log(f"Error processing {filename}: {e}")
        return None

# Main analysis functions
def calculate_basic_stats(df, algo_name, bucket_size):
    """Calculate basic statistics for a given algorithm and bucket size"""
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
        corr, p_value = pearsonr(subset['user_token_sum'], subset['normalized'])
    else:
        corr = p_value = np.nan
    
    # Format results
    result = {
        'algorithm': algo_name,
        'bucket_size': bucket_size,
        'count': count,
        'norm_min': norm_min,
        'norm_max': norm_max,
        'norm_mean': norm_mean,
        'norm_std': norm_std,
        'fairness_correlation': corr,
        'p_value': p_value
    }
    
    return result

def check_monotonicity(df, algo_name, bucket_size, max_users=100): 
    """Check if token count monotonically increases with normalized position"""
    # Check if necessary columns exist
    if 'user_id' not in df.columns or 'user_token_sum' not in df.columns or 'normalized' not in df.columns:
        return {
            'algorithm': algo_name,
            'bucket_size': bucket_size,
            'monotonic_pairs': 0,
            'non_monotonic_pairs': 0,
            'total_pairs': 0,
            'non_monotonic_pct': 0
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
            'non_monotonic_pct': 0
        }

    # Get all users for calculation (no sampling)
    unique_users = user_data['user_id'].unique()
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
        tokens = subset['user_token_sum'].values
        norms = subset['normalized'].values
        
        # Check each pair of consecutive requests
        monotonic_pairs = 0
        non_monotonic_pairs = 0
        
        for i in range(len(tokens) - 1):
            current_tokens = tokens[i]
            next_tokens = tokens[i+1]
            current_norm = norms[i]
            next_norm = norms[i+1]
            
            if current_tokens < next_tokens:
                # Tokens increased, normalized should increase
                if current_norm <= next_norm:
                    monotonic_pairs += 1
                else:
                    non_monotonic_pairs += 1
            elif current_tokens > next_tokens:
                # Tokens decreased, normalized should decrease
                if current_norm >= next_norm:
                    monotonic_pairs += 1
                else:
                    non_monotonic_pairs += 1
            # If tokens equal, any change in normalized is acceptable
            else:
                monotonic_pairs += 1
        
        # Add to totals
        total_monotonic_pairs += monotonic_pairs
        total_non_monotonic_pairs += non_monotonic_pairs
    
    # Calculate final metrics
    total_pairs = total_monotonic_pairs + total_non_monotonic_pairs
    
    if total_pairs > 0:
        non_monotonic_pct = (total_non_monotonic_pairs / total_pairs) * 100
    else:
        non_monotonic_pct = 0
        
    return {
        'algorithm': algo_name,
        'bucket_size': bucket_size,
        'monotonic_pairs': total_monotonic_pairs,
        'non_monotonic_pairs': total_non_monotonic_pairs,
        'total_pairs': total_pairs,
        'non_monotonic_pct': non_monotonic_pct
    }

def calculate_load_balance(df, algo_name, bucket_size): 
    """Calculate load balance metrics based on pod assignments"""
    if 'best_pod' not in df.columns:
        return {
            'algorithm': algo_name,
            'bucket_size': bucket_size, 
            'load_std_dev': np.nan,
            'load_max_min_ratio': np.nan
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

    # Create result
    result = {
        'algorithm': algo_name,
        'bucket_size': bucket_size, 
        'load_std_dev': std_dev,
        'load_max_min_ratio': max_min_ratio
    }
    
    # Add user_distribution if it exists in the dataframe
    if 'user_distribution' in subset.columns and len(subset['user_distribution'].unique()) == 1:
        result['user_distribution'] = subset['user_distribution'].iloc[0]
        
    return result

# Main analysis workflow
print(f"Loading benchmark data from: {input_dir}")
file_pattern = os.path.join(input_dir, "*.csv")
all_files = glob.glob(file_pattern)
if not all_files:
    print(f"No CSV files found in {input_dir}")
    exit(1)

print(f"Found {len(all_files)} files to analyze.")
write_to_log(f"Found {len(all_files)} files to analyze.")

# Process each CSV file
csv_results = []

# Create a list of tasks for parallel processing
if args.parallel:
    # Process files in parallel
    with mp.Pool(processes=args.max_processes) as pool:
        csv_results = list(tqdm(pool.imap(analyze_csv_file, all_files), total=len(all_files), desc="Processing files"))
        csv_results = [r for r in csv_results if r is not None]  # Filter out None results
else:
    # Process files sequentially
    for filename in tqdm(all_files, desc="Processing files"):
        result = analyze_csv_file(filename)
        if result is not None:
            csv_results.append(result)

# Combine all dataframes for comparison analysis
combined_dfs = []
for result in csv_results:
    combined_dfs.append(result['dataframe'])

# Concatenate all dataframes
if combined_dfs:
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    # Save combined data for later analysis
    combined_csv_path = os.path.join(comparison_dir, "combined_data.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined data to {combined_csv_path}")
    write_to_log(f"Saved combined data to {combined_csv_path}")
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
for algo_name in combined_df['algorithm'].unique():
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
    for algo_name in combined_df['algorithm'].unique():
        for bucket_size in sorted(combined_df['bucket_size'].unique()):
            monotonicity_tasks.append((combined_df, algo_name, bucket_size))
    
    # Run monotonicity checks
    if args.parallel:
        # Run in parallel
        with mp.Pool(processes=args.max_processes) as pool:
            all_monotonicity_list = pool.starmap(check_monotonicity, monotonicity_tasks)
    else:
        # Run sequentially
        for task in monotonicity_tasks:
            result = check_monotonicity(*task)
            all_monotonicity_list.append(result)
            
            if isinstance(result, dict) and 'non_monotonic_pct' in result:
                # Log to main log file
                write_to_log(f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
                print(f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
                
                # Find which CSV files contain this algorithm and bucket size
                for csv_result in csv_results:
                    csv_df = csv_result['dataframe']
                    if result['algorithm'] in csv_df['algorithm'].unique() and result['bucket_size'] in csv_df['bucket_size'].unique():
                        # Write to this CSV's analysis file
                        write_to_csv_analysis(csv_result['output_dir'], "\n=== Monotonicity Analysis ===\n")
                        write_to_csv_analysis(csv_result['output_dir'], f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
                        
                        # Save monotonicity results to CSV file in this CSV's directory
                        mono_df = pd.DataFrame([result])
                        mono_df.to_csv(os.path.join(csv_result['csv_dir'], "monotonicity.csv"), index=False)
            else:
                write_to_log(f"  Monotonicity check: No valid data available")
                print(f"  Monotonicity check: No valid data available")
    
    # Check if we have valid results before creating DataFrame
    if all_monotonicity_list and all(isinstance(item, dict) for item in all_monotonicity_list):
        # Convert to DataFrame for easier analysis
        all_monotonicity_df = pd.DataFrame(all_monotonicity_list)
        
        # Save monotonicity results to comparison directory
        all_monotonicity_df.to_csv(os.path.join(comparison_dir, "monotonicity_results.csv"), index=False)
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
required_lb_cols = ['node_id', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_lb_cols):
    load_balance_results = []
    
    # Group by algorithm and bucket size
    for name, group in combined_df.groupby(['algorithm', 'bucket_size']):
        algo_name, b_size = name
        
        # Calculate load balance
        node_counts = group['node_id'].value_counts()
        min_load = node_counts.min() if not node_counts.empty else 0
        max_load = node_counts.max() if not node_counts.empty else 0
        mean_load = node_counts.mean() if not node_counts.empty else 0
        std_load = node_counts.std() if not node_counts.empty else 0
        
        # Calculate load balance ratio (max/min)
        load_balance_ratio = max_load / min_load if min_load > 0 else float('inf')
        
        # Calculate coefficient of variation (std/mean)
        cv = std_load / mean_load if mean_load > 0 else float('inf')
        
        # Store results
        result = {
            'algorithm': algo_name,
            'bucket_size': b_size,
            'min_load': min_load,
            'max_load': max_load,
            'mean_load': mean_load,
            'std_load': std_load,
            'load_balance_ratio': load_balance_ratio,
            'coefficient_of_variation': cv
        }
        load_balance_results.append(result)
        
        # Find which CSV files contain this algorithm and bucket size
        for csv_result in csv_results:
            csv_df = csv_result['dataframe']
            if algo_name in csv_df['algorithm'].unique() and b_size in csv_df['bucket_size'].unique():
                # Write to this CSV's analysis file
                write_to_csv_analysis(csv_result['output_dir'], "\n=== Load Balance Analysis ===\n")
                write_to_csv_analysis(csv_result['output_dir'], f"Load balance for {algo_name} (Bucket: {b_size}):")
                write_to_csv_analysis(csv_result['output_dir'], f"  Min load: {min_load:.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Max load: {max_load:.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Mean load: {mean_load:.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Std dev: {std_load:.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Load balance ratio (max/min): {load_balance_ratio:.2f}")
                write_to_csv_analysis(csv_result['output_dir'], f"  Coefficient of variation (std/mean): {cv:.4f}")
                
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
    
    # Save load balance results to comparison directory
    if load_balance_results:
        lb_df = pd.DataFrame(load_balance_results)
        lb_df.to_csv(os.path.join(comparison_dir, "load_balance.csv"), index=False)
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
adaptive_variants = [algo for algo in combined_df['algorithm'].unique() if 'adaptive' in algo]
if adaptive_variants:
    write_to_log(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    print(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    
    # Compare fairness correlation across variants
    write_to_log("\nFairness Correlation Comparison (higher is better):")
    print("\nFairness Correlation Comparison (higher is better):")
    
    for algo in adaptive_variants:
        for b_size in sorted(combined_df['bucket_size'].unique()):
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
