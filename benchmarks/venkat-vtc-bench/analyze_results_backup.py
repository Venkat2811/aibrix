#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy.stats import pearsonr
import glob

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directory for analysis
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

input_dir = args.input_dir

# Create output directory structure
if args.output_dir:
    base_output_dir = args.output_dir
else:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("bench-venkat", "analysis", "output", f"analysis_{timestamp}")

# Create output directory structure
os.makedirs(base_output_dir, exist_ok=True)

# Create subdirectories for different output types
csv_dir = os.path.join(base_output_dir, "csv_results")
comparison_dir = os.path.join(base_output_dir, "comparison_graph_inputs")

# Create all directories
for directory in [csv_dir, comparison_dir]:
    os.makedirs(directory, exist_ok=True)
    
# Create a single text file for all analysis
analysis_file = os.path.join(base_output_dir, "analysis_results.txt")
with open(analysis_file, 'w') as f:
    f.write(f"Analysis started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input directory: {input_dir}\n")
    f.write(f"Output directory: {base_output_dir}\n")
    f.write(f"Parallel processing: {args.parallel}\n")
    if args.parallel:
        f.write(f"Max processes: {args.max_processes}\n")
    f.write("\n" + "="*80 + "\n\n")

# Function to write to text file
def write_to_text_file(content):
    with open(analysis_file, 'a') as f:
        f.write(content + '\n')

# Function to get a clean name for a file
def get_clean_name(filename):
    # Extract the base filename without path and extension
    base_name = os.path.basename(filename)
    base_name = os.path.splitext(base_name)[0]
    # Clean up any special characters
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
    return clean_name

# Function to analyze a single CSV file
def analyze_csv_file(filename):
    print(f"\nAnalyzing {filename}...")
    
    # Create output directory for this CSV file
    csv_name = get_clean_name(filename)
    csv_output_dir = os.path.join(base_output_dir, csv_name)
    
    # Create directory
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # Write header for this file's analysis to the main analysis file
    write_to_text_file(f"\n{'='*80}\n")
    write_to_text_file(f"ANALYSIS FOR: {csv_name}\n")
    write_to_text_file(f"{'='*80}\n")
    
    # Write analysis start info to the main analysis file
    write_to_text_file(f"Analysis of {filename}")
    write_to_text_file(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Load the CSV file
        df = pd.read_csv(filename)
        
        # Try to extract bucket size from filename if not in data
        if 'bucket_size' not in df.columns:
            try:
                b_size = int(filename.split('_b')[-1].split('.csv')[0])
                df['bucket_size'] = b_size
                write_to_text_file(f"Inferring bucket_size {b_size} from filename")
            except Exception as e:
                write_to_text_file(f"Warning: Could not infer bucket_size from filename: {e}")
        
        # Return the dataframe and output directories
        return {
            'filename': filename,
            'dataframe': df,
            'output_dir': csv_output_dir,
            'write_func': write_to_text_file
        }
    
    except Exception as e:
        write_to_text_file(f"Error loading CSV file: {e}")
        with open(log_file, 'a') as f:
            f.write(f"Error processing {filename}: {e}\n")
        return None

print(f"Loading benchmark data from: {input_dir}")
file_pattern = os.path.join(input_dir, "*.csv")
all_files = glob.glob(file_pattern)
if not all_files:
    print(f"Error: No CSV files found in '{input_dir}'. Did the benchmark run correctly?")
    with open(log_file, 'a') as f:
        f.write(f"Error: No CSV files found in '{input_dir}'. Did the benchmark run correctly?\n")
    exit()

print(f"Found {len(all_files)} files to analyze.")
with open(log_file, 'a') as f:
    f.write(f"Found {len(all_files)} files to analyze.\n")

# Process each CSV file
if args.parallel:
    # Process files in parallel
    with mp.Pool(processes=args.max_processes) as pool:
        results = pool.map(analyze_csv_file, all_files)
        # Filter out None results (failed processing)
        csv_results = [r for r in results if r is not None]
else:
    # Process files sequentially
    csv_results = []
    for filename in tqdm(all_files, desc="Processing files"):
        result = analyze_csv_file(filename)
        if result is not None:
            csv_results.append(result)

# Check if we have any valid results
if not csv_results:
    print("Error: No CSV files were processed successfully.")
    with open(log_file, 'a') as f:
        f.write("Error: No CSV files were processed successfully.\n")
    exit()

# Combine all dataframes for final comparison
combined_df = pd.concat([r['dataframe'] for r in csv_results], axis=0, ignore_index=True)

# Save the combined dataframe for later comparison
combined_csv_path = os.path.join(comparison_inputs_dir, "combined_data.csv")
combined_df.to_csv(combined_csv_path, index=False)
print(f"Saved combined data to {combined_csv_path}")

# --- Data cleaning/mapping (ensure algorithm names are consistent) --- #
# Map old algorithm names to new ones if needed
algo_mapping = {
    'vtc_fixed_bucket': 'vtc_fixed',
    'vtc_adaptive_bucket': 'vtc_adaptive'
}

# Apply mapping if old names are found
for old_name, new_name in algo_mapping.items():
    if old_name in combined_df['algorithm'].unique():
        print(f"Mapping algorithm name: {old_name} -> {new_name}")
        write_to_text_file("analysis_log.txt", f"Mapping algorithm name: {old_name} -> {new_name}")
        combined_df.loc[combined_df['algorithm'] == old_name, 'algorithm'] = new_name

# Add algorithm type column for easier grouping
combined_df['algo_type'] = combined_df['algorithm'].apply(
    lambda x: 'Adaptive Bucket' if 'adaptive' in x else 'Fixed Bucket'
)

# Add min threshold column if it exists in the data
if 'min_threshold' in combined_df.columns:
    # Create a min threshold group column for easier analysis
    combined_df['min_threshold_group'] = combined_df['min_threshold'].apply(
        lambda x: f'Min Threshold: {int(x)}'
    )
else:
    # Add a default min threshold column if it doesn't exist
    combined_df['min_threshold'] = 1000
    combined_df['min_threshold_group'] = 'Min Threshold: 1000'
    
# Add window size column if it exists in the data
if 'window_size' in combined_df.columns:
    # Create a window size group column for easier analysis
    combined_df['window_size_group'] = combined_df['window_size'].apply(
        lambda x: f'Window Size: {int(x)}'
    )
else:
    # Add a default window size column if it doesn't exist
    combined_df['window_size'] = 1000
    combined_df['window_size_group'] = 'Window Size: 1000'
    
# Add user distribution column if it exists in the data
if 'user_distribution' in combined_df.columns:
    # Create a user distribution group column for easier analysis
    combined_df['user_distribution_group'] = combined_df['user_distribution'].apply(
        lambda x: f'User Dist: {x}'
    )
else:
    # Add a default user distribution column if it doesn't exist
    combined_df['user_distribution'] = 'balanced'
    combined_df['user_distribution_group'] = 'User Dist: balanced'

# Add weight type column for easier analysis
def get_weight_type(algo_name):
    # First handle the min threshold variants
    if '_min' in algo_name:
        # Extract the base algorithm name and min threshold
        parts = algo_name.split('_min')
        base_algo = parts[0]
        min_threshold = parts[1] if len(parts) > 1 else '1000'
        
        # Get the base weight type
        if 'balanced' in base_algo:
            return f'Balanced (0.5/0.5) Min={min_threshold}'
        elif 'fairness_only' in base_algo:
            return f'Fairness Only (1.0/0.0) Min={min_threshold}'
        elif 'utilization_only' in base_algo:
            return f'Utilization Only (0.0/1.0) Min={min_threshold}'
        elif 'equal_weights' in base_algo:
            return f'Equal Weights (1.0/1.0) Min={min_threshold}'
        elif 'fairness_07_03' in base_algo:
            return f'Fairness Heavy (0.7/0.3) Min={min_threshold}'
        elif 'fairness_03_07' in base_algo:
            return f'Utilization Heavy (0.3/0.7) Min={min_threshold}'
    
    # Handle standard variants
    if 'balanced' in algo_name and '_min' not in algo_name:
        return 'Balanced (0.5/0.5)'
    elif 'fairness_only' in algo_name:
        return 'Fairness Only (1.0/0.0)'
    elif 'utilization_only' in algo_name:
        return 'Utilization Only (0.0/1.0)'
    elif 'equal_weights' in algo_name:
        return 'Equal Weights (1.0/1.0)'
    elif 'fairness_07_03' in algo_name:
        return 'Fairness Heavy (0.7/0.3)'
    elif 'fairness_03_07' in algo_name:
        return 'Utilization Heavy (0.3/0.7)'
    else:
        return 'Other'
        
combined_df['weight_type'] = combined_df['algorithm'].apply(get_weight_type)

print("Data loaded. Performing analysis...")

# Define a function to calculate basic statistics for a group
def calculate_basic_stats(name, group):
    algo_name, b_size = name
    result = f"\nAlgorithm: {algo_name}, Bucket Size: {b_size}:\n"
    result += f"  Data points: {len(group)}\n"
    result += f"  Normalized position range: {group['normalized'].min():.2f} to {group['normalized'].max():.2f}\n"
    result += f"  Mean normalized position: {group['normalized'].mean():.2f}\n"
    result += f"  Std dev of normalized position: {group['normalized'].std():.2f}\n"
    
    # Create a dictionary to store the stats for CSV output
    stats_dict = {
        'algorithm': algo_name,
        'bucket_size': b_size,
        'data_points': len(group),
        'normalized_min': group['normalized'].min(),
        'normalized_max': group['normalized'].max(),
        'normalized_mean': group['normalized'].mean(),
        'normalized_std': group['normalized'].std()
    }
    
    if 'user_token_sum' in group.columns and 'best_pod' in group.columns:
        valid_data = group[['user_token_sum', 'best_pod']].dropna()
        valid_data = valid_data[np.isfinite(valid_data['user_token_sum'])]
        if len(valid_data) > 1:
            try:
                corr, p_val = pearsonr(valid_data['user_token_sum'], valid_data['best_pod'])
                result += f"  Fairness (Token-Pod Correlation): {corr:.4f} (p={p_val:.3f})\n"
                stats_dict['fairness_correlation'] = corr
                stats_dict['p_value'] = p_val
            except ValueError as e:
                result += f"  Fairness (Token-Pod Correlation): Error calculating - {e}\n"
                stats_dict['fairness_correlation'] = None
                stats_dict['p_value'] = None
        else:
            result += "  Fairness (Token-Pod Correlation): Not enough data for correlation\n"
            stats_dict['fairness_correlation'] = None
            stats_dict['p_value'] = None
    else:
        result += "  Fairness (Token-Pod Correlation): Missing required columns.\n"
        stats_dict['fairness_correlation'] = None
        stats_dict['p_value'] = None
    
    return result, stats_dict

# Basic statistics - Group by algorithm AND bucket_size
print("\n=== Basic Statistics (Grouped by Algo & Bucket Size) ===")
write_to_text_file("basic_statistics.txt", "=== Basic Statistics (Grouped by Algo & Bucket Size) ===")
grouped_stats = combined_df.groupby(['algorithm', 'bucket_size'])

# List to store all basic stats for CSV output
all_basic_stats = []

if args.parallel:
    # Run in parallel
    with mp.Pool(processes=args.max_processes) as pool:
        results = pool.starmap(calculate_basic_stats, [(name, group) for name, group in grouped_stats])
        for result_text, stats_dict in results:
            print(result_text)
            write_to_text_file("basic_statistics.txt", result_text)
            all_basic_stats.append(stats_dict)
else:
    # Run sequentially
    for name, group in grouped_stats:
        result_text, stats_dict = calculate_basic_stats(name, group)
        print(result_text)
        write_to_text_file("basic_statistics.txt", result_text)
        all_basic_stats.append(stats_dict)

# Save basic stats to CSV
if all_basic_stats:
    basic_stats_df = pd.DataFrame(all_basic_stats)
    basic_stats_df.to_csv(os.path.join(csv_dir, "basic_statistics.csv"), index=False)

# Analysis 1: Distribution of normalized positions (Fairness Component)
print("\nGenerating distribution plots...")
write_to_text_file("analysis_summary.txt", "\nGenerating distribution plots...")
plt.figure(figsize=(15, 8))
# Use different colors for fixed vs adaptive
for algo in combined_df['algorithm'].unique():
    if 'adaptive' in algo:
        # Analysis 1: Distribution of normalized positions (Fairness Component)
        print("\nAnalyzing normalized position distributions...")
        write_to_text_file("normalized_distribution.txt", "\nAnalyzing normalized position distributions...")

        # Only generate plots if requested
        if args.generate_graphs:
            print("Generating distribution plots...")
            plt.figure(figsize=(15, 8))
            sns.histplot(data=combined_df, x='normalized', hue='algorithm', kde=True, bins=30)
            plt.title('Distribution of Normalized Positions by Algorithm', fontsize=16)
            plt.xlabel('Normalized Position', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.savefig(os.path.join(graphs_dir, "normalized_distribution.png"), dpi=300, bbox_inches='tight')

# Always calculate and print the distribution statistics
distribution_stats = []
for algo in combined_df['algorithm'].unique():
    algo_data = combined_df[combined_df['algorithm'] == algo]
    stats_text = f"\nNormalized position distribution for {algo}:\n"
    stats_text += f"  Mean: {algo_data['normalized'].mean():.4f}\n"
    stats_text += f"  Median: {algo_data['normalized'].median():.4f}\n"
    stats_text += f"  Std Dev: {algo_data['normalized'].std():.4f}\n"
    stats_text += f"  Min: {algo_data['normalized'].min():.4f}\n"
    stats_text += f"  Max: {algo_data['normalized'].max():.4f}"
    
    print(stats_text)
    write_to_text_file("normalized_distribution.txt", stats_text)
    
    # Save stats for CSV
    distribution_stats.append({
        'algorithm': algo,
        'normalized_mean': algo_data['normalized'].mean(),
        'normalized_median': algo_data['normalized'].median(),
        'normalized_std': algo_data['normalized'].std(),
        'normalized_min': algo_data['normalized'].min(),
        'normalized_max': algo_data['normalized'].max()
    })

# Save distribution stats to CSV
if distribution_stats:
    dist_stats_df = pd.DataFrame(distribution_stats)
    dist_stats_df.to_csv(os.path.join(csv_dir, "normalized_distribution_stats.csv"), index=False)

# Analysis 2: Normalized position by user category
print("Analyzing user categories...")
plt.figure(figsize=(15, 8))
# Add algo_type to the plot to distinguish fixed vs adaptive
sns.boxplot(x='algorithm', y='normalized', hue='category', data=combined_df)
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.title('Normalized Position by User Category and Algorithm', fontsize=16)
plt.xlabel('Algorithm', fontsize=14)
plt.ylabel('Normalized Position', fontsize=14)
plt.legend(title='User Category', fontsize=12)
plt.savefig(os.path.join(graphs_dir, "normalized_by_category.png"), dpi=300, bbox_inches='tight')

# Analysis 3: Relationship between token count and normalized position
print("Analyzing token count vs normalized position...")
# Sample for visualization (full dataset is too large)
sample_df = combined_df.sample(n=10000, random_state=42)

plt.figure(figsize=(18, 6))
for i, algo in enumerate(sample_df['algorithm'].unique()):
    plt.subplot(1, 3, i+1)
    algo_df = sample_df[sample_df['algorithm'] == algo]
    plt.scatter(algo_df['input_tokens'], algo_df['normalized'], alpha=0.5, s=10)
    plt.title(algo, fontsize=14)
    plt.xlabel('Total Tokens', fontsize=12)
    plt.ylabel('Normalized Position', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "tokens_vs_normalized.png"), dpi=300, bbox_inches='tight')

# Analysis 4: Effect of bucket size (skip if not present)
if 'bucket_size' in combined_df.columns:
    print("Analyzing bucket size effects...")
    plt.figure(figsize=(15, 8))
    for algo in combined_df['algorithm'].unique():
        algo_df = combined_df[combined_df['algorithm'] == algo]
        means = algo_df.groupby('bucket_size')['normalized'].mean()
        stds = algo_df.groupby('bucket_size')['normalized'].std()
        plt.errorbar(means.index, means.values, yerr=stds.values, label=algo, capsize=5, marker='o')
    plt.title('Effect of Bucket Size on Normalized Position', fontsize=16)
    plt.xlabel('Bucket Size', fontsize=14)
    plt.ylabel('Mean Normalized Position (with std dev)', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(graphs_dir, "bucket_size_effect.png"), dpi=300, bbox_inches='tight')
else:
    print("Skipping bucket size analysis (no 'bucket_size' column in advanced benchmark output)")

# Analysis 5: Monotonicity check - how often does the fairness score decrease when token sum increases?
print("\nChecking monotonicity of fairness score...")

def check_monotonicity(df, algo_name, bucket_size): 
    # Check if necessary columns exist
    if 'user_id' not in df.columns or 'user_token_sum' not in df.columns or 'normalized' not in df.columns:
        # print(f"Skipping monotonicity check for {algo_name} (Bucket: {bucket_size}): Missing required columns.")
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

    monotonicity_results = []
    unique_users = user_data['user_id'].unique()

    for user in tqdm(unique_users, desc=f"Mono {algo_name[:3]} b{bucket_size}", leave=False):
        subset = user_data[user_data['user_id'] == user].sort_values('timestamp') # Use timestamp if available, else arbitrary sort

        # Ensure we have token sums and normalized positions to compare
        if 'user_token_sum' in subset.columns and 'normalized' in subset.columns and len(subset) > 1:
            token_diff = np.diff(subset['user_token_sum'])
            normalized_diff = np.diff(subset['normalized'])

            # Find where token sum increased but normalized position decreased
            non_monotonic_indices = np.where((token_diff > 0) & (normalized_diff < 0))[0]
            # Find comparisons where token sum increased
            increasing_token_indices = np.where(token_diff > 0)[0]

            non_monotonic_count = len(non_monotonic_indices)
            total_comparisons = len(increasing_token_indices)

            if total_comparisons > 0:
                non_monotonic_percent = 100 * non_monotonic_count / total_comparisons
            else:
                non_monotonic_percent = 0 # Or NaN, depending on desired handling

            monotonicity_results.append({
                'user_id': user,
                'non_monotonic_percent': non_monotonic_percent
            })
        else:
             monotonicity_results.append({
                'user_id': user,
                'non_monotonic_percent': np.nan # Not enough data or missing columns
            })

    result_df = pd.DataFrame(monotonicity_results)
    # Calculate the mean non-monotonic percentage for the algorithm
    mean_perc = result_df['non_monotonic_percent'].mean() # Use mean, ignoring NaNs
    # Return bucket_size along with results
    return {
        'algorithm': algo_name,
        'bucket_size': bucket_size,
        'monotonic_pairs': 0,
        'non_monotonic_pairs': 0,
        'total_pairs': 0,
        'non_monotonic_pct': mean_perc
    }

# Run monotonicity check only if columns exist (advanced data)
all_monotonicity_list = []
required_mono_cols = ['user_id', 'user_token_sum', 'normalized', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_mono_cols):
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
                print(f"  {result['algorithm']} (Bucket: {result['bucket_size']}): {result['non_monotonic_pct']:.2f}% non-monotonic ({result['non_monotonic_pairs']}/{result['total_pairs']} pairs)")
            else:
                print(f"  Monotonicity check: No valid data available")
    
    # Check if we have valid results before creating DataFrame
    if all_monotonicity_list and all(isinstance(item, dict) for item in all_monotonicity_list):
        # Convert to DataFrame for easier analysis
        all_monotonicity_df = pd.DataFrame(all_monotonicity_list)
        
        # Save monotonicity results to CSV
        all_monotonicity_df.to_csv(os.path.join(csv_dir, "monotonicity_results.csv"), index=False)
        
        # Plot non-monotonic percentage by algorithm and bucket size if requested
        if args.generate_graphs and 'non_monotonic_pct' in all_monotonicity_df.columns:
            plt.figure(figsize=(12, 7))
            sns.barplot(x='algorithm', y='non_monotonic_pct', hue='bucket_size', data=all_monotonicity_df)
            plt.title('Non-Monotonic Percentage by Algorithm and Bucket Size (lower is better)', fontsize=16)
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Non-Monotonic Percentage', fontsize=14)
            plt.ylim(bottom=0)
            plt.legend(title='Bucket Size')
            plt.savefig(os.path.join(graphs_dir, "monotonicity.png"), dpi=300, bbox_inches='tight')
    else:
        print("  No valid monotonicity data to analyze")
else:
    print(f"\nSkipping Monotonicity analysis: Required columns ({', '.join(required_mono_cols)}) not all found.")

# Analysis 6: Predictability - How consistent is pod assignment for similar user states?
print("\nCalculating predictability...")

def calculate_predictability(df, algo_name, bucket_size, num_bins=10): 
    # Check if necessary columns exist
    if 'user_id' not in df.columns or 'user_token_sum' not in df.columns or 'best_pod' not in df.columns:
        # print(f"Skipping predictability check for {algo_name} (Bucket: {bucket_size}): Missing required columns.")
        return pd.DataFrame({'algorithm': [algo_name], 'bucket_size': [bucket_size], 'predictability_score': [np.nan]})

    # Create bins based on quantiles of user_token_sum
    try:
        df['token_bin'] = pd.qcut(df['user_token_sum'], num_bins, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Warning: Could not create {num_bins} bins for {algo_name} (Bucket: {bucket_size}) due to data distribution ({e}). Trying fewer bins or skipping.")
        # Fallback: Maybe try with fewer bins or just use raw values if binning fails drastically
        # For now, we'll return NaN
        return pd.DataFrame({'algorithm': [algo_name], 'bucket_size': [bucket_size], 'predictability_score': [np.nan]})

    # Group by user and token bin, then calculate std dev of assigned pod
    predictability_groups = df.groupby(['user_id', 'token_bin'])['best_pod']
    # Calculate std dev for each group, only if group has more than one assignment
    std_devs = predictability_groups.apply(lambda x: x.std() if len(x) > 1 else 0)

    # Overall predictability is the mean standard deviation (lower is better)
    # Ignore NaN values which might result from single-assignment groups or calculation issues
    mean_std_dev = std_devs.mean(skipna=True)

    return pd.DataFrame({'algorithm': [algo_name], 'bucket_size': [bucket_size], 'predictability_score': [mean_std_dev]})

# Run predictability check only if columns exist (advanced data)
all_predictability_list = []
required_pred_cols = ['user_id', 'user_token_sum', 'best_pod', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_pred_cols):
    print("\nCalculating predictability...")
    for name, group in tqdm(combined_df.groupby(['algorithm', 'bucket_size']), desc="Predictability Groups"):
        algo_name, b_size = name
        pred_df = calculate_predictability(group, algo_name, b_size)
        all_predictability_list.append(pred_df)

    if all_predictability_list:
        all_predictability_df = pd.concat(all_predictability_list, ignore_index=True)

        # ... (Print and plot, potentially using hue='bucket_size') ...
        print("\n=== Predictability Analysis ===")
        print("Average standard deviation of assigned pod (lower is more predictable):")
        all_predictability_df.sort_values(by=['algorithm', 'bucket_size'], inplace=True)
        for _, row in all_predictability_df.iterrows():
            print(f"  Algo: {row['algorithm']}, Bucket: {row['bucket_size']}, Predictability: {row['predictability_score']:.4f}")
        plt.figure(figsize=(12, 7))
        sns.barplot(x='algorithm', y='predictability_score', hue='bucket_size', data=all_predictability_df)
        plt.title('Predictability by Algorithm and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Mean Std Dev of Assigned Pod', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.savefig(os.path.join(graphs_dir, "predictability_analysis.png"), dpi=300, bbox_inches='tight')
    else:
        print("No predictability results generated.")
else:
    print(f"\nSkipping Predictability analysis: Required columns ({', '.join(required_pred_cols)}) not all found.")


# Analysis 7: Load Balance (Utilization) - Use actual pod assignments
print("\nCalculating load balance from actual assignments...")

def calculate_load_balance(df, algo_name, bucket_size): 
    if 'best_pod' not in df.columns:
        # print(f"Skipping load balance check for {algo_name} (Bucket: {bucket_size}): Missing 'best_pod' column.")
        return pd.DataFrame({'algorithm': [algo_name], 'bucket_size': [bucket_size], 'load_std_dev': [np.nan], 'load_max_min_ratio': [np.nan]})

    # Count requests per pod
    pod_counts = df['best_pod'].value_counts()

    # Calculate metrics
    std_dev = pod_counts.std()
    max_load = pod_counts.max()
    min_load = pod_counts.min()
    max_min_ratio = max_load / min_load if min_load > 0 else np.inf

    # Create result DataFrame
    result = {
        'algorithm': [algo_name],
        'bucket_size': [bucket_size], 
        'load_std_dev': [std_dev],
        'load_max_min_ratio': [max_min_ratio]
    }
    
    # Add user_distribution if it exists in the dataframe
    if 'user_distribution' in df.columns and len(df['user_distribution'].unique()) == 1:
        result['user_distribution'] = [df['user_distribution'].iloc[0]]
        
    return pd.DataFrame(result)

# Run load balance check only if 'best_pod' column exists
all_lb_list = []
required_lb_cols = ['best_pod', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_lb_cols):
    print("\nCalculating load balance from actual assignments...")
    for name, group in tqdm(combined_df.groupby(['algorithm', 'bucket_size']), desc="Load Balance Groups"):
        algo_name, b_size = name
        lb_df = calculate_load_balance(group, algo_name, b_size)
        all_lb_list.append(lb_df)

    if all_lb_list:
        all_lb_df = pd.concat(all_lb_list, ignore_index=True)

        # ... (Print and plot, potentially using hue='bucket_size') ...
        print("\n=== Load Balance Analysis (Utilization) ===")
        print("Std Dev and Max/Min Ratio of requests per pod (lower is better):")
        all_lb_df.sort_values(by=['algorithm', 'bucket_size'], inplace=True)
        for _, row in all_lb_df.iterrows():
             print(f"  Algo: {row['algorithm']}, Bucket: {row['bucket_size']}, StdDev={row['load_std_dev']:.2f}, Max/Min={row['load_max_min_ratio']:.2f}")
        # Plotting might need adjustment (e.g., facet grid or grouped bar plot)
        plt.figure(figsize=(12, 7))
        sns.barplot(x='algorithm', y='load_std_dev', hue='bucket_size', data=all_lb_df)
        plt.title('Load Balance (Std Dev) by Algorithm and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Standard Deviation', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.savefig(os.path.join(graphs_dir, "load_balance_std_dev.png"), dpi=300, bbox_inches='tight')

        plt.figure(figsize=(12, 7))
        plot_lb_df = all_lb_df.copy() # Handle inf for plotting
        plot_lb_df['load_max_min_ratio'] = plot_lb_df['load_max_min_ratio'].replace(np.inf, plot_lb_df[plot_lb_df['load_max_min_ratio'] != np.inf]['load_max_min_ratio'].max() * 1.1) # Replace inf with slightly higher than max finite for viz
        sns.barplot(x='algorithm', y='load_max_min_ratio', hue='bucket_size', data=plot_lb_df)
        plt.title('Load Balance (Max/Min Ratio) by Algorithm and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Max/Min Load Ratio', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.savefig(os.path.join(graphs_dir, "load_balance_ratio.png"), dpi=300, bbox_inches='tight')
    else:
        print("No load balance results generated.")
else:
    print(f"\nSkipping Load Balance analysis: Required columns ({', '.join(required_lb_cols)}) not all found.")


# Add specific analysis for adaptive bucket variants
print("\n=== Adaptive Bucket Variants Analysis ===")

# Get all adaptive variants
adaptive_variants = [algo for algo in combined_df['algorithm'].unique() if 'adaptive' in algo]

if adaptive_variants:
    print(f"Found {len(adaptive_variants)} adaptive variants: {', '.join(adaptive_variants)}")
    
    # Compare fairness correlation across variants
    print("\nFairness Correlation Comparison (higher is better):")
    
    # Create dataframe to store correlation results
    correlation_results = []
    
    for algo in adaptive_variants:
        for b_size in sorted(combined_df['bucket_size'].unique()):
            algo_group = combined_df[(combined_df['algorithm'] == algo) & (combined_df['bucket_size'] == b_size)]
            
            if len(algo_group) > 0:
                # Calculate correlation
                valid_data = algo_group[['user_token_sum', 'best_pod']].dropna()
                valid_data = valid_data[np.isfinite(valid_data['user_token_sum'])]
                
                if len(valid_data) > 1:
                    try:
                        corr, p_val = pearsonr(valid_data['user_token_sum'], valid_data['best_pod'])
                        weight_type = get_weight_type(algo)
                        
                        # Store result
                        correlation_results.append({
                            'algorithm': algo,
                            'weight_type': weight_type,
                            'bucket_size': b_size,
                            'correlation': corr,
                            'p_value': p_val
                        })
                        
                        print(f"  {weight_type} (Bucket Size {b_size}): {corr:.4f} (p={p_val:.3f})")
                    except ValueError as e:
                        print(f"  {algo} (Bucket Size {b_size}): Error calculating correlation - {e}")
    
    # Convert to DataFrame for plotting
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        
        # Plot correlation comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(x='weight_type', y='correlation', hue='bucket_size', data=corr_df)
        plt.title('Fairness Correlation by Weight Type and Bucket Size (higher is better)', fontsize=16)
        plt.xlabel('Weight Type', fontsize=14)
        plt.ylabel('Correlation (Tokens vs Pod)', fontsize=14)
        plt.ylim(-1, 1)  # Correlation range
        plt.legend(title='Bucket Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "adaptive_variants_correlation.png"), dpi=300, bbox_inches='tight')
        
        # If we have min threshold variants, create a separate plot for them if requested
        min_threshold_variants = [algo for algo in adaptive_variants if '_min' in algo]
        if min_threshold_variants and args.generate_graphs:
            # Filter for just the balanced algorithm with different min thresholds
            min_thresh_df = corr_df[corr_df['algorithm'].str.contains('_min')]
            
            if not min_thresh_df.empty:
                plt.figure(figsize=(14, 8))
                sns.barplot(x='min_threshold', y='correlation', hue='bucket_size', data=min_thresh_df)
                plt.title('Fairness Correlation by Min Threshold and Bucket Size (higher is better)', fontsize=16)
                plt.xlabel('Min Threshold', fontsize=14)
                plt.ylabel('Correlation (Tokens vs Pod)', fontsize=14)
                plt.ylim(-1, 1)  # Correlation range
                plt.legend(title='Bucket Size')
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, "min_threshold_correlation.png"), dpi=300, bbox_inches='tight')
    
    # Compare load balance metrics
    print("\nLoad Balance Comparison:")
    
    # Create dataframe to store load balance results
    load_balance_results = []
    
    for algo in adaptive_variants:
        for b_size in sorted(combined_df['bucket_size'].unique()):
            algo_group = combined_df[(combined_df['algorithm'] == algo) & (combined_df['bucket_size'] == b_size)]
            
            if len(algo_group) > 0:
                # Get the load balance metrics from the previous calculations
                lb_data = all_lb_df[(all_lb_df['algorithm'] == algo) & (all_lb_df['bucket_size'] == b_size)]
                
                if len(lb_data) > 0:
                    weight_type = get_weight_type(algo)
                    
                    # Store result
                    load_balance_results.append({
                        'algorithm': algo,
                        'weight_type': weight_type,
                        'bucket_size': b_size,
                        'load_std_dev': lb_data['load_std_dev'].values[0],
                        'load_max_min_ratio': lb_data['load_max_min_ratio'].values[0]
                    })
    
    # Convert to DataFrame for plotting
    if load_balance_results:
        lb_df = pd.DataFrame(load_balance_results)
        
        # Plot load balance comparison (std dev)
        plt.figure(figsize=(14, 8))
        sns.barplot(x='weight_type', y='load_std_dev', hue='bucket_size', data=lb_df)
        plt.title('Load Balance (Std Dev) by Weight Type and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Weight Type', fontsize=14)
        plt.ylabel('Standard Deviation', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "adaptive_variants_load_std.png"), dpi=300, bbox_inches='tight')
        
        # Plot load balance comparison (max/min ratio)
        plt.figure(figsize=(14, 8))
        # Handle inf for plotting
        plot_lb_df = lb_df.copy()
        plot_lb_df['load_max_min_ratio'] = plot_lb_df['load_max_min_ratio'].replace(
            np.inf, plot_lb_df[plot_lb_df['load_max_min_ratio'] != np.inf]['load_max_min_ratio'].max() * 1.1
        )
        sns.barplot(x='weight_type', y='load_max_min_ratio', hue='bucket_size', data=plot_lb_df)
        plt.title('Load Balance (Max/Min Ratio) by Weight Type and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Weight Type', fontsize=14)
        plt.ylabel('Max/Min Load Ratio', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "adaptive_variants_load_ratio.png"), dpi=300, bbox_inches='tight')
    
    # Compare monotonicity
    print("\nMonotonicity Comparison:")
    
    # Create dataframe to store monotonicity results
    monotonicity_results = []
    
    for algo in adaptive_variants:
        for b_size in sorted(combined_df['bucket_size'].unique()):
            mono_data = all_monotonicity_df[(all_monotonicity_df['algorithm'] == algo) & 
                                          (all_monotonicity_df['bucket_size'] == b_size)]
            
            if len(mono_data) > 0:
                weight_type = get_weight_type(algo)
                
                # Store result
                monotonicity_results.append({
                    'algorithm': algo,
                    'weight_type': weight_type,
                    'bucket_size': b_size,
                    'non_monotonic_pct': mono_data['non_monotonic_pct'].values[0]
                })
                
                print(f"  {weight_type} (Bucket Size {b_size}): {mono_data['non_monotonic_pct'].values[0]:.2f}% non-monotonic")
    
    # Convert to DataFrame for plotting
    if monotonicity_results:
        mono_df = pd.DataFrame(monotonicity_results)
        
        # Plot monotonicity comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(x='weight_type', y='non_monotonic_pct', hue='bucket_size', data=mono_df)
        plt.title('Non-Monotonic Percentage by Weight Type and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Weight Type', fontsize=14)
        plt.ylabel('Non-Monotonic Percentage', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "adaptive_variants_monotonicity.png"), dpi=300, bbox_inches='tight')
    
    # Compare predictability
    print("\nPredictability Comparison:")
    
    # Create dataframe to store predictability results
    predictability_results = []
    
    for algo in adaptive_variants:
        for b_size in sorted(combined_df['bucket_size'].unique()):
            pred_data = all_predictability_df[(all_predictability_df['algorithm'] == algo) & 
                                            (all_predictability_df['bucket_size'] == b_size)]
            
            if len(pred_data) > 0:
                weight_type = get_weight_type(algo)
                
                # Store result
                predictability_results.append({
                    'algorithm': algo,
                    'weight_type': weight_type,
                    'bucket_size': b_size,
                    'predictability': pred_data['predictability_score'].values[0]
                })
                
                print(f"  {weight_type} (Bucket Size {b_size}): {pred_data['predictability_score'].values[0]:.4f}")
    
    # Convert to DataFrame for plotting
    if predictability_results:
        pred_df = pd.DataFrame(predictability_results)
        
        # Plot predictability comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(x='weight_type', y='predictability', hue='bucket_size', data=pred_df)
        plt.title('Predictability by Weight Type and Bucket Size (higher is better)', fontsize=16)
        plt.xlabel('Weight Type', fontsize=14)
        plt.ylabel('Predictability Score', fontsize=14)
        plt.ylim(0, 1)  # Predictability range
        plt.legend(title='Bucket Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "adaptive_variants_predictability.png"), dpi=300, bbox_inches='tight')
        
    # Create summary table for the best bucket size for each variant
    print("\nSummary of Best Bucket Size for Each Variant:")
    
    # Add specific analysis for min threshold variants
    min_threshold_variants = [algo for algo in adaptive_variants if '_min' in algo]
    if min_threshold_variants:
        print("\n=== Min Threshold Variants Analysis ===")
        print(f"Found {len(min_threshold_variants)} min threshold variants: {', '.join(min_threshold_variants)}")
        
        # Extract min threshold values from algorithm names
        for algo in min_threshold_variants:
            min_threshold = int(algo.split('_min')[1])
            print(f"  {algo}: Min Threshold = {min_threshold}")
        
        # Create plots comparing different min thresholds for the balanced algorithm if requested
        min_thresh_data = combined_df[combined_df['algorithm'].str.contains('_min')]
        if not min_thresh_data.empty and args.generate_graphs:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='min_threshold', y='normalized', data=min_thresh_data)
            plt.title('Normalized Position Distribution by Min Threshold', fontsize=16)
            plt.xlabel('Min Threshold', fontsize=14)
            plt.ylabel('Normalized Position', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "min_threshold_normalized.png"), dpi=300, bbox_inches='tight')
    
    # Add analysis for window size variants
    if 'window_size' in combined_df.columns:
        print("\n=== Window Size Analysis ===")
        window_sizes = sorted(combined_df['window_size'].unique())
        print(f"Found {len(window_sizes)} window sizes: {', '.join(map(str, window_sizes))}")
        
        # Create plots comparing different window sizes
        if args.generate_graphs:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='window_size', y='normalized', hue='algorithm', data=combined_df)
            plt.title('Normalized Position Distribution by Window Size', fontsize=16)
            plt.xlabel('Window Size', fontsize=14)
            plt.ylabel('Normalized Position', fontsize=14)
            plt.legend(title='Algorithm')
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "window_size_normalized.png"), dpi=300, bbox_inches='tight')
        
        # Compare fairness correlation across window sizes
        print("\nFairness Correlation by Window Size:")
        window_corr_data = []
        for algo in adaptive_variants:
            for w_size in window_sizes:
                for b_size in sorted(combined_df['bucket_size'].unique()):
                    algo_data = combined_df[(combined_df['algorithm'] == algo) & 
                                          (combined_df['window_size'] == w_size) & 
                                          (combined_df['bucket_size'] == b_size)]
                    if len(algo_data) > 1:
                        try:
                            corr, p_val = pearsonr(algo_data['user_token_sum'], algo_data['best_pod'])
                            window_corr_data.append({
                                'algorithm': algo,
                                'window_size': w_size,
                                'bucket_size': b_size,
                                'correlation': corr,
                                'p_value': p_val
                            })
                            print(f"  {algo} (Window: {w_size}, Bucket: {b_size}): {corr:.4f}")
                        except:
                            pass
        
        if window_corr_data:
            window_corr_df = pd.DataFrame(window_corr_data)
            plt.figure(figsize=(14, 8))
            sns.barplot(x='window_size', y='correlation', hue='algorithm', data=window_corr_df)
            plt.title('Fairness Correlation by Window Size', fontsize=16)
            plt.xlabel('Window Size', fontsize=14)
            plt.ylabel('Correlation', fontsize=14)
            plt.ylim(-1, 1)
            plt.legend(title='Algorithm')
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "window_size_correlation.png"), dpi=300, bbox_inches='tight')
    
    # Add analysis for user distribution variants
    if 'user_distribution' in combined_df.columns:
        print("\n=== User Distribution Analysis ===")
        user_distributions = sorted(combined_df['user_distribution'].unique())
        print(f"Found {len(user_distributions)} user distributions: {', '.join(user_distributions)}")
        
        # Create plots comparing different user distributions if requested
        if args.generate_graphs:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='user_distribution', y='normalized', hue='algorithm', data=combined_df)
            plt.title('Normalized Position Distribution by User Distribution', fontsize=16)
            plt.xlabel('User Distribution', fontsize=14)
            plt.ylabel('Normalized Position', fontsize=14)
            plt.legend(title='Algorithm')
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "user_dist_normalized.png"), dpi=300, bbox_inches='tight')
        
        # Compare load balance across user distributions
        print("\nLoad Balance by User Distribution:")
        dist_lb_data = []
        for algo in adaptive_variants:
            for dist in user_distributions:
                # Check if user_distribution column exists in the dataframe
                if 'user_distribution' in all_lb_df.columns:
                    lb_data = all_lb_df[(all_lb_df['algorithm'] == algo) & 
                                      (all_lb_df['user_distribution'] == dist)]
                else:
                    # If user_distribution column doesn't exist, just filter by algorithm
                    lb_data = all_lb_df[all_lb_df['algorithm'] == algo]
                    print(f"  Note: user_distribution column not found in load balance data for {algo}")
                if len(lb_data) > 0:
                    dist_lb_data.append({
                        'algorithm': algo,
                        'user_distribution': dist,
                        'load_std_dev': lb_data['load_std_dev'].mean(),
                        'load_max_min_ratio': lb_data['load_max_min_ratio'].mean()
                    })
                    print(f"  {algo} (Dist: {dist}): Std Dev = {lb_data['load_std_dev'].mean():.4f}")
        
        if dist_lb_data:
            dist_lb_df = pd.DataFrame(dist_lb_data)
            if args.generate_graphs:
                plt.figure(figsize=(14, 8))
                sns.barplot(x='user_distribution', y='load_std_dev', hue='algorithm', data=dist_lb_df)
                plt.title('Load Balance (Std Dev) by User Distribution', fontsize=16)
                plt.xlabel('User Distribution', fontsize=14)
                plt.ylabel('Standard Deviation', fontsize=14)
                plt.legend(title='Algorithm')
                plt.tight_layout()
                plt.savefig(os.path.join(graphs_dir, "user_dist_load_balance.png"), dpi=300, bbox_inches='tight')
    
    # Combine all metrics
    summary_data = []
    
    for algo in adaptive_variants:
        for b_size in sorted(combined_df['bucket_size'].unique()):
            weight_type = get_weight_type(algo)
            
            # Get metrics for this algorithm and bucket size
            corr_value = np.nan
            if correlation_results:
                corr_data = [r for r in correlation_results if r['algorithm'] == algo and r['bucket_size'] == b_size]
                if corr_data:
                    corr_value = corr_data[0]['correlation']
            
            load_std = np.nan
            if load_balance_results:
                lb_data = [r for r in load_balance_results if r['algorithm'] == algo and r['bucket_size'] == b_size]
                if lb_data:
                    load_std = lb_data[0]['load_std_dev']
            
            non_mono_pct = np.nan
            if monotonicity_results:
                mono_data = [r for r in monotonicity_results if r['algorithm'] == algo and r['bucket_size'] == b_size]
                if mono_data:
                    non_mono_pct = mono_data[0]['non_monotonic_pct']
            
            pred_score = np.nan
            if predictability_results:
                pred_data = [r for r in predictability_results if r['algorithm'] == algo and r['bucket_size'] == b_size]
                if pred_data:
                    pred_score = pred_data[0]['predictability']
            
            # Store summary row
            summary_data.append({
                'algorithm': algo,
                'weight_type': weight_type,
                'bucket_size': b_size,
                'fairness_correlation': corr_value,
                'load_std_dev': load_std,
                'non_monotonic_pct': non_mono_pct,
                'predictability': pred_score
            })
    
    # Convert to DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Find best bucket size for each variant
        for algo in adaptive_variants:
            algo_data = summary_df[summary_df['algorithm'] == algo]
            if len(algo_data) > 0:
                weight_type = get_weight_type(algo)
                print(f"\n  {weight_type}:")
                
                # Best for fairness (highest correlation)
                best_fairness = algo_data.loc[algo_data['fairness_correlation'].idxmax()]
                print(f"    Best for Fairness: Bucket Size {best_fairness['bucket_size']} (Correlation: {best_fairness['fairness_correlation']:.4f})")
                
                # Best for load balance (lowest std dev)
                best_load = algo_data.loc[algo_data['load_std_dev'].idxmin()]
                print(f"    Best for Load Balance: Bucket Size {best_load['bucket_size']} (Std Dev: {best_load['load_std_dev']:.2f})")
                
                # Best for monotonicity (lowest non-monotonic percentage)
                best_mono = algo_data.loc[algo_data['non_monotonic_pct'].idxmin()]
                print(f"    Best for Monotonicity: Bucket Size {best_mono['bucket_size']} (Non-Monotonic: {best_mono['non_monotonic_pct']:.2f}%)")
                
                # Best for predictability (highest score)
                best_pred = algo_data.loc[algo_data['predictability'].idxmax()]
                print(f"    Best for Predictability: Bucket Size {best_pred['bucket_size']} (Score: {best_pred['predictability']:.4f})")
                
                # Overall recommendation (simple average of normalized metrics)
                algo_data['fairness_norm'] = (algo_data['fairness_correlation'] - algo_data['fairness_correlation'].min()) / \
                                            (algo_data['fairness_correlation'].max() - algo_data['fairness_correlation'].min() + 1e-10)
                algo_data['load_norm'] = 1 - (algo_data['load_std_dev'] - algo_data['load_std_dev'].min()) / \
                                        (algo_data['load_std_dev'].max() - algo_data['load_std_dev'].min() + 1e-10)
                algo_data['mono_norm'] = 1 - (algo_data['non_monotonic_pct'] - algo_data['non_monotonic_pct'].min()) / \
                                        (algo_data['non_monotonic_pct'].max() - algo_data['non_monotonic_pct'].min() + 1e-10)
                algo_data['pred_norm'] = (algo_data['predictability'] - algo_data['predictability'].min()) / \
                                        (algo_data['predictability'].max() - algo_data['predictability'].min() + 1e-10)
                
                # Calculate overall score (equal weights)
                algo_data['overall_score'] = (algo_data['fairness_norm'] + algo_data['load_norm'] + 
                                            algo_data['mono_norm'] + algo_data['pred_norm']) / 4
                
                best_overall = algo_data.loc[algo_data['overall_score'].idxmax()]
                print(f"    Overall Recommendation: Bucket Size {best_overall['bucket_size']} (Score: {best_overall['overall_score']:.4f})")
        
        # Save summary table
        summary_df.to_csv(os.path.join(csv_dir, "adaptive_variants_summary.csv"), index=False)
else:
    print("No adaptive variants found in the data.")

    # Create a comprehensive summary table with all parameters
    print("\n=== Comprehensive Analysis Summary ===")
    write_to_text_file("comprehensive_summary.txt", "\n=== Comprehensive Analysis Summary ===")
    summary_data = []
    
    # Get all combinations of parameters
    all_algos = sorted(combined_df['algorithm'].unique())
    all_buckets = sorted(combined_df['bucket_size'].unique())
    all_thresholds = sorted(combined_df['min_threshold'].unique()) if 'min_threshold' in combined_df.columns else [1000]
    all_windows = sorted(combined_df['window_size'].unique()) if 'window_size' in combined_df.columns else [1000]
    all_dists = sorted(combined_df['user_distribution'].unique()) if 'user_distribution' in combined_df.columns else ['balanced']
    
    # Create a summary table with all metrics
    for algo in all_algos:
        for bucket in all_buckets:
            for threshold in all_thresholds:
                for window in all_windows:
                    for dist in all_dists:
                        # Filter data for this combination
                        combo_data = combined_df[
                            (combined_df['algorithm'] == algo) &
                            (combined_df['bucket_size'] == bucket) &
                            (combined_df['min_threshold'] == threshold)
                        ]
                        
                        if 'window_size' in combined_df.columns:
                            combo_data = combo_data[combo_data['window_size'] == window]
                            
                        if 'user_distribution' in combined_df.columns:
                            combo_data = combo_data[combined_df['user_distribution'] == dist]
                        
                        if len(combo_data) > 0:
                            # Calculate metrics
                            fairness_corr = np.nan
                            try:
                                if len(combo_data) > 1:
                                    fairness_corr, _ = pearsonr(combo_data['user_token_sum'], combo_data['best_pod'])
                            except:
                                pass
                            
                            # Get load balance metrics
                            lb_data = None
                            if 'user_distribution' in combined_df.columns:
                                lb_data = all_lb_df[(all_lb_df['algorithm'] == algo) & 
                                                  (all_lb_df['bucket_size'] == bucket) &
                                                  (all_lb_df['user_distribution'] == dist)]
                            else:
                                lb_data = all_lb_df[(all_lb_df['algorithm'] == algo) & 
                                                  (all_lb_df['bucket_size'] == bucket)]
                            
                            load_std = np.nan
                            load_ratio = np.nan
                            if lb_data is not None and len(lb_data) > 0:
                                load_std = lb_data['load_std_dev'].values[0] if 'load_std_dev' in lb_data.columns else np.nan
                                load_ratio = lb_data['load_max_min_ratio'].values[0] if 'load_max_min_ratio' in lb_data.columns else np.nan
                            
                            # Get monotonicity metrics
                            mono_data = None
                            if 'user_distribution' in combined_df.columns:
                                mono_data = all_monotonicity_df[(all_monotonicity_df['algorithm'] == algo) & 
                                                            (all_monotonicity_df['bucket_size'] == bucket) &
                                                            (all_monotonicity_df['user_distribution'] == dist)]
                            else:
                                mono_data = all_monotonicity_df[(all_monotonicity_df['algorithm'] == algo) & 
                                                            (all_monotonicity_df['bucket_size'] == bucket)]
                            
                            non_mono_pct = np.nan
                            if mono_data is not None and len(mono_data) > 0:
                                non_mono_pct = mono_data['non_monotonic_pct'].values[0] if 'non_monotonic_pct' in mono_data.columns else np.nan
                            
                            # Add to summary data
                            summary_data.append({
                                'algorithm': algo,
                                'weight_type': get_weight_type(algo),
                                'bucket_size': bucket,
                                'min_threshold': threshold,
                                'window_size': window,
                                'user_distribution': dist,
                                'fairness_correlation': fairness_corr,
                                'load_std_dev': load_std,
                                'load_max_min_ratio': load_ratio,
                                'non_monotonic_pct': non_mono_pct,
                                'sample_size': len(combo_data)
                            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(csv_dir, "comprehensive_summary.csv"), index=False)
        print(f"Created comprehensive summary with {len(summary_df)} parameter combinations")
        
        # Create pivot tables for easier analysis
        if 'fairness_correlation' in summary_df.columns:
            # Pivot by algorithm and bucket size
            pivot_by_algo_bucket = summary_df.pivot_table(
                index=['algorithm', 'weight_type'], 
                columns=['bucket_size'], 
                values=['fairness_correlation', 'load_std_dev'],
                aggfunc='mean'
            )
            pivot_by_algo_bucket.to_csv(os.path.join(csv_dir, "pivot_by_algo_bucket.csv"))
            
            # If we have min_threshold, create a pivot for that too
            if 'min_threshold' in summary_df.columns:
                pivot_by_min_threshold = summary_df.pivot_table(
                    index=['algorithm', 'weight_type'], 
                    columns=['min_threshold'], 
                    values=['fairness_correlation', 'load_std_dev'],
                    aggfunc='mean'
                )
                pivot_by_min_threshold.to_csv(os.path.join(csv_dir, "pivot_by_min_threshold.csv"))
            print(f"Created pivot tables for fairness_correlation and load_std_dev")

# Function to generate final comparison graphs
def generate_final_comparison():
    print("\nGenerating final comparison graphs...")
    with open(log_file, 'a') as f:
        f.write("\nGenerating final comparison graphs...\n")
    
    # Load the combined data
    try:
        combined_data = pd.read_csv(os.path.join(comparison_inputs_dir, "combined_data.csv"))
        
        # Create a summary dataframe
        summary_data = []
        
        # Get all unique values for each parameter
        all_algos = combined_data['algorithm'].unique()
        all_buckets = sorted(combined_data['bucket_size'].unique())
        all_thresholds = sorted(combined_data['min_threshold'].unique()) if 'min_threshold' in combined_data.columns else [None]
        all_windows = sorted(combined_data['window_size'].unique()) if 'window_size' in combined_data.columns else [None]
        all_dists = sorted(combined_data['user_distribution'].unique()) if 'user_distribution' in combined_data.columns else [None]
        
        # Create comprehensive summary for all parameter combinations
        for algo in all_algos:
            for bucket in all_buckets:
                for threshold in all_thresholds:
                    for window in all_windows:
                        for dist in all_dists:
                            # Filter data for this combination
                            filter_conditions = [
                                (combined_data['algorithm'] == algo),
                                (combined_data['bucket_size'] == bucket)
                            ]
                            
                            if threshold is not None and 'min_threshold' in combined_data.columns:
                                filter_conditions.append((combined_data['min_threshold'] == threshold))
                            
                            if window is not None and 'window_size' in combined_data.columns:
                                filter_conditions.append((combined_data['window_size'] == window))
                                
                            if dist is not None and 'user_distribution' in combined_data.columns:
                                filter_conditions.append((combined_data['user_distribution'] == dist))
                            
                            # Apply all filters
                            combo_data = combined_data
                            for condition in filter_conditions:
                                combo_data = combo_data[condition]
                            
                            if len(combo_data) > 0:
                                # Calculate metrics
                                entry = {
                                    'algorithm': algo,
                                    'bucket_size': bucket,
                                    'weight_type': get_weight_type(algo),
                                    'sample_size': len(combo_data)
                                }
                                
                                if threshold is not None and 'min_threshold' in combined_data.columns:
                                    entry['min_threshold'] = threshold
                                
                                if window is not None and 'window_size' in combined_data.columns:
                                    entry['window_size'] = window
                                    
                                if dist is not None and 'user_distribution' in combined_data.columns:
                                    entry['user_distribution'] = dist
                                
                                # Calculate fairness correlation
                                if 'user_token_sum' in combo_data.columns and 'best_pod' in combo_data.columns:
                                    valid_data = combo_data[['user_token_sum', 'best_pod']].dropna()
                                    valid_data = valid_data[np.isfinite(valid_data['user_token_sum'])]
                                    if len(valid_data) > 1:
                                        try:
                                            corr, p_val = pearsonr(valid_data['user_token_sum'], valid_data['best_pod'])
                                            entry['fairness_correlation'] = corr
                                            entry['p_value'] = p_val
                                        except Exception:
                                            pass
                                
                                # Calculate load balance
                                if 'best_pod' in combo_data.columns:
                                    pod_counts = combo_data['best_pod'].value_counts()
                                    if len(pod_counts) > 1:
                                        entry['load_std_dev'] = pod_counts.std()
                                        entry['load_max_min_ratio'] = pod_counts.max() / pod_counts.min()
                                
                                # Calculate monotonicity
                                if 'user_token_sum' in combo_data.columns and 'normalized' in combo_data.columns:
                                    sorted_data = combo_data.sort_values('user_token_sum')
                                    total_pairs = 0
                                    non_monotonic_pairs = 0
                                    for i in range(len(sorted_data) - 1):
                                        if sorted_data.iloc[i]['user_token_sum'] < sorted_data.iloc[i+1]['user_token_sum']:
                                            total_pairs += 1
                                            if sorted_data.iloc[i]['normalized'] > sorted_data.iloc[i+1]['normalized']:
                                                non_monotonic_pairs += 1
                                    
                                    if total_pairs > 0:
                                        entry['non_monotonic_pct'] = (non_monotonic_pairs / total_pairs) * 100
                                
                                summary_data.append(entry)
        
        # Create summary dataframe
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(final_comparison_dir, "comprehensive_summary.csv"), index=False)
            print(f"Created comprehensive summary with {len(summary_df)} parameter combinations")
            
            # Create final comparison graphs if requested
            if args.generate_graphs and 'fairness_correlation' in summary_df.columns:
                # Create a summary plot comparing all algorithms by fairness and load balance
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 2, 1)
                sns.barplot(x='algorithm', y='fairness_correlation', data=summary_df)
                plt.title('Fairness Correlation by Algorithm (higher is better)', fontsize=14)
                plt.xticks(rotation=45)
                plt.ylim(-1, 1)
                
                plt.subplot(1, 2, 2)
                sns.barplot(x='algorithm', y='load_std_dev', data=summary_df)
                plt.title('Load Balance (Std Dev) by Algorithm (lower is better)', fontsize=14)
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(final_comparison_dir, "overall_algorithm_comparison.png"), dpi=300, bbox_inches='tight')
                
                # Create a heatmap of fairness correlation by bucket size and min threshold
                if 'min_threshold' in summary_df.columns:
                    for algo in summary_df['algorithm'].unique():
                        algo_data = summary_df[summary_df['algorithm'] == algo]
                        if len(algo_data) > 0:
                            try:
                                pivot = algo_data.pivot_table(
                                    index='bucket_size', 
                                    columns='min_threshold', 
                                    values='fairness_correlation'
                                )
                                
                                plt.figure(figsize=(12, 8))
                                sns.heatmap(pivot, annot=True, cmap='viridis', vmin=-1, vmax=1, center=0)
                                plt.title(f'Fairness Correlation for {algo}\nby Bucket Size and Min Threshold', fontsize=16)
                                plt.tight_layout()
                                plt.savefig(os.path.join(final_comparison_dir, f"{algo}_parameter_heatmap.png"), dpi=300, bbox_inches='tight')
                            except Exception as e:
                                print(f"Error creating heatmap for {algo}: {e}")
                
                # Create comparison plots for window sizes if available
                if 'window_size' in summary_df.columns:
                    plt.figure(figsize=(14, 8))
                    sns.lineplot(x='window_size', y='fairness_correlation', hue='algorithm', 
                                 data=summary_df, marker='o', ci=None)
                    plt.title('Fairness Correlation by Window Size', fontsize=16)
                    plt.xlabel('Window Size', fontsize=14)
                    plt.ylabel('Fairness Correlation', fontsize=14)
                    plt.ylim(-1, 1)
                    plt.tight_layout()
                    plt.savefig(os.path.join(final_comparison_dir, "window_size_comparison.png"), dpi=300, bbox_inches='tight')
                
                # Create comparison plots for user distributions if available
                if 'user_distribution' in summary_df.columns:
                    plt.figure(figsize=(14, 8))
                    sns.barplot(x='user_distribution', y='fairness_correlation', hue='algorithm', data=summary_df)
                    plt.title('Fairness Correlation by User Distribution', fontsize=16)
                    plt.xlabel('User Distribution', fontsize=14)
                    plt.ylabel('Fairness Correlation', fontsize=14)
                    plt.ylim(-1, 1)
                    plt.tight_layout()
                    plt.savefig(os.path.join(final_comparison_dir, "user_dist_comparison.png"), dpi=300, bbox_inches='tight')
    
    except Exception as e:
        print(f"Error generating final comparison: {e}")
        with open(log_file, 'a') as f:
            f.write(f"Error generating final comparison: {e}\n")

# Process each CSV file individually
for result in csv_results:
    # Extract data from the result
    df = result['dataframe']
    csv_name = result['csv_name']
    output_dir = result['output_dir']
    graphs_dir = result['graphs_dir']
    write_func = result['write_func']
    
    # Basic statistics
    write_func("\n=== Basic Statistics ===")
    write_func(f"Data points: {len(df)}")
    if 'normalized' in df.columns:
        write_func(f"Normalized position range: {df['normalized'].min():.2f} to {df['normalized'].max():.2f}")
        write_func(f"Mean normalized position: {df['normalized'].mean():.2f}")
        write_func(f"Std dev of normalized position: {df['normalized'].std():.2f}")
    
    # Fairness correlation
    if 'user_token_sum' in df.columns and 'best_pod' in df.columns:
        valid_data = df[['user_token_sum', 'best_pod']].dropna()
        valid_data = valid_data[np.isfinite(valid_data['user_token_sum'])]
        if len(valid_data) > 1:
            try:
                corr, p_val = pearsonr(valid_data['user_token_sum'], valid_data['best_pod'])
                write_func(f"Fairness (Token-Pod Correlation): {corr:.4f} (p={p_val:.3f})")
                
                # Save correlation data for final comparison
                corr_data = {
                    'csv_name': csv_name,
                    'correlation': corr,
                    'p_value': p_val
                }
                corr_df = pd.DataFrame([corr_data])
                corr_df.to_csv(os.path.join(comparison_inputs_dir, f"{csv_name}_correlation.csv"), index=False)
            except Exception as e:
                write_func(f"Fairness (Token-Pod Correlation): Error calculating - {e}")
    
    # Generate graphs if requested
    if args.generate_graphs:
        # Distribution of normalized positions
        if 'normalized' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='normalized', kde=True, bins=30)
            plt.title('Distribution of Normalized Positions', fontsize=16)
            plt.xlabel('Normalized Position', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.savefig(os.path.join(graphs_dir, "normalized_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Relationship between token count and normalized position
        if 'user_token_sum' in df.columns and 'normalized' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='user_token_sum', y='normalized')
            plt.title('Token Count vs Normalized Position', fontsize=16)
            plt.xlabel('Total Tokens', fontsize=14)
            plt.ylabel('Normalized Position', fontsize=14)
            plt.savefig(os.path.join(graphs_dir, "tokens_vs_normalized.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Write completion message
    write_func(f"\nAnalysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed analysis of {csv_name}")

# Generate final comparison after all individual files are processed
generate_final_comparison()

# Write final summary
with open(os.path.join(base_output_dir, "analysis_summary.txt"), 'a') as f:
    f.write(f"\nAnalysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Results saved to:\n")
    f.write(f"  - Individual CSV analyses: {base_output_dir}/<csv_name>/\n")
    f.write(f"  - Final comparison: {final_comparison_dir}\n")

print("\nAnalysis complete. Results saved to:")
print(f"  - Individual CSV analyses: {base_output_dir}/<csv_name>/")
print(f"  - Final comparison: {final_comparison_dir}")

# Optional: Show plots if running interactively
# plt.show()
