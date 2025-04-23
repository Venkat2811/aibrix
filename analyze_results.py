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

parser = argparse.ArgumentParser(description="Analyze benchmark results.")
parser.add_argument('--input_dir', type=str, default="./bench-venkat/analysis/advanced", help='Directory containing CSV results to analyze')
args = parser.parse_args()

input_dir = args.input_dir.rstrip('/')
analysis_dir = os.path.join(input_dir, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

print(f"Loading benchmark data from: {input_dir}")
file_pattern = os.path.join(input_dir, "sim_*_advanced_b*.csv")
all_files = glob.glob(file_pattern)
if not all_files:
    print(f"Error: No files found matching pattern '{file_pattern}'. Did the benchmark run correctly?")
    exit()

print(f"Found {len(all_files)} files to analyze.")
li = []
for filename in tqdm(all_files, desc="Loading files"):
    try:
        df = pd.read_csv(filename, index_col=None, header=0)
        # --- Infer bucket_size from filename if column missing (fallback) --- #
        if 'bucket_size' not in df.columns:
            try:
                b_size = int(filename.split('_b')[-1].split('.csv')[0])
                df['bucket_size'] = b_size
                print(f"Inferring bucket_size {b_size} for {filename}")
            except Exception as e:
                print(f"Warning: Could not infer bucket_size for {filename}: {e}")
        li.append(df)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue # Skip problematic files

# Combine dataframes
if not li:
    print("Error: No dataframes loaded successfully.")
    exit()
combined_df = pd.concat(li, axis=0, ignore_index=True)

# --- Data cleaning/mapping (ensure algorithm names are consistent) --- #
# Extract algo name and bucket size for clarity if needed, though columns should exist
# Example: combined_df['algorithm_name'] = combined_df['algorithm'].str.replace('_advanced_b.*', '', regex=True)

print("Data loaded. Performing analysis...")

# Basic statistics - Group by algorithm AND bucket_size
print("\n=== Basic Statistics (Grouped by Algo & Bucket Size) ===")
grouped_stats = combined_df.groupby(['algorithm', 'bucket_size'])
for name, group in grouped_stats:
    algo_name, b_size = name
    print(f"\nAlgorithm: {algo_name}, Bucket Size: {b_size}:")
    print(f"  Data points: {len(group)}")
    print(f"  Normalized position range: {group['normalized'].min():.2f} to {group['normalized'].max():.2f}")
    print(f"  Mean normalized position: {group['normalized'].mean():.2f}")
    print(f"  Std dev of normalized position: {group['normalized'].std():.2f}")
    if 'user_token_sum' in group.columns and 'best_pod' in group.columns:
        valid_data = group[['user_token_sum', 'best_pod']].dropna()
        valid_data = valid_data[np.isfinite(valid_data['user_token_sum'])]
        if len(valid_data) > 1:
            try:
                corr, p_val = pearsonr(valid_data['user_token_sum'], valid_data['best_pod'])
                print(f"  Fairness (Token-Pod Correlation): {corr:.4f} (p={p_val:.3f})")
            except ValueError as e:
                 print(f"  Fairness (Token-Pod Correlation): Error calculating - {e}")
        else:
            print("  Fairness (Token-Pod Correlation): Not enough data for correlation")
    else:
        print("  Fairness (Token-Pod Correlation): Missing required columns.")

# Analysis 1: Distribution of normalized positions (Fairness Component)
print("\nGenerating distribution plots...")
plt.figure(figsize=(15, 8))
for algo in combined_df['algorithm'].unique():
    sns.kdeplot(combined_df[combined_df['algorithm'] == algo]['normalized'], 
                label=algo, fill=True, alpha=0.3)
plt.title('Distribution of Normalized Positions Across Algorithms (Fairness Score)', fontsize=16)
plt.xlabel('Normalized Position (Fairness Score)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(fontsize=12)
plt.savefig(os.path.join(analysis_dir, "normalized_distribution.png"), dpi=300, bbox_inches='tight')

# Analysis 2: Normalized position by user category
print("Analyzing user categories...")
plt.figure(figsize=(15, 8))
sns.boxplot(x='algorithm', y='normalized', hue='category', data=combined_df)
plt.title('Normalized Position by User Category and Algorithm', fontsize=16)
plt.xlabel('Algorithm', fontsize=14)
plt.ylabel('Normalized Position', fontsize=14)
plt.legend(title='User Category', fontsize=12)
plt.savefig(os.path.join(analysis_dir, "normalized_by_category.png"), dpi=300, bbox_inches='tight')

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
plt.savefig(os.path.join(analysis_dir, "tokens_vs_normalized.png"), dpi=300, bbox_inches='tight')

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
    plt.savefig(os.path.join(analysis_dir, "bucket_size_effect.png"), dpi=300, bbox_inches='tight')
else:
    print("Skipping bucket size analysis (no 'bucket_size' column in advanced benchmark output)")

# Analysis 5: Monotonicity check - how often does the fairness score decrease when token sum increases?
print("\nChecking monotonicity of fairness score...")

def check_monotonicity(df, algo_name, bucket_size): # Added bucket_size param
    # Check if necessary columns exist
    if 'user_id' not in df.columns or 'user_token_sum' not in df.columns or 'normalized' not in df.columns:
        # print(f"Skipping monotonicity check for {algo_name} (Bucket: {bucket_size}): Missing required columns.")
        return pd.DataFrame({'algorithm': [algo_name], 'bucket_size': [bucket_size], 'mean_non_monotonic_percent': [np.nan]})

    monotonicity_results = []
    unique_users = df['user_id'].unique()

    for user in tqdm(unique_users, desc=f"Mono {algo_name[:3]} b{bucket_size}", leave=False):
        subset = df[df['user_id'] == user].sort_values('timestamp') # Use timestamp if available, else arbitrary sort

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
    return pd.DataFrame({'algorithm': [algo_name], 'bucket_size': [bucket_size], 'mean_non_monotonic_percent': [mean_perc]})

# Run monotonicity check only if columns exist (advanced data)
all_monotonicity_list = []
required_mono_cols = ['user_id', 'user_token_sum', 'normalized', 'algorithm', 'bucket_size']
if all(col in combined_df.columns for col in required_mono_cols):
    print("Calculating Monotonicity...")
    for name, group in tqdm(combined_df.groupby(['algorithm', 'bucket_size']), desc="Monotonicity Groups"):
        algo_name, b_size = name
        mono_df = check_monotonicity(group, algo_name, b_size)
        all_monotonicity_list.append(mono_df)

    if all_monotonicity_list:
        all_monotonicity_df = pd.concat(all_monotonicity_list, ignore_index=True)

        print("\n=== Monotonicity Analysis (Fairness Score) ===")
        print("Average % of times fairness score DECREASED when user token sum INCREASED:")
        # Sort for consistent output
        all_monotonicity_df.sort_values(by=['algorithm', 'bucket_size'], inplace=True)
        for _, row in all_monotonicity_df.iterrows():
             print(f"  Algo: {row['algorithm']}, Bucket: {row['bucket_size']}, Non-Monotonic: {row['mean_non_monotonic_percent']:.2f}%")

        # Visualize monotonicity results
        # Plotting might need adjustment (e.g., facet grid or grouped bar plot)
        plt.figure(figsize=(12, 7))
        sns.barplot(x='algorithm', y='mean_non_monotonic_percent', hue='bucket_size', data=all_monotonicity_df)
        plt.title('Non-Monotonicity by Algorithm and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('% Non-Monotonic Cases', fontsize=14)
        plt.ylim(bottom=0) # Start y-axis at 0
        plt.legend(title='Bucket Size')
        plt.savefig(os.path.join(analysis_dir, "monotonicity_analysis.png"), dpi=300, bbox_inches='tight')
    else:
        print("No monotonicity results generated.")
else:
     print(f"\nSkipping Monotonicity analysis: Required columns ({', '.join(required_mono_cols)}) not all found.")

# Analysis 6: Predictability - How consistent is pod assignment for similar user states?
print("\nCalculating predictability...")

def calculate_predictability(df, algo_name, bucket_size, num_bins=10): # Added bucket_size
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
        plt.savefig(os.path.join(analysis_dir, "predictability_analysis.png"), dpi=300, bbox_inches='tight')
    else:
        print("No predictability results generated.")
else:
    print(f"\nSkipping Predictability analysis: Required columns ({', '.join(required_pred_cols)}) not all found.")


# Analysis 7: Load Balance (Utilization) - Use actual pod assignments
print("\nCalculating load balance from actual assignments...")

def calculate_load_balance(df, algo_name, bucket_size): # Added bucket_size
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

    return pd.DataFrame({
        'algorithm': [algo_name],
        'bucket_size': [bucket_size], # Add bucket_size
        'load_std_dev': [std_dev],
        'load_max_min_ratio': [max_min_ratio]
    })

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
        plt.savefig(os.path.join(analysis_dir, "load_balance_std_dev.png"), dpi=300, bbox_inches='tight')

        plt.figure(figsize=(12, 7))
        plot_lb_df = all_lb_df.copy() # Handle inf for plotting
        plot_lb_df['load_max_min_ratio'] = plot_lb_df['load_max_min_ratio'].replace(np.inf, plot_lb_df[plot_lb_df['load_max_min_ratio'] != np.inf]['load_max_min_ratio'].max() * 1.1) # Replace inf with slightly higher than max finite for viz
        sns.barplot(x='algorithm', y='load_max_min_ratio', hue='bucket_size', data=plot_lb_df)
        plt.title('Load Balance (Max/Min Ratio) by Algorithm and Bucket Size (lower is better)', fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel('Max/Min Load Ratio', fontsize=14)
        plt.ylim(bottom=0)
        plt.legend(title='Bucket Size')
        plt.savefig(os.path.join(analysis_dir, "load_balance_ratio.png"), dpi=300, bbox_inches='tight')
    else:
        print("No load balance results generated.")
else:
    print(f"\nSkipping Load Balance analysis: Required columns ({', '.join(required_lb_cols)}) not all found.")


print("\nAnalysis complete. Plots saved to:", analysis_dir)


# Optional: Show plots if running interactively
# plt.show()
