import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import os
import random
import time

# Output directory for advanced results
output_dir = "./bench-venkat/analysis/adaptive-variants"
os.makedirs(output_dir, exist_ok=True)

# Simulation parameters
num_pods = 5
num_users = 300
simulation_steps = 4000  # Each step is a time tick
WINDOW_SIZES_TO_TEST = [500, 1000, 2000]  # Different window sizes to test
requests_per_step = 1000  # Number of requests per time step

# User behavior parameters - Two different distributions to test
user_distributions = {
    "balanced": [
        {"category": "low", "prob": 0.2, "burstiness": 0.1, "input_base": 30, "output_base": 500},
        {"category": "medium", "prob": 0.5, "burstiness": 0.2, "input_base": 50, "output_base": 2000},
        {"category": "high", "prob": 0.3, "burstiness": 0.3, "input_base": 100, "output_base": 4000},
        {"category": "extreme", "prob": 0.0, "burstiness": 0.4, "input_base": 200, "output_base": 8000},
    ],
    "high_usage": [
        {"category": "low", "prob": 0.1, "burstiness": 0.1, "input_base": 30, "output_base": 500},
        {"category": "medium", "prob": 0.3, "burstiness": 0.2, "input_base": 50, "output_base": 2000},
        {"category": "high", "prob": 0.5, "burstiness": 0.3, "input_base": 100, "output_base": 4000},
        {"category": "extreme", "prob": 0.1, "burstiness": 0.4, "input_base": 200, "output_base": 8000},
    ]
}

# Default user distribution to use
user_profiles = user_distributions["balanced"]

# Assign user categories
user_categories = np.random.choice(
    [p["category"] for p in user_profiles],
    size=num_users,
    p=[p["prob"] for p in user_profiles]
)
user_burstiness = {cat: next(p["burstiness"] for p in user_profiles if p["category"] == cat) for cat in set(user_categories)}
user_input_base = {cat: next(p["input_base"] for p in user_profiles if p["category"] == cat) for cat in set(user_categories)}
user_output_base = {cat: next(p["output_base"] for p in user_profiles if p["category"] == cat) for cat in set(user_categories)}

# Routing algorithms
BUCKET_SIZES_TO_TEST = [500, 1000, 2000, 4000, 8000, 16000]
MIN_THRESHOLD_VALUES = [500, 1000, 2500, 5000]  # Reduced set of min threshold values
RANDOM_FACTOR_SCALE = 0.1

# Track min/max tokens for adaptive bucket sizing
min_tokens = 100.0  # Minimum reasonable token count
max_tokens = 10000.0  # Default maximum token count

# Define algorithm function
def vtc_adaptive_bucket(tokens, npods, bsize, min_tok, max_tok, min_threshold=1000):
    """Adaptive bucket size using min/max token values"""
    # Calculate adaptive bucket size as average of min and max tokens
    # with a configurable minimum threshold to prevent extremely small bucket sizes
    adaptive_bsize = max(min_threshold, (min_tok + max_tok) / 2)
    return min(tokens / adaptive_bsize, npods - 1)

# Define algorithm variants with different weight combinations and min thresholds
algorithms = {}

# Base algorithm configurations with different weight combinations
base_configs = {
    "vtc_adaptive_balanced": {
        "fairness_weight": 0.5,
        "utilization_weight": 0.5,
    },
    "vtc_adaptive_fairness_only": {
        "fairness_weight": 1.0,
        "utilization_weight": 0.0,
    },
    "vtc_adaptive_utilization_only": {
        "fairness_weight": 0.0,
        "utilization_weight": 1.0,
    },
    "vtc_adaptive_equal_weights": {
        "fairness_weight": 1.0,
        "utilization_weight": 1.0,
    },
    "vtc_adaptive_fairness_07_03": {
        "fairness_weight": 0.7,
        "utilization_weight": 0.3,
    },
    "vtc_adaptive_fairness_03_07": {
        "fairness_weight": 0.3,
        "utilization_weight": 0.7,
    }
}

# Create algorithm variants with different min thresholds
for algo_name, config in base_configs.items():
    # Use the standard 1000 min threshold as the default variant
    algorithms[algo_name] = {
        "fairness_weight": config["fairness_weight"],
        "utilization_weight": config["utilization_weight"],
        "min_threshold": 1000,
        "func": lambda tokens, npods, bsize, fw=config["fairness_weight"], uw=config["utilization_weight"], mt=1000: 
               vtc_adaptive_bucket(tokens, npods, bsize, min_tokens, max_tokens, mt)
    }
    
    # Create min threshold variants for all weight combinations
    for threshold in MIN_THRESHOLD_VALUES:
        if threshold != 1000:  # Skip 1000 as it's already the default
            variant_name = f"{algo_name}_min{threshold}"
            algorithms[variant_name] = {
                "fairness_weight": config["fairness_weight"],
                "utilization_weight": config["utilization_weight"],
                "min_threshold": threshold,
                "func": lambda tokens, npods, bsize, fw=config["fairness_weight"], uw=config["utilization_weight"], mt=threshold: 
                       vtc_adaptive_bucket(tokens, npods, bsize, min_tokens, max_tokens, mt)
            }

# Simulation function
def simulate_algorithm(algo_key, bucket_size, window_size, user_dist, filename):
    # Set up user profiles based on distribution
    user_profiles = user_distributions[user_dist]
    
    # Assign user categories
    user_categories = np.random.choice(
        [p["category"] for p in user_profiles],
        size=num_users,
        p=[p["prob"] for p in user_profiles]
    )
    user_burstiness = {cat: next(p["burstiness"] for p in user_profiles if p["category"] == cat) for cat in set(user_categories)}
    user_input_base = {cat: next(p["input_base"] for p in user_profiles if p["category"] == cat) for cat in set(user_categories)}
    user_output_base = {cat: next(p["output_base"] for p in user_profiles if p["category"] == cat) for cat in set(user_categories)}
    
    # State: per-user token history (for sliding window), per-pod load
    user_token_history = [[] for _ in range(num_users)]
    user_token_sum = np.zeros(num_users)
    pod_loads = np.zeros(num_pods)
    
    # Create CSV file and write header
    csv_file = os.path.join(output_dir, filename)
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            "timestamp", "step", "user_id", "category", "input_tokens", "output_tokens",
            "user_token_sum", "normalized", "best_pod", "pod_loads", "bucket_size",
            "algorithm", "fairness_weight", "utilization_weight", "min_threshold",
            "window_size", "user_distribution"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # For adaptive bucket sizing
    global min_tokens, max_tokens
    
    # Get algorithm weights and parameters
    fairness_weight = algorithms[algo_key]["fairness_weight"]
    utilization_weight = algorithms[algo_key]["utilization_weight"]
    min_threshold = algorithms[algo_key].get("min_threshold", 1000)  # Default to 1000 if not specified
    algo_func = algorithms[algo_key]["func"]

    for step in tqdm(range(simulation_steps), desc=f"{algo_key} simulation"):
        # Select users for this step (simulate bursty arrivals)
        active_users = []
        for user in range(num_users):
            cat = user_categories[user]
            burst_prob = user_burstiness[cat]
            if random.random() < burst_prob:
                active_users.append(user)
        # Always ensure at least some requests per step
        while len(active_users) < requests_per_step:
            active_users.append(random.randint(0, num_users-1))

        # Shuffle to randomize order
        random.shuffle(active_users)
        for user in active_users:
            cat = user_categories[user]
            input_tokens = int(np.random.normal(user_input_base[cat], user_input_base[cat]*0.2))
            output_tokens = int(np.random.normal(user_output_base[cat], user_output_base[cat]*0.2))
            total_tokens = max(0, input_tokens) + max(0, output_tokens)
            # Update sliding window
            user_token_history[user].append((step, total_tokens))
            user_token_sum[user] += total_tokens
            # Remove expired tokens from sliding window
            while user_token_history[user] and user_token_history[user][0][0] <= step - window_size:
                _, expired = user_token_history[user].pop(0)
                user_token_sum[user] -= expired
                
            # Update min/max tokens for adaptive bucket sizing
            if user_token_sum[user] > 0:
                min_tokens = min(min_tokens, user_token_sum[user]) if min_tokens > 0 else user_token_sum[user]
                max_tokens = max(max_tokens, user_token_sum[user])
                
            # Compute normalized position
            normalized = algo_func(user_token_sum[user], num_pods, bucket_size)
            
            # All variants now use the hybrid scoring approach
            min_score = float('inf')
            best_pod = 0
            for pod in range(num_pods):
                fairness_score = abs(pod - normalized)
                utilization_score = pod_loads[pod] / max(1, pod_loads.max())
                random_factor = random.random() * RANDOM_FACTOR_SCALE
                score = (fairness_weight * fairness_score) + (utilization_weight * utilization_score) + random_factor
                if score < min_score:
                    min_score = score
                    best_pod = pod
                        
            # Assign to pod and update pod load
            pod_loads[best_pod] += 1
            
            # Create result dictionary
            result = {
                "timestamp": step, # Using step as a proxy for timestamp
                "step": step,
                "user_id": user,
                "category": cat,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "user_token_sum": user_token_sum[user],
                "normalized": normalized, # This is the fairness score component
                "best_pod": best_pod,     # Actual assigned pod after hybrid scoring
                "pod_loads": pod_loads.tolist(),
                "bucket_size": bucket_size,
                "algorithm": algo_key,
                "fairness_weight": fairness_weight,
                "utilization_weight": utilization_weight,
                "min_threshold": min_threshold,
                "window_size": window_size,
                "user_distribution": user_dist
            }
            
            # Append to CSV file instead of keeping in memory
            with open(os.path.join(output_dir, filename), 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                writer.writerow(result)
        # Decay pod loads to simulate request completion
        pod_loads *= 0.995  # Reduced decay for higher sustained load
        
        # Periodically log progress to reduce memory pressure
        if step % 100 == 0:
            print(f"{algo_key} (Bucket: {bucket_size}, Window: {window_size}, Dist: {user_dist}) - {step}/{simulation_steps} steps complete")
    
    print(f"{algo_key} (Bucket: {bucket_size}, Window: {window_size}, Dist: {user_dist}) simulation complete. Results saved to {output_dir}/{filename}")

if __name__ == "__main__":
    # Import CSV module
    import csv
    
    # Create all combinations to test
    all_combinations = []
    for algo_key in algorithms:
        for b_size in BUCKET_SIZES_TO_TEST:
            for w_size in WINDOW_SIZES_TO_TEST:
                for u_dist in user_distributions.keys():
                    filename = f"sim_{algo_key}_b{b_size}_w{w_size}_{u_dist}.csv"
                    all_combinations.append((algo_key, b_size, w_size, u_dist, filename))
    
    # Shuffle combinations to distribute workload more evenly
    random.shuffle(all_combinations)
    
    # Limit to 4 processes at a time
    max_processes = 10
    
    # Process combinations in batches
    for i in range(0, len(all_combinations), max_processes):
        batch = all_combinations[i:i+max_processes]
        procs = []
        
        for algo_key, b_size, w_size, u_dist, filename in batch:
            p = mp.Process(target=simulate_algorithm, args=(algo_key, b_size, w_size, u_dist, filename))
            procs.append(p)
            p.start()
            print(f"Started simulation for {algo_key} (Bucket: {b_size}, Window: {w_size}, Dist: {u_dist})")
        
        for p in procs:
            p.join()
        
        print(f"Completed batch {i//max_processes + 1}/{(len(all_combinations) + max_processes - 1)//max_processes}")
    
    print("All adaptive variant simulations complete!")
