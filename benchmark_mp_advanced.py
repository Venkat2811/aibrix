import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import os
import random
import time

# Output directory for advanced results
output_dir = "./bench-venkat/analysis/advanced-balanced"
os.makedirs(output_dir, exist_ok=True)

# Simulation parameters
num_pods = 5
num_users = 300
simulation_steps = 4000  # Each step is a time tick
window_size = 1000       # Sliding window size (in steps)
requests_per_step = 1000  # Number of requests per time step

# User behavior parameters
user_profiles = [
    {"category": "low", "prob": 0.2, "burstiness": 0.1, "input_base": 30, "output_base": 500},
    {"category": "medium", "prob": 0.5, "burstiness": 0.2, "input_base": 50, "output_base": 2000},
    {"category": "high", "prob": 0.3, "burstiness": 0.3, "input_base": 100, "output_base": 4000},
]

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
BUCKET_SIZES_TO_TEST = [1000, 2000, 4000, 8000]
FAIRNESS_WEIGHT = 0.5
UTILIZATION_WEIGHT = 0.5
RANDOM_FACTOR_SCALE = 0.1

algorithms = {
    "modulo": lambda tokens, npods, bsize: (tokens % (npods * bsize)) / bsize,
    "quotient_remainder": lambda tokens, npods, bsize: (tokens // bsize) % npods + (tokens % bsize) / bsize,
    "clamped_linear": lambda tokens, npods, bsize: min(tokens / bsize, npods - 1),
}

# Simulation function
def simulate_algorithm(algo_key, bucket_size, filename):
    # State: per-user token history (for sliding window), per-pod load
    user_token_history = [[] for _ in range(num_users)]
    user_token_sum = np.zeros(num_users)
    pod_loads = np.zeros(num_pods)
    results = []

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
            # Compute normalized position
            normalized = algorithms[algo_key](user_token_sum[user], num_pods, bucket_size)
            # Hybrid scoring
            min_score = float('inf')
            best_pod = 0
            for pod in range(num_pods):
                fairness_score = abs(pod - normalized)
                utilization_score = pod_loads[pod] / max(1, pod_loads.max())
                random_factor = random.random() * RANDOM_FACTOR_SCALE
                score = FAIRNESS_WEIGHT * fairness_score + UTILIZATION_WEIGHT * utilization_score + random_factor
                if score < min_score:
                    min_score = score
                    best_pod = pod
            # Assign to pod and update pod load
            pod_loads[best_pod] += 1
            # Record result
            results.append({
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
                "algorithm": algo_key
            })
        # Decay pod loads to simulate request completion
        pod_loads *= 0.995  # Reduced decay for higher sustained load
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"{algo_key} (Bucket: {bucket_size}) simulation complete. Results saved to {output_dir}/{filename}")

if __name__ == "__main__":
    procs = []
    for algo_key in algorithms:
        for b_size in BUCKET_SIZES_TO_TEST:
            filename = f"sim_{algo_key}_advanced_b{b_size}.csv"
            p = mp.Process(target=simulate_algorithm, args=(algo_key, b_size, filename))
            procs.append(p)
            p.start()
    for p in procs:
        p.join()
    print("All advanced simulations complete for all bucket sizes!")
