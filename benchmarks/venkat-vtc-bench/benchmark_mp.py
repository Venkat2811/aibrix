# Re-import necessary libraries after code execution state reset
import pandas as pd
import numpy as np
import threading
import os
import multiprocessing as mp
from tqdm import tqdm
import time

# Create output directory
output_dir = "./bench-venkat"
os.makedirs(output_dir, exist_ok=True)

# Simulation parameters
input_token_lengths = list(range(10, 4001, 200))  # 10 to 4000 step 200
output_token_lengths = list(range(200, 10001, 500))  # 200 to 10000 step 500
bucket_variants = 5
num_users = 20
requests_per_user = 100
num_pods = 5

# Load distribution
low_percent = 0.2
high_percent = 0.25
medium_percent = 1.0 - low_percent - high_percent

low_users = int(num_users * low_percent)
high_users = int(num_users * high_percent)
medium_users = num_users - low_users - high_users

# Utility function to simulate token loads for a user category
def generate_user_tokens(category, base_input, base_output):
    if category == "low":
        return np.random.randint(0.5 * base_input, 0.8 * base_input), np.random.randint(0.5 * base_output, 0.8 * base_output)
    elif category == "high":
        return np.random.randint(1.2 * base_input, 1.8 * base_input), np.random.randint(1.2 * base_output, 1.8 * base_output)
    else:  # medium
        return np.random.randint(0.8 * base_input, 1.2 * base_input), np.random.randint(0.8 * base_output, 1.2 * base_output)

# Normalization functions
def modulo_routing(user_tokens, num_pods, bucket_size):
    normalized = (user_tokens % (num_pods * bucket_size)) / bucket_size
    return normalized

def quotient_remainder_routing(user_tokens, num_pods, bucket_size):
    q = user_tokens // bucket_size
    r = (user_tokens % bucket_size) / bucket_size
    return (q % num_pods) + r

def clamped_linear_routing(user_tokens, num_pods, bucket_size):
    index = user_tokens / bucket_size
    return min(index, num_pods - 1)

# Calculate total iterations for progress bar
total_iterations = len(input_token_lengths) * len(output_token_lengths) * bucket_variants * num_users * requests_per_user

# Simulation function template
def simulate_variant(normalization_func, filename, func_name):
    results = []
    
    # Create a progress bar for this process
    pbar = tqdm(total=total_iterations, desc=f"Process: {func_name}")
    
    for input_len in input_token_lengths:
        for output_len in output_token_lengths:
            bucket_sizes = [
                int((input_len + output_len) * scale)
                for scale in [0.5, 0.75, 1.0, 1.25, 1.5]
            ]

            for bucket_size in bucket_sizes:
                user_categories = (
                    ["low"] * low_users + 
                    ["high"] * high_users + 
                    ["medium"] * medium_users
                )

                for user_index, category in enumerate(user_categories):
                    for _ in range(requests_per_user):
                        input_tokens, output_tokens = generate_user_tokens(category, input_len, output_len)
                        total_tokens = input_tokens + output_tokens
                        normalized = normalization_func(total_tokens, num_pods, bucket_size)
                        results.append({
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "normalized": normalized,
                            "bucket_size": bucket_size,
                            "user_category": category,
                            "input_len_base": input_len,
                            "output_len_base": output_len
                        })
                        pbar.update(1)

    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    pbar.close()
    print(f"Completed {func_name}, saved to {output_path}")

# Function to run in separate process
def run_simulation(normalization_func, filename, func_name):
    print(f"Starting process for {func_name}")
    simulate_variant(normalization_func, filename, func_name)

if __name__ == "__main__":
    # Create processes for each variant
    p1 = mp.Process(target=run_simulation, 
                   args=(modulo_routing, "sim_modulo_full.csv", "Modulo Routing"))
    
    p2 = mp.Process(target=run_simulation, 
                   args=(quotient_remainder_routing, "sim_qr_with_cycling_full.csv", "Quotient-Remainder Routing"))
    
    p3 = mp.Process(target=run_simulation, 
                   args=(clamped_linear_routing, "sim_clamped_linear_full.csv", "Clamped Linear Routing"))

    # Start processes
    print("Starting all simulation processes...")
    p1.start()
    p2.start()
    p3.start()

    # Wait for completion
    p1.join()
    p2.join()
    p3.join()
    
    print("All simulations completed!")