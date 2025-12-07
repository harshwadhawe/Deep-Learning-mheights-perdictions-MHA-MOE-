# This script generates a comprehensive dataset by iterating through all required
# parameter sets (N, K, M) for N=9, generating 50 random P matrices for each set,
# computing the exact m-Height h_m(C) using the parallel LP solver, and saving
# the aggregated input (X) and output (Y) data.

import numpy as np
from scipy.optimize import linprog
from itertools import combinations, product, permutations
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
import pickle 
import os

# --- CONFIGURATION ---
NUM_CORES = 60           # Number of CPU cores to utilize for parallel processing
N_SAMPLES_PER_SET = 10000   # Number of random samples to generate for each (N, K, M) set
COEFF_MIN = -100         # Min coefficient for P matrix elements
COEFF_MAX = 100          # Max coefficient for P matrix elements
HEIGHT_TOLERANCE = 1e-6  # Tolerance for filtering out near-zero results (m-Height > 1e-6)

# Parameters based on the problem context (N=9, K in {4,5,6}, M in {2, ..., N-K})
N_VAL = 9
K_OPTIONS = [4, 5, 6]

# 1. Define all parameter sets (N, K, M) required for the full dataset
PARAMETER_SETS = []
for K in K_OPTIONS:
    N_MINUS_K = N_VAL - K
    # M ranges from 2 up to N-K (inclusive)
    M_OPTIONS = list(range(2, N_MINUS_K + 1))
    for M in M_OPTIONS:
        PARAMETER_SETS.append((N_VAL, K, M))

TOTAL_EXPECTED_SAMPLES = len(PARAMETER_SETS) * N_SAMPLES_PER_SET


# --- UTILITIES ---

def generate_random_p_matrix(k, n, min_val, max_val):
    """Generates a random P matrix for G = [I_k | P] with specified coefficient range."""
    if n <= k:
        raise ValueError("n must be greater than k for a meaningful P matrix.")
    num_parity_cols = n - k
    # Using integer coefficients as they are common and simpler, adjust if real coefficients are needed.
    P_matrix = np.random.randint(min_val, max_val + 1, size=(k, num_parity_cols)).astype(float)
    return P_matrix

def construct_g_matrix(k, n, P_matrix):
    """Constructs the full Generator Matrix G = [I_k | P] (Systematic form assumption)."""
    if P_matrix.shape != (k, n - k):
        # This check should ideally not fail if generation is correct
        raise ValueError(f"P matrix shape {P_matrix.shape} is inconsistent with k={k} and n={n}.")
        
    I_k = np.identity(k, dtype=float)
    G_matrix = np.hstack((I_k, P_matrix))
    return G_matrix


# --- CORE LP ALGORITHM FUNCTIONS ---
# (Reused from previous steps)

def get_all_tuples(n, m):
    """Generates all valid (a, b, X, eps_bits) tuples for the m-Height LP search."""
    all_indices = set(range(n))
    
    # 1. Iterate over all ordered pairs (a, b)
    for a, b in permutations(all_indices, 2):
        remaining_indices = all_indices - {a, b}
        
        # 2. Iterate over all subsets X of size m-1 from the remaining indices
        for X_tuple in combinations(remaining_indices, m - 1):
            X = set(X_tuple)
            X_sorted = sorted(list(X))
            Y = sorted(list(remaining_indices - X))
            
            # 3. Iterate over all 2^m sign combinations (eps_bits)
            for signs in product([-1, 1], repeat=m):
                
                eps_bits = [0.0] * (m + 1)
                s0 = signs[0] # sign for a
                eps_bits[0] = s0
                
                # s_j for j in X (m-1 signs)
                for idx in range(m - 1):
                    eps_bits[idx + 1] = signs[idx + 1]
                
                yield (a, b, X, eps_bits, X_sorted, Y, s0)


def compute_height(G, m, case_id=None):
    """Core function to compute h_m(C) for a single G matrix."""
    k, n = G.shape
    best_value = 0.0

    for a, b, X, eps_bits, X_sorted, Y, s0 in get_all_tuples(n, m):
        
        # Objective: max c^T u == minimize (-c)^T u
        c = np.array([s0 * G[i, a] for i in range(k)], dtype=float)

        A_ub = []
        b_ub = []

        # Constraints for j in X
        for idx, j in enumerate(X_sorted):
            s_j = eps_bits[idx + 1]
            A_ub.append([(s_j * G[i, j] - s0 * G[i, a]) for i in range(k)])
            b_ub.append(0.0)
            A_ub.append([(-s_j * G[i, j]) for i in range(k)])
            b_ub.append(-1.0)

        # Equality: sum_i g_{i,b} u_i = 1
        A_eq = np.array([[G[i, b] for i in range(k)]], dtype=float)
        b_eq = np.array([1.0], dtype=float)

        # For j in Y: |sum_i g_{i,j} u_i| <= 1
        for j in Y:
            A_ub.append([G[i, j] for i in range(k)])
            b_ub.append(1.0)
            A_ub.append([-G[i, j] for i in range(k)])
            b_ub.append(1.0)
            
        A_ub_array = np.array(A_ub, dtype=float) if A_ub else None
        b_ub_array = np.array(b_ub, dtype=float) if b_ub else None

        # Solve LP: maximize c^T u == minimize (-c)^T u
        bounds = [(None, None)] * k
        
        try:
            res = linprog(
                c=-c,
                A_ub=A_ub_array,
                b_ub=b_ub_array,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={'presolve': True, 'tol': 1e-6, 'disp': False} 
            )
        except ValueError:
            continue 

        if res.success and res.fun is not None:
            value = -res.fun
            
            if value > 1e-6:
                if value > best_value:
                    best_value = value
                    
    return (case_id, float(best_value))


def solve_all_cases_parallel(datasets_batch, max_workers):
    """
    Runs compute_height for a batch of datasets in parallel using a process pool.
    """
    
    tasks = []
    # datasets_batch format: (n, k, m, P_matrix)
    for i, (n, k, m, P_matrix) in enumerate(datasets_batch):
        G_matrix = construct_g_matrix(k, n, P_matrix)
        # Pass G_matrix, m, and a unique case ID (1 to N_SAMPLES_PER_SET)
        tasks.append((G_matrix, m, i + 1)) 

    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {executor.submit(compute_height, G, m, case_id): case_id for G, m, case_id in tasks}
        
        for future in as_completed(future_to_case):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                case_id = future_to_case[future]
                print(f'Case {case_id} generated an exception: {exc}', file=sys.stderr)

    results.sort(key=lambda x: x[0])
    return results


def save_pickle(data, filename):
    """Saves data to a pickle file with error handling."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"-> Successfully saved {filename} with {len(data)} entries.")
    except Exception as e:
        print(f"Error saving {filename}: {e}", file=sys.stderr)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    np.random.seed(301) # Ensure reproducibility
    
    # Global lists to hold the entire dataset
    X_full_dataset = [] # Input: [n, k, m, P_matrix]
    Y_full_dataset = [] # Output: m_height
    
    print(f"\n{'='*70}")
    print(f"Starting Full Dataset Generation ({TOTAL_EXPECTED_SAMPLES} total samples)")
    print(f"Parameter Sets to process: {len(PARAMETER_SETS)}")
    print(f"Using {NUM_CORES} CPU cores for parallel solving.")
    print(f"{'='*70}\n")
    
    generation_start_time = time.time()
    
    total_successful_samples = 0
    total_samples_processed = 0 # New tracker for overall progress
    
    for set_idx, (N, K, M) in enumerate(PARAMETER_SETS):
        
        start_time_set = time.time()
        print(f"[{set_idx + 1}/{len(PARAMETER_SETS)}] Processing set (N={N}, K={K}, M={M}) with {N_SAMPLES_PER_SET} samples...")
        
        # 1. Generate the batch of random P matrices for the current set
        datasets_batch = []
        for i in range(N_SAMPLES_PER_SET):
            P_matrix = generate_random_p_matrix(K, N, COEFF_MIN, COEFF_MAX)
            # Store the tuple: (n, k, m, P_matrix)
            datasets_batch.append((N, K, M, P_matrix))
            
        # 2. Execute solving in parallel for this batch
        final_results = solve_all_cases_parallel(datasets_batch, NUM_CORES)
        
        # 3. Process and filter results
        successful_batch_count = 0
        
        for case_id, m_height in final_results:
            # Retrieve the input parameters for the current sample
            n, k, m, P_matrix = datasets_batch[case_id - 1] 
            
            if m_height > HEIGHT_TOLERANCE:
                # Append to global dataset if non-zero
                X_full_dataset.append([n, k, m, P_matrix]) 
                Y_full_dataset.append(m_height)
                successful_batch_count += 1

        total_successful_samples += successful_batch_count
        
        # --- PROGRESS TRACKING ---
        total_samples_processed += N_SAMPLES_PER_SET
        progress_percent = (total_samples_processed / TOTAL_EXPECTED_SAMPLES) * 100
        # --- END PROGRESS TRACKING ---
        
        end_time_set = time.time()
        
        print(f"\n\n\n\n\n  -> Set (N={N}, K={K}, M={M}) completed in {end_time_set - start_time_set:.2f}s. Successful: {successful_batch_count}/{N_SAMPLES_PER_SET}")
        print(f"  -> GLOBAL PROGRESS: {total_samples_processed}/{TOTAL_EXPECTED_SAMPLES} samples processed ({progress_percent:.2f}%)\n\n\n\n\n")


    # 4. Final Summary and Saving
    
    print(f"\n{'='*70}")
    print("--- FULL DATASET GENERATION COMPLETE ---")
    
    # Final check of data integrity
    if len(X_full_dataset) != len(Y_full_dataset):
          print(f"WARNING: X ({len(X_full_dataset)}) and Y ({len(Y_full_dataset)}) lengths mismatch!")
          
    print(f"Total successful non-zero samples aggregated: {total_successful_samples}")
    print(f"Total expected samples: {TOTAL_EXPECTED_SAMPLES}")
    print(f"Zero/Failed calculations filtered: {TOTAL_EXPECTED_SAMPLES - total_successful_samples}\n")
    
    OUTPUT_X_FILE = 'x_full_dataset_6k_samples.pkl'
    OUTPUT_Y_FILE = 'y_full_dataset_6k_samples.pkl'
    
    save_pickle(X_full_dataset, OUTPUT_X_FILE)
    save_pickle(Y_full_dataset, OUTPUT_Y_FILE)
    
    generation_end_time = time.time()
    
    print(f"\nTOTAL EXECUTION TIME FOR ALL SETS: {generation_end_time - generation_start_time:.2f} seconds.")
    print(f"{'='*70}")