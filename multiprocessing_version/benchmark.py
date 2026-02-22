import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from main import mc_price_option_parallel

# --- Setup Output Paths ---
FOLDER_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(FOLDER_DIR) 
RESULTS_DIR = os.path.join(PROJECT_ROOT, "logs", "multiprocessing")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Constants ---
S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.12, 0.5
REPEATS = 5
WORKERS = 8 

I_SCENARIOS = [10_000, 25_000, 50_000, 75_000, 100_000]
FIXED_M = 1000
M_SCENARIOS = [100, 500, 1000, 2500, 5000]
FIXED_I = 50_000

def run_experiment(variable_name, scenarios, m_val, i_val, pool):
    print(f"\n>> Benchmarking Parallel {variable_name} Scaling ({WORKERS} workers)...")
    print(f"{variable_name:<12} | {'Avg Time (s)':<15} | {'Std Dev (s)':<15}")
    print("-" * 50)
    
    exp_results = []
    for val in scenarios:
        current_m = val if variable_name == "M" else m_val
        current_i = val if variable_name == "I" else i_val
        
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            _ = mc_price_option_parallel(S0, K, r, sigma, T, current_m, current_i, pool)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        exp_results.append({
            "Dimension": variable_name, "Value": val,
            "Avg_Time": avg_time, "Std_Dev": np.std(times)
        })
        print(f"{val:<12} | {avg_time:<15.4f} | {np.std(times):<15.4f}")
    return exp_results

if __name__ == "__main__":
    with mp.Pool(processes=WORKERS) as shared_pool:
        res_i = run_experiment("I", I_SCENARIOS, FIXED_M, None, shared_pool)
        res_m = run_experiment("M", M_SCENARIOS, None, FIXED_I, shared_pool)

    df = pd.DataFrame(res_i + res_m)
    csv_path = os.path.join(RESULTS_DIR, "multiprocessing_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot I-Scaling
    df_i = df[df["Dimension"] == "I"]
    ax1.errorbar(df_i["Value"], df_i["Avg_Time"], yerr=df_i["Std_Dev"], fmt='-o', color='blue', capsize=5)
    ax1.set_title(f"Parallel I-Scaling (Fixed M={FIXED_M}, {WORKERS} Workers)")
    ax1.set_xlabel("Number of Paths (I)")
    ax1.set_ylabel("Time (s)")
    ax1.grid(True, linestyle='--')

    # Plot M-Scaling
    df_m = df[df["Dimension"] == "M"]
    ax2.errorbar(df_m["Value"], df_m["Avg_Time"], yerr=df_m["Std_Dev"], fmt='-s', color='red', capsize=5)
    ax2.set_title(f"Parallel M-Scaling (Fixed I={FIXED_I}, {WORKERS} Workers)")
    ax2.set_xlabel("Number of Steps (M)")
    ax2.set_ylabel("Time (s)")
    ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "multiprocessing_performance.png")
    plt.savefig(plot_path)
    print(f"Parallel scaling plot saved to '{plot_path}'")