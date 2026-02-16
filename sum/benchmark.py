import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from main import mc_price_option

# --- Setup Output Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../logs/sum")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Experiment Settings ---
SCENARIOS = [10_000, 50_000, 100_000]
M_STEPS = 1000  
REPEATS = 5     

# Fixed Constants
S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.12, 0.5

results = []

print(f"{'Paths (I)':<12} | {'Avg Time (s)':<15} | {'Std Dev (s)':<15}")
print("-" * 50)

for I in SCENARIOS:
    times = []
    
    for _ in range(REPEATS):
        start_time = time.perf_counter()
        _ = mc_price_option(S0, K, r, sigma, T, M_STEPS, I)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    results.append({
        "Paths": I,
        "Avg_Time": avg_time,
        "Std_Dev": std_time,
        "Min_Time": np.min(times),
        "Max_Time": np.max(times)
    })
    
    print(f"{I:<12} | {avg_time:<15.4f} | {std_time:<15.4f}")

# --- Save Data ---
csv_path = os.path.join(RESULTS_DIR, "sum_benchmark.csv")
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"\nData saved to '{csv_path}'")

# --- Generate Performance Plot ---
plt.figure(figsize=(10, 6))
plt.errorbar(df["Paths"], df["Avg_Time"], yerr=df["Std_Dev"], 
             fmt='-o', capsize=5, label='Summation (Pure NumPy)', color='orange')

plt.title("Sum-Optimized Performance: Execution Time vs Input Size")
plt.xlabel("Number of Paths (I)")
plt.ylabel("Execution Time (seconds)")
plt.grid(True, which="both", linestyle='--')
plt.legend()
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "sum_performance.png")
plt.savefig(plot_path)
print(f"Plot saved to '{plot_path}'")