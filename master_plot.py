import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_DIR = "logs"

# All benchmark sources with (label, csv_path)
csv_sources = {
    "Baseline":             os.path.join(LOG_DIR, "baseline", "baseline_benchmark.csv"),
    "Cython":               os.path.join(LOG_DIR, "cython_version", "cython_benchmark.csv"),
    "Multiprocessing":      os.path.join(LOG_DIR, "multiprocessing_version", "multiprocessing_benchmark.csv"),
    "Numba":                os.path.join(LOG_DIR, "numba_version", "numba_benchmark.csv"),
    "Numba (prange)":       os.path.join(LOG_DIR, "numba_version_prange", "numba_benchmark.csv"),
    "NumPy (vectorised)":   os.path.join(LOG_DIR, "sum", "sum_benchmark.csv"),
}

# CuPy data from notebook run (total time inc. PCIe transfer)
cupy_I = {
    "Dimension": ["I"] * 5,
    "Value":     [10000, 25000, 50000, 75000, 100000],
    "Avg_Time":  [0.1228, 0.1714, 0.2534, 0.3325, 0.4197],
    "Std_Dev":   [0.0] * 5,
}
cupy_M = {
    "Dimension": ["M"] * 5,
    "Value":     [100, 500, 1000, 2500, 5000],
    "Avg_Time":  [0.0285, 0.1236, 0.2531, 0.6262, 1.2888],
    "Std_Dev":   [0.0] * 5,
}
cupy_df = pd.concat([pd.DataFrame(cupy_I), pd.DataFrame(cupy_M)], ignore_index=True)

# Load all data
data = {}
for label, path in csv_sources.items():
    data[label] = pd.read_csv(path)
data["CuPy (GPU)"] = cupy_df

# Colours and markers for each version
styles = {
    "Baseline":           {"color": "#1f77b4", "marker": "o"},
    "Cython":             {"color": "#ff7f0e", "marker": "s"},
    "Multiprocessing":    {"color": "#2ca02c", "marker": "^"},
    "Numba":              {"color": "#d62728", "marker": "D"},
    "Numba (prange)":     {"color": "#9467bd", "marker": "v"},
    "NumPy (vectorised)": {"color": "#8c564b", "marker": "P"},
    "CuPy (GPU)":         {"color": "#e377c2", "marker": "*"},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for label, df in data.items():
    s = styles[label]

    # Path scaling (I)
    df_i = df[df["Dimension"] == "I"].sort_values("Value")
    ax1.plot(df_i["Value"], df_i["Avg_Time"],
             marker=s["marker"], color=s["color"], label=label, linewidth=2, markersize=7)

    # Step scaling (M)
    df_m = df[df["Dimension"] == "M"].sort_values("Value")
    ax2.plot(df_m["Value"], df_m["Avg_Time"],
             marker=s["marker"], color=s["color"], label=label, linewidth=2, markersize=7)

# Formatting
ax1.set_title("Path Scaling  (fixed M = 1 000)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Number of Paths (I)", fontsize=12)
ax1.set_ylabel("Avg Time (s)", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=10)

ax2.set_title("Step Scaling  (fixed I = 50 000)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Number of Time Steps (M)", fontsize=12)
ax2.set_ylabel("Avg Time (s)", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=10)

fig.suptitle("Monte Carlo Option Pricing — All Implementations", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "master_benchmark.png"), dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved to {os.path.join(LOG_DIR, 'master_benchmark.png')}")
