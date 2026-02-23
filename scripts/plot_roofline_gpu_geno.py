import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# Plot parameters
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")
markers = Line2D.filled_markers

# Kernel plaqsum
aos_I = 0.7397  #flops/byte
aos_P = 432     #flops

# Files
input_file_gpu = "../output/time_gpu.csv"
input_file_cpu = "../output/time_threads.csv"
plot_file = "../output/time_gpu.pdf"

# GENO details
threads = [1, 8, 16]
perf_1core = 12*2   # in GFlops/s, with AVX on
memb_1core = 30     # in GB/s
socket_bw = 460.8   # in GB/s

peak_performances_cpu = [perf_1core*t for t in threads]
memory_bandwidths_cpu = [memb_1core*t if memb_1core*t<socket_bw else socket_bw for t in threads]


# add kernel lines plaq_sum
plt.vlines(aos_I, 0.001, 1e5, linestyles='dashed', color="black", label="plaq_sum", alpha=0.7, zorder=-1)


# Plot cpu roofline and performance
df_soa = pd.read_csv(input_file_cpu)
df_soa["op_int"]= aos_I
for i, t in enumerate(threads):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.7)

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_y = peak_performances_cpu[i]*0.5
    plt.text(90, label_y, f"CPU {t} cores", color=line.get_color(), fontsize=9, ha="left", va="bottom")
    
    aost = df_soa[df_soa["threads"] == t]
    plt.scatter(aost["op_int"], aost["total_mflops"]*1e-3, marker=markers[-i], zorder=4, label=f"{t} cores")


# Plot roofline GPU
df_gpu = pd.read_csv(input_file_gpu)
df_gpu["op_int"]= aos_I

labels_gpu = ["FP64 A2000", "FP32 A2000"]
peak_performances_gpu = np.array([124.8, 7987.2])   # in GFlops/s
memory_bandwidths_gpu = np.array([288, 288])        # in GB/s
colors_gpu = ["tab:pink", "tab:brown"]
for i in range(2):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_gpu[i], peak_performances_gpu[i])
    line, = plt.plot(x, y, alpha=0.9, color=colors_gpu[i])

    #  add label
    ridge_x = peak_performances_gpu[i] / memory_bandwidths_gpu[i]
    label_y = peak_performances_gpu[i] * 0.9
    plt.text(90, label_y, labels_gpu[i], color=line.get_color(), fontsize=9, ha="left", va="top")
plt.scatter(df_gpu["op_int"]+0.5, df_gpu["total_mflops"]*1e-3, zorder=3, color=colors_gpu[0], label=labels_gpu[0])


# Problem size label
for x, y, v in zip(df_gpu["op_int"], df_gpu["total_mflops"]*1e-3, df_gpu["volume"]):
    plt.text(x+0.5,y,str(v),fontsize=9,color="tab:pink",ha="left",va="center")


# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e0, 1e5])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=9)


# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file)
