import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# Plot parameters
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")
markers = Line2D.filled_markers

# Kernel parameters, plaqsum
aos_I = 0.7397  #flops/byte
aos_P = 432     #flops

computer = sys.argv[1] if len(sys.argv) > 1 else "geno"


input_file_gpu = f"../output/time_gpu_{computer}.csv"
input_file_cpu = f"../output/time_threads_{computer}.csv"
plot_file = f"../output/time_gpu_{computer}.pdf"

if computer=="geno":
    threads = [1,2,4,8,16]
    perf_1core = 12*2   # in GFlops/s, AVX on
    memb_1core = 30     # in GB/s
    socket_bw = 460.8   # in GB/s

    labels_gpu = ["FP64 A2000", "FP32 A2000"]
    peak_performances_gpu = [124.8, 7987.2]   # in GFlops/s
    memory_bandwidths_gpu = [288, 288]        # in GB/s
elif computer=="daint":
    threads = [1,4,8,16,32,64]
    perf_1core = 24.8*2 # in GFlops/s, AVX on
    memb_1core = 28     # in GB/s
    socket_bw = 480     # in GB/s

    labels_gpu = ["FP64 H200", "FP32 H200"]
    peak_performances_gpu = [34000, 67000]  # in GFlops/s
    memory_bandwidths_gpu = [4000, 4000]  # in GB/s

peak_performances_cpu = [perf_1core*t for t in threads]
memory_bandwidths_cpu = [memb_1core*t if memb_1core*t<socket_bw else socket_bw for t in threads]


# add kernel lines plaq_sum
plt.vlines(aos_I, 0.001, 1e5, linestyles='dashed', color="black", label="plaq_sum_dble", alpha=0.7)

marker_count = 0
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
    plt.scatter(aost["op_int"], aost["total_mflops"]*1e-3, marker=markers[marker_count], label=f"{t} cores", zorder=10-marker_count)
    marker_count+=1


# Plot roofline GPU
df_gpu = pd.read_csv(input_file_gpu)
df_gpu["op_int"]= aos_I

colors_gpu = ["tab:pink", "tab:grey"]
for i in range(2):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_gpu[i], peak_performances_gpu[i])
    line, = plt.plot(x, y, alpha=0.9, color=colors_gpu[i])

    #  add label
    ridge_x = peak_performances_gpu[i] / memory_bandwidths_gpu[i]
    label_y = peak_performances_gpu[i] * 0.9
    plt.text(90, label_y, labels_gpu[i], color=line.get_color(), fontsize=9, ha="left", va="top")
plt.scatter(df_gpu["op_int"]+0.5, df_gpu["total_mflops"]*1e-3, color=colors_gpu[0], label=labels_gpu[0], marker=markers[marker_count], zorder=marker_count)


# Problem size label
for x, y, v in zip(df_gpu["op_int"], df_gpu["total_mflops"]*1e-3, df_gpu["volume"]):
    plt.text(x+0.8,y,str(v),fontsize=9,color=colors_gpu[0],ha="left",va="center")


# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e0, 1e5])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=9)
plt.tick_params(which='minor', length=2, color='grey')
plt.tick_params(which='major', length=3, color='grey')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file)
