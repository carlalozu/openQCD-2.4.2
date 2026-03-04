import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Create the log-log roofline plot
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")

# Kernel parameters, plaqsum
aos_I = 0.7397 #flops/byte
aos_P = 432 #flops

computer = sys.argv[1] if len(sys.argv) > 1 else "geno"

input_file = f"../output/time_threads_{computer}.csv"
plot_file = f"../output/time_threads_{computer}.pdf"

if computer=="geno":
    threads = [1,2,4,8,16]
    perf_1core = 12*2   # in GFlops/s, AVX on
    memb_1core = 30     # in GB/s
    socket_bw = 460.8   # in GB/s
elif computer=="daint":
    threads = [1,2,4,8,16,32,64]
    perf_1core = 24.8*2 # in GFlops/s, AVX on
    memb_1core = 28     # in GB/s
    socket_bw = 480     # in GB/s

# add kenrel lines su3matmat
plt.vlines(aos_I, 0.001, 1e5, linestyles='dashed', colors="black", label="plaq_sum_dble", alpha=0.7, zorder=-1)

# CPU architecture
peak_performances_cpu = [perf_1core*t for t in threads]  # in GFlops/s
memory_bandwidths_cpu = [memb_1core*t if memb_1core*t<socket_bw else socket_bw for t in threads]  # in GB/s
markers = Line2D.filled_markers

df_soa = pd.read_csv(input_file)
df_soa["op_int"]= aos_I
marker_count = 0

for i, t in enumerate(threads):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.7)

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_x = 90
    label_y = peak_performances_cpu[i]*0.5
    plt.text(label_x, label_y, f"CPU {t} cores", color=line.get_color(), fontsize=9, ha="left", va="bottom")

    aost = df_soa[df_soa["threads"] == t]
    plt.scatter(aost["op_int"], aost["total_mflops"]*1e-3, marker=markers[marker_count], label=f"{t} cores", zorder=10-marker_count)
    marker_count+=1


for x, y, v in zip(aost["op_int"], aost["total_mflops"]*1e-3, aost["volume"]):
    plt.text(x+0.5,y,str(v),fontsize=9,color=line.get_color(),ha="left",va="center")

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

