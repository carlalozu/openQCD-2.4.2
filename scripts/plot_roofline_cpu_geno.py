import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Create the log-log roofline plot
plt.figure(figsize=(5, 3))
plt.style.use("seaborn-v0_8-whitegrid")

# Parameters
aos_I = 0.7397 #flops/byte
aos_P = 432 #flops

input_file = "time_threads.csv"
plot_file = "time_threads.pdf"

threads = [1,2,4,8,16]

perf_1core = 12*2 # in GFlops/s, AVX on
memb_1core = 30 # in GB/s
socket_bw = 460.8 # in GB/s

# CPU architecture
peak_performances_cpu = [perf_1core*t for t in threads]  # in GFlops/s
memory_bandwidths_cpu = [memb_1core*t if memb_1core*t<socket_bw else socket_bw for t in threads]  # in GB/s
markers = Line2D.filled_markers

for i, t in enumerate(threads):
    x = np.linspace(0.001,  2**10, 100000)
    y = np.minimum(x * memory_bandwidths_cpu[i], peak_performances_cpu[i])
    line, = plt.plot(x, y, alpha=0.7)

    #  add label
    ridge_x = peak_performances_cpu[i] / memory_bandwidths_cpu[i]
    label_x = 90
    label_y = peak_performances_cpu[i]*0.5
    plt.text(label_x, label_y, f"CPU {t} cores", color=line.get_color(), fontsize=10, ha="left", va="bottom")


# add kenrel lines su3matmat
plt.vlines(aos_I, 0.001, 1e5, linestyles='dashed', colors="black", label="plaq_sum", alpha=0.7, zorder=-1)

df_soa = pd.read_csv(input_file)
df_soa["op_int"]= aos_I
for i, t in enumerate(threads):
    aost = df_soa[df_soa["threads"] == t]
    plt.scatter(aost["op_int"]+0.005*t, aost["total_mflops"]*1e-3, marker=markers[-i], zorder=4)
    # print(aost.head(10))

for x, y, v in zip(aost["op_int"], aost["total_mflops"]*1e-3, aost["volume"]):
    plt.text(x+0.5,y,str(v),fontsize=9,color="tab:brown",ha="left",va="center")

# Add labels and legend
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOPs/s)')
plt.ylim([1e-1, 1e4])
plt.xlim([1e-2, 1e3])
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=9)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file)

