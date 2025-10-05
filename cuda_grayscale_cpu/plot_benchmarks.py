import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark.csv")

plt.figure(figsize=(7,5))
plt.plot(df["Width"], df["CPU_Time(ms)"], marker='o', label="CPU Grayscale")
plt.title("CPU Grayscale Performance")
plt.xlabel("Image Width (pixels)")
plt.ylabel("Execution Time (ms)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cpu_grayscale_performance.png", dpi=300)
plt.show()
