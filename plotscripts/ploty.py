import re
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Load and Parse Log File ===
with open("log.txt", "r") as file:
    log_data = file.read()

eval_pattern = re.compile(
    r"Eval on experience (\d+) \(Task (\d+)\) from test stream ended.\n(.*?)-- >> End of eval phase << --",
    re.DOTALL,
)
eval_matches = eval_pattern.findall(log_data)

metrics_keys = [
    "AP", "AP50", "AP75", "APl", "APm", "APs",
    "AR@1", "AR@10", "AR@100", "ARl@100", "ARm@100", "ARs@100"
]

metrics_data = []
for exp_id, task_id, block in eval_matches:
    metrics = {"experience": int(exp_id), "task": int(task_id)}
    for key in metrics_keys:
        match = re.search(fr"{key} = ([0-9.]+)", block)
        metrics[key] = float(match.group(1)) if match else None
    metrics_data.append(metrics)

df = pd.DataFrame(metrics_data).sort_values(by="experience").reset_index(drop=True)
df.to_csv("metrics_over_time.csv", index=False)

# === 2. Plot & Save Detection Metric Graphs ===

# (a) Average Precision (AP)
plt.figure()
plt.plot(df["experience"], df["AP"], label="AP (IoU=0.5:0.95)")
plt.plot(df["experience"], df["AP50"], label="AP50")
plt.plot(df["experience"], df["AP75"], label="AP75")
plt.title("Average Precision Over Time")
plt.xlabel("Experience")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig("ap_trend.png")

# (b) AP by object size
plt.figure()
plt.plot(df["experience"], df["APl"], label="AP (Large)")
plt.plot(df["experience"], df["APm"], label="AP (Medium)")
plt.plot(df["experience"], df["APs"], label="AP (Small)")
plt.title("AP by Object Size")
plt.xlabel("Experience")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig("ap_by_size.png")

# (c) Average Recall (AR)
plt.figure()
plt.plot(df["experience"], df["AR@1"], label="AR@1")
plt.plot(df["experience"], df["AR@10"], label="AR@10")
plt.plot(df["experience"], df["AR@100"], label="AR@100")
plt.title("Average Recall Over Time")
plt.xlabel("Experience")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig("ar_trend.png")

# (d) AR by object size
plt.figure()
plt.plot(df["experience"], df["ARl@100"], label="AR (Large)")
plt.plot(df["experience"], df["ARm@100"], label="AR (Medium)")
plt.plot(df["experience"], df["ARs@100"], label="AR (Small)")
plt.title("AR by Object Size")
plt.xlabel("Experience")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig("ar_by_size.png")

# === 3. Compute Continual Learning Metrics ===

average_accuracy = df["AP"].mean()
bwt = df["AP"].iloc[-1] - df["AP"].iloc[0]
fwt = df["AP"].iloc[1] - df["AP"].iloc[0]

cl_metrics = {
    "Average Accuracy (AA)": average_accuracy,
    "Backward Transfer (BWT)": bwt,
    "Forward Transfer (FWT)": fwt
}

# Save metrics to text file
with open("cl_metrics.txt", "w") as f:
    for k, v in cl_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# === 4. Plot & Save CL Metric Bar Chart ===

plt.figure(figsize=(6, 4))
plt.bar(cl_metrics.keys(), cl_metrics.values(), color=["skyblue", "lightgreen", "salmon"])
plt.title("Continual Learning Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
for i, (k, v) in enumerate(cl_metrics.items()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("cl_metrics.png")
