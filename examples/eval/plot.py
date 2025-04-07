import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Microsoft YaHei", "SimHei"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelpad": 12,
    }
)

models = [
    "Qwen2.5-7B-Instruct",
    "step_256",
    "step_512",
    "step_768",
    "step_1024",
    "step_1536",
    "step_2048",
    "step_2560",
    "step_3072",
    "step_3584",
    "step_4096",
    "step_4608",
    "step_5120",
]

HUMAN_EVAL = [
    "82.3/76.8",
    "83.5/77.4",
    "81.1/75.0",
    "86.0/81.1",
    "85.4/78.7",
    "83.5/78.0",
    "81.1/78.4",
    "84.1/78.0",
    "82.9/76.0",
    "80.5/74.4",
    "84.8/75.0",
    "82.9/76.8",
    "85.4/77.4",
]
MBPP = [
    "80.7/68.5",
    "81.2/69.0",
    "81.2/68.8",
    "82.8/69.8",
    "84.4/71.7",
    "83.1/70.6",
    "81.2/69.3",
    "83.9/70.6",
    "84.1/70.9",
    "83.1/70.9",
    "86.2/71.7",
    "84.9/70.9",
    "85.2/70.9",
]
LiveCodeBench = [
    "14.6[46.0/9.3/1.7]",
    "15.7[49.2/10.5/1.7]",
    "19.0[54.0/14.0/4.2]",
    "19.0[52.4/14.0/5.0]",
    "18.6[52.4/14.0/4.2]",
    "19.8[55.6/15.1/4.2]",
    "19.4[50.8/17.4/4.2]",
    "19.4[57.1/11.6/5.0]",
    "18.6[55.6/10.5/5.0]",
    "18.3[55.6/10.5/4.2]",
    "20.1[57.1/14.0/5.0]",
    "18.6[58.7/10.5/3.4]",
    "19.0[57.1/12.8/3.4]"
]
humaneval = [float(x.split("/")[0]) for x in HUMAN_EVAL]
humanevalplus = [float(x.split("/")[1]) for x in HUMAN_EVAL]
mbpp = [float(x.split("/")[0]) for x in MBPP]
mbppplus = [float(x.split("/")[1]) for x in MBPP]
livecodebench = [float(x.split("[")[0]) for x in LiveCodeBench]
baseline = {
    # 'Humaneval': 82.3,
    "Humaneval+": 76.8,
    # 'MBPP': 80.7,
    "MBPP+": 68.5,
    "LiveCodeBench": 14.6,
}

fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

colors = sns.color_palette("husl", 5)
line_styles = ["-", "--", "-.", ":", "-"]
markers = ["o", "s", "D", "^", "v"]
data_lst = [humanevalplus, mbppplus, livecodebench]
label_lst = ["Humaneval+", "MBPP+", "LiveCodeBench"]

for i, (data, label) in enumerate(zip(data_lst, label_lst)):
    ax.plot(
        models,
        data,
        color=colors[i],
        linestyle=line_styles[i],
        marker=markers[i],
        markersize=8,
        markeredgecolor="white",
        linewidth=2.5,
        label=label,
    )

for i, (key, value) in enumerate(baseline.items()):
    ax.axhline(
        y=value,
        color=colors[i],
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label=f"{key} Baseline",
    )

ax.set_title("Model Performance Evolution with Baseline Comparison", pad=20)
ax.set_xlabel("Model Versions", labelpad=15)
ax.set_ylabel("Score (%)", labelpad=15)

ax.set_ylim(0, 90)
ax.set_yticks(np.arange(0, 91, 5))
plt.xticks(rotation=45, ha="right")

handles, labels = ax.get_legend_handles_labels()
# order = [0,5,1,6,2,7,3,8,4,9]
order = [0, 1, 2, 3, 4, 5]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    title="Metrics",
    borderpad=1,
)

ax.grid(axis="y", alpha=0.4, linestyle="--")
ax.grid(axis="x", visible=False)

plt.subplots_adjust(right=0.78, bottom=0.15)

plt.savefig("figs/enhanced_performance_plot.pdf", bbox_inches="tight")
plt.show()
