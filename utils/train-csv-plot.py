from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import read_csv

df = read_csv("train.csv")

epochs = df["epoch"]

metrics = {
    "train_loss": (df["train_loss"], "Training Loss"),
    "valid_loss": (df["valid_loss"], "Validation Loss"),
    "cer": (df["cer"], "Character Error Rate"),
    "ier": (df["ier"], "Item Error Rate"),
    "train_leven": (df["train_leven"], "Training Levenshtein"),
    "val_leven": (df["val_leven"], "Validation Levenshtein")
}

output_summary: list[tuple[str, ...]] = [("", "Min", "Max", "Mean", "Median")]
output_col_max_len: tuple[int, ...] = tuple([0 for _ in output_summary[0]])

for metric_data in metrics.values():
    output_summary.append(tuple_ := tuple([metric_data[1]] + [str(getattr(metric_data[0], func)()) for func in ["min", "max", "mean", "median"]]))
    output_col_max_len = tuple([max(old, len(new)) for old, new in zip(output_col_max_len, tuple_)])

for data in output_summary:
    print("  ".join(f"{col:<{length}}" for col, length in zip(data, output_col_max_len)))

ax1: Axes
ax2: Axes
fig1, (ax1, ax2) = plt.subplots(2, 1)

for key in ["train_loss", "valid_loss"]:
    ax1.plot(epochs, metrics[key][0], label=metrics[key][1])

ax1.set_title("Epoch vs Training Loss and Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

for key in ["cer", "ier", "train_leven", "val_leven"]:
    ax2.plot(epochs, metrics[key][0], label=metrics[key][1])

ax2.set_title("Epoch vs Character Error Rate, Item Error Rate, Training and Validation Levenshtein")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Metrics")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
