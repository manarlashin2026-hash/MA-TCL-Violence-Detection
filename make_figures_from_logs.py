"""Fig. 6 (training dynamics) and Fig. 10 (RWF-2000 comparison) from program outputs."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                            "mathtext.fontset": "stix", "font.size": 11})
hist_path, boot_tcl, boot_notcl, out = sys.argv[1:5]
h = json.load(open(hist_path))
ep = np.arange(1, len(h["train_loss"]) + 1)
best = h["best_epoch"] + 1
BLUE, RED, GREEN, PURPLE, INK = "#2E6DA4", "#C0392B", "#1A9E7A", "#7B5EA7", "#333333"

fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
def deco(a, ylabel, tag, title):
    a.set_xlabel("Epoch"); a.set_ylabel(ylabel); a.grid(alpha=0.25)
    a.set_title(f"({tag}) {title}", fontsize=12)
    a.axvline(best, color="#888888", ls=":", lw=1.2)
    a.set_xticks(ep[::2] if len(ep) > 12 else ep); a.set_xlim(0.5, len(ep) + 0.5)
a = ax[0, 0]; a.plot(ep, h["train_loss"], "-o", color=BLUE, lw=2, ms=4)
deco(a, "Training loss (weighted CE)", "a", "Training loss")
a = ax[0, 1]
a.plot(ep, np.array(h["train_acc"]) * 100, "-o", color=BLUE, lw=2, ms=4, label="Training")
a.plot(ep, np.array(h["val_acc"]) * 100, "-s", color=RED, lw=2, ms=4, label="Test (official partition)")
deco(a, "Accuracy (%)", "b", "Training and test accuracy"); a.legend(loc="lower right", fontsize=10)
a = ax[0, 2]; a.semilogy(ep, h["learning_rate"], "-o", color=PURPLE, lw=2, ms=4)
deco(a, "Learning rate (log scale)", "c", "Learning-rate schedule")
a = ax[1, 0]; a.plot(ep, h["val_f1"], "-o", color=GREEN, lw=2, ms=4)
deco(a, "Macro-F1", "d", "Test macro-F1")
a = ax[1, 1]
a.plot(ep, np.array(h["val_precision"]) * 100, "-o", color=BLUE, lw=2, ms=4, label="Macro precision")
a.plot(ep, np.array(h["val_recall"]) * 100, "-s", color=RED, lw=2, ms=4, label="Macro recall")
deco(a, "Score (%)", "e", "Test precision and recall"); a.legend(loc="lower right", fontsize=10)
a = ax[1, 2]
gap = (np.array(h["train_acc"]) - np.array(h["val_acc"])) * 100
a.plot(ep, gap, "-o", color=INK, lw=2, ms=4); a.axhline(0, color="#888888", lw=1)
deco(a, "Training minus test accuracy (pp)", "f", "Generalization gap")
fig.tight_layout()
fig.savefig(os.path.join(out, "training_curves_rwf2000_official.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "training_curves_rwf2000_official.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig. 10 ----------------
bt = json.load(open(boot_tcl)); bn = json.load(open(boot_notcl))
acc_tcl = 100 * np.mean(np.array(h["val_acc"][-1]))              # last-epoch accuracy of the reported run
labels = ["ConvLSTM*", "C3D*", "I3D (RGB)*", "Flow Gated\nNetwork*", "VideoMAE\n(TCL disabled)$^\\dagger$", "MA-TCL\n(proposed)"]
vals = [77.00, 82.75, 85.75, 87.25, None, None]
hn = json.load(open(os.path.join(os.path.dirname(boot_notcl), "training_history.json")))
vals[4] = 100 * hn["val_acc"][-1]; vals[5] = acc_tcl
ci = {4: (100 * bn["accuracy"]["ci_lower"], 100 * bn["accuracy"]["ci_upper"]),
      5: (100 * bt["accuracy"]["ci_lower"], 100 * bt["accuracy"]["ci_upper"])}
colors = ["#B8BEC6"] * 4 + ["#7F9DBE", RED]
fig, a = plt.subplots(figsize=(10, 5.6))
x = np.arange(len(vals))
bars = a.bar(x, vals, width=0.62, color=colors, edgecolor="white", linewidth=2)
for i, (lo, hi) in ci.items():
    a.errorbar(i, vals[i], yerr=[[vals[i] - lo], [hi - vals[i]]], fmt="none", ecolor=INK, capsize=6, lw=1.4)
for i, v in enumerate(vals):
    ytxt = (ci[i][1] if i in ci else v) + 0.6
    a.text(i, ytxt, f"{v:.2f}%", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold" if i == 5 else None)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=10.5)
a.set_ylabel("Test accuracy (%)"); a.set_ylim(70, 96); a.grid(axis="y", alpha=0.25)
a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
fig.text(0.01, -0.02, "* Reported by Cheng et al. (ICPR 2020) on the official partition.   "
         "$^\\dagger$ Trained in this work with the identical recipe and TCL input set to zero.   "
         "Error bars: 95% bootstrap CI.", fontsize=9, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(out, "benchmark_comparison_rwf2000.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "benchmark_comparison_rwf2000.png"), dpi=300, bbox_inches="tight")
print("acc_tcl", acc_tcl, "acc_notcl", vals[4], "best_epoch", best)
