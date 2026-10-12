"""
Revision figures for the MA-TCL manuscript (IEEE Access resubmission).

Everything below is computed from the videos and the trained checkpoints;
nothing is typed in by hand. Run once from the repository root, after the three training scripts:

    python make_figures_from_videos.py

Outputs (folder ./revision_figures):
  Fig. 4  fig4a_scfight_tcl_distributions.(pdf|png)
          fig4b_rwf_tcl_distributions.(pdf|png)
          tcl_features_scfight.csv, tcl_features_rwf.csv, tcl_statistics.txt
  Fig. 9  fig9_tcl_trajectories.(pdf|png)      (p_t, m_t of real clips)
  Fig. 7  fig7_gradcam_scfight.(pdf|png)        (SC-Fight-300 model, validation clips)
  Fig. 8  fig8_gradcam_rwf.(pdf|png)            (RWF-2000 model, official test clips)
          predictions_scfight.csv, predictions_rwf.csv
  Check   checks.json  (re-computed accuracies vs. training logs, paired test
                        between the TCL and no-TCL RWF-2000 runs)

TCL features use exactly the extractor of the training scripts.
Grad-CAM uses exactly the method of Code_5_GradCAM.py
(last encoder block, layernorm_after; token-pooled gradients; temporal mean).
Clip selection rules are fixed in advance (see SELECTION RULES below).
"""

import os
import json
import csv
import gc
from multiprocessing import Pool

import cv2
import numpy as np

# =============================================================================
# PATHS
# =============================================================================
ROOT = "."
SC_FIGHT_DIR = os.path.join(ROOT, "data", "SC-Fight-300", "fight")
SC_NOFIGHT_DIR = os.path.join(ROOT, "data", "SC-Fight-300", "noFight")
RWF_BASE = os.path.join(ROOT, "data", "RWF-2000")

SC_MODEL = os.path.join(ROOT, "fight_model_ma_tcl_final", "best_ma_tcl_model.pth")
RWF_TCL_RUN = os.path.join(ROOT, "rwf2000_official_split_results")
RWF_NOTCL_RUN = os.path.join(ROOT, "rwf2000_official_split_results_noTCL")

OUT_DIR = "./revision_figures"
MODEL_CKPT = "MCG-NJU/videomae-base"
NUM_FRAMES = 16
TAU_P, TAU_M = 0.35, 0.15
N_PROCESSES = max(1, min(6, (os.cpu_count() or 2) - 1))
VIDEO_EXT = ('.avi', '.mp4', '.mov', '.mkv')

# SELECTION RULES (fixed before looking at any output)
#  Fig. 9 : for each dataset and class, the evaluation clip whose phi_r is the
#           class median (ties -> alphabetical file name).
#  Fig. 7/8: TP = correct violent clip with the highest P(violence);
#           TN = correct non-violent clip with the highest P(non-violence);
#           FP = non-violent clip predicted violent with the highest P(violence);
#           FN = violent clip predicted non-violent with the highest P(non-violence).
#           Shown frame: sampled frame 8 of 16 (as in the original figures).
SHOW_FRAME = 8


# =============================================================================
# TCL EXTRACTOR  (identical to the training scripts)
# =============================================================================
class MotionAwareTCLExtractor:
    @staticmethod
    def compute_frame_motion(frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        motion_simple = diff.mean() / 255.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_flow = min(1.0, magnitude.mean() / 10.0)
        except Exception:
            motion_flow = motion_simple
        return motion_simple * 0.4 + motion_flow * 0.6

    @staticmethod
    def detect_people_proximity_with_motion(frame, prev_frame=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]

        edges_fine = cv2.Canny(center_region, 30, 100)
        edges_coarse = cv2.Canny(center_region, 100, 200)
        edges = edges_fine * 0.6 + edges_coarse * 0.4
        edge_density = edges.sum() / (center_region.shape[0] * center_region.shape[1])

        texture_var = np.std(center_region)
        hist = cv2.calcHist([center_region], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10)) / 8.0

        proximity = min(1.0, max(0.0, edge_density * 0.008 + (texture_var / 255.0) * 0.4 + entropy * 0.3))
        motion = MotionAwareTCLExtractor.compute_frame_motion(prev_frame, frame) if prev_frame is not None else 0.0
        return proximity, motion

    @staticmethod
    def per_frame_scores(frames):
        prox, mot, prev = [], [], None
        for f in frames:
            p, m = MotionAwareTCLExtractor.detect_people_proximity_with_motion(f, prev)
            prox.append(p); mot.append(m); prev = f
        return np.array(prox), np.array(mot)

    @staticmethod
    def signature_from_scores(prox_arr, motion_arr, proximity_threshold=TAU_P, motion_threshold=TAU_M):
        n = len(prox_arr)
        high_prox = (prox_arr > proximity_threshold).astype(int)
        max_streak = current_streak = 0
        streaks = []
        for val in high_prox:
            if val == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)

        duration = max_streak / n if n else 0.0
        consistency = min(1.0, 1.0 / (prox_arr.std() + 0.1) / 10.0)
        avg_prox = prox_arr.mean()
        event_score = 1.0 / (len(streaks) if streaks else 1)
        high_prox_idx = np.where(prox_arr > proximity_threshold)[0]
        motion_contact = motion_arr[high_prox_idx].mean() if len(high_prox_idx) > 0 else 0.0
        motion_consist = min(1.0, 1.0 / (motion_arr.std() + 0.1) / 5.0)
        peak_motion = motion_arr.max()
        risk_score = ((prox_arr > proximity_threshold) & (motion_arr > motion_threshold)).sum() / n

        num_bins = 8
        bin_size = n // num_bins
        binned = []
        for i in range(num_bins):
            start = i * bin_size
            end = start + bin_size if i < num_bins - 1 else n
            binned.append(prox_arr[start:end].mean() * 0.5 + motion_arr[start:end].mean() * 0.5 if end > start else 0)
        return np.array([duration, consistency, avg_prox, event_score,
                         motion_contact, motion_consist, peak_motion, risk_score] + binned, dtype=np.float32)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                                "mathtext.fontset": "stix", "savefig.dpi": 300})


FEATURE_NAMES = ["phi_d", "phi_c", "phi_a", "phi_e", "phi_mc", "phi_mv", "phi_max", "phi_r"] + \
                [f"bin{k+1}" for k in range(8)]


def load_video_frames(path, num_frames=NUM_FRAMES):
    """Same sampling as the training scripts (uniform, sparse seek)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    frames = []
    for i in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1])
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    return frames


def tcl_worker(path):
    try:
        frames = load_video_frames(path)
        if frames is None:
            return path, None, None, None
        p, m = MotionAwareTCLExtractor.per_frame_scores(frames)
        sig = MotionAwareTCLExtractor.signature_from_scores(p, m)
        return path, sig.tolist(), p.tolist(), m.tolist()
    except Exception as e:
        print(f"  [warn] {path}: {e}")
        return path, None, None, None


# =============================================================================
# DATA LISTS (same ordering / split rules as the training scripts)
# =============================================================================
def list_dir_videos(d):
    return [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(VIDEO_EXT)]


def scfight_lists():
    from sklearn.model_selection import train_test_split
    files, labels = [], []
    for p in list_dir_videos(SC_FIGHT_DIR):
        files.append(p); labels.append(1)
    for p in list_dir_videos(SC_NOFIGHT_DIR):
        files.append(p); labels.append(0)
    tr_p, va_p, tr_l, va_l = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)
    return files, labels, set(va_p)


def rwf_lists():
    out = {}
    for split in ("train", "val"):
        paths, labels = [], []
        for folder, lab in (("Fight", 1), ("NonFight", 0)):
            d = os.path.join(RWF_BASE, split, folder)
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(VIDEO_EXT):
                    continue
                p = os.path.join(d, f)
                cap = cv2.VideoCapture(p)
                ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 10
                cap.release()
                if ok:
                    paths.append(p); labels.append(lab)
        out[split] = (paths, labels)
    return out


# =============================================================================
# STEP 1 : TCL FEATURES FOR ALL CLIPS
# =============================================================================
def compute_tcl_table(paths, labels, csv_path, eval_set):
    cache = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cache[row["path"]] = row
    todo = [p for p in paths if p not in cache]
    def save_partial():
        keys0 = ["path"] + FEATURE_NAMES + ["p_t", "m_t"]
        tmp = csv_path + ".partial"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys0, extrasaction="ignore")
            w.writeheader()
            for r in cache.values():
                w.writerow({k: r[k] for k in keys0})
        os.replace(tmp, csv_path)

    if todo:
        print(f"  computing TCL for {len(todo)} clips with {N_PROCESSES} processes ...")
        with Pool(N_PROCESSES) as pool:
            for k, (path, sig, p, m) in enumerate(pool.imap_unordered(tcl_worker, todo, chunksize=4), 1):
                if sig is not None:
                    row = {"path": path}
                    row.update({n: v for n, v in zip(FEATURE_NAMES, sig)})
                    row["p_t"] = " ".join(f"{x:.6f}" for x in p)
                    row["m_t"] = " ".join(f"{x:.6f}" for x in m)
                    cache[path] = row
                if k % 100 == 0:
                    print(f"    {k}/{len(todo)}")
                if k % 200 == 0:
                    save_partial()          # progress is kept if the run is interrupted
        save_partial()
    rows = []
    for path, lab in zip(paths, labels):
        if path in cache:
            r = dict(cache[path]); r["label"] = lab; r["eval"] = int(path in eval_set)
            rows.append(r)
    keys = ["path", "label", "eval"] + FEATURE_NAMES + ["p_t", "m_t"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    return rows


def as_array(rows, name, label=None):
    return np.array([float(r[name]) for r in rows if label is None or int(r["label"]) == label])


# =============================================================================
# STEP 2 : FIG. 4  (distributions of phi_d, phi_c, phi_r)
# =============================================================================
def fig4(rows, title, out_base, stats_lines):
    _style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu

    panels = [("phi_d", r"Contact duration $\phi_d$"),
              ("phi_c", r"Proximity consistency $\phi_c$"),
              ("phi_r", r"Risk score $\phi_r$")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    stats_lines.append(f"\n{title}  (n_violent={int(sum(int(r['label']) for r in rows))}, "
                       f"n_nonviolent={int(sum(1 - int(r['label']) for r in rows))})")
    for ax, (key, lab) in zip(axes, panels):
        v = as_array(rows, key, 1); n = as_array(rows, key, 0)
        lo, hi = min(v.min(), n.min()), max(v.max(), n.max())
        bins = np.linspace(lo, hi, 18) if hi > lo else 10
        ax.hist(n, bins=bins, density=True, alpha=0.45, color="#3A6EA5", label="Non-violent")
        ax.hist(v, bins=bins, density=True, alpha=0.45, color="#C0392B", label="Violent")
        ax.axvline(n.mean(), color="#3A6EA5", ls="--", lw=1.5)
        ax.axvline(v.mean(), color="#C0392B", ls="--", lw=1.5)
        ax.set_xlabel(lab, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.grid(alpha=0.25)
        y = np.r_[np.ones(len(v)), np.zeros(len(n))]
        auc = roc_auc_score(y, np.r_[v, n])
        p = mannwhitneyu(v, n, alternative="two-sided").pvalue
        ratio = v.mean() / n.mean() if n.mean() > 0 else float("nan")
        ax.set_title(f"{lab}\nmean V/NV = {v.mean():.3f}/{n.mean():.3f}, AUC = {auc:.2f}", fontsize=10)
        stats_lines.append(f"  {key}: violent mean {v.mean():.4f} (sd {v.std():.4f}) | non-violent mean "
                           f"{n.mean():.4f} (sd {n.std():.4f}) | ratio {ratio:.3f} | relative gap "
                           f"{(ratio - 1) * 100:+.1f}% | AUC {auc:.3f} | Mann-Whitney p {p:.2e}")
    axes[0].legend(fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 3 : FIG. 9  (p_t, m_t of representative real clips)
# =============================================================================
def median_clip(rows, label):
    cand = sorted([r for r in rows if int(r["eval"]) == 1 and int(r["label"]) == label],
                  key=lambda r: (float(r["phi_r"]), os.path.basename(r["path"])))
    return cand[(len(cand) - 1) // 2] if cand else None


def fig9(sc_rows, rwf_rows, out_base, stats_lines):
    _style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [("SC-Fight-300", "violent", median_clip(sc_rows, 1)),
             ("SC-Fight-300", "non-violent", median_clip(sc_rows, 0)),
             ("RWF-2000", "violent", median_clip(rwf_rows, 1)),
             ("RWF-2000", "non-violent", median_clip(rwf_rows, 0))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.4), sharey=True)
    stats_lines.append("\nFig. 9 clips (class-median phi_r among evaluation clips):")
    for ax, (ds, cls, r), tag in zip(axes.ravel(), picks, "abcd"):
        if r is None:
            ax.set_visible(False); continue
        p = np.array([float(x) for x in r["p_t"].split()])
        m = np.array([float(x) for x in r["m_t"].split()])
        t = np.arange(1, len(p) + 1)
        both = (p > TAU_P) & (m > TAU_M)
        for k in np.where(both)[0]:
            ax.axvspan(t[k] - 0.5, t[k] + 0.5, color="#C0392B", alpha=0.12, lw=0)
        ax.plot(t, p, "-o", color="#C0392B", ms=3.5, lw=2, label=r"Proximity $p_t$")
        ax.plot(t, m, "-s", color="#16A085", ms=3.5, lw=2, label=r"Motion $m_t$")
        ax.axhline(TAU_P, color="#C0392B", ls="--", lw=1, label=r"$\tau_p=0.35$")
        ax.axhline(TAU_M, color="#16A085", ls="--", lw=1, label=r"$\tau_m=0.15$")
        name = os.path.splitext(os.path.basename(r["path"]))[0]
        ax.set_title(f"({tag}) {ds}, {cls} clip {name}: $\\phi_r$ = {float(r['phi_r']):.3f}, "
                     f"$\\phi_d$ = {float(r['phi_d']):.3f}", fontsize=10)
        ax.set_xlim(0.5, len(p) + 0.5); ax.set_ylim(0, 1.03)
        ax.set_xticks(t)
        ax.set_xlabel("Sampled frame index $t$", fontsize=10)
        ax.set_ylabel("Normalized score", fontsize=10)
        ax.grid(alpha=0.25)
        stats_lines.append(f"  ({tag}) {ds} {cls}: {r['path']} | phi_r {float(r['phi_r']):.4f} | "
                           f"phi_d {float(r['phi_d']):.4f} | conjunction frames {int(both.sum())}/16")
    h, l = axes[0, 0].get_legend_handles_labels()
    import matplotlib.patches as mpatches
    h.append(mpatches.Patch(color="#C0392B", alpha=0.12)); l.append(r"$p_t>\tau_p \wedge m_t>\tau_m$")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 4 : MODEL, PREDICTIONS AND GRAD-CAM (identical to Code_5_GradCAM.py)
# =============================================================================
def build_model(device):
    import torch
    import torch.nn as nn
    from transformers import VideoMAEModel

    class MotionAwareTCLFusionModule(nn.Module):
        def __init__(self, video_dim=768, tcl_dim=16):
            super().__init__()
            self.tcl_encoder = nn.Sequential(
                nn.Linear(tcl_dim, 64), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 256))
            self.attention = nn.MultiheadAttention(video_dim, 8, dropout=0.2, batch_first=True)
            self.risk_gate = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 1), nn.Sigmoid())
            self.fusion = nn.Sequential(
                nn.Linear(video_dim + 256, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.6),
                nn.Linear(512, 256), nn.Dropout(0.5))

        def forward(self, video_feats, tcl_feats):
            tcl_enc = self.tcl_encoder(tcl_feats)
            risk_w = self.risk_gate(tcl_enc)
            video_seq = video_feats.unsqueeze(1)
            tcl_query = tcl_enc.unsqueeze(1)
            # NOTE: with one pooled video token and one TCL token, the attention weight is 1 at
            # inference, so the output does not depend on the key (and hence not on this projection);
            # TCL reaches the classifier through risk_gate and the concatenation below (Sec. III-C).
            tcl_proj = nn.Linear(256, video_feats.size(-1)).to(video_feats.device)
            tcl_key = tcl_proj(tcl_query)
            attended, attn_w = self.attention(video_seq, tcl_key, video_seq)
            attended = attended.squeeze(1) * (1.0 + risk_w)
            combined = torch.cat([attended, tcl_enc], dim=1)
            return self.fusion(combined), attn_w

    class MotionAwareTCLViolenceDetector(nn.Module):
        def __init__(self, backbone=MODEL_CKPT, num_classes=2):
            super().__init__()
            self.backbone = VideoMAEModel.from_pretrained(backbone)
            self.tcl_fusion = MotionAwareTCLFusionModule(self.backbone.config.hidden_size, 16)
            self.classifier = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))

        def forward(self, pixels, tcl_feats):
            video_out = self.backbone(pixels)
            video_feats = video_out.last_hidden_state.mean(dim=1)
            fused, _ = self.tcl_fusion(video_feats, tcl_feats)
            return self.classifier(fused)

    return MotionAwareTCLViolenceDetector().to(device)


def load_weights(model, path, device):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return ck


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.h1 = target_layer.register_forward_hook(self.save_activation)
        self.h2 = target_layer.register_full_backward_hook(self.save_gradient)

    def remove(self):
        self.h1.remove(); self.h2.remove()

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def generate_cam(self, pixels, tcl, target_class):
        import torch
        self.model.eval()
        out = self.model(pixels, tcl)
        self.model.zero_grad()
        out[0, target_class].backward()
        g, a = self.gradients, self.activations
        if g.dim() == 3: g = g.squeeze(0)
        if a.dim() == 3: a = a.squeeze(0)
        pooled = torch.mean(g, dim=0)
        heat = torch.mean(a * pooled.unsqueeze(0), dim=-1).cpu().numpy()
        heat = np.maximum(heat, 0)
        if heat.max() > 0:
            heat = heat / heat.max()
        return heat


def heat_to_2d(heat):
    n = 196
    if len(heat) > n and len(heat) % n == 0:
        heat = heat.reshape(-1, n).mean(axis=0)
    else:
        heat = heat[:n]
    return heat.reshape(14, 14)


def overlay(frame, heat2d, alpha=0.6):
    import matplotlib.cm as mcm
    h, w = frame.shape[:2]
    hm = cv2.resize(heat2d.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.power(np.maximum(hm, 0), 0.7)
    col = (mcm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, col, alpha, 0)


def predict_all(model, processor, rows, device):
    import torch
    preds = []
    for r in rows:
        frames = load_video_frames(r["path"])
        if frames is None:
            continue
        tcl = torch.tensor([float(r[n]) for n in FEATURE_NAMES], dtype=torch.float32).unsqueeze(0).to(device)
        px = processor(frames, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            if device.type == "cuda":          # same numerics as the training-time evaluation (AMP)
                with torch.amp.autocast("cuda"):
                    logits = model(px, tcl)
            else:
                logits = model(px, tcl)
            prob = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        preds.append({"path": r["path"], "label": int(r["label"]), "pred": int(prob.argmax()),
                      "p_violent": float(prob[1])})
    return preds


def pick_cases(preds):
    tp = [p for p in preds if p["label"] == 1 and p["pred"] == 1]
    tn = [p for p in preds if p["label"] == 0 and p["pred"] == 0]
    fp = [p for p in preds if p["label"] == 0 and p["pred"] == 1]
    fn = [p for p in preds if p["label"] == 1 and p["pred"] == 0]
    key_v = lambda p: (-p["p_violent"], os.path.basename(p["path"]))
    key_n = lambda p: (p["p_violent"], os.path.basename(p["path"]))
    out = []
    if tp: out.append(("TP", sorted(tp, key=key_v)[0]))
    if tn: out.append(("TN", sorted(tn, key=key_n)[0]))
    if fp: out.append(("FP", sorted(fp, key=key_v)[0]))
    if fn: out.append(("FN", sorted(fn, key=key_n)[0]))
    return out


def gradcam_strip(model, processor, rows_by_path, cases, out_base, device):
    _style()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"TP": "TP: violence detected", "TN": "TN: normal activity",
             "FP": "FP: false alarm", "FN": "FN: missed violence"}
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 4.2))
    axes = np.atleast_1d(axes)
    cam = GradCAM(model, model.backbone.encoder.layer[-1].layernorm_after)
    try:
        for ax, (kind, p), tag in zip(axes, cases, "abcd"):
            r = rows_by_path[p["path"]]
            frames = load_video_frames(p["path"])
            tcl = torch.tensor([float(r[n]) for n in FEATURE_NAMES], dtype=torch.float32).unsqueeze(0).to(device)
            px = processor(frames, return_tensors="pt")["pixel_values"].to(device)
            heat = cam.generate_cam(px, tcl, p["pred"])
            img = overlay(frames[SHOW_FRAME - 1], heat_to_2d(heat))
            ax.imshow(img); ax.axis("off")
            gt = "VIOLENT" if p["label"] == 1 else "NON-VIOLENT"
            pr = "VIOLENT" if p["pred"] == 1 else "NON-VIOLENT"
            conf = p["p_violent"] if p["pred"] == 1 else 1 - p["p_violent"]
            ax.set_title(f"{os.path.splitext(os.path.basename(p['path']))[0]} | GT: {gt}\n"
                         f"Pred: {pr} ({conf * 100:.1f}%)", fontsize=9)
            ax.text(0.5, -0.08, f"({tag}) {names[kind]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=12, fontweight="bold")
            torch.cuda.empty_cache() if device.type == "cuda" else None
    finally:
        cam.remove()
    fig.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_preds(preds, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "pred", "p_violent"])
        w.writeheader(); w.writerows(preds)


# =============================================================================
# STEP 5 : PAIRED COMPARISON OF THE TWO RWF-2000 RUNS (stored predictions)
# =============================================================================
def paired_test():
    import torch
    from scipy.stats import binomtest
    a = torch.load(os.path.join(RWF_TCL_RUN, "last_checkpoint.pth"), map_location="cpu", weights_only=False)
    b = torch.load(os.path.join(RWF_NOTCL_RUN, "last_checkpoint.pth"), map_location="cpu", weights_only=False)
    yt_a, yp_a = np.array(a["val_true"]), np.array(a["val_preds"])
    yt_b, yp_b = np.array(b["val_true"]), np.array(b["val_preds"])
    same = bool(np.array_equal(yt_a, yt_b))
    ca, cb = yp_a == yt_a, yp_b == yt_b
    n01 = int(np.sum(ca & ~cb))   # TCL correct, no-TCL wrong
    n10 = int(np.sum(~ca & cb))   # TCL wrong, no-TCL correct
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"labels_identical_order": same, "acc_tcl": float(ca.mean()), "acc_notcl": float(cb.mean()),
            "tcl_correct_notcl_wrong": n01, "tcl_wrong_notcl_correct": n10,
            "mcnemar_exact_p": float(p)}


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stats_lines = ["TCL STATISTICS (exact training-time extractor, 16 uniformly sampled frames)"]
    checks = {}

    print("[1/6] Listing clips ...")
    sc_files, sc_labels, sc_eval = scfight_lists()
    rwf = rwf_lists()
    rwf_paths = rwf["train"][0] + rwf["val"][0]
    rwf_labels = rwf["train"][1] + rwf["val"][1]
    rwf_eval = set(rwf["val"][0])
    print(f"  SC-Fight-300: {len(sc_files)} clips ({len(sc_eval)} validation)")
    print(f"  RWF-2000: {len(rwf_paths)} clips ({len(rwf_eval)} official test)")

    print("[2/6] TCL features (SC-Fight-300) ...")
    sc_rows = compute_tcl_table(sc_files, sc_labels, os.path.join(OUT_DIR, "tcl_features_scfight.csv"), sc_eval)
    print("[2/6] TCL features (RWF-2000) ...")
    rwf_rows = compute_tcl_table(rwf_paths, rwf_labels, os.path.join(OUT_DIR, "tcl_features_rwf.csv"), rwf_eval)

    print("[3/6] Fig. 4 ...")
    fig4(sc_rows, f"SC-Fight-300 (all {len(sc_rows)} clips)", os.path.join(OUT_DIR, "fig4a_scfight_tcl_distributions"), stats_lines)
    fig4(rwf_rows, f"RWF-2000 (all {len(rwf_rows)} clips)", os.path.join(OUT_DIR, "fig4b_rwf_tcl_distributions"), stats_lines)

    print("[4/6] Fig. 9 ...")
    fig9(sc_rows, rwf_rows, os.path.join(OUT_DIR, "fig9_tcl_trajectories"), stats_lines)
    with open(os.path.join(OUT_DIR, "tcl_statistics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")

    print("[5/6] Predictions and Grad-CAM ...")
    import torch
    from transformers import VideoMAEImageProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    model = build_model(device)

    for tag, rows, ckpt_path, out_fig, out_csv in (
        ("scfight", sc_rows, SC_MODEL, "fig7_gradcam_scfight", "predictions_scfight.csv"),
        ("rwf", rwf_rows, os.path.join(RWF_TCL_RUN, "last_checkpoint.pth"), "fig8_gradcam_rwf", "predictions_rwf.csv"),
    ):
        ck = load_weights(model, ckpt_path, device)
        ev = [r for r in rows if int(r["eval"]) == 1]
        preds = predict_all(model, processor, ev, device)
        write_preds(preds, os.path.join(OUT_DIR, out_csv))
        acc = float(np.mean([p["label"] == p["pred"] for p in preds]))
        logged = ck.get("accuracy")
        if logged is None and "val_true" in ck:
            logged = float(np.mean(np.array(ck["val_true"]) == np.array(ck["val_preds"])))
        checks[tag] = {"checkpoint": ckpt_path, "n_eval": len(preds), "recomputed_accuracy": acc,
                       "accuracy_in_training_log": None if logged is None else float(logged),
                       "confusion": {"TN": sum(p["label"] == 0 and p["pred"] == 0 for p in preds),
                                     "FP": sum(p["label"] == 0 and p["pred"] == 1 for p in preds),
                                     "FN": sum(p["label"] == 1 and p["pred"] == 0 for p in preds),
                                     "TP": sum(p["label"] == 1 and p["pred"] == 1 for p in preds)}}
        print(f"  {tag}: recomputed accuracy {acc*100:.2f}% (training log: "
              f"{'n/a' if logged is None else f'{float(logged)*100:.2f}%'})")
        cases = pick_cases(preds)
        checks[tag]["gradcam_cases"] = [(k, p["path"], p["p_violent"]) for k, p in cases]
        gradcam_strip(model, processor, {r["path"]: r for r in rows}, cases,
                      os.path.join(OUT_DIR, out_fig), device)
        gc.collect()

    print("[6/6] Paired comparison of the RWF-2000 runs ...")
    try:
        checks["rwf_tcl_vs_notcl"] = paired_test()
    except Exception as e:
        checks["rwf_tcl_vs_notcl"] = {"error": str(e)}

    with open(os.path.join(OUT_DIR, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    print("\nDONE. Results in", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
