"""Fig. 9: per-frame proximity/motion profiles and conjunction rate over ALL clips (no clip selection)."""
import csv, sys, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                            "mathtext.fontset": "stix", "font.size": 11})
sc_csv, rwf_csv, out = sys.argv[1:4]
TAU_P, TAU_M = 0.35, 0.15
RED, BLUE = "#C0392B", "#2E6DA4"
def load(p):
    P, M, L = [], [], []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            P.append([float(x) for x in r["p_t"].split()]); M.append([float(x) for x in r["m_t"].split()]); L.append(int(r["label"]))
    return np.array(P), np.array(M), np.array(L)
fig, ax = plt.subplots(2, 2, figsize=(13, 7.4))
t = np.arange(1, 17)
report = []
for row, (name, path) in enumerate((("SC-Fight-300", sc_csv), ("RWF-2000", rwf_csv))):
    P, M, L = load(path)
    a = ax[row, 0]
    for lab, col, txt in ((1, RED, "violent"), (0, BLUE, "non-violent")):
        a.plot(t, P[L == lab].mean(0), "-o", color=col, ms=3.5, lw=2, label=f"$\\bar p_t$ ({txt})")
        a.plot(t, M[L == lab].mean(0), "--s", color=col, ms=3.5, lw=2, label=f"$\\bar m_t$ ({txt})")
    a.axhline(TAU_P, color="#555555", ls=":", lw=1.2); a.text(16.4, TAU_P, r"$\tau_p$", va="center", fontsize=10)
    a.axhline(TAU_M, color="#555555", ls=":", lw=1.2); a.text(16.4, TAU_M, r"$\tau_m$", va="center", fontsize=10)
    a.set_xlim(0.5, 16.9); a.set_ylim(0, 0.75); a.set_xticks(t)
    a.set_xlabel("Sampled frame index $t$"); a.set_ylabel("Mean score")
    a.set_title(f"({'ac'[row]}) {name}: mean proximity and motion scores", fontsize=11); a.grid(alpha=0.25)
    a.legend(fontsize=8.5, ncol=2, loc="upper right")
    b = ax[row, 1]
    C = (P > TAU_P) & (M > TAU_M)
    for lab, col, txt in ((1, RED, "violent"), (0, BLUE, "non-violent")):
        b.plot(t, 100 * C[L == lab].mean(0), "-o", color=col, ms=3.5, lw=2, label=f"{txt} (n = {(L == lab).sum()})")
        report.append(f"{name} {txt}: n={(L==lab).sum()} mean conj rate {100*C[L==lab].mean():.1f}% | frames2-16 {100*C[L==lab][:,1:].mean():.1f}% | mean p {P[L==lab].mean():.3f} mean m(t>=2) {M[L==lab][:,1:].mean():.3f} | frac m>tau_m {100*(M[L==lab][:,1:]>TAU_M).mean():.1f}% frac p>tau_p {100*(P[L==lab]>TAU_P).mean():.1f}% | clips with phi_r>0 {100*(C[L==lab].sum(1)>0).mean():.1f}%")
    b.set_xlim(0.5, 16.5); b.set_ylim(0, 50); b.set_xticks(t)
    b.set_xlabel("Sampled frame index $t$"); b.set_ylabel(r"Clips with $p_t>\tau_p \wedge m_t>\tau_m$ (%)")
    b.set_title(f"({'bd'[row]}) {name}: frames counted by the Risk Score", fontsize=11); b.grid(alpha=0.25); b.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, "tcl_temporal_profiles.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "tcl_temporal_profiles.png"), dpi=300, bbox_inches="tight")
print("\n".join(report))
