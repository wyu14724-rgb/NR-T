"""Generate paper figures F2 and F4 (English labels only).

Outputs to ../../figures/ as PDF (vector) and PNG (300 dpi).

F2  paired scoring comparison, 3 datasets x 3 seeds
F4  calibration contamination stress test (NR-T arm)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT.parent / "results"
OUT = ROOT.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})

SEEDS = [42, 43, 44]
SEED_COLOR = {42: "#4C72B0", 43: "#DD8452", 44: "#55A868"}
DS_COLOR = {"ton": "#4C72B0", "unsw": "#DD8452", "cic": "#55A868"}
DS_MARKER = {"ton": "o", "unsw": "s", "cic": "^"}
DS_LABEL = {"ton": "ToN-IoT", "unsw": "UNSW-NB15", "cic": "CIC-IDS2017"}


def load_arms():
    """Return {dataset: {seed: {arm: macro_tpr1}}} from the exact sources used in the paper."""
    frames = []
    ton = pd.read_csv(RES / "paired_three_arms.csv")
    ton = ton[ton["dataset"] == "ton"].assign(dataset="ton")
    frames.append(ton)
    unsw = pd.read_csv(RES / "paired_2025_unsw_minmax.csv").assign(dataset="unsw")
    frames.append(unsw)
    cic = pd.read_csv(RES / "paired_2025_cic.csv").assign(dataset="cic")
    frames.append(cic)
    df = pd.concat(frames, ignore_index=True)

    out = {}
    for ds, g in df.groupby("dataset"):
        out[ds] = {}
        for seed, gs in g.groupby("seed"):
            out[ds][int(seed)] = dict(zip(gs["arm"], gs["macro_tpr1"]))
    return out


def fig_f2():
    arms = load_arms()
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.42), sharey=True)
    x_nat, x_nrt = 0.0, 1.0

    for ax, ds in zip(axes, ["ton", "unsw", "cic"]):
        sub = arms[ds]
        diff = []
        for seed in SEEDS:
            a = sub[seed]
            ax.plot([x_nat, x_nrt], [a["native"], a["lat1nn"]],
                    color=SEED_COLOR[seed], lw=1.3, alpha=0.9, zorder=3)
            ax.plot(x_nat, a["native"], "o", ms=5, color=SEED_COLOR[seed], zorder=4)
            ax.plot(x_nrt, a["lat1nn"], "s", ms=5, color=SEED_COLOR[seed], zorder=4)
            diff.append(a["lat1nn"] - a["native"])

        raw = sub[SEEDS[0]]["raw1nn"]
        ax.hlines(raw, x_nat - 0.32, x_nrt + 0.32, colors="#8C8C8C",
                  linestyles=(0, (4, 2)), lw=1.2, zorder=2)

        mean_pp = 100.0 * sum(diff) / len(diff)
        detail = " / ".join(f"{100*d:+.1f}" for d in diff)
        ax.text(0.5, 0.985, f"mean $\\Delta$ = {mean_pp:+.1f} pp\n(per seed: {detail})",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
                color="#333333", linespacing=1.3)

        ax.set_title(DS_LABEL[ds], fontsize=9, pad=5)
        ax.set_xticks([x_nat, x_nrt])
        ax.set_xticklabels(["Prototype\nresidual", "NR-T\n(latent 1-NN)"])
        ax.set_xlim(-0.45, 1.45)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        if ds == "ton":
            ax.annotate("collapsed\ncheckpoint", xy=(x_nat - 0.03, sub[44]["native"]),
                        xytext=(0.28, 0.13), fontsize=7.2, color=SEED_COLOR[44],
                        arrowprops=dict(arrowstyle="->", lw=0.7, color=SEED_COLOR[44]))

    axes[0].set_ylabel("Macro TPR at\nFPR $\\leq$ 1%")
    axes[0].set_ylim(0.0, 0.62)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=SEED_COLOR[s], marker="o", ms=4, lw=1.3,
                      label=f"seed {s}") for s in SEEDS]
    handles.append(Line2D([], [], color="#8C8C8C", ls=(0, (4, 2)), lw=1.2,
                          label="raw-feature 1-NN (seed-invariant)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
               columnspacing=1.4)

    fig.subplots_adjust(left=0.10, right=0.992, bottom=0.175, top=0.80, wspace=0.30)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"F2_paired_scoring.{ext}", dpi=300)
    plt.close(fig)


def fig_f4():
    df = pd.read_csv(RES / "calib_stress.csv")
    df = df[(df["arm"] == "lat1nn") & (df["scenario"].isin(["honest", "contamination"]))]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ds in ["ton", "unsw", "cic"]:
        g = df[df["dataset"] == ds].sort_values("rate")
        x = 100.0 * g["rate"].to_numpy()
        axes[0].plot(x, g["recall"].to_numpy(), ls="-", marker=DS_MARKER[ds], ms=4.5, lw=1.3,
                     color=DS_COLOR[ds], label=DS_LABEL[ds])
        axes[1].plot(x, 100.0 * g["fpr"].to_numpy(), ls="-", marker=DS_MARKER[ds], ms=4.5, lw=1.3,
                     color=DS_COLOR[ds], label=DS_LABEL[ds])

    axes[0].set_xlabel("Attack samples mixed into calibration batch (%)")
    axes[0].set_ylabel("Macro recall at the locked threshold")
    axes[0].set_ylim(0.0, 0.62)
    axes[0].set_title("(a) Detection collapses", fontsize=9.5)

    axes[1].set_xlabel("Attack samples mixed into calibration batch (%)")
    axes[1].set_ylabel("Measured FPR (%)")
    axes[1].set_ylim(0.0, 5.5)
    axes[1].axhline(1.0, color="#8C8C8C", ls=(0, (4, 2)), lw=1.0)
    axes[1].text(9.6, 1.22, "target $\\alpha$ = 1%", ha="right", va="bottom",
                 fontsize=7.2, color="#666666")
    axes[1].set_title("(b) FPR falls below target", fontsize=9.5)

    for ax in axes:
        ax.set_xticks([0, 1, 5, 10])
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F4_calibration_contamination.{ext}", dpi=300)
    plt.close(fig)


def fig_f1():
    """Method overview: one frozen representation, two scoring objects, one threshold rule."""
    from matplotlib.patches import (Arc, Ellipse, FancyArrowPatch, FancyBboxPatch,
                                    Polygon, Rectangle)

    INK, GREY, HAIR = "#1F2328", "#8A9199", "#D5DAE0"
    PAPER, PEDGE = "#FBFCFD", "#E1E6EB"
    BASE_F, BASE_E = "#F3F4F6", "#B9C0C6"
    OURS_F, OURS_E = "#EAF2F8", "#2E6DA4"
    ACC = "#1F4E79"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    def rbox(x0, y0, x1, y1, fc, ec, lw=0.9, r=0.9, z=2, ls="-"):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls))

    def arr(p0, p1, color=INK, lw=0.85, ms=6.5, z=4):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color,
            lw=lw, shrinkA=0, shrinkB=1.2, zorder=z))

    def line(pts, color=INK, lw=0.85, z=3, ls="-"):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=z,
                solid_capstyle="round")

    def txt(x, y, s, fs=7.0, color=INK, ha="center", va="center", style="normal",
            weight="normal", rot=0, lsp=1.45, z=6, bbox=None):
        ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, style=style,
                fontweight=weight, rotation=rot, linespacing=lsp, zorder=z, bbox=bbox)

    def panel(x0, x1, tag, title):
        rbox(x0, 4, x1, 62, PAPER, PEDGE, lw=0.8, r=1.3, z=1)
        txt(x0 + 2.0, 59.4, tag, fs=6.2, color=ACC, ha="left", weight="bold")
        txt(x0 + 5.2, 59.4, title, fs=6.0, color="#6B737B", ha="left", weight="bold")

    def padlock(cx, cy, s=1.0, color=INK):
        ax.add_patch(Rectangle((cx - 0.62 * s, cy - 0.62 * s), 1.24 * s, 0.95 * s,
                               facecolor=color, edgecolor="none", zorder=6))
        ax.add_patch(Arc((cx, cy + 0.33 * s), 0.95 * s, 1.05 * s, theta1=0, theta2=180,
                         lw=0.8 * s, color=color, zorder=6))
        ax.add_patch(Rectangle((cx - 0.08 * s, cy - 0.34 * s), 0.16 * s, 0.44 * s,
                               facecolor="white", edgecolor="none", zorder=7))

    # ------------------------------------------------ panel 1: representation
    panel(2, 25, "1", "FROZEN REPRESENTATION")

    rbox(5.0, 50.4, 22.0, 56.2, "#FFFFFF", HAIR, lw=0.9, r=0.8)
    txt(13.5, 54.3, "$x$", fs=8.6, weight="bold")
    txt(13.5, 51.9, "raw flow record", fs=5.8, color=GREY)

    arr((13.5, 50.2), (13.5, 46.4))

    rbox(3.2, 26.8, 23.8, 46.2, "#FFFFFF", INK, lw=1.0, r=0.9)
    txt(13.5, 40.8, "$f_{\\theta}$", fs=11.0)
    txt(13.5, 37.0, "encoder", fs=6.2, color=GREY)
    line([(5.2, 34.4), (21.8, 34.4)], color=HAIR, lw=0.6)
    padlock(6.8, 31.4, s=1.05)
    txt(8.9, 31.3, "parameters frozen", fs=6.0, ha="left")

    arr((13.5, 26.6), (13.5, 22.6))

    rbox(5.0, 13.8, 22.0, 22.4, "#FFFFFF", INK, lw=1.0, r=0.8)
    txt(13.5, 20.0, "$h(x)\\in\\mathbb{R}^{128}$", fs=7.2)
    txt(13.5, 16.4, "latent code", fs=5.8, color=GREY)

    txt(13.5, 9.4, "same checkpoint as the baseline;\n"
                   "weights are never updated",
        fs=5.7, color=GREY, style="italic")

    # bus: one latent code feeds both scoring objects
    line([(22.0, 18.0), (26.7, 18.0), (26.7, 55.0)])
    arr((26.7, 55.0), (28.5, 55.0))

    # ------------------------------------------------ panel 2: scoring object
    panel(28.5, 72, "2", "SCORING OBJECT  (same $h(x)$, different distance set)")

    # shared bus: one latent code feeding both objects
    line([(28.6, 55.0), (70.0, 55.0)], color=INK, lw=0.85)
    arr((39.5, 55.0), (39.5, 43.5), color=INK, lw=0.8, ms=5.5)
    arr((60.5, 55.0), (60.5, 43.5), color=OURS_E, lw=0.8, ms=5.5)

    # column headers
    txt(30.0, 48.4, "BASELINE", fs=6.0, color=GREY, ha="left", weight="bold")
    txt(30.0, 46.0, "prototype object", fs=5.2, color=GREY, ha="left")
    txt(70.6, 48.4, "THIS WORK", fs=6.0, color=OURS_E, ha="right", weight="bold")
    txt(70.6, 46.0, "sample object", fs=5.2, color=OURS_E, ha="right")

    line([(50.2, 5.0), (50.2, 52.0)], color=HAIR, lw=0.6, ls=(0, (3, 2)))

    # left arm: distance to a predicted mixture point
    hull = [(35.6, 29.4), (42.6, 29.0), (44.9, 33.4), (41.5, 38.6), (36.3, 37.8)]
    ax.add_patch(Polygon(hull, closed=True, facecolor="#EDEFF2", edgecolor=BASE_E,
                         lw=0.6, zorder=3))
    for vx, vy in hull:
        ax.plot([vx], [vy], marker="o", ms=2.0, color=GREY, zorder=5)
    txt(38.6, 31.2, "$\\mathrm{conv}(B)$", fs=5.0, color=GREY)
    line([(39.5, 42.6), (39.7, 35.6)], color=INK, lw=0.75, ls=(0, (2.2, 1.8)))
    ax.plot([39.5], [43.0], marker="o", ms=3.8, color=INK, zorder=6)
    ax.plot([39.7], [35.4], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=6)
    txt(37.6, 43.2, "$h(x)$", fs=5.6, ha="right")
    txt(41.2, 34.9, "$\\hat{h}$", fs=6.0)
    txt(37.6, 40.2, "$R(x)$", fs=5.6)
    txt(39.5, 26.3, "$\\hat{h}$ moves with $x$", fs=5.3, color=GREY, style="italic")

    # right arm: distance to the nearest stored normal code
    rng = np.random.RandomState(11)
    cx, cy = 60.5, 33.4
    pts = rng.normal(0, 1, (70, 2)) * np.array([2.5, 1.9]) + np.array([cx, cy])
    keep = ((pts[:, 0] - cx) / 5.2) ** 2 + ((pts[:, 1] - cy) / 4.3) ** 2 < 1.0
    pts = pts[keep]
    ax.add_patch(Ellipse((cx, cy), 10.4, 8.6, facecolor="none", edgecolor=HAIR,
                         lw=0.6, ls=(0, (3, 2)), zorder=3))
    ax.plot(pts[:, 0], pts[:, 1], ls="none", marker="o", ms=1.5, color=GREY, zorder=4)
    txt(cx - 4.6, cy + 2.9, "$\\mathcal{H}$", fs=6.0, color=GREY)

    hx, hy = 60.5, 43.0
    order = np.argsort(np.hypot(pts[:, 0] - hx, pts[:, 1] - hy))
    for k in order[1:4]:
        line([(hx, hy), tuple(pts[k])], color=GREY, lw=0.45, ls=(0, (1.5, 1.8)))
    nn = pts[order[0]]
    arr((hx, hy - 0.4), tuple(nn), color=OURS_E, lw=0.85, ms=6)
    ax.add_patch(Ellipse((nn[0], nn[1]), 1.7, 1.7, facecolor="none",
                         edgecolor=OURS_E, lw=0.6, zorder=5))
    ax.plot([nn[0]], [nn[1]], marker="o", ms=3.0, color=OURS_E, zorder=6)
    ax.plot([hx], [hy], marker="o", ms=3.8, color=OURS_E, zorder=6)
    txt(hx + 1.6, 43.2, "$h(x)$", fs=5.6, color=OURS_E)
    txt(58.4, 39.7, "$N(x)$", fs=5.6, color=OURS_E)
    txt(60.5, 26.3, "$\\mathcal{H}$ fixed after training", fs=5.3,
        color=GREY, style="italic")

    # formulas
    txt(30.0, 21.6, "$w(x)=\\mathrm{softmax}(\\phi_{\\theta}(x))$", fs=6.2, ha="left")
    txt(30.0, 18.6, "$\\hat{h}=B^{\\top}w(x)$", fs=6.2, ha="left")
    txt(30.0, 15.6, "$R(x)=\\|\\,h(x)-\\hat{h}\\,\\|^{2}$", fs=6.2, ha="left")
    txt(30.0, 12.8, "$B\\in\\mathbb{R}^{K\\times 128}$, $K=5$ basis vectors",
        fs=4.9, color=GREY, ha="left", va="top")

    txt(51.2, 21.6, "$\\mathcal{H}=\\{h_i\\}_{i=1}^{n}$", fs=6.2, ha="left")
    txt(51.2, 18.6, "$N(x)=\\min_{j}\\|\\,h(x)-h_j\\,\\|^{2}$", fs=6.2, ha="left")
    txt(51.2, 15.6, "$k=1$ (1-NN)", fs=6.2, ha="left")
    txt(51.2, 12.8, "training normal codes", fs=4.9, color=GREY, ha="left", va="top")

    arr((72.0, 33.0), (75.3, 33.0))
    txt(73.5, 34.7, "$s(x)$", fs=5.4)

    # ------------------------------------------------ panel 3: calibration
    panel(75.5, 98, "3", "CALIBRATION")

    rbox(76.6, 46.0, 97.0, 55.0, "#FFFFFF", HAIR, lw=0.9, r=0.8)
    txt(86.8, 52.4, "calibration normals", fs=6.0)
    txt(86.8, 48.8, "held out from training", fs=5.3, color=GREY)

    arr((86.8, 45.8), (86.8, 42.9))

    rbox(76.6, 31.6, 97.0, 42.8, "#FFFFFF", OURS_E, lw=1.0, r=0.8)
    txt(86.8, 39.5, "$\\tau=F_{s}^{-1}(0.99)$", fs=7.4)
    txt(86.8, 35.8, "upper order statistic of $s(x)$", fs=5.4, color=GREY)
    txt(86.8, 33.4, "for $s=R$ or $s=N$", fs=5.4, color=GREY)

    arr((86.8, 31.4), (86.8, 28.3))

    rbox(76.6, 18.2, 97.0, 28.3, "#FFFFFF", INK, lw=1.0, r=0.8)
    txt(86.8, 24.9, "$s(x)>\\tau$", fs=7.4)
    txt(86.8, 21.0, "flag as anomaly", fs=5.6, color=GREY)

    txt(86.8, 12.7, "one rule, both\nscoring objects", fs=5.6, color=GREY, style="italic")

    txt(50.0, 1.5, "Only the scoring object $s(x)$ is varied; the encoder $f_{\\theta}$, "
                   "the calibration normals and the 99% rule are shared ($\\tau$ recomputed per $s$).",
        fs=6.2, color="#5B636B", style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F1_method_overview.{ext}", dpi=300)
    plt.close(fig)


NORMAL_NAMES = {"normal", "Normal", "BENIGN"}


def eval_half(y, seed=42):
    """Second half of each class, matching baselines.split (seed 42)."""
    rng = np.random.RandomState(seed)
    te = []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        te.append(ix[len(ix) // 2:])
    return np.concatenate(te)


def per_class_tpr(scores, y, cats, cls, alpha=0.01):
    """TPR of one attack class at FPR <= alpha, matching baselines.macro_metrics."""
    from sklearn.metrics import roc_curve
    m = cats == cls
    pair = (y == 0) | m
    fpr, tpr, _ = roc_curve(m[pair].astype(int), scores[pair])
    return float(tpr[fpr <= alpha].max()) if (fpr <= alpha).any() else 0.0


def fig_f3():
    """Per-attack-class TPR at the 1% operating point, prototype residual vs NR-T."""
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.54))
    arms = [("native", "Prototype residual", "#9E9E9E"),
            ("lat1nn", "NR-T", "#4C72B0")]
    # column width is 1.9 in: the two longest CIC-IDS2017 names are abbreviated
    tick_name = {"Web Attack \u2013 Brute Force": "Web\u2013Brute Force",
                 "DoS Slowhttptest": "DoS SlowHTTP"}

    for ax, ds in zip(axes, ["ton", "unsw", "cic"]):
        z = np.load(RES / f"scores_{ds}_s42.npz", allow_pickle=True)
        te = eval_half(z["y_test"])
        y = z["y_test"][te]
        cats = z["cats_test"].astype(str)[te]
        classes = sorted(set(cats.tolist()) - NORMAL_NAMES)

        tpr = {}
        for arm, _, _ in arms:
            s = z[f"test_{arm}"].astype(np.float64)[te]
            tpr[arm] = {c: per_class_tpr(s, y, cats, c) for c in classes}

        order = sorted(classes, key=lambda c: tpr["lat1nn"][c])
        x = np.arange(len(order))
        w = 0.38
        for k, (arm, label, color) in enumerate(arms):
            ax.bar(x + (k - 0.5) * w, [tpr[arm][c] for c in order], width=w,
                   color=color, edgecolor="black", lw=0.4, label=label,
                   hatch=("//" if k == 1 else ""))

        macro = {arm: float(np.mean([tpr[arm][c] for c in classes])) for arm, _, _ in arms}
        ax.text(0.03, 0.97,
                f"seed 42; macro TPR\n{macro['native']:.3f} $\\rightarrow$ {macro['lat1nn']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                color="#333333", linespacing=1.25)

        ax.set_xticks(x)
        ax.set_xticklabels([tick_name.get(c, c) for c in order],
                           rotation=45, ha="right", fontsize=7.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(DS_LABEL[ds], fontsize=9)
        ax.grid(axis="y", color="#EDEDED", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel("TPR at FPR $\\leq$ 1%", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.30, top=0.845, wspace=0.32)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"F3_per_class_tpr.{ext}", dpi=300)
    plt.close(fig)

    for ds in ["ton", "unsw", "cic"]:
        z = np.load(RES / f"scores_{ds}_s42.npz", allow_pickle=True)
        te = eval_half(z["y_test"])
        y, cats = z["y_test"][te], z["cats_test"].astype(str)[te]
        classes = sorted(set(cats.tolist()) - NORMAL_NAMES)
        for arm in ("native", "lat1nn"):
            s = z[f"test_{arm}"].astype(np.float64)[te]
            m = float(np.mean([per_class_tpr(s, y, cats, c) for c in classes]))
            print(f"{ds:5s} {arm:7s} macro TPR@1% = {m:.4f}")


def macro_tpr_at(scores, y, cats, classes, alpha):
    """Macro TPR at FPR <= alpha, same convention as baselines.macro_metrics."""
    return float(np.mean([per_class_tpr(scores, y, cats, c, alpha) for c in classes]))


def fig_f6():
    """Macro TPR across the low-FPR range: is 1% a cherry-picked operating point?"""
    alphas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
    arms = [("native", "Prototype residual", "#9E9E9E", (0, (4, 2)), "o"),
            ("lat1nn", "NR-T (latent 1-NN)", "#2E6DA4", "solid", "s")]

    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.49), sharey=True)
    for ax, ds in zip(axes, ["ton", "unsw", "cic"]):
        z = np.load(RES / f"scores_{ds}_s42.npz", allow_pickle=True)
        te = eval_half(z["y_test"])
        y = z["y_test"][te]
        cats = z["cats_test"].astype(str)[te]
        classes = sorted(set(cats.tolist()) - NORMAL_NAMES)

        curves = {}
        for arm, label, color, ls, mk in arms:
            s = z[f"test_{arm}"].astype(np.float64)[te]
            v = np.array([macro_tpr_at(s, y, cats, classes, a) for a in alphas])
            curves[arm] = v
            ax.plot(100 * alphas, v, ls=ls, marker=mk, ms=4.0, lw=1.4,
                    color=color, label=label, zorder=3,
                    mfc=color if arm == "lat1nn" else "white", mew=0.9)

        gap = 100.0 * (curves["lat1nn"] - curves["native"])
        # compact wording: the full "Δ at α=…" form would run into the 1% line
        ax.text(0.03, 0.97,
                f"$\\Delta$($\\alpha$=1%) {gap[3]:+.1f} pp\n"
                f"$\\Delta$($\\alpha$=5%) {gap[-1]:+.1f} pp",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                color="#333333", linespacing=1.4)

        ax.axvline(1.0, color="#C0392B", ls=(0, (2, 2)), lw=0.9, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 6.5)
        ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"], fontsize=8)
        ax.minorticks_off()
        ax.set_title(DS_LABEL[ds], fontsize=9)
        ax.grid(axis="y", color="#EDEDED", lw=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Macro TPR at\nFPR $\\leq\\alpha$", fontsize=8.5)
    axes[0].set_ylim(0.0, 0.88)
    for ax in axes:
        ax.set_xlabel("False-positive budget $\\alpha$ (%)", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#C0392B", ls=(0, (2, 2)), lw=0.9,
                          label="operating point ($\\alpha$ = 1%)"))
    labels.append("operating point ($\\alpha$ = 1%)")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
               columnspacing=1.4)

    fig.subplots_adjust(left=0.088, right=0.992, bottom=0.195, top=0.80, wspace=0.30)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"F6_alpha_sweep.{ext}", dpi=300)
    plt.close(fig)

    for ds in ["ton", "unsw", "cic"]:
        z = np.load(RES / f"scores_{ds}_s42.npz", allow_pickle=True)
        te = eval_half(z["y_test"])
        y = z["y_test"][te]
        cats = z["cats_test"].astype(str)[te]
        classes = sorted(set(cats.tolist()) - NORMAL_NAMES)
        row = []
        for arm, *_ in arms:
            s = z[f"test_{arm}"].astype(np.float64)[te]
            row.append([macro_tpr_at(s, y, cats, classes, a) for a in alphas])
        print(f"{ds:5s} " + " ".join(f"{a:.3f}:{row[0][i]:.3f}->{row[1][i]:.3f}"
                                    for i, a in enumerate(alphas)))


def fig_f5():
    """Reference-set size: the 1% tail degrades while the overall ranking largely does not."""
    names = {"ton": "ToN-IoT", "unsw": "UNSW-NB15", "cic": "CIC-IDS2017"}
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 3.82), sharex="col")
    changes = {}

    for j, ds in enumerate(["ton", "unsw", "cic"]):
        df = pd.read_csv(RES / f"refsize_{ds}.csv")
        g = (df.groupby("n_ref")
               .agg(mtpr=("mtpr1", "mean"), mtpr_sd=("mtpr1", "std"),
                    auc=("auroc", "mean"), auc_sd=("auroc", "std"))
               .reset_index()
               .sort_values("n_ref"))
        x = g["n_ref"].to_numpy()

        for row, (col, sdcol) in enumerate([("mtpr", "mtpr_sd"), ("auc", "auc_sd")]):
            ax = axes[row, j]
            y = g[col].to_numpy()
            sd = g[sdcol].fillna(0.0).to_numpy()
            if row == 0:
                ax.axhline(y[0], color="#C4C4C4", ls=(0, (3, 2)), lw=0.8, zorder=1)
            ax.fill_between(x, y - sd, y + sd, color=DS_COLOR[ds], alpha=0.18, lw=0)
            ax.plot(x, y, ls="-", marker=DS_MARKER[ds], ms=4.5, lw=1.4, color=DS_COLOR[ds], zorder=3)

            # change at the smallest reference set, relative to the full set;
            # where the label goes is decided once all y-limits are final
            d = 100.0 * (y[0] - y[-1])
            changes[(row, j)] = (ax, x[0], y[0], d, DS_COLOR[ds])

            ax.set_xscale("log")
            ax.minorticks_off()
            ax.grid(axis="y", color="#E6E6E6", lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)

        # per-dataset window: the largest n_ref differs by a factor of two;
        # only three ticks (smallest / middle / full) fit the column width
        span = {"ton": (2.4e3, 4.2e4, [3.0e3, 1.5e4, 3.0e4],
                        ["3k", "15k", "30k"]),
                "unsw": (3.8e3, 7.2e4, [5.04e3, 2.52e4, 5.04e4],
                         ["5.0k", "25.2k", "50.4k"]),
                "cic": (4.4e3, 8.6e4, [6e3, 3e4, 6e4],
                        ["6k", "30k", "60k"])}[ds]
        for ax in axes[:, j]:
            ax.set_xlim(span[0], span[1])
            ax.set_xticks(span[2])
            ax.set_xticklabels(span[3], fontsize=8)

        axes[0, j].set_title(names[ds], fontsize=9, pad=5)

        # AUROC level differs per dataset (0.82 / 0.89 / 0.97): same window width,
        # so the slopes stay comparable instead of a shared axis flattening two of them
        lo = float((g["auc"] - g["auc_sd"].fillna(0.0)).min())
        hi = float((g["auc"] + g["auc_sd"].fillna(0.0)).max())
        c = 0.5 * (lo + hi)
        axes[1, j].set_ylim(c - 0.025, c + 0.025)
        axes[1, j].yaxis.set_major_locator(mticker.MaxNLocator(4))

    for ax in axes[0]:
        ax.set_ylim(0.12, 0.57)
        ax.set_yticks([0.15, 0.25, 0.35, 0.45, 0.55])

    # "change" labels sit below the smallest-n_ref point, except in the panels
    # where that point is too close to the bottom edge (top-row CIC-IDS2017):
    # there the label moves to the free top-left corner of the panel
    for ax, x0, y0, d, color in changes.values():
        lo_, hi_ = ax.get_ylim()
        if (y0 - lo_) / (hi_ - lo_) < 0.22:
            ax.annotate(f"{d:+.1f} pp", xy=(0.03, 0.95), xycoords="axes fraction",
                        ha="left", va="top", fontsize=7.6, color=color, weight="bold")
        else:
            ax.annotate(f"{d:+.1f} pp", xy=(x0, y0), xytext=(7, -10),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=7.6, color=color, weight="bold")

    axes[0, 0].set_ylabel("Macro TPR at\nFPR $\\leq$ 1%", fontsize=8.5)
    axes[1, 0].set_ylabel("AUROC", fontsize=8.5)
    fig.supxlabel("Reference-set size $n_{\\mathrm{ref}}$ (training normal codes, log scale)",
                  fontsize=8)

    fig.subplots_adjust(left=0.10, right=0.992, bottom=0.115, top=0.925,
                        wspace=0.34, hspace=0.38)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"F5_reference_size.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_f1()
    fig_f2()
    fig_f3()
    fig_f4()
    fig_f5()
    fig_f6()
    print("written to", OUT)
