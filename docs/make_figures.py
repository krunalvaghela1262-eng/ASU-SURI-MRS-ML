"""
Generate the explanatory README figures from the basis sets in this repository.

    python docs/make_figures.py
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
BASIS = ROOT / "basis_set_analysis"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, GREY = "#1d2330", "#8a8f98"
COL = {"NAA": "#1f5fbf", "Cr": "#2ca02c", "PCh": "#7b3fbf", "Lac": "#d62728", "mI": "#e07b00"}
NAME = {"NAA": "NAA", "Cr": "Creatine", "PCh": "Choline", "Lac": "Lactate", "mI": "Myo-inositol"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9ced6",
                     "axes.titleweight": "bold", "axes.titlesize": 12})


def style(ax):
    ax.grid(True, color="#e6e9ee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(tag):
    s = sio.loadmat(BASIS / f"GE_UnEdited_PRESS_TE30_{tag}.mat", squeeze_me=True, struct_as_record=False)["outA"]
    return np.asarray(s.fids, complex), np.asarray(s.t, float), np.asarray(s.ppm, float)


fids, t, ppm = {}, None, None
for k in COL:
    fids[k], t, ppm = load(k)
CONC = {"NAA": 12.0, "Cr": 8.0, "PCh": 2.5, "Lac": 1.5, "mI": 6.0}          # typical brain levels (mM)
SCALE = {"Lac": 4.10799, "mI": 5.40013}                                       # basis-set correction factors


def spectrum(fid):
    """FID-A convention: spec = fftshift(ifft(fid)), matching the ppm axis stored in the basis set."""
    return np.fft.fftshift(np.fft.ifft(fid)) * len(fid)


def signal(alpha=2.0, df=0.0, dtheta=0.0, snr=None, rng=None):
    s = sum(CONC[k] * SCALE.get(k, 1.0) * fids[k] for k in COL)
    s = s * np.exp((-alpha + 2j * np.pi * df) * t) * np.exp(1j * dtheta)
    if snr is not None:
        peak = np.abs(spectrum(s)).max()
        sigma = peak / snr / np.sqrt(len(t))
        s = s + sigma * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    return s


band = (ppm > 0.6) & (ppm < 4.3)


# ------------------------------------------------------------------ banner
def banner():
    fig = plt.figure(figsize=(16, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis("off")
    ax.imshow(np.linspace(0, 1, 512)[None, :], extent=[0, 16, 0, 4], aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", ["#1a0b12", "#3a1020", "#5a1a1e"]))
    sp = np.real(spectrum(signal(alpha=4)))[band]
    x = ppm[band]
    xs = 15.6 - (x - x.min()) / (x.max() - x.min()) * 7.3
    y = 0.85 + 2.3 * sp / sp.max()
    ax.plot(xs, y, color="#ffc627", lw=2.2)
    ax.plot(xs, y, color="#ffc627", lw=9, alpha=0.10)
    for k, p in (("NAA", 2.01), ("Cr", 3.03), ("Cho", 3.2), ("mI", 3.55), ("Lac", 1.31)):
        xp = 15.6 - (p - x.min()) / (x.max() - x.min()) * 7.3
        ax.text(xp, 3.35 if k != "Cho" else 3.6, k, color="#ffe9b0", fontsize=10, ha="center")
    ax.text(0.55, 2.75, "Machine Learning for MR Spectroscopy", color="white", fontsize=29, weight="bold",
            va="center")
    ax.text(0.58, 2.0, "Quantifying brain metabolites from noisy, damped\nMRS signals with CNN, PINN, SVM, KNN and XGBoost",
            color="#ffd9c2", fontsize=14, va="center", linespacing=1.4)
    ax.text(0.58, 1.05, "SURI 2026  ·  Arizona State University  ·  Mentor: Prof. Vikram Kodibagkar",
            color="#ffc627", fontsize=12.5, va="center", weight="bold")
    fig.savefig(OUT / "banner.png", dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------ signal model animation
def signal_model_gif():
    rng = np.random.default_rng(3)
    clean = signal(alpha=1.0)
    steps = ([("ideal basis-set mixture", dict(alpha=1.0))] * 4
             + [(f"damping α = {a:g} s⁻¹  (broader peaks)", dict(alpha=a)) for a in (4, 8, 12, 16, 20)]
             + [(f"+ frequency shift Δf = {f:g} Hz", dict(alpha=12, df=f)) for f in (2, 4)]
             + [(f"+ noise, SNR = {s:g}", dict(alpha=12, df=4, snr=s, rng=rng)) for s in (60, 30, 15, 8, 5, 5, 5)])
    ref = np.real(spectrum(clean))[band]
    top = ref.max()
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=80, gridspec_kw=dict(width_ratios=[1.0, 2.0]))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.14, wspace=0.18)

    def draw(i):
        title, kw = steps[i]
        s = signal(**kw)
        for ax in axs:
            ax.clear(); style(ax)
        ax = axs[0]
        tt = t * 1000
        ax.plot(tt[:1600], np.real(s[:1600]), color="#1f5fbf", lw=0.7)
        ax.set_xlabel("time (ms)"); ax.set_title("time signal (FID)", loc="left")
        ax.set_yticks([])
        ax = axs[1]
        ax.plot(ppm[band], ref, color=GREY, lw=1, alpha=0.5, label="ideal")
        ax.plot(ppm[band], np.real(spectrum(s))[band], color="#d62728", lw=1.2, label="measured")
        ax.set_xlim(4.3, 0.6); ax.set_ylim(-0.2 * top, 1.08 * top); ax.set_yticks([])
        for k, p in (("NAA", 2.01), ("Cr", 3.03), ("Cho", 3.21), ("mI", 3.55), ("Lac", 1.33)):
            ax.text(p, 1.0 * top, k, ha="center", fontsize=9, color=INK)
        ax.set_xlabel("chemical shift (ppm)")
        ax.set_title("spectrum: what the model has to un-mix", loc="left")
        ax.legend(loc="center right", fontsize=9, frameon=False)
        fig.suptitle(f"Why quantification is hard: {title}", x=0.05, ha="left", fontsize=13, weight="bold",
                     color=INK)

    anim = FuncAnimation(fig, draw, frames=len(steps))
    anim.save(OUT / "signal_model.gif", writer=PillowWriter(fps=2))
    plt.close(fig)


# ------------------------------------------------------------------ basis-set correction
def basis_correction():
    # area per proton of each metabolite's dominant peak (results/basis_set/dominant_peak_analysis.txt)
    labels = ["Choline", "Creatine", "NAA", "Lactate", "Myo-inositol"]
    before = np.array([0.44702, 0.48555, 0.45924, 0.10431, 0.072453])
    after = before * np.array([1, 1, 1, SCALE["Lac"], SCALE["mI"]])
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), dpi=110, gridspec_kw=dict(width_ratios=[1.1, 1.4]))
    ax = axs[0]; style(ax)
    xx = np.arange(len(labels))
    ax.bar(xx - 0.2, before, 0.38, color="#c9ced6", label="original basis set")
    ax.bar(xx + 0.2, after, 0.38, color=["#7b3fbf", "#2ca02c", "#1f5fbf", "#d62728", "#e07b00"],
           label="after correction")
    ax.axhspan(0.44, 0.49, color="#2ca02c", alpha=0.08)
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("peak area per proton")
    ax.set_title("Every proton should give the same signal", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    for i in (3, 4):
        ax.annotate(f"× {[SCALE['Lac'], SCALE['mI']][i - 3]:.2f}", xy=(i + 0.2, after[i]), xytext=(i + 0.2, after[i] + 0.04),
                    ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 0.58)

    ax = axs[1]; style(ax)
    for k in ("Lac", "mI"):
        sp = np.real(spectrum(fids[k] * np.exp(-2 * t)))
        ax.plot(ppm[band], sp[band], color=COL[k], lw=1, alpha=0.35, ls="--")
        ax.plot(ppm[band], SCALE[k] * sp[band], color=COL[k], lw=1.6, label=f"{NAME[k]}  (× {SCALE[k]:.2f})")
    ax.set_xlim(4.3, 0.6); ax.set_yticks([])
    ax.set_xlabel("chemical shift (ppm)")
    ax.set_title("Lactate and myo-inositol basis spectra: before (dashed) and after", loc="left")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "basis_set_correction.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (banner, basis_correction, signal_model_gif):
        fn()
        print("done:", fn.__name__)
