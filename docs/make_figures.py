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


# ------------------------------------------------------------------ metabolite maps (illustrative MRSI)
def brain_phantom(n=44):
    """Synthetic axial slice: tissue labels 0=outside, 1=CSF, 2=white matter, 3=grey matter."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    head = (x / 0.80) ** 2 + (y / 0.95) ** 2 < 1
    brain = (x / 0.72) ** 2 + (y / 0.87) ** 2 < 1
    wm = (x / 0.56) ** 2 + (y / 0.71) ** 2 < 1
    vent = (((x + 0.13) / 0.08) ** 2 + ((y + 0.05) / 0.32) ** 2 < 1) | (((x - 0.13) / 0.08) ** 2 + ((y + 0.05) / 0.32) ** 2 < 1)
    lab = np.zeros((n, n), int)
    lab[head] = 1
    lab[brain] = 3
    lab[wm] = 2
    lab[vent] = 1
    return lab, x, y


def metabolite_maps_gif():
    rng = np.random.default_rng(7)
    lab, x, y = brain_phantom()
    n = lab.shape[0]
    # typical concentrations (mM) per tissue: CSF, WM, GM
    tissue = {"NAA": (0.3, 9.0, 10.5), "Cr": (0.2, 5.8, 8.2), "PCh": (0.05, 2.1, 1.5)}
    lesion_c = {"NAA": 3.0, "Cr": 5.0, "PCh": 4.8}               # tumour-like: NAA down, choline up
    vmax = {"NAA": 12.0, "Cr": 9.0, "PCh": 5.0}
    noise = {k: rng.normal(0, 0.06 * vmax[k], (n, n)) for k in tissue}
    lesion_d = np.hypot(x - 0.30, y + 0.30)

    def maps(g):
        """Concentration maps for lesion growth g in [0, 1]."""
        out = {}
        w = np.clip((0.06 + 0.16 * g - lesion_d) / 0.05, 0, 1) * (lab >= 2) * (g > 0)
        for k, (c_csf, c_wm, c_gm) in tissue.items():
            base = np.choose(lab, [0, c_csf, c_wm, c_gm]).astype(float)
            m = (1 - w) * base + w * lesion_c[k] + noise[k] * (lab >= 1)
            m[lab == 0] = np.nan
            out[k] = m
        return out

    lr = int(round((-0.30 + 1) / 2 * (n - 1)))          # row through the lesion centre
    lc = int(round((0.30 + 1) / 2 * (n - 1)))
    path = [(lr, c) for c in range(n) if lab[lr, c] >= 2]
    frames = []
    for i, (r, c) in enumerate(path):
        g = min(1.0, i / (0.6 * len(path)))
        frames.append((r, c, g))
    frames += [(lr, lc, 1.0)] * 8                        # finish on the lesion

    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("#000000")
    fig = plt.figure(figsize=(13, 4.6), dpi=78, facecolor="black")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.9], left=0.02, right=0.98, top=0.82, bottom=0.13,
                          wspace=0.12)
    axm = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axs = fig.add_subplot(gs[0, 3])
    healthy = maps(0.0)

    def voxel_fid(conc):
        return sum(conc[k] * fids[k] for k in ("NAA", "Cr", "PCh")) * np.exp(-6 * t)

    ref_spec = None

    def draw(i):
        nonlocal ref_spec
        r, c, g = frames[i]
        mp = maps(g)
        for ax, k in zip(axm, ("NAA", "Cr", "PCh")):
            ax.clear()
            ax.imshow(mp[k], cmap=cmap, vmin=0, vmax=vmax[k], interpolation="nearest")
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="white", lw=2))
            ax.set_title({"NAA": "NAA", "Cr": "Creatine", "PCh": "Choline"}[k], color="white", fontsize=14,
                         weight="bold")
            ax.axis("off")
        conc = {k: max(mp[k][r, c], 0) if not np.isnan(mp[k][r, c]) else 0 for k in mp}
        spec = np.real(spectrum(voxel_fid(conc)))[band]
        if ref_spec is None:
            ref_spec = np.real(spectrum(voxel_fid({"NAA": 10.5, "Cr": 8.2, "PCh": 2.1})))[band]
        axs.clear()
        axs.set_facecolor("black")
        axs.plot(ppm[band], ref_spec, color="#6b7280", lw=1, label="healthy grey matter")
        axs.plot(ppm[band], spec, color="#ffc627", lw=1.8, label="selected voxel")
        top = 1.12 * max(ref_spec.max(), np.real(spectrum(voxel_fid(lesion_c)))[band].max())
        axs.set_xlim(4.0, 1.0); axs.set_ylim(-0.08 * top, top)
        for k, p, yy in (("NAA", 2.01, 0.93), ("Cr", 3.0, 0.93), ("Cho", 3.24, 0.82)):
            axs.text(p, yy * top, k, ha="center", color="white", fontsize=11, weight="bold")
        axs.set_yticks([])
        axs.tick_params(colors="#c9ced6")
        for s in axs.spines.values():
            s.set_color("#4b5563")
        axs.set_xlabel("chemical shift (ppm)", color="#c9ced6")
        axs.set_title("spectrum in the white square", color="white", fontsize=13, loc="left")
        axs.legend(loc="center left", fontsize=9, facecolor="black", edgecolor="#4b5563", labelcolor="white")
        tissue_name = {0: "outside", 1: "CSF", 2: "white matter", 3: "grey matter"}[lab[r, c]]
        in_lesion = g > 0 and lesion_d[r, c] < 0.06 + 0.16 * g and lab[r, c] >= 2
        status = "lesion: NAA ↓  choline ↑" if in_lesion else tissue_name
        title.set_text(f"Metabolite maps: quantify every voxel  ·  voxel = {status}")
        title.set_color("#ff8a80" if in_lesion else "white")

    title = fig.text(0.02, 0.93, "", color="white", fontsize=14, weight="bold", ha="left")
    fig.text(0.02, 0.03, "Illustration on a synthetic brain slice (not patient data). Spectra built from "
             "this project's basis sets.", color="#9ca3af", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(OUT / "metabolite_maps.gif", writer=PillowWriter(fps=5), savefig_kwargs={"facecolor": "black"})
    plt.close(fig)


if __name__ == "__main__":
    for fn in (banner, basis_correction, signal_model_gif, metabolite_maps_gif):
        fn()
        print("done:", fn.__name__)
