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


# ------------------------------------------------------------------ metabolite maps (simulated whole-brain MRSI)
# Tissue concentrations (mM, approximate literature values for healthy adult brain):
#   creatine is higher in grey matter, choline higher in white matter, NAA similar in both; CSF ~ 0.
TISSUE_CONC = {          # (grey matter, white matter, CSF)
    "NAA": (10.0, 9.0, 0.2),
    "tCr": (8.0, 5.5, 0.2),
    "tCho": (1.5, 2.0, 0.05),
}
MAP_MAX = {"NAA": 12.0, "tCr": 10.0, "tCho": 2.6}
MAP_LABEL = {"NAA": "NAA", "tCr": "tCr", "tCho": "tCho"}


def simulate_mrsi(block=3, noise=0.03, seed=11):
    """Metabolite maps from the MNI152 tissue probability maps, at MRSI-like resolution (block × 2 mm voxels)."""
    from nilearn import datasets
    t1 = datasets.load_mni152_template(resolution=2).get_fdata()
    gm = datasets.load_mni152_gm_template(resolution=2).get_fdata()
    wm = datasets.load_mni152_wm_template(resolution=2).get_fdata()
    mask = datasets.load_mni152_brain_mask(resolution=2).get_fdata() > 0.5
    gm, wm = np.clip(gm, 0, 1), np.clip(wm, 0, 1)
    csf = np.clip(mask - gm - wm, 0, 1)
    tot = np.maximum(gm + wm + csf, 1e-6)
    rng = np.random.default_rng(seed)

    def blocky(v):
        """Average over block³ voxels (MRSI point-spread), then repeat back to 2 mm for display."""
        s = [d - d % block for d in v.shape]
        v = v[:s[0], :s[1], :s[2]]
        b = v.reshape(s[0] // block, block, s[1] // block, block, s[2] // block, block).mean(axis=(1, 3, 5))
        return np.repeat(np.repeat(np.repeat(b, block, 0), block, 1), block, 2), s

    maps = {}
    for k, (c_gm, c_wm, c_csf) in TISSUE_CONC.items():
        conc = (gm * c_gm + wm * c_wm + csf * c_csf) / tot * mask
        m, s = blocky(conc)
        m = m * (1 + noise * rng.standard_normal(m.shape))
        maps[k] = m
    cover, s = blocky(mask.astype(float))
    t1 = t1[:s[0], :s[1], :s[2]]
    # Like a real whole-brain MRSI slab, skip the skull base (cerebellum / brainstem) where lipids and
    # susceptibility artefacts spoil spectra: those regions show plain anatomy.
    z = np.arange(s[2])[None, None, :]
    return t1, maps, (cover > 0.5) & (z >= int(0.36 * s[2]))


def metabolite_maps_gif():
    t1, maps, cover = simulate_mrsi()
    nx, ny, nz = t1.shape
    cmap = matplotlib.colormaps["turbo"]
    n_frames = 36
    phase = np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    pos = {"axial": (nz * 0.57 + nz * 0.15 * phase).astype(int),
           "coronal": (ny * 0.56 + ny * 0.16 * phase).astype(int),
           "sagittal": (nx * 0.50 - nx * 0.20 * np.abs(phase)).astype(int)}

    def cut(vol, view, i):
        if view == "axial":
            return np.rot90(vol[:, :, i])
        if view == "coronal":
            return np.rot90(vol[:, i, :])
        return np.rot90(vol[i, :, :])

    mets = list(TISSUE_CONC)
    views = ("axial", "coronal", "sagittal")
    fig = plt.figure(figsize=(10.5, 9.6), dpi=72, facecolor="black")
    gs = fig.add_gridspec(3, 5, width_ratios=[0.42, 1, 1, 1.18, 0.07], left=0.01, right=0.95, top=0.9,
                          bottom=0.06, wspace=0.04, hspace=0.06)
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(3)] for r in range(3)]
    cax = [fig.add_subplot(gs[r, 4]) for r in range(3)]
    for r, k in enumerate(mets):
        lab = fig.add_subplot(gs[r, 0]); lab.axis("off")
        lab.text(0.95, 0.5, MAP_LABEL[k], color="white", fontsize=22, weight="bold", ha="right", va="center")
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, MAP_MAX[k]), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax[r])
        cb.set_label("[mM]", color="white", fontsize=11)
        cb.ax.tick_params(colors="white", labelsize=9)
        cb.outline.set_edgecolor("#4b5563")
    title = fig.text(0.02, 0.955, "", color="white", fontsize=15, weight="bold")
    fig.text(0.02, 0.015, "Simulated from the MNI152 template's grey/white-matter maps with literature concentrations "
             "(not patient data).", color="#9ca3af", fontsize=9)

    def draw(f):
        for r, k in enumerate(mets):
            for c, v in enumerate(views):
                ax = axes[r][c]
                ax.clear(); ax.axis("off"); ax.set_facecolor("black")
                i = pos[v][f]
                ax.imshow(cut(t1, v, i), cmap="gray", interpolation="bilinear")
                m = np.ma.masked_where(~cut(cover, v, i), cut(maps[k], v, i))
                ax.imshow(m, cmap=cmap, vmin=0, vmax=MAP_MAX[k], alpha=0.78, interpolation="nearest")
                if r == 0:
                    ax.set_title(v, color="#c9d4ff", fontsize=13)
        z_mm = (pos["axial"][f] - nz / 2) * 2
        title.set_text(f"Whole-brain metabolite maps  ·  sweeping through the brain  (axial z ≈ {z_mm:+.0f} mm)")

    from PIL import Image
    frames = []
    for f in range(n_frames):                      # render frames ourselves: opaque black, no GIF transparency
        draw(f)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=255))
    frames[0].save(OUT / "metabolite_maps.gif", save_all=True, append_images=frames[1:], duration=170, loop=0,
                   optimize=True)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (banner, basis_correction, signal_model_gif, metabolite_maps_gif):
        fn()
        print("done:", fn.__name__)
