# -*- coding: utf-8 -*-
"""
Spectral distortion plot for the six neutrino species.

Runs a QKE density-matrix BBN calculation and plots the final
Δf(y) / f_FD(y) = [f_α(y) - f_FD(y, T_ν_com)] / f_FD(y, T_ν_com)
for each species α ∈ {ν_e, ν̄_e, ν_μ, ν̄_μ, ν_τ, ν̄_τ} as a function
of the comoving momentum y.

f_FD is the Fermi-Dirac distribution at the comoving neutrino
temperature T_ν_com = T_ν_ini · a_ini (constant in comoving frame
before distortions develop), so Δf/f_FD shows the DEVIATION from
thermal caused by incomplete decoupling, finite-T QED, and ν-e /
ν-ν interactions.

The distortion has a characteristic "bump" near y ~ 3 T_ν_com from
ν-e annihilation heating the tail, attenuated by oscillations.

Writes the plot to validation/spectral_distortion.png and prints
summary statistics (max distortion per species) to stdout.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")
os.chdir("/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu")

import PRyM.PRyM_init as PRyMini

PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
PRyMini.numba_flag = True
PRyMini.compute_bckg_flag = False
PRyMini.compute_nTOp_flag = False
PRyMini.verbose_flag = False
PRyMini.general_nu_flag = True
PRyMini.boltzmann_nu_flag = True
PRyMini.nu_oscillation_flag = False
PRyMini.qke_density_matrix_flag = True  # full 6-species internal state
PRyMini.massive_electron_flag = False

import PRyM.PRyM_main as PRyMmain

print("Running QKE density-matrix BBN for spectral distortion ...", flush=True)
t0 = time.time()
c = PRyMmain.PRyMclass()
print(f"Done in {time.time()-t0:.1f}s. Neff = {c.Neff_f:.4f}\n", flush=True)

# Extract final state
solver = c._boltz_solver
y_grid = solver.y_grid               # shape (Ny,), comoving momentum
rho_final = c._boltz_rho_final       # shape (2, 9, Ny) for QKE
a_ini = c._boltz_a_ini
Tnu_ini = c._boltz_Tnu_ini
Tnu_com = Tnu_ini * a_ini            # comoving temperature [MeV]

# Diagonal density-matrix entries = species distributions
# rho layout (see DensityMatrixSolver docstring):
#   rho_all[sector, 0] = rho_ee     (ν_e or ν̄_e)
#   rho_all[sector, 1] = rho_μμ
#   rho_all[sector, 2] = rho_ττ
# sector 0 = neutrinos, sector 1 = antineutrinos.
f_species = {
    r"$\nu_e$":       rho_final[0, 0],
    r"$\bar\nu_e$":   rho_final[1, 0],
    r"$\nu_\mu$":     rho_final[0, 1],
    r"$\bar\nu_\mu$": rho_final[1, 1],
    r"$\nu_\tau$":    rho_final[0, 2],
    r"$\bar\nu_\tau$":rho_final[1, 2],
}

# Thermal FD at the comoving Tν
x = y_grid / Tnu_com
f_FD = np.where(x < 500.0, 1.0 / (np.exp(np.minimum(x, 500.0)) + 1.0), 0.0)

# Avoid division where FD is essentially zero (y > ~30 Tν_com)
f_min = 1.0e-12
mask = f_FD > f_min

# --- Plot ---
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

fig, (ax_f, ax_d) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
colors = {
    r"$\nu_e$":       "#d62728",
    r"$\bar\nu_e$":   "#ff9896",
    r"$\nu_\mu$":     "#1f77b4",
    r"$\bar\nu_\mu$": "#aec7e8",
    r"$\nu_\tau$":    "#2ca02c",
    r"$\bar\nu_\tau$":"#98df8a",
}

# Left: f(y) × y² for each species + FD reference
ax_f.plot(y_grid / Tnu_com, f_FD * (y_grid / Tnu_com)**2,
          "k--", lw=2, label="thermal FD", alpha=0.6)
for name, f in f_species.items():
    ax_f.plot(y_grid / Tnu_com, f * (y_grid / Tnu_com)**2,
              color=colors[name], lw=1.5, label=name)
ax_f.set_xlim(0, 10)
ax_f.set_xlabel(r"$y / T_\nu^{\rm com}$")
ax_f.set_ylabel(r"$y^2\, f_\alpha(y) \ /\ (T_\nu^{\rm com})^2$")
ax_f.set_title("Number density spectrum")
ax_f.grid(True, alpha=0.3)
ax_f.legend(loc="upper right", fontsize=9)

# Right: relative distortion Δf/f_FD
print("Max |Δf/f_FD| per species (region y/Tν_com ∈ [0.5, 8]):")
yrange = (y_grid / Tnu_com > 0.5) & (y_grid / Tnu_com < 8.0)
for name, f in f_species.items():
    dist = np.zeros_like(f)
    dist[mask] = (f[mask] - f_FD[mask]) / f_FD[mask] * 100.0  # percent
    ax_d.plot(y_grid / Tnu_com, dist, color=colors[name], lw=1.5, label=name)
    peak = np.max(np.abs(dist[yrange]))
    print(f"  {name:18s}  max |Δf/f_FD| = {peak:6.3f} %")
ax_d.axhline(0.0, color="k", lw=0.8, alpha=0.5)
ax_d.set_xlim(0, 10)
ax_d.set_xlabel(r"$y / T_\nu^{\rm com}$")
ax_d.set_ylabel(r"$\Delta f_\alpha / f_{\rm FD}\ [\%]$")
ax_d.set_title("Relative distortion from thermal FD")
ax_d.grid(True, alpha=0.3)
ax_d.legend(loc="upper left", fontsize=9)

plt.suptitle(f"PRyMordial-nu QKE spectral distortions at $T_\\gamma={c._boltz_Tg_final:.4f}$ MeV "
             f"(Neff = {c.Neff_f:.4f})")
plt.tight_layout()

out = "/Users/tait/Library/CloudStorage/Dropbox/Claude/PRyMordial-nu/validation/spectral_distortion.png"
plt.savefig(out, dpi=140)
print(f"\nSaved plot -> {out}")
