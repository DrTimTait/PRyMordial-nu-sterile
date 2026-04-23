"""Stage E.2 sprint 9: MSW-passage adiabatic over-pumping diagnostic.

Per-step per-y-mode Hamiltonian and diagonal-population accounting for
the active-sterile pair that carries Point A's mixing (theta_24, pair
(alpha=1=numu, s=3), pair index 4) inside evolve_step_ode_etdrk2 at
Point A (sin^2(2theta)=1e-1, dm^2=0.93) of the Hannestad+2012 benchmark,
extended window T_boltz_start=30 MeV, projection on, n_B=10000 — the
sprint-7 Phase-5 configuration that shows dNeff=+1.57 at Σρ_ss ≈ 30
("thermal population, wrong y-distribution").

Metrics recorded at two sub-step positions per time step:
  positions : {"pre_step", "post_clip"}
  per (sector s in {0,1}, y-mode):
    H_aa(y), H_ss(y)        in eV, diagonal Hamiltonian entries
    Re_H_as(y), Im_H_as(y)  in eV, off-diagonal entry (vacuum + medium)
    rho_aa(y), rho_ss(y)    diagonal populations

Post-processor outputs:
  1. Per y-mode resonance-localisation table: T_res (Tg at
     argmin_step |H_aa - H_ss|), step_res, |H_as| at crossing, local
     step-indexed adiabaticity γ_LZ_step = |H_as|^2 / |dΔ/dstep|,
     end-of-run ρ_ss(y), and integrated Δρ_ss within ±N_window steps
     of the crossing.
  2. Resonance-window heatmap summary: Δρ_ss(y, step) averaged across
     the resonance window and bucketed by y. Identifies whether
     high-y modes receive anomalously large deposition (Suspect 2
     signature).
  3. End-state y-distribution vs. thermal Fermi-Dirac target at T_ν,
     normalised to measured Σρ_ss. Residual ρ_ss(y) − ρ_FD(y, T_ν)
     locates the bias.
  4. Verdict line: if the max-residual y-mode bucket coincides with
     the high-y end of the adiabatic window, Suspect 2 (MSW-passage
     adiabatic over-pumping) is corroborated; flat or low-y bias →
     escalate to Suspect 3 (Phase-A thermal-IC inadequacy).

Acceptance (sprint-9 scope-a):
  diagnostic reproduces Point A Neff=4.57772, Yp=0.29894,
  Σρ_ss=29.960 bit-identically — instrumentation is non-perturbative.
"""
import os
import sys
import time
import importlib

import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini

_OUT_NPZ = os.path.join(_WT, "validation/diagnostics/diag_msw_passage_pointA.npz")
_OUT_TXT = os.path.join(_WT, "validation/diagnostics/diag_msw_passage.out")


def _base_flags():
    """Point A config, mirroring diag_energy_balance.py with MSW flag swapped."""
    PRyMini.smallnet_flag = True
    PRyMini.julia_flag = False
    PRyMini.numba_flag = True
    PRyMini.compute_bckg_flag = False
    PRyMini.compute_nTOp_flag = False
    PRyMini.verbose_flag = False
    PRyMini.general_nu_flag = True
    PRyMini.boltzmann_nu_flag = True
    PRyMini.qke_density_matrix_flag = True
    PRyMini.qke_full_ode_flag = True
    PRyMini.qke_ode_etdrk2_flag = True
    PRyMini.massive_electron_flag = False
    PRyMini.n_B_override = 10000  # sprint-7 Phase-5 n_B; sprint-9 target
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-1)) / 2.0  # Point A: sin²2θ=1e-1
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "mirizzi"
    PRyMini.T_boltz_start = 30.0
    PRyMini.T_start = 30.1 * PRyMini.MeV_to_Kelvin
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_v_nunu_active_only = True
    # Sprint-9 MSW instrumentation.
    PRyMini.qke_msw_diag_flag = True
    PRyMini.qke_msw_diag_path = _OUT_NPZ
    PRyMini.qke_msw_diag_pair_idx = 4  # (alpha=1=numu, s=3) — theta_24 channel


_SCALAR_KEYS = ("step", "label", "a", "Tg", "pair_idx", "alpha", "sterile")
_ARRAY_KEYS = (
    "H_aa_0", "H_aa_1", "H_ss_0", "H_ss_1",
    "Re_H_as_0", "Re_H_as_1", "Im_H_as_0", "Im_H_as_1",
    "rho_aa_0", "rho_aa_1", "rho_ss_0", "rho_ss_1",
)


def _rows_to_arrays(hist):
    """Convert list[dict] into dict[str, np.ndarray].

    Scalars become (n_rows,); per-y arrays become (n_rows, Ny).
    """
    out = {}
    for k in _SCALAR_KEYS:
        vals = [row[k] for row in hist]
        if k == "label":
            out[k] = np.array(vals, dtype=object)
        else:
            out[k] = np.asarray(vals)
    for k in _ARRAY_KEYS:
        out[k] = np.stack([np.asarray(row[k]) for row in hist], axis=0)
    return out


def _mask_by_label(arrs, label):
    return arrs["label"] == label


def _fd_thermal(y_grid, T_nu_MeV, a_final, eta):
    """Target Fermi-Dirac distribution at T_ν, expressed on the comoving-y
    grid: f(y) = 1 / (exp(y / (a_final * T_nu_MeV)) + 1). Scaled by eta
    so that Σ f * y^2 matches the measured Σ ρ_ss * y^2.
    """
    x = y_grid / (a_final * T_nu_MeV)
    fd = 1.0 / (np.exp(np.minimum(x, 500.0)) + 1.0)
    return eta * fd


def _resonance_per_y(arrs, y_grid, sector=0):
    """For each y-mode, locate the resonance step and compute:
      T_res(y), step_res(y), |H_as|(y) at crossing, γ_LZ_step(y),
      ρ_ss(y) at end of run, Δρ_ss window integral around crossing.

    Uses pre_step rows in sector `sector` (Point A drives conversion
    symmetrically across sectors — report sector=0 primary).
    """
    pre = _mask_by_label(arrs, "pre_step")
    post = _mask_by_label(arrs, "post_clip")
    H_aa = arrs[f"H_aa_{sector}"][pre]          # (n_steps, Ny)
    H_ss = arrs[f"H_ss_{sector}"][pre]
    Re_H_as = arrs[f"Re_H_as_{sector}"][pre]
    Im_H_as = arrs[f"Im_H_as_{sector}"][pre]
    rho_ss_pre = arrs[f"rho_ss_{sector}"][pre]
    rho_ss_post = arrs[f"rho_ss_{sector}"][post]
    Tg = arrs["Tg"][pre]
    n_steps, Ny = H_aa.shape
    Delta = H_aa - H_ss                          # (n_steps, Ny)
    absH_as = np.sqrt(Re_H_as**2 + Im_H_as**2)

    step_res = np.argmin(np.abs(Delta), axis=0)  # (Ny,)
    step_idx = np.arange(n_steps)

    T_res = Tg[step_res]                          # (Ny,)
    H_as_res = absH_as[step_res, np.arange(Ny)]   # (Ny,)

    # Step-indexed local adiabaticity: γ_step = |H_as|^2 / |dΔ/dstep| at s_res.
    # Uses centered difference where possible; endpoints fall back to one-sided.
    dDelta = np.empty_like(Delta)
    dDelta[1:-1] = 0.5 * (Delta[2:] - Delta[:-2])
    dDelta[0] = Delta[1] - Delta[0]
    dDelta[-1] = Delta[-1] - Delta[-2]
    abs_dDelta = np.abs(dDelta)
    # Protect against exact zero at pathological modes.
    abs_dDelta_at_res = np.maximum(
        abs_dDelta[step_res, np.arange(Ny)], 1.0e-30)
    gamma_step = (H_as_res**2) / abs_dDelta_at_res

    # Δρ_ss window integral: sum (ρ_ss post - ρ_ss pre) over ±N_win steps
    # around step_res per y. N_win = 50 is ~0.5% of the run; catches the
    # local deposition at the crossing without folding in the later bath-
    # pump phase.
    N_win = 50
    n_post_steps = rho_ss_post.shape[0]
    delta_win = np.zeros(Ny)
    for j in range(Ny):
        s0 = max(0, step_res[j] - N_win)
        s1 = min(n_post_steps, step_res[j] + N_win + 1)
        if s1 > s0:
            delta_win[j] = float(np.sum(
                rho_ss_post[s0:s1, j] - rho_ss_pre[s0:s1, j]))

    # End-of-run ρ_ss(y) (last post_clip row).
    rho_ss_end = rho_ss_post[-1] if rho_ss_post.shape[0] > 0 else np.full(Ny, np.nan)

    return {
        "step_res": step_res,
        "T_res": T_res,
        "H_as_res": H_as_res,
        "gamma_step": gamma_step,
        "delta_win": delta_win,
        "rho_ss_end": rho_ss_end,
        "n_steps": n_steps,
    }


def _format_per_y_table(y_grid, res, top_n=20):
    """Print a compact per-y table. If Ny > top_n, show 10 evenly-spaced
    modes plus the 5 with max |delta_win| and the 5 with min |delta_win|."""
    Ny = y_grid.size
    lines = []
    hdr = (f"{'y':>8s}  {'step_res':>8s}  {'T_res[MeV]':>11s}  "
           f"{'|H_as|[eV]':>11s}  {'gamma_step':>11s}  {'delta_win':>11s}  "
           f"{'rho_ss_end':>11s}")
    lines.append("-" * len(hdr))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    if Ny <= top_n:
        idxs = list(range(Ny))
    else:
        # Evenly-spaced 10 + max-5 + min-5, dedup, sort.
        ev = np.linspace(0, Ny - 1, 10).astype(int).tolist()
        order_by_win = np.argsort(np.abs(res["delta_win"]))
        lo = order_by_win[:5].tolist()
        hi = order_by_win[-5:].tolist()
        idxs = sorted(set(ev + lo + hi))
    for j in idxs:
        lines.append(f"{y_grid[j]:>8.3f}  {int(res['step_res'][j]):>8d}  "
                     f"{res['T_res'][j]:>11.4e}  "
                     f"{res['H_as_res'][j]:>11.4e}  "
                     f"{res['gamma_step'][j]:>11.4e}  "
                     f"{res['delta_win'][j]:>+11.4e}  "
                     f"{res['rho_ss_end'][j]:>11.4e}")
    return lines


def _residual_vs_thermal(y_grid, rho_ss_end, T_nu_MeV, a_final):
    """Compare ρ_ss(y) end-of-run to a thermal FD distribution at T_ν,
    matched to the same number integral ∫ y^2 ρ_ss dy. Residual
    ρ_ss(y) − ρ_FD(y) shows y-localisation of the deposition bias.
    """
    x = y_grid / (a_final * T_nu_MeV)
    fd = 1.0 / (np.exp(np.minimum(x, 500.0)) + 1.0)
    num_meas = float(np.sum(y_grid**2 * rho_ss_end))
    num_fd = float(np.sum(y_grid**2 * fd))
    eta = num_meas / num_fd if num_fd > 0 else 0.0
    rho_fd_scaled = eta * fd
    residual = rho_ss_end - rho_fd_scaled
    return rho_fd_scaled, residual, eta


def _bucket_by_y(y_grid, series, n_buckets=5):
    """Average `series` over y into `n_buckets` consecutive buckets.
    Returns (centers, means, maxes, mins)."""
    Ny = y_grid.size
    edges = np.linspace(0, Ny, n_buckets + 1).astype(int)
    centers = np.zeros(n_buckets)
    means = np.zeros(n_buckets)
    maxes = np.zeros(n_buckets)
    mins = np.zeros(n_buckets)
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            centers[b] = float(np.mean(y_grid[lo:hi]))
            means[b] = float(np.mean(series[lo:hi]))
            maxes[b] = float(np.max(series[lo:hi]))
            mins[b] = float(np.min(series[lo:hi]))
    return edges, centers, means, maxes, mins


def _run():
    _base_flags()
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)

    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt = time.time() - t0

    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        rho = c._boltz_rho_final
        sum_ss = float(rho[:, 3, :].sum())

    Neff = res[0]
    Yp = res[4]
    DoH = res[5]
    print(f"Point A run: Neff={Neff:.5f}  Yp={Yp:.5f}  D/H={DoH:.4f}  "
          f"Σρ_ss={sum_ss:.3f}  ({dt:.0f}s)", flush=True)

    dm = c._boltz_dm_solver
    assert dm is not None, "dm_solver not stored on PRyMclass"
    hist = dm._msw_hist
    assert len(hist) > 0, "_msw_hist empty — instrumentation skipped?"
    arrs = _rows_to_arrays(hist)
    y_grid = np.asarray(dm.y_grid).copy()

    # Save raw history + y_grid to .npz.
    save_kwargs = {k: arrs[k] for k in arrs if k != "label"}
    save_kwargs["label"] = arrs["label"].astype(str)
    save_kwargs["y_grid"] = y_grid
    np.savez(_OUT_NPZ, **save_kwargs)

    return Neff, Yp, DoH, sum_ss, dt, arrs, y_grid, c


def _summarise(arrs, y_grid, c, Neff, Yp, DoH, sum_ss, dt):
    """Build the .out summary string from the run history."""
    lines = []
    lines.append("=" * 100)
    lines.append("Stage E.2 sprint 9 scope (a): MSW-passage adiabatic over-pumping diagnostic")
    lines.append(f"Point A (sin²2θ=1e-1, δm²=0.93), w30 projection, n_B=10000  ({dt:.0f}s)")
    lines.append("=" * 100)
    lines.append(f"Point A: Neff={Neff:.5f}  Yp={Yp:.5f}  D/H={DoH:.4f}  Σρ_ss={sum_ss:.3f}")

    n_pre = int(np.sum(_mask_by_label(arrs, "pre_step")))
    n_post = int(np.sum(_mask_by_label(arrs, "post_clip")))
    lines.append(f"snapshots: pre_step={n_pre}  post_clip={n_post}  "
                 f"y-modes={y_grid.size}  pair_idx={int(arrs['pair_idx'][0])}  "
                 f"alpha={int(arrs['alpha'][0])}  sterile={int(arrs['sterile'][0])}")
    lines.append("")

    # Sector 0 resonance analysis (per brief: Point A sweeps symmetrically
    # in both sectors; sector 0 is primary — ν_μ channel).
    res0 = _resonance_per_y(arrs, y_grid, sector=0)
    lines.append("-" * 100)
    lines.append("Sector 0 (neutrino) resonance per y-mode:")
    lines.extend(_format_per_y_table(y_grid, res0, top_n=20))
    lines.append("")

    # Sector 1 (anti-neutrino).
    res1 = _resonance_per_y(arrs, y_grid, sector=1)
    lines.append("-" * 100)
    lines.append("Sector 1 (anti-neutrino) resonance per y-mode:")
    lines.extend(_format_per_y_table(y_grid, res1, top_n=20))
    lines.append("")

    # Bucketed window-integrated deposition Δρ_ss(y). Compares high-y vs. low-y.
    lines.append("-" * 100)
    lines.append("Δρ_ss window-integrated deposition bucketed by y (sector 0):")
    edges, centers, means, maxes, mins = _bucket_by_y(
        y_grid, res0["delta_win"], n_buckets=5)
    lines.append(f"{'bucket':>8s}  {'y_lo':>8s}  {'y_hi':>8s}  "
                 f"{'<Δρss_win>':>12s}  {'max':>12s}  {'min':>12s}")
    lines.append("-" * 74)
    for b in range(5):
        lines.append(f"{b:>8d}  {y_grid[edges[b]]:>8.3f}  "
                     f"{y_grid[min(edges[b+1], y_grid.size - 1)]:>8.3f}  "
                     f"{means[b]:>+12.4e}  {maxes[b]:>+12.4e}  "
                     f"{mins[b]:>+12.4e}")
    lines.append("")

    # End-state ρ_ss(y) vs thermal FD at measured T_ν end-of-run.
    # T_nu at end-of-run ≈ final Tg (Phase B is thermal above decoupling;
    # below decoupling the neutrinos are collisionless, T_ν ≈ T_γ * (4/11)^(1/3)
    # at SM freeze-out, but we use the raw Tg at the last post_clip as the
    # bath temperature reference). a at end-of-run is the last `a` in arrs.
    post = _mask_by_label(arrs, "post_clip")
    a_final = float(arrs["a"][post][-1])
    Tg_final = float(arrs["Tg"][post][-1])
    # Use the sector-0 end-state directly.
    rho_ss_end = res0["rho_ss_end"]
    # Target: bath-temperature T_ν ≈ Tg at end of Phase B.
    rho_fd, residual, eta = _residual_vs_thermal(
        y_grid, rho_ss_end, Tg_final, a_final)
    lines.append("-" * 100)
    lines.append(f"End-state ρ_ss(y) vs. thermal FD target  "
                 f"(Tg_final={Tg_final:.4e} MeV, a_final={a_final:.4e}, "
                 f"eta_FD={eta:.4f})")
    lines.append(f"{'y':>8s}  {'ρ_ss(y)':>12s}  {'ρ_FD(y)':>12s}  "
                 f"{'residual':>12s}  {'resid/ρ_FD':>12s}")
    lines.append("-" * 64)
    Ny = y_grid.size
    pick = np.linspace(0, Ny - 1, 20).astype(int)
    for j in pick:
        rel = residual[j] / rho_fd[j] if rho_fd[j] > 1.0e-30 else float("nan")
        lines.append(f"{y_grid[j]:>8.3f}  {rho_ss_end[j]:>12.4e}  "
                     f"{rho_fd[j]:>12.4e}  {residual[j]:>+12.4e}  "
                     f"{rel:>+12.4e}")
    lines.append("")

    # Residual bucketed.
    edges_r, centers_r, means_r, maxes_r, mins_r = _bucket_by_y(
        y_grid, residual, n_buckets=5)
    lines.append("Residual ρ_ss(y) − ρ_FD(y) bucketed by y:")
    lines.append(f"{'bucket':>8s}  {'y_lo':>8s}  {'y_hi':>8s}  "
                 f"{'<residual>':>12s}  {'max':>12s}  {'min':>12s}")
    lines.append("-" * 74)
    for b in range(5):
        lines.append(f"{b:>8d}  {y_grid[edges_r[b]]:>8.3f}  "
                     f"{y_grid[min(edges_r[b+1], y_grid.size - 1)]:>8.3f}  "
                     f"{means_r[b]:>+12.4e}  {maxes_r[b]:>+12.4e}  "
                     f"{mins_r[b]:>+12.4e}")
    lines.append("")

    # Verdict.
    lines.append("-" * 100)
    lines.append("Sprint-9 scope-(a) verdict:")
    # Compare the highest-y bucket residual-mean magnitude to the lowest-y.
    hi_mean = means_r[-1]
    lo_mean = means_r[0]
    mid_mean = means_r[len(means_r) // 2]
    lines.append(f"  <residual>_hi-y  = {hi_mean:+.4e}")
    lines.append(f"  <residual>_mid-y = {mid_mean:+.4e}")
    lines.append(f"  <residual>_lo-y  = {lo_mean:+.4e}")
    if abs(hi_mean) > 2.0 * abs(lo_mean) and hi_mean > 0:
        verdict = ("Suspect 2 CORROBORATED: high-y residual dominates. "
                   "MSW-passage over-pumping localises to high-y modes — "
                   "consistent with ETDRK2 eigenbasis bias at resonance. "
                   "Escalate to sprint-10 fix at _etdrk2_expm_phi.")
    elif abs(lo_mean) > 2.0 * abs(hi_mean) and lo_mean > 0:
        verdict = ("Unexpected: low-y residual dominates. Not the "
                   "canonical Suspect-2 signature. Investigate collision-"
                   "integral y-weighting before escalating.")
    elif abs(hi_mean) < 1.0e-3 * max(1.0, np.max(np.abs(means_r))):
        verdict = ("Suspect 2 NOT CORROBORATED: residual flat in y. "
                   "Escalate sprint-10 to Suspect 3 (Phase-A thermal-IC "
                   "inadequacy) — build Phase-0 QKE from ~100 MeV.")
    else:
        verdict = ("Ambiguous: residual pattern non-canonical. Review "
                   "full per-y table before deciding sprint-10 direction.")
    lines.append(verdict)
    lines.append("-" * 100)

    return lines


if __name__ == "__main__":
    print("=" * 100)
    print("Stage E.2 sprint 9 scope (a): MSW-passage diagnostic")
    print("Point A (sin²2θ=1e-1, δm²=0.93), w30 projection, n_B=10000")
    print(f"Output: {_OUT_NPZ}")
    print(f"        {_OUT_TXT}")
    print("=" * 100, flush=True)

    Neff, Yp, DoH, sum_ss, dt, arrs, y_grid, c = _run()
    lines = _summarise(arrs, y_grid, c, Neff, Yp, DoH, sum_ss, dt)
    text = "\n".join(lines)
    print(text, flush=True)
    with open(_OUT_TXT, "w") as fh:
        fh.write(text + "\n")
