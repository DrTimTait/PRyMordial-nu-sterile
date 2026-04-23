"""Stage E.2 sprint 8: active-sterile energy-balance diagnostic.

Per-step N/E/C accounting for the three (alpha, s) 2x2 blocks inside
evolve_step_ode_etdrk2 at Point A (sin^2(2theta) = 1e-1, dm^2 = 0.93) of
the Hannestad+2012 benchmark, extended window T_boltz_start = 30 MeV,
projection on, n_B = 10000 (the sprint-7 Phase-5 configuration that
shows dNeff = +1.57 at Σρ_ss ≈ 30 — "thermal population, wrong energy").

Metrics recorded at three sub-step positions per time step under the
uniform midpoint rule (dy constant on the Ny y_grid):
  N_{s,αs} = dy * Σ_y y^2 * (rho_aa + rho_ss)     number
  E_{s,αs} = dy * Σ_y y^3 * (rho_aa + rho_ss)     energy-weighted
  C_{s,αs} = dy * Σ_y y^3 * |rho_as|              coherence sidebar

Positions: "pre_step", "post_corrector", "post_clip".

Acceptance (see doc/STAGE_E2_SPRINT8_BRIEF.md):
  dE/E(pre_step -> post_clip) > +1e-3 (positive drift)   -> Candidate A
  dE/E(pre_step -> post_clip) < -1e-3 (negative drift)   -> Candidate B
  |dE/E(pre_step -> post_clip)| < 1e-4 (zero drift)      -> halt, scope (a)
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

_OUT_NPZ = os.path.join(_WT, "validation/diagnostics/diag_energy_balance_pointA.npz")
_OUT_TXT = os.path.join(_WT, "validation/diagnostics/diag_energy_balance.out")


def _base_flags():
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
    PRyMini.n_B_override = 10000  # sprint-7 Phase-5 n_B; brief target
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-1)) / 2.0  # Point A: sin²2θ = 1e-1
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
    # Stage E.2 sprint 8: turn on energy-balance instrumentation.
    PRyMini.qke_energy_diag_flag = True
    PRyMini.qke_energy_diag_path = _OUT_NPZ


def _rows_to_arrays(hist):
    """Convert list[dict] to dict[str, np.ndarray] keyed by column name."""
    keys = list(hist[0].keys())
    out = {}
    for k in keys:
        vals = [row[k] for row in hist]
        if k == "label":
            out[k] = np.array(vals, dtype=object)
        else:
            out[k] = np.array(vals)
    return out


def _drift_summary(arrs):
    """Compute drift summary from row arrays. Returns list of lines to print."""
    labels = arrs["label"]
    steps = arrs["step"]
    is_pre = labels == "pre_step"
    is_post_corr = labels == "post_corrector"
    is_post_clip = labels == "post_clip"

    n_pre = int(is_pre.sum())
    n_post_corr = int(is_post_corr.sum())
    n_post_clip = int(is_post_clip.sum())

    lines = []
    lines.append(f"sub-step rows: pre_step={n_pre}  post_corrector={n_post_corr}  post_clip={n_post_clip}")
    lines.append(f"step range: {int(steps.min())} .. {int(steps.max())}")
    lines.append("")

    # Per (sector, alpha) drift: compare first pre_step to last post_clip.
    # Intra-step mean drift: average over all steps of (post_clip - pre_step)/pre_step.
    header = f"{'sector':>6s}  {'pair':>6s}  {'N0':>12s}  {'N_end':>12s}  {'dN/N0':>12s}  " \
             f"{'E0':>12s}  {'E_end':>12s}  {'dE/E0':>12s}  {'C_max':>12s}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in (0, 1):
        for alpha in (0, 1, 2):
            N_key = f"N_{s}_{alpha}s"
            E_key = f"E_{s}_{alpha}s"
            C_key = f"C_{s}_{alpha}s"
            # First pre_step row (initial conditions) and last post_clip row (end state).
            N_pre = arrs[N_key][is_pre]
            N_clip = arrs[N_key][is_post_clip]
            E_pre = arrs[E_key][is_pre]
            E_clip = arrs[E_key][is_post_clip]
            C_all = arrs[C_key]
            N0 = N_pre[0] if len(N_pre) else float('nan')
            N_end = N_clip[-1] if len(N_clip) else float('nan')
            E0 = E_pre[0] if len(E_pre) else float('nan')
            E_end = E_clip[-1] if len(E_clip) else float('nan')
            dN = (N_end - N0) / N0 if N0 != 0 else float('nan')
            dE = (E_end - E0) / E0 if E0 != 0 else float('nan')
            C_max = float(np.max(np.abs(C_all))) if len(C_all) else float('nan')
            lines.append(f"{s:>6d}  {f'({alpha},s)':>6s}  "
                         f"{N0:>12.4e}  {N_end:>12.4e}  {dN:>+12.4e}  "
                         f"{E0:>12.4e}  {E_end:>12.4e}  {dE:>+12.4e}  "
                         f"{C_max:>12.4e}")
    lines.append("")

    # Intra-step mean drift (pre_step -> post_clip per step), averaged over steps
    # that exist in both sets. Gives an average per-step drift rate.
    lines.append("Intra-step average drift (pre_step -> post_clip, per step):")
    lines.append(f"{'sector':>6s}  {'pair':>6s}  "
                 f"{'<dN/N>/step':>14s}  {'<dE/E>/step':>14s}")
    lines.append("-" * 54)
    for s in (0, 1):
        for alpha in (0, 1, 2):
            N_key = f"N_{s}_{alpha}s"
            E_key = f"E_{s}_{alpha}s"
            N_pre = arrs[N_key][is_pre]
            N_clip = arrs[N_key][is_post_clip]
            E_pre = arrs[E_key][is_pre]
            E_clip = arrs[E_key][is_post_clip]
            n_steps = min(len(N_pre), len(N_clip))
            if n_steps == 0:
                continue
            dN_per = (N_clip[:n_steps] - N_pre[:n_steps]) / np.where(
                N_pre[:n_steps] != 0, N_pre[:n_steps], 1.0)
            dE_per = (E_clip[:n_steps] - E_pre[:n_steps]) / np.where(
                E_pre[:n_steps] != 0, E_pre[:n_steps], 1.0)
            lines.append(f"{s:>6d}  {f'({alpha},s)':>6s}  "
                         f"{float(np.mean(dN_per)):>+14.4e}  "
                         f"{float(np.mean(dE_per)):>+14.4e}")
    lines.append("")
    return lines


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
    hist = dm._energy_hist
    assert len(hist) > 0, "_energy_hist empty -- instrumentation skipped?"
    arrs = _rows_to_arrays(hist)

    # Save raw history to .npz for post-processing.
    save_kwargs = {k: v for k, v in arrs.items() if k != "label"}
    save_kwargs["label"] = arrs["label"].astype(str)
    np.savez(_OUT_NPZ, **save_kwargs)

    # Print + save drift summary.
    lines = _drift_summary(arrs)
    return Neff, Yp, DoH, sum_ss, dt, lines


if __name__ == "__main__":
    print("=" * 96)
    print("Stage E.2 sprint 8 Phase 1: active-sterile energy-balance diagnostic")
    print("Point A (sin²2θ=1e-1, δm²=0.93), w30 projection, n_B=10000")
    print(f"Output: {_OUT_NPZ}")
    print(f"        {_OUT_TXT}")
    print("=" * 96, flush=True)

    Neff, Yp, DoH, sum_ss, dt, lines = _run()

    summary = [
        "",
        "=" * 96,
        "Stage E.2 sprint 8 Phase 1: active-sterile energy-balance diagnostic",
        f"Point A (sin²2θ=1e-1, δm²=0.93), w30 projection, n_B=10000  ({dt:.0f}s)",
        "=" * 96,
        f"Point A: Neff={Neff:.5f}  Yp={Yp:.5f}  D/H={DoH:.4f}  Σρ_ss={sum_ss:.3f}",
        "",
    ]
    summary.extend(lines)

    # Scope-(a) / scope-(b) verdict.
    summary.append("-" * 96)
    summary.append("Sprint-8 Phase-1 verdict (energy-weighted total drift dE/E0):")
    summary.append("  > +1e-3  -> Suspect-1 Candidate A (double-counting in L add-back)")
    summary.append("  < -1e-3  -> Suspect-1 Candidate B (missing ρ_ss back-reaction)")
    summary.append("  |...| < 1e-4 -> zero drift; halt, scope (a), escalate to sprint 9")
    summary.append("-" * 96)

    text = "\n".join(summary)
    print(text, flush=True)
    with open(_OUT_TXT, "w") as fh:
        fh.write(text + "\n")
