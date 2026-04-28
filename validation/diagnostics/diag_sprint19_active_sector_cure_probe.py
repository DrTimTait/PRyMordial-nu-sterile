"""Stage E.2 sprint 19 part 2 cure verification (gate 4).

Re-runs the sprint-19 closure config at both production n_B=12000 and reduced
n_B=3500, but with the new `qke_post_phaseB_clamp_flag` enabled. Confirms the
cure brings the active sector back to near-SM (Neff ≤ 4.0, Yp ≤ 0.255) at
production n_B while preserving the sterile-sector closure δNeff_ss ∈
[0.02, 0.10].

Outputs to `_active_sector_cure_probe.{out,npz}` (separate from part-1
`_active_sector_probe.{out,npz}` so the part-1 reference data is preserved).

Pre-cure references (from sprint-19 part-1 .out):
  Production n_B=12000 (no cure): Neff=417.1947  Yp=0.36186  δNeff_ss=0.0542
  Reduced    n_B=3500  (no cure): Neff=3.9090    Yp=0.24850  δNeff_ss=0.0304

Post-cure expectations (from offline replay of trace data):
  Production n_B=12000 (cure):   Neff~2.88  δNeff_ss~0.054 (in band)
  Reduced    n_B=3500  (cure):   Neff~2.40  δNeff_ss~0.030 (in band)

Cost: ~95 min wall-clock (Run A ~73 min + Run B ~22 min, sequential).
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


REF_PRODUCTION_PRECURE = {"n_B": 12000, "Neff": 417.1947, "Yp": 0.36186,
                           "delta_neff_ss": 0.0542}
REF_REDUCED_PRECURE = {"n_B": 3500, "Neff": 3.9090, "Yp": 0.24850,
                        "delta_neff_ss": 0.0304}

# Gate 4 targets per cure_design §5.
ACTIVE_NEFF_MAX = 4.0
ACTIVE_YP_MAX = 0.255
STERILE_BAND = (0.02, 0.10)


def _set_flags(n_B):
    """Sprint-19 closure config + cure flag enabled."""
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
    PRyMini.qke_etdrk4_flag = False
    PRyMini.qke_ode_etdrk2_flag = True
    PRyMini.qke_lsoda_driver_flag = False
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = True
    PRyMini.Dm2_41 = 0.93
    PRyMini.theta_14 = 0.0
    PRyMini.theta_24 = np.arcsin(np.sqrt(1.0e-4)) / 2.0
    PRyMini.theta_34 = 0.0
    PRyMini.xi_nue_init = 0.0
    PRyMini.xi_numu_init = 0.0
    PRyMini.xi_nutau_init = 0.0
    PRyMini.qke_damping_formula = "symmetric"
    PRyMini.qke_v_nunu_active_only = True
    PRyMini.qke_damping_scale = 1.0
    PRyMini.qke_v_nc_scale = 1.0
    PRyMini.qke_v_thermal_scale = 1.0
    PRyMini.qke_v_nunu_scale = 1.0
    PRyMini.qke_phase0_flag = False
    PRyMini.T_boltz_start = 100.0
    PRyMini.T_start = 105.0 * PRyMini.MeV_to_Kelvin
    PRyMini.n_B_override = int(n_B)
    PRyMini.n_B_phase0_override = 0
    PRyMini.qke_active_probe_flag = False
    PRyMini.qke_post_phaseB_trace_flag = False
    PRyMini.qke_post_phaseB_clamp_flag = True  # the cure


def _delta_neff_ss_from_rho(rho_final):
    if rho_final is None or rho_final.shape[1] < 4:
        return None
    Ny = rho_final.shape[2]
    y_max = PRyMini.y_max_boltz
    dy = y_max / Ny
    y_grid = np.linspace(dy / 2.0, y_max - dy / 2.0, Ny)
    y3 = y_grid ** 3
    rho_ss = rho_final[:, 3, :]
    rho_e = rho_final[:, 0, :]
    m3_ss = float((rho_ss * y3[None, :]).sum())
    m3_e = float((rho_e * y3[None, :]).sum())
    if m3_e <= 0.0:
        return None
    return m3_ss / m3_e


def _run_one(n_B, label):
    print(f"\n--- {label}: n_B = {n_B} ---", flush=True)
    _set_flags(n_B)
    import PRyM.PRyM_thermo as PRyMthermo
    importlib.reload(PRyMthermo)
    import PRyM.PRyM_boltzmann as PRyMboltz
    importlib.reload(PRyMboltz)
    import PRyM.PRyM_main as PRyMmain
    importlib.reload(PRyMmain)

    t0 = time.time()
    c = PRyMmain.PRyMclass()
    res = c.PRyMresults()
    dt_run = time.time() - t0

    Neff = float(res[0])
    Yp = float(res[4])
    delta_neff_ss = None
    sum_ss = 0.0
    if (hasattr(c, "_boltz_rho_final") and c._boltz_rho_final is not None
            and c._boltz_rho_final.shape[1] >= 4):
        sum_ss = float(c._boltz_rho_final[:, 3, :].sum())
        delta_neff_ss = _delta_neff_ss_from_rho(c._boltz_rho_final)

    print(f"    Neff={Neff:.4f}  Yp={Yp:.5f}  sum_ss(raw)={sum_ss:.4f}  "
          f"δNeff_ss={delta_neff_ss}  ({dt_run:.0f}s)", flush=True)

    return {
        "n_B": int(n_B), "label": label,
        "Neff": Neff, "Yp": Yp, "sum_raw": sum_ss,
        "delta_neff_ss": (float(delta_neff_ss)
                          if delta_neff_ss is not None else float("nan")),
        "wall_clock_s": float(dt_run),
    }


if __name__ == "__main__":
    header = [
        "=" * 110,
        "Stage E.2 sprint 19 part 2 cure verification (gate 4)",
        "  Config: damping='symmetric', active_only=True, qke_phase0_flag=False, "
        "qke_post_phaseB_clamp_flag=True (CURE ON)",
        "  Phase B from T=100 MeV to T=0.005 MeV",
        "  Pre-cure references (sprint-19 part 1):",
        f"    Production n_B={REF_PRODUCTION_PRECURE['n_B']}: "
        f"Neff={REF_PRODUCTION_PRECURE['Neff']:.4f}  Yp={REF_PRODUCTION_PRECURE['Yp']:.5f}  "
        f"δNeff_ss={REF_PRODUCTION_PRECURE['delta_neff_ss']:.4f}",
        f"    Reduced n_B={REF_REDUCED_PRECURE['n_B']}: "
        f"Neff={REF_REDUCED_PRECURE['Neff']:.4f}  Yp={REF_REDUCED_PRECURE['Yp']:.5f}  "
        f"δNeff_ss={REF_REDUCED_PRECURE['delta_neff_ss']:.4f}",
        f"  Gate 4 targets: Neff ≤ {ACTIVE_NEFF_MAX}, Yp ≤ {ACTIVE_YP_MAX}, "
        f"δNeff_ss ∈ {STERILE_BAND}",
        "=" * 110,
    ]
    for line in header:
        print(line, flush=True)

    t_total_0 = time.time()
    prod = _run_one(REF_PRODUCTION_PRECURE["n_B"], "Run A (production)")
    red = _run_one(REF_REDUCED_PRECURE["n_B"], "Run B (reduced)")
    t_total = time.time() - t_total_0

    def _verdict(r):
        n_ok = r["Neff"] <= ACTIVE_NEFF_MAX
        y_ok = r["Yp"] <= ACTIVE_YP_MAX
        s_ok = (not np.isnan(r["delta_neff_ss"])
                and STERILE_BAND[0] <= r["delta_neff_ss"] <= STERILE_BAND[1])
        return n_ok and y_ok and s_ok, n_ok, y_ok, s_ok

    summary = ["", "=" * 110, "Sprint 19 part 2 cure verification result",
               "=" * 110]
    all_pass = True
    for r in (prod, red):
        ok, n_ok, y_ok, s_ok = _verdict(r)
        all_pass = all_pass and ok
        marker = "PASS" if ok else "FAIL"
        summary.append(
            f"  [{marker}] {r['label']:25s} n_B={r['n_B']}: "
            f"Neff={r['Neff']:.4f} ({'✓' if n_ok else '✗'})  "
            f"Yp={r['Yp']:.5f} ({'✓' if y_ok else '✗'})  "
            f"δNeff_ss={r['delta_neff_ss']:.4f} ({'✓' if s_ok else '✗'})  "
            f"sum_raw={r['sum_raw']:.4f}  ({r['wall_clock_s']:.0f}s)")
    summary += [
        "",
        f"  Gate 4 verdict: {'PASS' if all_pass else 'FAIL'}",
        "",
        f"  Pre/post comparison (production n_B=12000):",
        f"    Neff: {REF_PRODUCTION_PRECURE['Neff']:.4f} → {prod['Neff']:.4f}",
        f"    Yp:   {REF_PRODUCTION_PRECURE['Yp']:.5f} → {prod['Yp']:.5f}",
        f"  Pre/post comparison (reduced n_B=3500):",
        f"    Neff: {REF_REDUCED_PRECURE['Neff']:.4f} → {red['Neff']:.4f}",
        f"    Yp:   {REF_REDUCED_PRECURE['Yp']:.5f} → {red['Yp']:.5f}",
        "",
        f"  n_B convergence: prod Neff / red Neff = "
        f"{prod['Neff']/max(red['Neff'], 1e-12):.3f}  (target ~1.0)",
        "",
        f"  Total wall-clock: {t_total:.0f}s ({t_total/60:.1f} min)",
    ]
    text = "\n".join(summary)
    print(text, flush=True)

    out_txt = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_active_sector_cure_probe.out")
    with open(out_txt, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(text + "\n")

    out_npz = os.path.join(_WT,
        "validation/diagnostics/diag_sprint19_active_sector_cure_probe.npz")
    np.savez_compressed(
        out_npz,
        production_n_B=np.int64(prod["n_B"]),
        production_Neff=np.float64(prod["Neff"]),
        production_Yp=np.float64(prod["Yp"]),
        production_sum_raw=np.float64(prod["sum_raw"]),
        production_delta_neff_ss=np.float64(prod["delta_neff_ss"]),
        production_wall_clock_s=np.float64(prod["wall_clock_s"]),
        reduced_n_B=np.int64(red["n_B"]),
        reduced_Neff=np.float64(red["Neff"]),
        reduced_Yp=np.float64(red["Yp"]),
        reduced_sum_raw=np.float64(red["sum_raw"]),
        reduced_delta_neff_ss=np.float64(red["delta_neff_ss"]),
        reduced_wall_clock_s=np.float64(red["wall_clock_s"]),
        gate4_pass=np.bool_(all_pass),
    )
    print(f"\nSaved: {out_txt}", flush=True)
    print(f"Saved: {out_npz}", flush=True)
