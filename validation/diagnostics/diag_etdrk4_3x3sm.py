"""Stage E.2 sprint 16 gate 3: ETDRK4 vs ETDRK2 on 3x3 SM.

Sanity-check that the new Krogstad ETDRK4 driver produces results
consistent with ETDRK2 in the regime where ETDRK2 is known correct
(Standard Model, no sterile, no MSW resonance). Compares Neff and Yp
across the two drivers on identical SM flags. Pass criterion (per
sprint 16 brief): Neff matches to <1e-4, Yp <1e-5; ETDRK4 wall-clock
<5x ETDRK2.
"""
import os
import sys
import time
import importlib

import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)


def _base_flags():
    import PRyM.PRyM_init as PRyMini
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
    PRyMini.massive_electron_flag = False
    PRyMini.sterile_flag = False
    PRyMini.qke_phase0_flag = False
    PRyMini.T_boltz_start = 5.0
    # Gate-3 is a driver sanity check (does ETDRK4 produce SM-consistent Neff/Yp
    # without crashing). The default n_B=2000 makes a full BBN run >10 min per
    # driver; reduce to 500 to keep the sanity check tractable. This is enough
    # to exercise the ETDRK4 dispatch path through the full BBN integration
    # while keeping wall-clock under ~5 min per driver. The Hannestad Point C
    # gate (gate 5) uses production n_B=10000.
    PRyMini.n_B_override = 500


def _run(driver):
    """driver in {'etdrk2', 'etdrk4'}."""
    import PRyM.PRyM_init as PRyMini
    _base_flags()
    PRyMini.qke_etdrk4_flag = (driver == 'etdrk4')
    PRyMini.qke_ode_etdrk2_flag = (driver == 'etdrk2')
    PRyMini.qke_lsoda_driver_flag = False

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
    return res, dt


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("Stage E.2 sprint 16 gate 3: ETDRK4 vs ETDRK2, 3x3 SM", flush=True)
    print("=" * 80, flush=True)

    print("\n[1/2] ETDRK2 baseline...", flush=True)
    res2, dt2 = _run('etdrk2')
    Neff2, Yp2, DoH2 = res2[0], res2[4], res2[5]
    print(f"      Neff={Neff2:.6f}  Yp={Yp2:.6f}  D/H={DoH2:.4f}  ({dt2:.1f}s)",
          flush=True)

    print("\n[2/2] ETDRK4...", flush=True)
    res4, dt4 = _run('etdrk4')
    Neff4, Yp4, DoH4 = res4[0], res4[4], res4[5]
    print(f"      Neff={Neff4:.6f}  Yp={Yp4:.6f}  D/H={DoH4:.4f}  ({dt4:.1f}s)",
          flush=True)

    dNeff = abs(Neff4 - Neff2)
    dYp = abs(Yp4 - Yp2)
    speed = dt4 / dt2 if dt2 > 0 else float('inf')

    print("\n" + "=" * 80, flush=True)
    print(f"  |dNeff| = {dNeff:.2e}  (target <1e-4)", flush=True)
    print(f"  |dYp|   = {dYp:.2e}  (target <1e-5)", flush=True)
    print(f"  speed   = {speed:.2f}x  (target <5x)", flush=True)
    print("=" * 80, flush=True)

    pass_neff = dNeff < 1e-4
    pass_yp = dYp < 1e-5
    pass_speed = speed < 5.0
    if pass_neff and pass_yp and pass_speed:
        print("  GATE 3: PASS", flush=True)
    else:
        print(
            f"  GATE 3: FAIL "
            f"(Neff={'OK' if pass_neff else 'FAIL'}, "
            f"Yp={'OK' if pass_yp else 'FAIL'}, "
            f"speed={'OK' if pass_speed else 'FAIL'})",
            flush=True)

    out_txt = os.path.join(_WT, "validation/diagnostics/diag_etdrk4_3x3sm.out")
    with open(out_txt, "w") as fh:
        fh.write(f"ETDRK2: Neff={Neff2:.6f} Yp={Yp2:.6f} D/H={DoH2:.4f} ({dt2:.1f}s)\n")
        fh.write(f"ETDRK4: Neff={Neff4:.6f} Yp={Yp4:.6f} D/H={DoH4:.4f} ({dt4:.1f}s)\n")
        fh.write(f"|dNeff|={dNeff:.2e}  |dYp|={dYp:.2e}  speed={speed:.2f}x\n")
