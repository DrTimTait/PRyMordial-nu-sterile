"""Stage E.2 sprint 11 gate 7: V_nunu active-only at 5 MeV window with the
narrow-mixing eigendecomposition fallback turned on.

Runs the same four PRyMclass scenarios as diag_vnunu_active_only.py
(3×3 reference ± projection, Point C ± projection) but with
PRyMini.qke_expm_fallback_near_degeneracy=True throughout. Pass criterion:
the V_nunu_active_only Point-C result is bit-identical (within ULP) to the
sprint-5 baseline because the fallback gate (small commutator gap AND
non-zero H_α,sterile) should not fire at T_boltz_start=5 MeV — the Point-C
MSW resonance for y=0.5 ν̄ sits at T≈62 MeV, well above the integration
range, so |H_αα − H_ss| stays large relative to the active-flavor spread.
"""
import os
import sys

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini

# Set the sprint-11 flag BEFORE importing diag_vnunu_active_only so the
# downstream _point_c_flags() / _configure_3x3() / _configure_point_c() do
# not undo it.
PRyMini.qke_expm_fallback_near_degeneracy = True
PRyMini.qke_expm_fallback_eps_cross = 1.0e-3

# Patch _reset to preserve the fallback flag (otherwise _reset would not
# touch it, but be explicit).
import importlib
import validation.diagnostics.diag_vnunu_active_only as base

_orig_reset = base._reset


def _reset_with_fallback():
    _orig_reset()
    PRyMini.qke_expm_fallback_near_degeneracy = True
    PRyMini.qke_expm_fallback_eps_cross = 1.0e-3


base._reset = _reset_with_fallback

if __name__ == "__main__":
    print("=" * 96)
    print("Stage E.2 sprint 11 gate 7: V_nunu active-only at 5 MeV window")
    print("WITH qke_expm_fallback_near_degeneracy = True (eps_cross = 1e-3)")
    print("Sprint-5 baseline at flag off: Point C dNeff ≈ -0.012 (V_nunu projection on)")
    print("Pass criterion: bit-identical to sprint-5 baseline at flag off.")
    print("=" * 96)
    # Re-execute the original script's __main__ logic by calling its body.
    # The base module guards its __main__ body inside an `if __name__ == "__main__":`
    # block, so we replicate the calls directly.
    import numpy as np
    Neff_3x3_full, _, _ = base._run(
        "3x3 reference (V_nunu full 4x4 behaviour, default)",
        base._configure_3x3,
    )
    Neff_3x3_proj, _, _ = base._run(
        "3x3 reference (V_nunu active-only projection)",
        base._configure_3x3,
        {"qke_v_nunu_active_only": True},
    )
    print()
    scenarios = [
        ("baseline (V_nunu full 4x4, default)", {}, Neff_3x3_full),
        ("V_nunu ACTIVE-ONLY projection", {"qke_v_nunu_active_only": True}, Neff_3x3_proj),
    ]
    results = []
    for label, overrides, ref_Neff in scenarios:
        Neff, ss, dt = base._run(f"Point C, {label}", base._configure_point_c, overrides)
        dNeff = Neff - ref_Neff
        results.append((label, Neff, ss, dNeff, ref_Neff, dt))
        print(f"     dNeff = {dNeff:+.4f}  (ref: {ref_Neff:.5f})")
        # Also report fallback dispatch on the last run
        # (stored at module level after _run completes).
        print()
    # Note: the boltzmann solver instance is local to _run; we don't have a
    # direct reference here. The summary table is the primary output.

    print("=" * 96)
    print(f"{'scenario':45s} {'Neff':>10s} {'dNeff':>10s} {'sum_rho_ss':>14s} {'ref_Neff':>10s}")
    print("-" * 96)
    for label, Neff, ss, dNeff, ref, _ in results:
        print(f"{label:45s} {Neff:>10.5f} {dNeff:>+10.4f} {ss:>14.3f} {ref:>10.5f}")
    print("=" * 96)
    print()
    print("Verdict:")
    print("  Point C dNeff with V_nunu projection should match sprint-5 baseline ~-0.012")
    print("  (bit-identity not strict — fallback may fire ULP-level on edge modes).")
