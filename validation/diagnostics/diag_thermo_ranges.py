"""Stage E.2 sprint-2 (scope b): thermo interpolant range probe (Suspect 2).

doc/STAGE_E2_SPRINT2_BRIEF.md Suspect 2: "Phase B's Froustey entropy
equation is out of range". `PRyMthermo.spl(Tg)` depends on the QED
correction interpolators `PofT`, `dPdT`, `d2PdT2` (PRyM_thermo.py:40-65).
If those are extrapolated at T >> table top, the initial
`a_boltz_ini = (spl_ref / spl(Tg))^(1/3)` is wrong and Phase B inherits
that.

Pre-inspection showed `PRyMrates/thermo/QED_P_int.txt` +
`QED_dP_intdT.txt` + `QED_d2P_intdT2.txt` all top out at T = 40 MeV.
The baseline interp1ds use `fill_value="extrapolate"` -> 60 MeV is
garbage. The O(e^4) tables already use `fill_value=0.0`
(PRyM_thermo.py:49-58).

This script prints:
  - table top / bottom for the baseline three QED tables
  - P_QED(T), dPdT_QED(T), d2PdT2_QED(T), rho_e(T), p_e(T), spl(T)
    at T in {1, 5, 10, 30, 40, 60, 100} MeV
  - the fractional magnitude P_QED(T) / P_photon(T), and the same
    for dPdT vs drho_g_dT. At T >> m_e the QED correction should be
    an O(alpha/pi) ~ 0.002 fractional piece on top of the photon gas.
  - the denominator `drho_g_dT + drho_e_dT + T*d2PdT2` from
    PRyM_main.py:252 to confirm it stays safely nonzero even with
    `d2PdT2` zero-clamped above 40 MeV.

Runtime: seconds. Output is documentation / confirmation, not an
investigation.
"""
import os
import sys
import numpy as np

_WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _WT)
os.chdir(_WT)

import PRyM.PRyM_init as PRyMini

PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
PRyMini.numba_flag = False  # avoid JIT warmup for a seconds-level probe
PRyMini.verbose_flag = False

import PRyM.PRyM_thermo as PRyMthermo


def _table_range(interp, name):
    x = interp.x
    return f"{name:20s}  T in [{x.min():.5f}, {x.max():.2f}] MeV ({len(x)} nodes)"


print("=" * 88)
print("Stage E.2 sprint-2 (scope b): thermo interpolant range probe (Suspect 2)")
print("=" * 88)
print()
print("Tabulated ranges (baseline QED tables):")
print("  " + _table_range(PRyMthermo._PofT_base, "PofT_base"))
print("  " + _table_range(PRyMthermo._dPdT_base, "dPdT_base"))
print("  " + _table_range(PRyMthermo._d2PdT2_base, "d2PdT2_base"))
if PRyMini.two_loop_QED_flag:
    print()
    print("Tabulated ranges (O(e^4) QED tables, two_loop_QED_flag=True):")
    print("  " + _table_range(PRyMthermo._PofT_e4, "PofT_e4"))
    print("  " + _table_range(PRyMthermo._dPdT_e4, "dPdT_e4"))
    print("  " + _table_range(PRyMthermo._d2PdT2_e4, "d2PdT2_e4"))
else:
    print()
    print("(two_loop_QED_flag = False -> e^4 tables not loaded)")

print()
print("Point samples:")
print(f"  {'T (MeV)':>8s}  {'PofT':>12s}  {'dPdT':>12s}  {'d2PdT2':>12s}  "
      f"{'rho_e':>12s}  {'p_e':>12s}  {'spl':>12s}")
print("  " + "-" * 90)
for T in [1.0, 5.0, 10.0, 30.0, 40.0, 60.0, 100.0]:
    print(f"  {T:>8.2f}  {float(PRyMthermo.PofT(T)):>12.4e}  "
          f"{float(PRyMthermo.dPdT(T)):>12.4e}  "
          f"{float(PRyMthermo.d2PdT2(T)):>12.4e}  "
          f"{PRyMthermo.rho_e(T):>12.4e}  "
          f"{PRyMthermo.p_e(T):>12.4e}  "
          f"{PRyMthermo.spl(T):>12.4e}")

print()
print("Fractional magnitude of QED corrections vs photon gas:")
print(f"  {'T (MeV)':>8s}  {'P_QED / P_gamma':>18s}  {'dP/dT_QED / drho_g/dT':>25s}")
print("  " + "-" * 60)
for T in [1.0, 5.0, 10.0, 30.0, 40.0, 60.0, 100.0]:
    P_gamma = PRyMthermo.rho_g(T) / 3.0  # photon pressure = rho/3
    P_QED = float(PRyMthermo.PofT(T))
    dPdT_QED = float(PRyMthermo.dPdT(T))
    drhog_dT = float(PRyMthermo.drho_g_dT(T))
    print(f"  {T:>8.2f}  {P_QED / P_gamma:>+18.4e}  "
          f"{dPdT_QED / drhog_dT:>+25.4e}")

print()
print("Phase A dTgdt denominator  drho_g_dT + drho_e_dT + T*d2PdT2  "
      "(PRyM_main.py:252):")
print(f"  {'T (MeV)':>8s}  {'drho_g_dT':>14s}  {'drho_e_dT':>14s}  "
      f"{'T*d2PdT2':>14s}  {'sum':>14s}")
print("  " + "-" * 76)
for T in [1.0, 5.0, 10.0, 30.0, 40.0, 60.0, 100.0]:
    a = float(PRyMthermo.drho_g_dT(T))
    b = PRyMthermo.drho_e_dT(T)
    c = T * float(PRyMthermo.d2PdT2(T))
    print(f"  {T:>8.2f}  {a:>14.4e}  {b:>14.4e}  {c:>+14.4e}  {a+b+c:>14.4e}")

print()
print("=" * 88)
print("Expected at T > 40 MeV (extrapolation region):")
print("  - PofT / dPdT / d2PdT2 are linearly extrapolated (NOT physical).")
print("  - P_QED / P_gamma should trend toward O(alpha/pi) ~ 2e-3 at T >> m_e,")
print("    not inflate with T. Deviations indicate extrapolation garbage.")
print("  - drho_g_dT dominates the dTgdt denominator by a huge margin (safe).")
print("=" * 88)
