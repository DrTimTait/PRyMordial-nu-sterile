#!/usr/bin/env python3
"""
Build QED plasma correction tables from NUDEC_BSM v2 data files.

Source:
    Escudero, Jackson, Laine, Sandner, arXiv:2511.04747 (2025),
    https://github.com/MiguelEA/nudec_BSM (v2 branch)

Produces two sets of tables:

  Baseline (O(e^2) + O(e^3), always loaded):
    QED_P_int.txt
    QED_dP_intdT.txt
    QED_d2P_intdT2.txt

  O(e^4) correction (loaded when PRyMini.two_loop_QED_flag = True):
    QED_P_int_e4.txt
    QED_dP_intdT_e4.txt
    QED_d2P_intdT2_e4.txt

Both sets are log-interpolated onto the PRyMordial-nu reference T grid
(descending, 10001 points, 40 MeV down to 5e-3 MeV). Keeping baseline
and correction in the SAME renormalization scheme is essential — mixing
v1 (Escudero 2020) p^(2)+p^(3) with v2 p^(4) produces an inconsistent
combination that blows up Neff by several percent.

Run:
    python build_QED_e4.py

Requires network to download the three NUDEC_BSM v2 .dat files (~1 MB each).
Output files are committed to the repo; rerun only when NUDEC_BSM publishes
updated tables or the reference T grid changes.
"""
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NUDEC_BASE = "https://raw.githubusercontent.com/MiguelEA/nudec_BSM/v2/BasicModules_data/"
# NUDEC v2 columns (all files): T, p^(0), p^(2)non-log, p^(2)log, p^(3), p^(4), p^(5)
# We split into baseline (p^(2)nonlog + p^(2)log + p^(3)) and e4 (p^(4)).
COLS_BASELINE = (2, 3, 4)   # p^(2) non-log + p^(2) log + p^(3)
COL_E4 = 5                  # p^(4)
SPECS = [
    # (remote_filename, baseline_out, e4_out, baseline_label, e4_label)
    ("QED_p_int.dat",
     "QED_P_int.txt",       "QED_P_int_e4.txt",
     "P_int (e^2+e^3) [MeV^4]",
     "P_int (e^4) [MeV^4]"),
    ("QED_dp_dT_int.dat",
     "QED_dP_intdT.txt",    "QED_dP_intdT_e4.txt",
     "dP_int/dT (e^2+e^3) [MeV^3]",
     "dP_int/dT (e^4) [MeV^3]"),
    ("QED_d2p_dT2_int.dat",
     "QED_d2P_intdT2.txt",  "QED_d2P_intdT2_e4.txt",
     "d^2P_int/dT^2 (e^2+e^3) [MeV^2]",
     "d^2P_int/dT^2 (e^4) [MeV^2]"),
]

OUT_HEADER = (
    "# Source: NUDEC_BSM v2 (https://github.com/MiguelEA/nudec_BSM/tree/v2)\n"
    "# Escudero, Jackson, Laine, Sandner, arXiv:2511.04747 (2025)\n"
    "# me = 0.51099895 MeV, alpha = 1/137.035999084\n"
)


def _download(url: str) -> np.ndarray:
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")
    # NUDEC v2 files start with 3 comment lines, then data rows
    return np.loadtxt(raw.splitlines())


def _log_interp_preserving_sign(x_new, x, y):
    """Interpolate y(x) onto x_new using log-|y| + sign in log-x space.

    Handles the p^(n) tables that span ~60 orders of magnitude with mixed
    signs over the T range. Exactly-zero entries are treated as +tiny.
    """
    log_x = np.log(x)
    log_x_new = np.log(x_new)
    sign_y = np.sign(y)
    log_abs_y = np.log(np.maximum(np.abs(y), 1e-300))
    sign_interp = np.interp(log_x_new, log_x, sign_y)
    log_abs_interp = np.interp(log_x_new, log_x, log_abs_y)
    out = np.sign(sign_interp) * np.exp(log_abs_interp)
    # Clamp points outside the source grid to nearest endpoint
    below = log_x_new < log_x[0]
    above = log_x_new > log_x[-1]
    if np.any(below):
        out[below] = y[0]
    if np.any(above):
        out[above] = y[-1]
    return out


def main() -> int:
    # Reference grid: existing PRyMordial-nu QED_P_int.txt (descending T)
    ref_path = os.path.join(HERE, "QED_P_int.txt")
    if not os.path.isfile(ref_path):
        print(f"error: reference grid not found at {ref_path}", file=sys.stderr)
        return 1
    ref_T = np.loadtxt(ref_path)[:, 0]  # descending
    print(f"reference T grid: {len(ref_T)} points, "
          f"{ref_T[0]:.3e} -> {ref_T[-1]:.3e} MeV")
    ref_T_asc = ref_T[::-1]

    for remote_name, baseline_name, e4_name, baseline_label, e4_label in SPECS:
        data = _download(NUDEC_BASE + remote_name)
        # NUDEC v2 has a T=0 row first; drop it
        assert data[0, 0] == 0.0, f"expected T=0 in first row of {remote_name}"
        data = data[1:]
        T_nudec = data[:, 0]              # ascending

        # Baseline: p^(2) non-log + p^(2) log stored as column "p^(2)" below
        # (the existing PRyM_thermo.py loader sums cols 1+2; we keep the same
        # 3-column layout: T, p^(2), p^(3)).
        p2_nonlog = _log_interp_preserving_sign(
            ref_T_asc, T_nudec, data[:, 2])[::-1]
        p2_log = _log_interp_preserving_sign(
            ref_T_asc, T_nudec, data[:, 3])[::-1]
        p3 = _log_interp_preserving_sign(
            ref_T_asc, T_nudec, data[:, 4])[::-1]
        p2_total = p2_nonlog + p2_log

        baseline_path = os.path.join(HERE, baseline_name)
        with open(baseline_path, "w") as f:
            f.write(OUT_HEADER)
            f.write("# Baseline: O(e^2) non-log + O(e^2) log summed into col 2;\n")
            f.write("# O(e^3) in col 3. Loader sums col 2 + col 3.\n")
            f.write(f"# T (MeV)           {baseline_label} p^(2)   p^(3)\n")
            for T, p2v, p3v in zip(ref_T, p2_total, p3):
                f.write(f"{T:.6E}        {p2v:+.6E}        {p3v:+.6E}\n")
        print(f"  wrote {baseline_path}: {len(ref_T)} rows "
              f"[p^(2) range {p2_total.min():+.3e} to {p2_total.max():+.3e}]")

        # O(e^4) correction — single column p^(4) at col 1
        p4 = _log_interp_preserving_sign(
            ref_T_asc, T_nudec, data[:, COL_E4])[::-1]
        e4_path = os.path.join(HERE, e4_name)
        with open(e4_path, "w") as f:
            f.write(OUT_HEADER)
            f.write(f"# T (MeV)           {e4_label}\n")
            for T, v in zip(ref_T, p4):
                f.write(f"{T:.6E}        {v:+.6E}\n")
        print(f"  wrote {e4_path}: {len(ref_T)} rows "
              f"[range {p4.min():+.3e} to {p4.max():+.3e}]")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
