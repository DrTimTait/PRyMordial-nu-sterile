# PRyMordial-nu regression tests

Pytest-style regression tests that freeze the BBN outputs for known
configurations. Reference values were captured on main commit `1d6e05e`
(Task 2 complete; Bennett+2021 recommended Neff = 3.0440 matched to
10⁻⁴ in QKE mode).

## Running

```bash
# Fast tests only (~15 s): thermal path, general-nu FD, O(e⁴) variant
pytest -m "not slow"

# Full suite including Boltzmann + QKE (~4-5 minutes total)
pytest

# Specific test
pytest tests/test_regression.py::test_mode5_qke_density_matrix -v
```

## What's frozen

| Test | Mode | Runtime | Neff expected | Tolerance |
|------|------|--------:|--------------:|----------:|
| `test_mode1_standard_thermal` | thermal | ~5 s | 3.044389 | 1e-5 |
| `test_mode2_general_nu_fd` | general nu (FD) | ~3 s | 3.044389 | 1e-5 |
| `test_mode6_standard_plus_oe4` | thermal + O(e⁴) | ~5 s | 3.044338 | 2e-5 |
| `test_mode3_boltzmann_diagonal` [slow] | n=3 Boltzmann | ~100 s | 3.0398 | 3e-4 |
| `test_mode5_qke_density_matrix` [slow] | full QKE | ~130 s | 3.0445 | 3e-4 |

Yp and D/H are also asserted with appropriate tolerances (see
`test_regression.py` for exact values).

## When tests fail

A failing regression test usually means one of:

1. **Intentional physics change** that shifted the numerics — update the
   expected values after confirming the shift is correct. Log the shift
   in the commit message.
2. **Numerical-precision drift** beyond the tolerance — widen the
   tolerance or investigate the source (e.g., a numba version change,
   scipy ODE solver bump).
3. **A real bug** — investigate and fix, the tests have caught something.

## Adding new tests

Follow the pattern in `test_regression.py`: call `_reset_flags()`, set
the flags of interest, call `_run_mode()`, and assert with
`pytest.approx(..., abs=...)`. For slow Boltzmann/QKE tests add the
`@pytest.mark.slow` decorator.
