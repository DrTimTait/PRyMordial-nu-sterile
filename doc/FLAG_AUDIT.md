# PRyMini flag audit

Findings and recommendations from a review of `PRyM/PRyM_init.py` flag
coupling, triggered by a Stage E.2 diagnostic that hit a silent-null
failure mode (hypothesis A test yielded indistinguishable results across
three variants because `_offdiag_collision_gain` is linear in the
coherence input and we'd turned PMNS off, leaving active-active
coherences identically zero).

This is not a bug report — the code is correct. It's an ergonomics /
observability report: which flag combinations produce quiet,
hard-to-debug outcomes, and what inexpensive guards would catch them.

## What went wrong this session

`diag_as_gain.py` (first PMNS-off pass) produced dNeff = +0.2928 for
all three hypothesis variants to 4 decimal places. My first read was
"the monkey-patch isn't landing in the code path". After building a
separate probe to confirm the patch WAS reaching `_assemble_collision_N`
(stub called twice per invocation), I realised the patched function is
linear in `rho_offdiag`: with PMNS off (θ_12=θ_13=θ_23=0) and thermal
initial conditions, active-active coherences never develop, so the
gain term is identically zero regardless of what we do to it. The
diagnostic was physically unobservable in that regime.

Cost: ~1 h of wasted compute and ~15 min of confused debugging before
the physics implication landed.

## Categories of flag-coupling issues

### 1. Documented implication chains — partially enforced

Several flags' docstrings state "Implies X = True". `PRyMclass.__init__`
(lines 32-49 of `PRyM_main.py`) does auto-enable some of this chain:

- `qke_density_matrix_flag = True` → sets `boltzmann_nu_flag = True`
- `boltzmann_nu_flag = True` → sets `general_nu_flag = True`,
  `compute_bckg_flag = True`, and flips `compute_nTOp_flag = False`
  (unless `NP_nTOp_flag` says otherwise)

Gaps:

- `qke_full_ode_flag = True` is NOT cross-checked against
  `qke_density_matrix_flag = True`. If a user sets the Stage D flag
  without the QKE flag, nothing propagates.
- `qke_ode_etdrk2_flag = True` similarly not checked against
  `qke_full_ode_flag = True`.
- The auto-enable is silent. A user who explicitly sets
  `boltzmann_nu_flag = False` while also setting
  `qke_density_matrix_flag = True` gets their explicit False
  overridden with no warning.

The call site in `PRyM_main.py:207` is
`if PRyMini.general_nu_flag and PRyMini.boltzmann_nu_flag:` — after
the auto-enable, both are True when QKE is on. Fine functionally,
silent on conflict.

### 2. Flags that are ignored in certain modes

| Flag | Ignored when |
|---|---|
| `nu_nubar_symmetric_flag` | `qke_density_matrix_flag = True` ("QKE path is always full 6-species regardless of this flag", line 191-193) |
| `nlo_weak_rate_scale` | `nlo_weak_flag = False` |
| `xi_nue_init, xi_numu_init, xi_nutau_init` | `sterile_flag = False` ("Only honored when sterile_flag=True; ignored otherwise", line 235) |

No warning when the user sets these to non-default values in the
ignoring mode. A user who sets `xi_nue_init = 0.1` to seed an
asymmetry and forgets `sterile_flag = True` gets silent no-op.

### 3. Physics regimes where flags become non-load-bearing

These are the silent-null failure modes. Setting the flag to a legal
value produces a config in which some downstream calculation is
linear-in-zero and therefore zero regardless of any other knob.

| Config | Consequence |
|---|---|
| `theta_12 = theta_13 = theta_23 = 0` (our PMNS-off case) | Active-active off-diagonals have no Hamiltonian driving; `_offdiag_collision_gain` returns zero regardless of its inputs. Variants of the gain function are indistinguishable. |
| `sterile_flag = True` with all of `theta_14 = theta_24 = theta_34 = 0` | Sterile is uncoupled; ρ_ss stays at 0; Neff is bit-identical to `sterile_flag = False`. |
| `Dm2_41 = 0.0` with `sterile_flag = True` | Same as above effectively (no MSW, vacuum term vanishes). |
| `numba_flag = True` without numba installed | Graceful fallback (good — explicit try/except in PRyM_thermo), but the flag name misleadingly suggests hard requirement. |

### 4. Defaults tuned for a narrow regime

| Flag | Default | Breaks when |
|---|---|---|
| `n_B_override = None` (→ max(2000, 2·n_sampling)) | 2000 steps | `T_boltz_start` is pushed to ≫ 5 MeV: the collision rate grows as T^5 but steps are spread log-uniform in *a*, so CFL violations accumulate (observed at T_boltz_start = 60 MeV: Neff = 13.66, unphysical). |
| `T_boltz_start = 5.0` MeV | 5 MeV | Any literature comparison that integrates from higher T (Hannestad: 60 MeV). |
| `y_max_boltz = 100.0` MeV, `Ny_boltz = 100` | linear | Gauss-Laguerre cross-checks or any IR-weighted quantity may want log-spaced. |

### 5. String-typed flags with no validation

| Flag | Values |
|---|---|
| `qke_damping_formula` | `"symmetric"`, `"mirizzi"`, `"gariazzo"` (last raises NotImplementedError) |
| `nu_oscillation_method` | `"collision_mixing"`, `"relaxation"` |
| `rates_dir` | set from `nacreii_flag` — not user-facing but still stringly-typed |

A typo (e.g. `"mirizz"`) only fails at the call site (`_compute_D_pair_matrix` raises ValueError). Downstream raise is fine but delayed — ideally caught at import time.

## Recommendations, prioritized

1. **Add a `validate_configuration()` function** in `PRyM_init.py`,
   called once at the top of `PRyMclass.__init__`. Minimum scope:
   enforce the implication chains in Category 1 (raise `ValueError`
   with a clear actionable message). Optionally warn on Category 2
   "silently ignored" combinations. Low risk, small diff, high value.

2. **Validate string-typed flags at import time.** For each
   string-typed flag, check membership in the allowed set and raise
   immediately. Catches typos before any compute runs.

3. **Emit a "physics configuration" summary when
   `verbose_flag = True`.** Print the active-physics list (PMNS on/off,
   sterile on/off, non-thermal ν on/off, driver = ETDRK2 / Strang /
   symmetric) at PRyMclass start. Turns Category 3 silent-nulls into
   explicit "your config is X, expect consequence Y" notices.

4. **Warn when window flags are pushed outside the tested range.**
   `T_boltz_start > 10 MeV` or `T_start > 60 MeV` with default `n_B`
   should log a warning that step-size policy may fail CFL. Ideally
   auto-scale `n_B` proportional to log(T_start / T_boltz_end).

5. **Consider promoting the qke_* chain to a single enum.**
   `qke_driver ∈ {"sigl_raffelt", "strang", "ode_etd1", "ode_etdrk2"}`
   replaces the four boolean flags with documented implications. Makes
   illegal combinations syntactically impossible. Larger diff — park
   for a standalone refactor rather than bundling here.

6. **Long-term: structured config object.** Replace the flat module-
   level globals with a dataclass (`PRyMConfig`) whose `__post_init__`
   validates and whose instances are passed explicitly. This is a
   weeks-of-work change with user-facing-API implications; mentioning
   only for completeness.

## What this session will land

A single `PRyM_init.validate_configuration()` helper called from
`PRyMclass.__init__` that covers the CHEAP items:

- **String-flag validation** (recommendation 2): `qke_damping_formula`
  must be one of `{"symmetric", "mirizzi", "gariazzo"}`;
  `nu_oscillation_method` must be one of `{"collision_mixing",
  "relaxation"}`. Raise `ValueError` with an actionable message at
  `PRyMclass()` construction rather than at first call-site.
- **Silently-ignored knob warnings** (recommendation covering
  Category 2): warn when `xi_nue_init / xi_numu_init / xi_nutau_init`
  are non-zero while `sterile_flag = False`; warn when
  `nlo_weak_rate_scale != 1.003` while `nlo_weak_flag = False`.
- **QKE chain gap** (recommendation 1 gap): warn when
  `qke_ode_etdrk2_flag = True` but `qke_full_ode_flag = False`, and
  when `qke_full_ode_flag = True` but `qke_density_matrix_flag = False`.
  Warn rather than raise to preserve current "best effort" behaviour.
- **PMNS-off verbose summary** (Category 3): when `verbose_flag` is
  True, print an explicit "PMNS active-active mixing: OFF" line if
  `theta_12 = theta_13 = theta_23 = 0`. Catches the exact silent-null
  that burned this session.

Larger items (recommendations 4-6) are parked — they're refactors
that want their own review cycle.
