# Reproducibility package: Experiments 2 and 5

This folder reproduces the two numerical illustrations used in the **“Movement Stability of D-Optimal Sensor Placement Under Projector Perturbations.”**

- **Experiment 2:** exact finite-instance movement certificates, comparing `P_act` with `P_cert`.
- **Experiment 5:** complete objective landscape and exact distance-margin curve for one deterministic representative random-geometric instance.

The package contains only the code and outputs needed for those two paper experiments. It does **not** contain the older exploratory experiments.

## 1. Environment

Recommended: Python 3.10.x.

Create a clean virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run the paper experiments

From this folder:

```bash
python mmsf_experiments.py \
  --profile paper \
  --experiments 2,5 \
  --output paper_outputs
```

The paper profile uses exactly two baseline seeds per graph family in Experiment 2.

Expected Experiment 2 totals after the `P_max` correction used in the manuscript:

- attempted perturbation cases: **1,224**
- cases satisfying both sharp smallness conditions: **1,189**
- genuinely nontrivial certificates (`P_cert < P_max`): **1,035**
- cases with positive actual movement: **33**
- integer-radius sharp cases (`P_cert = P_act`): **561**
- certificate implication failures: **0**

The selected Experiment 5 representative should report:

- zero-based perturbed edge: **(3, 5)**
- fractional change: **+0.10**
- `P_act = 1`
- `P_cert = 4`
- `P_max = 5`

The exact floating-point values are written to `exp5_summary.csv` rather than hard-coded in the README.

## 3. Generate the two paper figures

After the experiment command finishes:

```bash
python make_paper_figures.py \
  --input paper_outputs \
  --output figures
```

This writes:

```text
figures/Fig1.pdf
figures/Fig1.png
figures/Fig2.pdf
figures/Fig2.png
```

Use the **PDF** files in LaTeX because they are vector graphics.

Suggested manuscript inclusions:

```latex
\includegraphics[width=0.78\linewidth]{figures/Fig1.pdf}
```

and

```latex
\includegraphics[width=\linewidth]{figures/Fig2.pdf}
```

## 4. Main output files

`paper_outputs/exp2_certificates.csv` contains one row per attempted one-edge perturbation. Important columns include:

- `smallness_pass`
- `P_act`
- `P_cert`
- `P_max`
- `nontrivial_certificate`
- `integer_radius_sharp`
- `certificate_slack_hops`
- theorem quantities `rho`, `kappa_star`, `sigma_star`, `eta1`, `eta2`, `B_t`

`paper_outputs/exp2_summary.csv` contains an overall row (`family=ALL`) and one row per graph family.

`paper_outputs/exp5_landscape.csv` contains every feasible configuration in the representative instance.

`paper_outputs/exp5_margin_curve.csv` contains the exact integer-radius margin curve `Gamma_t(P)` and the markers used in Fig. 2.

`paper_outputs/numerical_accuracy.csv` reports numerical separation/slack values alongside the tolerances used by the implementation.

`paper_outputs/validation_log.csv` records implementation consistency checks.

`paper_outputs/manifest.json` records the runtime environment and tolerance constants.

## 5. Numerical conventions

The implementation uses:

- positive-definiteness tolerance: `1e-11`
- optimizer absolute tolerance: `1e-9`
- optimizer relative tolerance: `1e-9`
- near-zero objective-gap tolerance: `1e-10`
- inequality comparison tolerance: `1e-12`

All graph distances and bottleneck matching radii are exact integers. Every `m`-subset is enumerated; here `m=2`, so the bottleneck matching calculation is also exact by direct enumeration of the two bijections.

## 6. Important definition: nontrivial certificate

For a fixed reference configuration,

```text
P_max = max_S d_match(S_t^*, S).
```

At `P_max`, the far family is empty and the distance margin is `+infinity`. Therefore this package defines a certificate as genuinely nontrivial exactly when

```text
P_cert < P_max.
```

This is intentionally **not** defined using the graph diameter.

## 7. Interpretation

These computations do not statistically validate the theorem. They illustrate the proved sufficient certificate, check the implementation on finite exhaustive instances, and quantify the conservatism of the resulting integer movement radius.
