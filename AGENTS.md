# AGENTS.md — VeriAero-CEGIS v3.2 Permanent Operating Rules

**Project:** VeriAero-CEGIS v3.2 — *2027 Presidential SI Challenge (High School Trailblazer Track)*  
**Primary Architecture Reference (Authoritative Single Source of Truth):** Google Doc *"VeriAero-CEGIS v3.2 — Hybrid Generative + Active-Verification Architecture Reference"* (`https://docs.google.com/document/d/164wnufHsDWYd_gkCkLGPMfK_u_BedSRPSKcKkYVyadI/edit`)  
**Primary Compute Host:** **C2-WSL2** (Windows 11 + WSL2 Ubuntu 24.04, Intel Core i9-10850K 10C/20T, 32 GB RAM, NVIDIA RTX 3080 10 GB VRAM)  
**GitHub Single Source of Truth:** `https://github.com/aqswde40/VeriAero-CEGIS.git`  
**Hard Cutoffs:**
- **Technical Freeze:** December 9, 2026
- **Team Registration:** December 18, 2026 (5:00 p.m. ET)
- **State Submission:** December 23, 2026 (5:00 p.m. ET)

Every AI coding agent session working in this repository MUST follow all rules in this document without exception.

---

## 1. Student Knowledge Profile & Pedagogical Contract (MANDATORY)

### Student Background
- The student is an **11th grader** with strong foundations in:
  - **AP Calculus BC** (derivatives, integration, separable/numeric ODEs, slope fields, Taylor intuition)
  - **AP Physics C: Mechanics** (Newton's laws, torque & static equilibrium, rotational inertia, work-energy, beam bending intuition)
  - **AP Physics C: Electricity & Magnetism** (DC circuits, internal resistance, $P = VI$, $I^2R$ Joule heating, thermal energy balance)
  - **AP Computer Science A** (object-oriented Java/Python design, interfaces/protocols, arrays/lists/maps, unit testing)
- The student has **NO** graduate-level mathematics, **NO** beyond-AP software engineering background, and **NO** prior AI-building experience.

### What AI MUST Automate Aggressively
To hit the December 23 deadline cleanly without wasting time on low-learning-value chores, AI MUST aggressively automate:
1. Environment provisioning and dependency locking (`uv`, Python 3.12, CUDA PyTorch wheels).
2. Compiling and smoke-testing external native binaries (`XFoil 6.99` into `bin/xfoil_patched`, `SU2 8.6.0` into `bin/SU2_CFD`).
3. Directory/package boilerplate scaffolding and repetitive data parsing.
4. Unit-test harness wiring (`pytest`), plot/table rendering scripts, and diagnostics formatting.
5. Long unattended or overnight simulation and training sweeps (with `HostProvenance` and `BudgetLedger` logging).

### What AI MUST NOT Turn Into a Black Box
Never silently implement complex core AI or physics logic without giving the student a concise, AP-level intuitive mental model:
- **Conditional Flow Matching (Stage C)** -> Explain as an **AP Calculus BC slope field / ODE** ($\frac{dx}{dt} = v_\theta(x, t)$) that pushes random noise along smooth trajectories toward feasible aircraft designs.
- **Battery Voltage Sag & $I^2R$ Thermal Runaway** -> Explain using **AP Physics C: E&M DC circuits** ($V_{\text{term}} = V_{\text{oc}} - I R_{\text{pack}}$; solving $P_{\text{req}} = (V_{\text{oc}} - I R_{\text{pack}})I$ via the quadratic formula, where discriminant $< 0$ means physical voltage collapse!) plus $I^2R$ Joule heating feeding a first-order ODE.
- **Wing Root Bending & 6-Rotor OEI Trim** -> Explain using **AP Physics C: Mechanics** torque/force equilibrium ($\sum F_z = W$, $\sum \tau_x = \sum \tau_y = \sum \tau_z = 0$) and beam cross-section second moment of area $I$.
- **Modular Contracts (`ProblemSchema`, `DesignIR`, `DisciplineOracle`)** -> Explain using **AP Computer Science A interfaces** (`DisciplineOracle` is an interface contract separating *what* physics an oracle checks from *how* Stage C/B search over numbers).

### Mandatory Interactive Checkpoints
At the end of **every major module** (`Tier L`, `Tier H1`, `Stage 0`, `Stage C`, `Stage B`, `Ablation`), **STOP** and present:
1. **Plain-English 5-bullet intuitive summary** ("What this actually does and why it matters").
2. **One short hands-on student verification task** (running a quick sanity script, inspecting a plot/number, or tweaking one physical input to see what breaks) so the student builds real engineering intuition and can defend every part of the system live to judges.
3. **A pre-formatted row** for the student to review and append to `docs/ai_disclosure_log.csv` with exact columns:
   `date_utc,tool_name_and_version,task_category,exact_scope_of_assistance,student_independent_verification,defect_caught_by_student,git_commit_or_artifact_ref`
   *(Nondisclosure risks disqualification; full transparency boosts the 15-pt Process and Learning score.)*

---

## 2. Locked VeriAero-CEGIS v3.2 Architectural Guardrails

### 2.1 Domain-Agnostic Modular Kernel
- **Stage C** (`DeepSets Catalog-Hole CFM + Limited-Interval Dampened CFG`) and **Stage B** (`Multi-TR Augmented Tchebycheff ParEGO-TuRBO` on Tier-H1 residuals over snapped features `s_snap`) MUST **NEVER** hard-code UAV physics formulas, wing geometry equations, or fixed vector indices.
- All communication between optimization/generative modules and physical/catalog domains MUST go strictly through:
  - `ProblemSchema` (26-D variable bounds, kinds, 3 Pareto objectives `verified_range_km`, `domestic_bom_cost_usd`, `cold_oei_margin` with HV ref points, constraints with `m_safe_scale`, scenario defaults)
  - `DesignIR` (`schema_version`, `x_norm`, `s_snap`, decoded `params`, `selected_skus` across 4 categories `cell`, `lift_motor`, `esc`, `cruise_motor`, `topology`, 18-param Kulfan `airfoil_cst`, `.json`/`.dat` export)
  - `SubsystemCatalog` + `CompatibilityGraph` (origin-audited SKUs with `ALLIED` / `PRC` / `US_DESIGNED_PRC_MFG` / `OTHER`, `price_source`, separated `KDE Direct XF`, `Vertiq 81-08 G2` 12S integrated-ESC cap, >=4 allied 14S–16S ESCs, zero Cobra SKUs)
  - `DisciplineOracle` (`tier` in `L`, `H1`, `H2`, future `H3`, `approx_cost_s`, `provides_outputs`, supporting sub-oracles)

### 2.2 Honest Multi-Fidelity Physics (Zero Tier-L Strawman)
- **Tier L (`src/veriaero/physics/tier_l_uav.py`):** All fast closed-form and analytical physics lives here:
  - Closed 25.00 kg mass balance (no artificial clamping on perturbed designs)
  - True trapezoidal wing $\text{MAC} = \frac{2}{3} c_r \frac{1+\lambda+\lambda^2}{1+\lambda} = 0.3144\text{ m}$ at baseline
  - Complete parasitic drag buildup + `NeuralFoil 0.3.3` on 18-parameter Kulfan CST (`8 upper + 8 lower + LE weight + TE=0.002`, `r_LE/c >= 0.008` from 4-D frozen UIUC PCA) + 3D induced drag ($L/D \approx 12.45$, still-air baseline range $\approx 151.2\text{ km}$)
  - Euler-Bernoulli root beam bending stress under separated `+3.0g` maneuver vs dynamic Pratt discrete gust (`FS = 1.5`, $\sigma_{\text{allow}} = 350\text{ MPa}$)
  - DCIR + Arrhenius $0\,^\circ\text{C}$ resistance scaling ($E_a = 25\text{ kJ/mol}$, sensitivity band $20\text{--}30\text{ kJ/mol}$) + exact quadratic voltage sag
  - Two-phase cruise-preheated steady-state ($T_{\text{cruise,eq}}$ for $h \in [5.0, 10.0]\text{ W/(m}^2\cdot\text{K)}$) feeding the 90-second OEI burst thermal ODE at conservative cylindrical 21700 $c_p = 850\text{ J/(kg}\cdot\text{K)}$ with live Arrhenius $R_{\text{pack}}(T)$
  - 6-rotor twin-boom static OEI trim convex QP (`cvxpy`) across all 6 single-rotor failures and $\pm 5\%\text{ MAC}$ CG shifts
- **Tier H1:** Strictly expensive coupled physics (`OpenAeroStruct 2.12.1` CFRP `wingbox` FEM with `with_viscous=False`, `with_wave=False`, cached `Problem.setup()`, separated `+3.0g` maneuver vs Pratt gust, `FS=1.5` + 8 Gauss-Legendre spanwise patched `XFoil 6.99` stations, `N=200`, `N_crit=9.0`, free + tripped `x_tr=0.05`).
- **Tier H2:** Strictly 24–36 steady-state `SU2 8.6.0` RANS runs (`TIME_DOMAIN= NO`, `SOLVER= RANS`, `KIND_TURB_MODEL= SST`, `KIND_TRANS_MODEL= LM`, `L2ROE`, `BOUNDED_SCALAR`, `WEIGHTED_LEAST_SQUARES`, structured quad C-grid, dual low-Re pass gate `STRICT_PASS` / `LSB_DAMPED_PASS`).

### 2.3 Tri-State Solver Logging & Quarantine Firewall
- Every `DisciplineOracle.evaluate()` MUST return a tri-state `SolverStatus`:
  - `RESOLVED_PASS`: converged finite evaluation satisfying all physical & compatibility constraints (`g(x) <= tol`)
  - `RESOLVED_VIOLATION`: converged finite evaluation violating at least one physical or compatibility constraint (`g(x) > tol`)
  - `UNRESOLVED_NUMERICAL`: solver crash, timeout, non-convergence, or non-finite output (`NaN`/`Inf`)
- **NEVER** count `UNRESOLVED_NUMERICAL` as a physical failure (`RESOLVED_VIOLATION`), and **ALWAYS** debit every oracle call (including `UNRESOLVED_NUMERICAL`) on `BudgetLedger`.
- **Import Firewall:** Code inside `src/` MUST **NEVER** import anything from `eval_quarantine/`. Enforced by `tests/test_quarantine_firewall.py`.

### 2.4 Single-Host Timing Lock & December Scope Freeze
- 100% of locked confirmatory 5-arm ablations ($n = 12$ scenarios) MUST execute on owned **C2-WSL2** (RTX 3080 10 GB VRAM, 8 pinned CPU workers, `OPENBLAS_NUM_THREADS=1`) with automatic `HostProvenance` logging on every run.
- **NEVER** build Post-December Tier H3 items (`OpenVSP`/`VSPAERO`, `CalculiX`, `JSBSim`) before December 23, 2026.
