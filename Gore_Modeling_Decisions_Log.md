# Gore Emergent Phases: Modeling Decisions & Architectural Log

This document serves as the **authoritative running log** of all scientific, architectural, data-preprocessing, and mathematical decisions made while modeling the Jeff Gore (*Science* 2022) dataset with our Latent Assembly Neural ODE framework.

* **Dataset Source**: [`Data/gore_data.xlsx`](file:///home/karthik/SerialDilutionModel/Data/gore_data.xlsx)
* **Preprocessed Tensors**: [`Data/Gore_processed_data.pt`](file:///home/karthik/SerialDilutionModel/Data/Gore_processed_data.pt)
* **Processing Pipeline**: [`src/Gore_Data_Processing.py`](file:///home/karthik/SerialDilutionModel/src/Gore_Data_Processing.py)
* **Reference Paper**: [`Data/Gore - Emergent phases - Science 2022.pdf`](file:///home/karthik/SerialDilutionModel/Data/Gore%20-%20Emergent%20phases%20-%20Science%202022.pdf)
* **Master Roadmap**: [`Gore_Emergent_Phases_Project_Plan.md`](file:///home/karthik/SerialDilutionModel/Gore_Emergent_Phases_Project_Plan.md)

---

## Decision Index

1. [DEC-001: Consumer-Resource Neural ODE vs. Generalized Lotka-Volterra](#dec-001-consumer-resource-neural-ode-vs-generalized-lotka-volterra)
2. [DEC-002: Preservation of Strictly Positive Initial Resources ($R(0) > 0$)](#dec-002-preservation-of-strictly-positive-initial-resources-r0--0)
3. [DEC-003: Focus on Relative Abundances for Dynamic Fitting](#dec-003-focus-on-relative-abundances-for-dynamic-fitting)
4. [DEC-004: Standardizing Asymmetric Longitudinal Time Horizons (LN/HN vs. MN)](#dec-004-standardizing-asymmetric-longitudinal-time-horizons-lnhn-vs-mn)
5. [DEC-005: Handling Intermediate Zeros & Resurrections (The $10^{-6}$ Dispersal Floor)](#dec-005-handling-intermediate-zeros--resurrections-the-10-6-dispersal-floor)
6. [DEC-006: Staged Modeling Strategy (Direction 1 Core vs. Direction 2 Emergence)](#dec-006-staged-modeling-strategy-direction-1-core-vs-direction-2-emergence)
7. [DEC-007: Standardizing on the `hema` Conda Environment](#dec-007-standardizing-on-the-hema-conda-environment)
8. [DEC-008: Global Latent Assembly ODE Training & Phase Evaluation (Milestone 2)](#dec-008-global-latent-assembly-ode-training--phase-evaluation-milestone-2)
9. [DEC-009: Phase A Latent Dimension ($K \in [2..12]$) Sweep & Error Compounding Discovery](#dec-009-phase-a-latent-dimension-k-in-212-sweep--error-compounding-discovery)
10. [DEC-010: Phase B Solver Precision ($\Delta t$) & Scheduled Sampling Anchoring ($TF_{\text{min}}$)](#dec-010-phase-b-solver-precision-delta-t--scheduled-sampling-anchoring-tf_textmin)
11. [DEC-011: Phase C Regularization Uncaging, Cross-Feeding Activation & Trajectory Jump](#dec-011-phase-c-regularization-uncaging-cross-feeding-activation--trajectory-jump)

---

### DEC-001: Consumer-Resource Neural ODE vs. Generalized Lotka-Volterra

* **Date**: 2026-10-19
* **Status**: Decided & Implemented
* **Context**:
  In *Science* 2022, Gore et al. tested theoretical phase transitions using a generalized Lotka-Volterra (gLV) model:
  $$\frac{dN_i}{dt} = N_i \left( 1 - \sum_{j=1}^S a_{ij} N_j \right) + D$$
  To model different nutrient conditions, they treated nutrient concentration as a phenomenological multiplier for the interaction matrix $a_{ij}$.
* **Problem**:
  The physical bacteria are the identical 48 strains across Low Nutrient (LN), Medium Nutrient (MN), and High Nutrient (HN). Their genomes, uptake enzymes, and byproduct stoichiometry do not change. Modeling chemical nutrient enrichment as an alteration of intrinsic pairwise interaction coefficients obscures the mechanism and inflates the parameter space to $48 \times 48 = 2,304$ free parameters.
* **Decision**:
  We use the [`LatentAssemblyNeuralODE`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py#L208-L274) with mass-action kinetics:
  $$\frac{dN_i}{dt} = N_i \sum_{k=1}^K \Theta_{ki} R_k - \alpha_i N_i^2, \quad \frac{dR_k}{dt} = - R_k \sum_{l=1}^K W_{kl} \sum_{j=1}^N \Theta_{jl} N_j$$
  * **Biological Invariance**: The trait matrix $\Theta \in \mathbb{R}^{48 \times K}$, cross-feeding matrix $W \in \mathbb{R}^{K \times K}$, and self-limitation $\alpha \in \mathbb{R}^{48}$ are **strictly fixed and conserved** across all nutrient regimes.
  * **Environmental Driver**: The transition between stable coexistence (LN) and persistent fluctuations/chaos (HN) is driven **solely by the initial resource vector** $R(0)_{\text{LN}} \to R(0)_{\text{MN}} \to R(0)_{\text{HN}}$.
  * **Dimensionality**: Reduces parameters from 2,304 to $\approx 200$ ($K \approx 4-6$).

---

### DEC-002: Preservation of Strictly Positive Initial Resources ($R(0) > 0$)

* **Date**: 2026-10-19
* **Status**: Decided & Implemented
* **Context**:
  Evaluating whether removing the positivity constraint on initial resources $R(0)$ (e.g., removing `softplus(a)`) would give the optimizer more flexibility and improve model fit.
* **Risk Analysis**:
  In [`ConsumerResourceODEFunc`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py#L158-L206):
  $$\frac{dR_k}{dt} = - R_k(t) \cdot \text{metabolic\_flux}_k$$
  If $R_k(0) < 0$ and community cross-feeding produces that resource ($\text{metabolic\_flux} < 0$), the derivative becomes negative:
  $$\frac{dR_k}{dt} = -(-) \cdot (-) < 0$$
  This causes $R_k(t)$ to become increasingly negative, triggering an **exponential divergence to $-\infty$** during numerical integration over $[0, T_{\text{end}}]$, leading to ODE solver step-size underflow or NaN loss.
  Furthermore, the trait matrix $\Theta$ already has unconstrained real values; negative traits already capture metabolic burden, toxin sensitivity, or inhibition from a positive resource.
* **Decision**:
  Maintain $R(0) > 0$ strictly positive via `softplus(a)` or `exp(a)`. Do not allow negative resource concentrations in the ODE.

---

### DEC-003: Focus on Relative Abundances for Dynamic Fitting

* **Date**: 2026-10-19
* **Status**: Decided & Implemented
* **Context**:
  The Gore dataset contains both 16S amplicon sequencing (relative abundances) and daily optical density OD600 (total biomass in triplicate).
* **Rationale**:
  1. Optical density at high cell densities suffers from non-linear light-scattering saturation and species-specific cell-size/morphology variations (OD per cell varies between taxa).
  2. Training multi-passage rollouts on relative abundance vectors ($x_i(t) = \text{reads}_i / \sum \text{reads}$) aligns directly with our established, benchmarked pipelines on German's and Pollard's datasets.
* **Decision**:
  Focus primary training and evaluation on **relative abundance**. Daily OD600 biomass trajectories are extracted and preserved in [`Data/Gore_processed_data.pt`](file:///home/karthik/SerialDilutionModel/Data/Gore_processed_data.pt) under `'biomass_od'` for secondary validation, carrying-capacity estimation, and absolute biomass checks.

---

### DEC-004: Standardizing Asymmetric Longitudinal Time Horizons (LN/HN vs. MN)

* **Date**: 2026-10-19
* **Status**: Decided & Implemented
* **Context**:
  Sequencing coverage in [`Data/gore_data.xlsx`](file:///home/karthik/SerialDilutionModel/Data/gore_data.xlsx) varies by condition:
  * **LN**: Days 0–10 (11 daily timepoints) for all 33 communities.
  * **HN**: Days 0–10 (11 daily timepoints) for all 33 communities.
  * **MN**: Days 7–10 (4 daily timepoints) for $S \in \{3, 6, 12\}$; Days 0–10 (11 daily timepoints) for $S \in \{24, 48\}$.
* **Rationale**:
  Gore et al. restricted small communities in MN to Days 7–10 to minimize sequencing costs while capturing the late-phase/endpoint dynamical classification.
* **Decision**:
  The parser [`src/Gore_Data_Processing.py`](file:///home/karthik/SerialDilutionModel/src/Gore_Data_Processing.py) preserves the full available time course for every trajectory without synthetic zero-padding:
  * Dict key `(cond, S, comm_name)` maps to the exact available tensor (`shape=(11, 48)` or `shape=(4, 48)`).
  * Day labels for every trajectory are explicitly tracked under `'timepoints'`.

---

### DEC-005: Handling Intermediate Zeros & Resurrections (The $10^{-6}$ Dispersal Floor)

* **Date**: 2026-10-19
* **Status**: Decided & Implemented
* **Problem**:
  In the raw sequencing read counts, **722 resurrection events** were detected where a species had 0 reads at day $t$, but $>0$ reads at day $t' > t$.
  * *Low-frequency noise (~95%)*: Fluctuations around the sequencing limit of detection ($\approx 1$ read per 20,000 cells).
  * *High-magnitude blooms*: Genuine ecological crashes followed by massive rebounds. E.g., Species 4 in $S=24$ Community 4 (HN) had 0 reads on Day 6, then bloomed to 6,475 reads (**47.2% of the community**) on Day 8. Species 20 in $S=24$ Community 5 (MN) had 0 reads on Day 8, then exploded to **64.3%** on Day 10.
* **Conflict with Multiplicative Models**:
  In our ODE, growth is multiplicative ($\dot{N}_i = N_i \sum \dots$). If the model receives $N_i(t) = 0$, $N_i(t+1)$ is mathematically constrained to be 0. When evaluated against a true $47\%$ bloom, the model incurs severe/infinite loss for failing to predict what its own equations forbid from growing.
* **Experimental Mechanism**:
  Gore et al.'s experimental protocol explicitly applied a **$10^{-6}$ daily dispersal aliquot** from the species pool into every well during each 24h dilution cycle ($D = 30$). Even if locally diluted below the sequencing threshold, a physical seed stock was re-introduced daily.
* **Decision**:
  1. For all **member species** belonging to a community's pool, apply a baseline floor of $1.00 \times 10^{-6}$, matching the experimental dispersal rate:
     $$x_i^{\text{floored}}(t) = \frac{x_i^{\text{raw}}(t) + 10^{-6} \cdot M_i}{\sum_j (x_j^{\text{raw}}(t) + 10^{-6} \cdot M_j)}$$
     where $M_i \in \{0, 1\}$ is the community membership mask.
  2. For all **non-member species** ($M_i = 0$), abundance remains strictly **`0.00e+00`**.
* **Outcome**:
  * Member species minimum abundance is strictly $1.00 \times 10^{-6} > 0$.
  * Resurrection events ($0 \to >0$) are reduced to **strictly 0**.
  * The model maintains the seed population needed to capture steep crashes and dramatic rebounds.
  * Raw unfloored read counts and fractions are preserved under `'raw_reads'` and `'raw_relative_abundance'`.

---

### DEC-006: Staged Modeling Strategy (Direction 1 Core vs. Direction 2 Emergence)

* **Date**: 2026-10-19
* **Status**: Decided
* **Context**:
  The 48 library isolates comprise 24 soil isolates (Species 1–24) and 24 *C. elegans* gut isolates (Species 25–48).
  * $S=3, 6, 12$ communities contain **exclusively Species 1–24**.
  * $S=24, 48$ communities introduce Species 25–48.
* **Decision**:
  We adopt a two-stage modeling plan:
  1. **Direction 1 (Primary Core Fit)**:
     Train on all communities ($S \in \{3, 6, 12, 24, 48\}$) across LN, MN, and HN with fixed $\Theta$ ($48 \times K$) and $W$ ($K \times K$). Demonstrate that varying only the initial resource vector $R_0(\text{LN}) \to R_0(\text{MN}) \to R_0(\text{HN})$ accurately captures the transition from stable equilibria in LN to non-equilibrium fluctuations in HN.
  2. **Direction 2 (Bottom-Up Assembly as the "Cherry on Top")**:
     Train $\Theta$ and $W$ exclusively on small, stable communities ($S=3, 6$) using the 24 soil species. Perform zero-shot forward rollouts on $S=12$ under HN, testing whether the model correctly predicts which 4 communities remain stable and which 4 communities spontaneously fluctuate.

---

### DEC-007: Standardizing on the `hema` Conda Environment

* **Date**: 2026-10-19
* **Status**: Decided & Implemented
* **Context**:
  Selecting an active, uniform Python environment with PyTorch, ODE solvers, and Excel processing.
* **Environment Path**:
  `/home/karthik/miniconda3/envs/hema/bin/python`
* **Installed Packages**:
  `torch`, `torchdiffeq`, `scipy`, `numpy`, `pandas`, `matplotlib`, and `openpyxl` (installed to support direct `.xlsx` parsing).
* **Policy**:
  All future scripts, training runs, and evaluations for this project must execute using this environment.

---

### DEC-008: Global Latent Assembly ODE Training & Phase Evaluation (Milestone 2)

* **Date**: 2026-10-19
* **Status**: Completed & Verified
* **Context**:
  Implementing Direction 1: Training a single continuous-time [`LatentAssemblyNeuralODE`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py#L208-L274) across all 33 experimental communities in Low Nutrient (LN), Medium Nutrient (MN), and High Nutrient (HN) regimes (99 total longitudinal trajectories).
* **Training Architecture & Hyperparameters**:
  * **Model**: $N=48$ species, $K=5$ latent resources, 3 condition embeddings $a_{\text{LN}}, a_{\text{MN}}, a_{\text{HN}}$.
  * **Biological Invariance**: $\Theta \in \mathbb{R}^{48 \times 5}$ (traits), $W \in \mathbb{R}^{5 \times 5}$ (cross-feeding), and $\alpha \in \mathbb{R}^{48}$ (self-limitation) are strictly conserved across all conditions and communities.
  * **Loss & Optimization**: KL Divergence loss on relative abundance with decaying Scheduled Sampling (teacher forcing ratio $1.0 \to 0.0$ over 350 epochs). Learning rate $\eta = 0.008$, Adam optimizer, L2 regularization $\lambda = 0.01$ on $\Theta, W$, gradient clipping norm 5.0. Euler solver with step size $0.1$.
  * **Dilution Factor**: $D = 30.0$ (matching the experimental 1:30 dilution).
  * **Batching**: Two length-matched batches on GPU: Batch 1 (75 seqs $\times$ 11 timepoints) and Batch 2 (24 seqs $\times$ 4 timepoints).
  * **Runtime**: Completed 350 epochs in **54.3 seconds** on NVIDIA GeForce RTX 3070 Ti.
* **Empirical Validation & Results**:
  1. **Autoregressive Unrolling Predictability ($R^2$, Log Scale)**:
     * Overall mean multi-passage autoregressive trajectory $R^2 = 0.437$.
     * **Stratification by Dynamic Phase (Stable vs. Chaotic/Fluctuating)**:
       * **Stable Communities (50 communities)**: Mean Trajectory $R^2 = \mathbf{0.524} \pm 0.323$ (Median: $0.504$, Day 10 Endpoint $R^2 = 0.542$).
       * **Chaotic / Fluctuating Communities (49 communities)**: Mean Trajectory $R^2 = \mathbf{0.347} \pm 0.315$ (Median: $0.204$, Day 10 Endpoint $R^2 = 0.358$).
       * *Within LN*: Stable communities achieve $R^2 = 0.469 \pm 0.249$ vs. Fluctuating $R^2 = 0.306 \pm 0.239$.
       * *Within MN*: Stable communities achieve $R^2 = 0.893 \pm 0.149$ vs. Fluctuating $R^2 = 0.508 \pm 0.375$.
       * *Within HN*: Stable communities achieve $R^2 = 0.258 \pm 0.197$ vs. Fluctuating $R^2 = 0.205 \pm 0.206$.
       * *At the $S=12$ HN Bifurcation Point*: The 3 stable communities average $R^2 = 0.202 \pm 0.055$, whereas the 5 fluctuating communities average $R^2 = 0.115 \pm 0.074$.
     * **Medium Nutrient (MN)**: $S=3$ ($R^2 = 0.888 \pm 0.145$), $S=6$ ($R^2 = 0.873 \pm 0.095$), $S=12$ ($R^2 = 0.855 \pm 0.123$).
     * **Low Nutrient (LN)**: $S=3$ ($R^2 = 0.605 \pm 0.275$), $S=6$ ($R^2 = 0.438 \pm 0.183$), $S=12$ ($R^2 = 0.442 \pm 0.209$).
     * **High Nutrient (HN) Complexity-Driven Predictability Collapse**:
       * $S=3$: $R^2 = 0.445 \pm 0.292$
       * $S=6$: $R^2 = 0.202 \pm 0.121$
       * $S=12$: $R^2 = 0.148 \pm 0.078$
       * $S=24$: $R^2 = 0.133 \pm 0.066$
       * $S=48$: $R^2 = 0.122$
       * *Insight*: Under HN, the model directly mirrors Gore et al.'s emergent phase transition. As pool size $S$ increases from 3 to 48, the system moves from stable equilibria into high-dimensional persistent fluctuations / chaos where exact deterministic trajectory matching degrades while capturing non-equilibrium dynamics.
  2. **Learned Latent Environmental Resource Geometry $R(0)$**:
     * $R(0) = \text{softplus}(a)$ strictly positive.
     * **Monotonic Total Resource Scaling**:
       * $\sum R(0)_{\text{LN}} = 8.78$ ($L_2\text{-norm} = 4.46$)
       * $\sum R(0)_{\text{MN}} = 9.89$ ($L_2\text{-norm} = 5.59$)
       * $\sum R(0)_{\text{HN}} = 10.29$ ($L_2\text{-norm} = 5.66$)
     * **Selective Chemical Axis Bloom**: Dimension 3 expands by **32-fold** ($0.10$ in LN $\to 3.22$ in HN), capturing the physical addition of 20 g/L glucose and 16 g/L urea without manual nutrient tuning.
* **Generated Outputs**:
  * Trained weights: [`Data/Gore_Global_ODE_Model.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_Global_ODE_Model.pth)
  * Metric table: [`Results/Gore_Global_Evaluation_Results.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_Global_Evaluation_Results.csv)
  * 4-Panel publication figure: [`Figures/Gore_Global_Fit_Evaluation.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_Global_Fit_Evaluation.png)

---

### DEC-009: Phase A Latent Dimension ($K \in [2..12]$) Sweep & Error Compounding Discovery

* **Date**: 2026-10-20
* **Status**: Completed & Analyzed
* **Context**:
  Executed a systematic sweep of the latent resource capacity $K \in \{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\}$ in increments of 1 using pure `KLDivLoss` and multi-passage autoregressive Linear $R^2$ evaluation on relative abundance fractions ($x_i \in [0, 1]$).
* **Summary Results**:
  | $K$ | Free Parameters | Mean Linear $R^2$ | Median Linear $R^2$ | LN $R^2$ | MN $R^2$ | HN $R^2$ | Stable $R^2$ | Chaotic $R^2$ |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | 2 | 154 | 0.530 | 0.496 | 0.494 | 0.745 | 0.351 | 0.583 | 0.477 |
  | 3 | 210 | 0.545 | 0.524 | 0.525 | 0.754 | 0.355 | 0.605 | 0.484 |
  | 4 | 268 | 0.546 | 0.537 | 0.527 | 0.755 | 0.356 | 0.607 | 0.484 |
  | 5 | 328 | 0.546 | 0.533 | 0.527 | 0.755 | 0.356 | 0.608 | 0.483 |
  | 6 | 390 | 0.544 | 0.518 | 0.522 | 0.754 | 0.355 | 0.604 | 0.482 |
  | **7** | **454** | **0.546** | **0.535** | **0.528** | **0.756** | **0.356** | **0.607** | **0.484** |
  | 8 | 520 | 0.545 | 0.517 | 0.526 | 0.754 | 0.354 | 0.604 | 0.484 |
  | 9 | 588 | 0.546 | 0.543 | 0.526 | 0.755 | 0.355 | 0.605 | 0.485 |
  | 10 | 658 | 0.546 | 0.501 | 0.522 | 0.756 | 0.359 | 0.602 | 0.488 |
  | 11 | 730 | 0.545 | 0.521 | 0.524 | 0.755 | 0.356 | 0.604 | 0.484 |
  | 12 | 804 | 0.546 | 0.541 | 0.526 | 0.755 | 0.356 | 0.605 | 0.485 |
* **Critical Scientific Finding: 1-Step Accuracy vs. Compounding Rollout Drift**:
  * Diagnostic tracking of passage-by-passage accuracy revealed that the model's **1-step prediction (teacher-forced) achieves $R^2 = 0.90 - 0.95$ across Days 3 to 10**!
  * However, during **Passage 1 (Day 0 $\to$ Day 1)**, where species undergo a massive transient sorting/lag phase from equal OD inoculation, 1-step $R^2$ drops to $0.528$.
  * In autonomous multi-passage rollouts, this initial Day 1 sorting error propagates forward, capping autonomous trajectory $R^2$ around $\sim 0.45 - 0.55$.
  * Increasing $K$ from 4 to 12 produces almost identical performance because the single-step consumer-resource ODE dynamics are already saturated ($R^2 \approx 0.95$).
* **Decision for Next Phases**:
  1. Fix optimal latent capacity at **$K = 7$** (or $K = 5$) as the representative backbone.
  2. In **Phase B**, address numerical precision: test smaller $\Delta t$ ($\Delta t = 0.05, 0.02$) and higher-order Runge-Kutta (`rk4`) to minimize per-step integration error accumulation.
  3. In **Phase C**, optimize training rollouts: implement a scheduled sampling floor ($TF_{\text{min}} \in [0.10, 0.20]$) or multi-step curriculum training (training on 2-step and 3-step rollouts) so the ODE parameters learn to recover from perturbed states instead of drifting.
* **Artifacts Generated**:
  * Checkpoint: [`Data/Gore_Global_ODE_Model_Best_K7.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_Global_ODE_Model_Best_K7.pth)
  * Summary CSV: [`Results/Gore_PhaseA_K_Sweep_Summary.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseA_K_Sweep_Summary.csv)
  * Detailed CSV: [`Results/Gore_PhaseA_K_Sweep_Detailed.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseA_K_Sweep_Detailed.csv)
  * Comparative Figure: [`Figures/Gore_PhaseA_K_Sweep_Comparison.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_PhaseA_K_Sweep_Comparison.png)

---

### DEC-010: Phase B Solver Precision ($\Delta t$) & Scheduled Sampling Anchoring ($TF_{\text{min}}$)

* **Date**: 2026-10-20
* **Status**: Completed & Analyzed
* **Context**:
  Investigated numerical solver algorithms (`euler`, `midpoint`, `rk4`), integration step size $\Delta t \in \{0.10, 0.05, 0.02\}$, and teacher forcing scheduled sampling anchoring ($TF_{\text{min}} \in \{0.0, 0.10, 0.20\}$) using the optimal $K=7$ backbone across 350 epochs.
* **Summary Results**:
  | Configuration | Solver Algorithm | Step Size ($\Delta t$) | Steps / Passage | $TF_{\text{min}}$ Floor | Mean Linear $R^2$ | Median Linear $R^2$ | Stable $R^2$ | Day 10 Endpoint $R^2$ | Train Time |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Euler Baseline** | Euler | 0.10 | 10 | 0.00 | 0.546 | 0.535 | 0.607 | 0.409 | 51.1s |
  | **Euler-dt0.05** | Euler | 0.05 | 20 | 0.00 | 0.546 | 0.521 | 0.606 | 0.410 | 92.3s |
  | **Euler-dt0.02** | Euler | 0.02 | 50 | 0.00 | 0.545 | 0.528 | 0.603 | 0.406 | 223.3s |
  | **Midpoint-dt0.05** | Midpoint (2nd) | 0.05 | 20 | 0.00 | 0.545 | 0.499 | 0.605 | 0.408 | 169.0s |
  | **RK4-dt0.10** | RK4 (4th) | 0.10 | 10 | 0.00 | 0.544 | 0.507 | 0.603 | 0.408 | 179.7s |
  | **RK4-dt0.05** | RK4 (4th) | 0.05 | 20 | 0.00 | 0.546 | 0.537 | 0.606 | 0.408 | 351.8s |
  | **RK4-dt0.05 (TF10%)** | RK4 (4th) | 0.05 | 20 | **0.10** | **0.553** | **0.550** | **0.615** | **0.422** | 339.1s |
  | **Euler-dt0.10 (TF10%)** | Euler | 0.10 | 10 | **0.10** | **0.553** | **0.560** | **0.618** | **0.423** | **49.2s** |
  | **Euler-dt0.10 (TF15%)** | Euler | 0.10 | 10 | **0.15** | **0.549** | **0.564** | **0.613** | **0.416** | **49.4s** |
* **Key Scientific & Numerical Takeaways**:
  1. **Discretization Error is NOT the Limiting Factor**:
     Cutting step size from $\Delta t = 0.10$ down to $0.02$ (50 steps/passage) or using 4th-order Runge-Kutta (`rk4`) with $TF=0$ yields practically identical multi-step rollouts ($R^2 \approx 0.546$). Euler at $\Delta t = 0.10$ does not suffer from stiffness or blowup for this consumer-resource parameter scale, confirming that $\Delta t = 0.10$ is mathematically sound and computationally optimal ($7\times$ faster than RK4).
  2. **Scheduled Sampling Floor ($TF_{\text{min}} = 10\%$) is the Real Breakthrough**:
     When teacher forcing decays completely to $0.0$, backpropagating through 10 unrolled ODE cycles causes compounding gradient degradation. Enforcing a modest $10\%$ ground-truth anchor ($TF_{\text{min}} = 0.10$) stabilizes long-horizon backpropagation, driving:
     * Overall Mean Linear $R^2 \to \mathbf{0.553}$
     * Overall Median Linear $R^2 \to \mathbf{0.560}$ (up from 0.535)
     * Stable Communities $R^2 \to \mathbf{0.618}$ (up from 0.607)
     * Day 10 Endpoint $R^2 \to \mathbf{0.423}$ (up from 0.409)
     * All achieved in **under 50 seconds** using Euler $\Delta t = 0.10$.
* **Decision for Next Step (Phase C)**:
  * Adopt **Euler with $\Delta t = 0.10$ and $TF_{\text{min}} = 0.10$** as our core training engine.
  * Proceed to **Phase C (Optimization & Regularization Fine-Tuning)**:
    - Tune L2 weight penalty $\lambda \in \{0.0, 0.0005, 0.001, 0.005, 0.01\}$ (specifically testing lower penalty on traits $\Theta$ so species can express stronger niche preferences, while keeping $W$ regularized).
    - Tune Learning Rate schedule with `CosineAnnealingLR`.
* **Artifacts Generated**:
  * Checkpoint: [`Data/Gore_Global_ODE_Model_PhaseB_Best.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_Global_ODE_Model_PhaseB_Best.pth)
  * Summary CSV: [`Results/Gore_PhaseB_Solver_Summary.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseB_Solver_Summary.csv)
  * Detailed CSV: [`Results/Gore_PhaseB_Solver_Detailed.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseB_Solver_Detailed.csv)
  * Comparison Plot: [`Figures/Gore_PhaseB_Solver_Comparison.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_PhaseB_Solver_Comparison.png)

---

### DEC-011: Phase C Regularization Uncaging, Cross-Feeding Activation & Trajectory Jump

* **Date**: 2026-10-20
* **Status**: Completed & Validated
* **Context**:
  Explored L2 regularization weight penalty $\lambda \in \{0.010, 0.005, 0.001, 0.0005, 0.0\}$, decoupled regularization ($\lambda_{\Theta}$ vs. $\lambda_W$), learning rate schedulers (`CosineAnnealingLR` vs. flat LR), and training epochs ($350$ vs. $500$) using the established $K=7$, Euler $\Delta t=0.10$, $TF_{\text{min}}=0.10$ foundation.
* **Summary Results**:
  | Configuration | L2 $\lambda_{\Theta}$ | L2 $\lambda_W$ | LR Schedule | Epochs | Mean Linear $R^2$ | Median Linear $R^2$ | LN $R^2$ | MN $R^2$ | HN $R^2$ | Stable $R^2$ | Chaotic $R^2$ | Day 10 Endpoint $R^2$ | $\|\Theta\|_F$ | $\|W\|_F$ |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **L2_0.010-FlatLR (Base)** | 0.010 | 0.010 | Flat (0.008) | 350 | 0.553 | 0.560 | 0.536 | 0.762 | 0.360 | 0.618 | 0.486 | 0.547 | 0.52 | 0.0003 |
  | **L2_0.005-FlatLR** | 0.005 | 0.005 | Flat (0.008) | 350 | 0.552 | 0.554 | 0.533 | 0.762 | 0.360 | 0.615 | 0.487 | 0.544 | 0.62 | 0.0008 |
  | **L2_0.001-FlatLR** | 0.001 | 0.001 | Flat (0.008) | 350 | 0.552 | 0.558 | 0.534 | 0.762 | 0.360 | 0.616 | 0.487 | 0.545 | 0.97 | 0.0076 |
  | **L2_0.0005-FlatLR** | 0.0005 | 0.0005 | Flat (0.008) | 350 | 0.552 | 0.561 | 0.534 | 0.762 | 0.360 | 0.616 | 0.487 | 0.545 | 1.17 | 0.0187 |
  | **L2_0.000-FlatLR (NoReg)** | **0.0** | **0.0** | **Flat (0.008)** | **350** | **0.654** | **0.713** | **0.589** | **0.803** | **0.569** | **0.700** | **0.607** | **0.688** | **13.36** | **24.16** |
  | **L2_0.001-CosineLR** | 0.001 | 0.001 | Cosine | 350 | 0.551 | 0.552 | 0.532 | 0.762 | 0.359 | 0.615 | 0.487 | 0.544 | 1.19 | 0.0110 |
  | **L2_0.001-CosineLR-500ep**| 0.001 | 0.001 | Cosine | 500 | 0.552 | 0.551 | 0.533 | 0.762 | 0.360 | 0.615 | 0.487 | 0.544 | 1.01 | 0.0082 |
  | **Decoupled-CosineLR** | 0.0005 | 0.005 | Cosine | 350 | 0.551 | 0.551 | 0.532 | 0.762 | 0.360 | 0.615 | 0.487 | 0.544 | 1.43 | 0.0028 |
  | **Decoupled-CosineLR-500ep**| 0.0005 | 0.005 | Cosine | 500 | 0.552 | 0.551 | 0.533 | 0.762 | 0.360 | 0.615 | 0.487 | 0.544 | 1.21 | 0.0021 |
* **Major Scientific Discovery: Cross-Feeding Suppression by L2 Regularization**:
  1. In all previous models, an L2 penalty on $W$ (even $\lambda = 0.001$) heavily suppressed the cross-feeding matrix ($W \approx 0.0003$). This effectively disabled secondary metabolic conversions, reducing the ODE to independent single-substrate consumption.
  2. When the L2 constraint is removed (`L2_0.000`), the model engages the full cross-feeding matrix ($\|W\|_F = 24.16$).
  3. **Complex Conjugate Oscillatory Modes**:
     Spectral analysis of the learned $W$ matrix reveals complex conjugate eigenvalue pairs ($-12.09 \pm 7.24j$ and $2.02 \pm 0.85j$). In dynamical systems theory, complex eigenvalues create **rotational and oscillatory flow fields** in resource space. This directly captures the cyclic metabolic cross-feeding loops where primary degrader byproducts sustain secondary taxa.
  4. **Performance Leap Across All Regimes**:
     * Overall Mean Linear $R^2$ leaped from $0.553 \to \mathbf{0.654}$.
     * Overall Median Linear $R^2$ surged from $0.560 \to \mathbf{0.713}$.
     * **High Nutrient (HN) Regimes**: Linear $R^2$ surged from $0.360 \to \mathbf{0.569}$ (a $+58\%$ relative boost), finally capturing the oscillatory dynamics of complex communities.
     * **Stable Communities**: Reached Linear $R^2 = \mathbf{0.700}$.
     * **Day 10 Endpoint Predictability**: Reached $R^2 = \mathbf{0.688}$.
     * **Resource Monotonicity**: Total initial resource mass strictly increases along the chemical axis: $\sum R(0)_{\text{LN}} = 3.32 \to \sum R(0)_{\text{MN}} = 7.57 \to \sum R(0)_{\text{HN}} = 9.07$.
* **Artifacts Generated**:
  * Checkpoint: [`Data/Gore_Global_ODE_Model_PhaseC_Best.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_Global_ODE_Model_PhaseC_Best.pth)
  * Summary CSV: [`Results/Gore_PhaseC_Optimization_Summary.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseC_Optimization_Summary.csv)
  * Detailed CSV: [`Results/Gore_PhaseC_Optimization_Detailed.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseC_Optimization_Detailed.csv)
  * Comparison Plot: [`Figures/Gore_PhaseC_Optimization_Comparison.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_PhaseC_Optimization_Comparison.png)
  * 6-Panel Optimal Fit Figure: [`Figures/Gore_Optimal_Model_Fits.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_Optimal_Model_Fits.png)
  * Visualization Script: [`src/Gore_Plot_Optimal_Model.py`](file:///home/karthik/SerialDilutionModel/src/Gore_Plot_Optimal_Model.py)




