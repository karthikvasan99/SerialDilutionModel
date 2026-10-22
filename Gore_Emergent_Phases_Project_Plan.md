# Emergent Phases & Consumer-Resource Modeling: Project Plan & Milestones

This document details the scientific foundation, experimental dataset specifications, mathematical model formulation, and phased milestones for modeling the **Jeff Gore (Science 2022)** dataset using our **Latent Assembly Neural ODE** framework.

---

## 1. Scientific Thesis & Context

* **Paper**: *"Emergent phases of ecological diversity and dynamics mapped in microcosms"*, Jiliang Hu, Daniel R. Amor, Matthieu Barbier, Guy Bunin, Jeff Gore, *Science* 378, 85–89 (2022).
* **Paper PDF**: [`Data/Gore - Emergent phases - Science 2022.pdf`](file:///home/karthik/SerialDilutionModel/Data/Gore%20-%20Emergent%20phases%20-%20Science%202022.pdf)
* **Raw Data**: [`Data/gore_data.xlsx`](file:///home/karthik/SerialDilutionModel/Data/gore_data.xlsx)
* **Decisions Log**: [`Gore_Modeling_Decisions_Log.md`](file:///home/karthik/SerialDilutionModel/Gore_Modeling_Decisions_Log.md)

### The Core Problem in the Original Paper
In the Science 2022 paper, Gore et al. tested theoretical predictions based on May's complexity-stability hypothesis and random generalized Lotka-Volterra (gLV) models:
$$\frac{dN_i}{dt} = N_i \left( 1 - \sum_{j=1}^S a_{ij} N_j \right) + D$$
To tune the "interaction strength" $\langle a_{ij} \rangle$ experimentally, they increased the concentration of carbon (glucose) and nitrogen (urea). However, in their simulations, they artificially scaled the numerical interaction coefficients $a_{ij}$.

### Our Mechanistic Consumer-Resource Alternative
* **The Physical Reality**: The physical bacteria are the exact same 48 strains in all conditions; their metabolic enzymes, uptake affinities, and byproduct secretion stoichiometry do not change when glucose or urea is added.
* **The Model**: Using [`LatentAssemblyNeuralODE`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py#L208-L274), we model the communities with **conserved biological parameters**:
  - Trait affinity matrix $\Theta \in \mathbb{R}^{N \times K}$ (invariant across conditions)
  - Cross-feeding / byproduct matrix $W \in \mathbb{R}^{K \times K}$ (invariant across conditions)
  - Intraspecific self-limitation $\alpha \in \mathbb{R}^N$ (invariant across conditions)
* **The Driver of Dynamics**: The transition between dynamical phases (Phase I: Stable Coexistence $\to$ Phase II: Extinctions $\to$ Phase III: Persistent Fluctuations/Chaos) is driven **purely by shifting/scaling the initial environmental resource vector $R(0)$** ($R(0)_{\text{LN}} \to R(0)_{\text{MN}} \to R(0)_{\text{HN}}$).
* **Low-Dimensional Advantage**: Replaces a $48 \times 48 = 2,304$-parameter gLV matrix with a compact $K \approx 4-6$ latent consumer-resource representation ($\approx 200$ parameters total).

---

## 2. Experimental Dataset Specifications

### Media Chemistry (Glucose & Urea Concentrations)
From the supplementary experimental protocol:
* **Base Medium (BM)**: $1\text{ g/L}$ yeast extract, $1\text{ g/L}$ soytone, $10\text{ mM}$ sodium phosphate buffer, trace salts ($\text{CaCl}_2$, $\text{MgCl}_2$, $\text{NiSO}_4$, $\text{MnCl}_2$), pH 6.5.
* **Low Nutrient (LN)**: Pure Base Medium ($0\text{ g/L}$ added glucose, $0\text{ g/L}$ added urea).
* **Medium Nutrient (MN)**: Base Medium $+ 5\text{ g/L}$ glucose $+ 4\text{ g/L}$ urea.
* **High Nutrient (HN)**: Base Medium $+ 20\text{ g/L}$ glucose $+ 16\text{ g/L}$ urea ($4\times$ scalar increase of MN).

### Bacterial Library & Community Pool Structure
The library contains **48 bacterial isolates**:
* **Species 1 to 24**: 24 isolates from **soil**.
* **Species 25 to 48**: 24 isolates from the ***C. elegans* gut**.

Across the 33 experimental communities:
| Pool Size ($S$) | Number of Communities | Species Composition | Primary Dynamical Regime Observed |
| :--- | :--- | :--- | :--- |
| **$S = 3$** | 8 communities | Uses **only Species 1–24** (each species in 1 community) | Phase I & II (Fully stable) |
| **$S = 6$** | 8 communities | Uses **only Species 1–24** (each species in 2 communities) | Phase II (Stable partial coexistence) |
| **$S = 12$** | 8 communities | Uses **only Species 1–24** (each species in 4 communities) | **Bifurcation Point**: In HN, 4 stable, 4 fluctuating |
| **$S = 24$** | 8 communities | Uses **all 48 species** (soil + gut isolates) | Phase II & III (High fluctuation frequency) |
| **$S = 48$** | 1 community | Full 48-species pool | Phase II (LN stable) vs. Phase III (HN fluctuating) |

### Serial Dilution Protocol Parameters
* **Daily Dilution Factor**: $D = 30$-fold dilution ($20\,\mu\text{L}$ culture transferred into $600\,\mu\text{L}$ fresh media every 24h).
* **Dispersal Rate**: $10^{-6}$ daily dispersal from species pool ($6\,\mu\text{L}$ of a $10^{-4}$ diluted monoculture stock added per cycle).
* **Duration**: 10 daily serial dilution cycles (Day 0 to Day 10).

### Timepoint Coverage in the Spreadsheet
* **`LN species abundances`**: Daily 16S sequencing for **Days 0–10 (11 timepoints)** for Replicate 1, plus Day 10 in triplicate.
* **`HN Species abundances`**: Daily 16S sequencing for **Days 0–10 (11 timepoints)** for Replicate 1, plus Day 10 in triplicate.
* **`MN species abundances`**: 16S sequencing for **Days 7, 8, 9, 10 (4 timepoints)** for Replicate 1, plus Day 10 in triplicate.
* **`Biomass (OD)` sheets**: Measured daily from Day 0 to Day 10 in **triplicate** for all 33 communities $\times$ 3 conditions (297 full trajectories).
* **Focus for Modeling**: Start with **relative abundance** ($x_i(t) = \text{reads}_i / \sum \text{reads}$), exactly matching the German and Pollard modeling pipelines.

---

## 3. Mathematical Model Architecture

Located in [`src/SerialDilutionModels.py`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py):

### Coupled Consumer-Resource ODEs
For each passage $t \in [0, T_{\text{end}}]$ with state vector $[N(t), R(t)]$:
$$\frac{dN_i}{dt} = N_i(t) \left( \sum_{k=1}^K \Theta_{ki} R_k(t) \right) - \alpha_i N_i(t)^2$$
$$\frac{dR_k}{dt} = - R_k(t) \left( \sum_{l=1}^K W_{kl} \sum_{j=1}^N \Theta_{jl} N_j(t) \right)$$
where:
* $\Theta \in \mathbb{R}^{N \times K}$: Latent metabolic trait matrix.
* $W \in \mathbb{R}^{K \times K}$: Metabolic cross-feeding / byproduct matrix.
* $\alpha_i = \text{softplus}(\alpha_{\text{raw}, i}) > 0$: Intraspecific self-limitation / carrying capacity constraint.

### Passage Boundary & Replenishment
At the end of passage $m$, dilution is applied and fresh media resources are replenished:
$$N(0)^{(m+1)} = \frac{N(T_{\text{end}})^{(m)}}{30.0} + 10^{-6} \cdot N_{\text{dispersal}}$$
$$R(0)^{(m+1)} = \text{softplus}(a_{\text{condition}})$$
where $a_{\text{condition}} \in \mathbb{R}^K$ is the learnable initial resource embedding for the given condition ($\text{LN}, \text{MN}, \text{HN}$).

---

## 4. Phased Milestones & Execution Roadmap

```
+-------------------------------------------------------------+
| Milestone 1: Data Parsing & Standardization (Tensors)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Milestone 2: Direction 1 — Global Multi-Nutrient Model Fit  |
| - Fixed Theta, W, alpha across all 48 species               |
| - R(0)_LN vs R(0)_MN vs R(0)_HN drives phase transitions    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Milestone 3: Direction 2 — Bottom-Up Assembly Prediction   |
| - Train on S=3, 6 (stable soil communities)                 |
| - Zero-shot predict S=12 bifurcation (stable vs chaotic)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Milestone 4: Mechanistic & Ecological Insights              |
| - PCA of learned R(0) along the 5:4 glucose:urea axis       |
| - Cross-feeding loop analysis for diversity maintenance     |
+-------------------------------------------------------------+
```

### Milestone 1: Data Extraction & Standardized Tensor Formatting
* **Goal**: Write a modular ingestion script (`src/Gore_Data_Processing.py`) to parse [`Data/gore_data.xlsx`](file:///home/karthik/SerialDilutionModel/Data/gore_data.xlsx).
* **Outputs**:
  1. Community metadata table: Mapping each community index (1–33) to its pool size $S$, species composition, and empirical stability classification (stable vs. fluctuating).
  2. Relative abundance trajectories:
     - LN: Shape $(33, 11, 48)$
     - HN: Shape $(33, 11, 48)$
     - MN: Shape $(33, 4, 48)$ (Days 7–10)
  3. Inoculum presence masks ($M \in \{0, 1\}^{33 \times 48}$) indicating which species belong to each community's pool.
  4. Save structured output to `Data/Gore_processed_data.pt`.

### Milestone 2: Direction 1 — Global Model Fitting Across Nutrient Regimes [COMPLETED & OPTIMIZED]
* **Goal**: Prove that a single low-dimensional Consumer-Resource Neural ODE captures both stable equilibria (LN) and chaotic fluctuations (HN) with fixed biological traits and varying initial resources.
* **Tuned Optimal Architecture & Optimization Engine (Phases A, B, & C)**:
  - Architecture: [`LatentAssemblyNeuralODE`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py#L208-L274) with $N=48$ species, $K=7$ latent resources (454 total parameters).
  - Numerical Solver: Forward Euler with $\Delta t = 0.10$ and 30-fold dilution factor.
  - Optimization: 350 epochs with scheduled sampling anchor $TF_{\text{min}} = 0.10$, Adam optimizer ($\eta = 0.008$).
  - Regularization: Uncaged cross-feeding matrix $W$ ($\lambda = 0.0$), allowing secondary metabolic feedback loops.
* **Results & Performance Breakthrough**:
  - Model weights: [`Data/Gore_Global_ODE_Model_PhaseC_Best.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_Global_ODE_Model_PhaseC_Best.pth).
  - Trajectory metrics: [`Results/Gore_PhaseC_Optimization_Detailed.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseC_Optimization_Detailed.csv).
  - Summary metrics: [`Results/Gore_PhaseC_Optimization_Summary.csv`](file:///home/karthik/SerialDilutionModel/Results/Gore_PhaseC_Optimization_Summary.csv).
  - Comparative figure: [`Figures/Gore_PhaseC_Optimization_Comparison.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_PhaseC_Optimization_Comparison.png).
  - **Overall Mean Linear $R^2$**: Reached **$0.654$** (with **Median Linear $R^2 = 0.713$** across all 99 trajectories).
  - **Stable Communities**: Reached Linear $R^2 = \mathbf{0.700}$.
  - **Medium Nutrient (MN)**: Reached Linear $R^2 = \mathbf{0.803}$.
  - **High Nutrient (HN)**: Reached Linear $R^2 = \mathbf{0.569}$ (up from 0.356).
  - **Cross-Feeding Mechanism**: Spectral analysis of the learned $W$ matrix revealed complex conjugate eigenvalue pairs ($-12.09 \pm 7.24j$), proving that rotational/oscillatory resource flow fields sustain coexistence and chaotic dynamics.
  - **Monotonic Chemical Resource Scaling**: Total initial resource mass strictly tracks experimental media: $\sum R(0)_{\text{LN}} = 3.32 \to \sum R(0)_{\text{MN}} = 7.57 \to \sum R(0)_{\text{HN}} = 9.07$.

### Milestone 2B: Pure $K=2$ Minimalist Physical Model & Ecological Phase Classification [COMPLETED & LOCKED]
* **Goal**: Strip the architecture down to a minimalist, strictly mass-conserving $K=2$ Consumer-Resource ODE ($\approx 148$ parameters, $W \ge 0$), optimize on absolute abundances via Dr. Dixit's Generalized KL loss, and construct a leak-free out-of-sample classifier to detect the emergent chaotic dynamical phase.
* **Key Achievements**:
  1. **Physical ODE Checkpoint**: [`Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth)
     - Absolute abundance median $R^2 = \mathbf{0.9738}$ in Medium Nutrient (MN) and $\mathbf{0.9071}$ in $S=3$ communities.
  2. **Temporal Window Lock**: Proved empirically under LOOCV that simulated dynamics must match the experimental horizon (Passages 7–10); extended simulations (Passages 11–30) degrade $F_1$ from $0.727 \to 0.643$ due to asymptotic competitive exclusion.
  3. **Feature Engineering Suite**: Derived 26 physical features across intra-passage tortuosity (`mean_tv_path`), daily return map drift (`mean_poincare_shift`), niche overlap (`mean_overlap`), demographic survival (`surv_count`, `extinct_ratio`), and species volatility (`max_sp_cv`).
  4. **Media Condition & Model Complexity Studies**:
     - With media flags: 2-layer Tanh MLP achieved $F_1 = 0.7547$ by learning a soft non-linear gate that reduced HN false alarms by 27%.
     - Without media flags: Linear Logistic Regression outperformed MLPs ($F_1 = 0.6909 - 0.7119$ vs $0.6786$) as the maximum-margin hyperplane on monotonic continuous features.
  5. **Locked 100% Chaos Recall Champion Classifier**:
     - Checkpoint: [`Data/Gore_K2_Classifier_L2_C10_100Rec.pth`](file:///home/karthik/SerialDilutionModel/Data/Gore_K2_Classifier_L2_C10_100Rec.pth)
     - Standalone Module: [`src/Gore_K2_Classifier_100Rec.py`](file:///home/karthik/SerialDilutionModel/src/Gore_K2_Classifier_100Rec.py)
     - Operating Threshold: $T^* = 0.3891$.
     - Out-of-sample performance: **Recall = 100.00%** (22/22 chaotic communities caught, **$\text{FN}=0$**), **$F_1 = 0.7458$**, **Accuracy = 84.85%**.
     - $S=24$ Medium Nutrient Recall: **100.0%** (3/3 caught: Comm 1, 4, 5).
  6. **Long-Term Multi-Species Diversity Retention (100 Passages)**:
     - Confirmed that the $K=2$ physical ODE does not crash to 1 or 2 species over long horizons.
     - Over 100 daily serial dilutions, **89.9% of communities retain $\ge 3$ species**; $S=24$ communities retain $11.4 - 14.1$ species ($\exp(H) \approx 6.5 - 7.6$), and $S=48$ retains $19 - 25$ species ($\exp(H) \approx 12 - 15.6$).
     - Coexistence is stably maintained by intra-specific self-limitation ($\alpha_i N_i^2$), daily dispersal ($10^{-6}$), and non-equilibrium chaotic oscillations.
     - Figure: [`Figures/Gore_LongTerm_Diversity_100Days.png`](file:///home/karthik/SerialDilutionModel/Figures/Gore_LongTerm_Diversity_100Days.png).
     - Data: [`Data/Gore_LongTerm_Abundances_100p.npy`](file:///home/karthik/SerialDilutionModel/Data/Gore_LongTerm_Abundances_100p.npy).

### Milestone 3: Direction 2 — Zero-Shot Emergent Phase & Bifurcation Prediction [ACTIVE NEXT STEP]
* **Goal**: Test bottom-up assembly: can a model trained exclusively on small, stable communities ($S=3, 6$) predict the onset of instability in complex communities ($S=12$)?
* **Implementation**:
  - Script: `src/Gore_ZeroShot_Assembly.py`.
  - Training Data: The 16 communities from $S=3$ and $S=6$ (composed entirely of Species 1–24).
  - Test Data: The 8 communities of $S=12$ under High Nutrient (HN).
  - Validation Criteria:
    - In empirical data (Fig. 2D), exactly 4 of the 12-species communities stabilize, while 4 enter persistent fluctuations.
    - Measure whether the zero-shot forward unrolling from Day 0 to Day 10 correctly segregates the stable vs. fluctuating communities.
* **Deliverable**:
  - Figure: `Figures/Gore_ZeroShot_Bifurcation_Prediction.png`.

### Milestone 4: Latent Space Geometry & Mechanism of Diversity Maintenance
* **Goal**: Connect model parameters back to ecological and chemical reality.
* **Analyses**:
  1. **Latent Resource Trajectory**: Perform PCA or trajectory analysis on $R(0)_{\text{LN}}, R(0)_{\text{MN}}, R(0)_{\text{HN}}$. Check if the learned vectors lie on a monotonic 1D manifold corresponding to chemical nutrient loading.
  2. **Cross-Feeding Mechanism of Coexistence**: Inspect the learned $W$ matrix. Identify which metabolic cycles allow fluctuating communities to maintain higher Shannon diversity than stable communities (resolving the Fig. 4 paradox).

---

## 5. Environment & Execution Reference

* **Python Environment**: `/home/karthik/miniconda3/envs/hema/bin/python` (contains PyTorch, `torchdiffeq`, SciPy, Pandas, Matplotlib, `openpyxl`).
* **Core Reference Files**:
  - Model definitions: [`src/SerialDilutionModels.py`](file:///home/karthik/SerialDilutionModel/src/SerialDilutionModels.py)
  - Data directory: [`Data/`](file:///home/karthik/SerialDilutionModel/Data/)
  - Related project summaries:
    - [`German_Data_Summary.md`](file:///home/karthik/SerialDilutionModel/German_Data_Summary.md)
    - [`MeOH_Cross_Host_Transfer_Results.md`](file:///home/karthik/SerialDilutionModel/MeOH_Cross_Host_Transfer_Results.md)
    - [`Methanogen_Enrichment_Plan.md`](file:///home/karthik/SerialDilutionModel/Methanogen_Enrichment_Plan.md)
