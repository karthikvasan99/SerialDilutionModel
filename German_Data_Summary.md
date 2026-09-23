# German's Dataset Analysis Summary

This document summarizes the comprehensive modeling and evaluation pipeline implemented for **German's dataset** (a rumen enrichment microbiome dataset) within the `SerialDilutionModel` repository.

## 1. Data Processing and Exploration

* **Filtering and Preprocessing (`German_microbiome_alignment.py`)**: We processed the raw data and applied a strict relative abundance threshold (`>= 1e-3` in all T1 samples). We saved the cleaned data as PyTorch tensors (`Data/German_processed_data.pt`).
* **Exploratory Visualization (`German_composition_visualization.py`, `German_composition_analysis.py`)**: We generated PCoA plots and composition panels to visualize the microbiome community dynamics, taxa shifts, and guilds across the sequential passages.

![Microbiome Dynamics Overview](/home/karthik/.gemini/antigravity-cli/brain/c2bbdb23-bccc-4a1d-a2a8-c7c7665a2548/German_microbiome_dynamics_overview.png)

![Passage Dynamics (12 Panels)](/home/karthik/.gemini/antigravity-cli/brain/c2bbdb23-bccc-4a1d-a2a8-c7c7665a2548/German_passage_dynamics_12panels.png)

![Taxa Shifts and Guilds](/home/karthik/.gemini/antigravity-cli/brain/c2bbdb23-bccc-4a1d-a2a8-c7c7665a2548/German_taxa_shifts_and_guilds.png)

## 2. Modeling Frameworks

We built and evaluated two primary forms of a Latent Assembly (Consumer-Resource) Model to predict compositional trajectories across passages:

* **Discrete Latent Assembly Model (`German_evaluation.py`)**: A step-by-step sequential model using "Soft Scheduled Sampling" to handle autoregressive multi-passage rollouts.
* **Continuous-Time Neural ODE Model (`German_ODE_evaluation.py`)**: A continuous-time adaptation of the model using `torchdiffeq`, representing microbial growth as an ordinary differential equation over time.

## 3. Hyperparameter Optimization & Validation

* **Extensive Grid Searches (`German_grid_search.py`, `German_ODE_grid_search.py`)**: We ran parallelized grid searches for both the Discrete and ODE models to tune the number of latent resource dimensions ($K$), the L2 proximal regularization penalty, and the number of training epochs.
* **Generalization Testing**: We used leave-one-out autoregressive rollouts (e.g., training on the *BRRS H2:CO2 5.0 bar* condition and validating on the *30.0 bar* condition) to evaluate how well the model predicts unseen environments.

## 4. Latent Space Interpretation (Environment Vectors)

* **PCA of Learned Environments (`German_ODE_environment_pca.py`)**: After training the full ODE model (`Results/German_ODE_full_model.pth`), we extracted the learned "environment vectors" ($a$) that the model assigns to different substrate/pressure conditions.
* **Clustering & Visualizing**: We applied PCA to these vectors to show that the neural network learns to map similar environmental treatments to similar coordinates in its latent resource space.

![PCA of ODE Environment Vectors](/home/karthik/.gemini/antigravity-cli/brain/c2bbdb23-bccc-4a1d-a2a8-c7c7665a2548/German_ODE_environment_vectors_pca.png)

## 5. Baselines and Ablation Tests

* **Null Environment Test (`German_null_environment_evaluation.py`)**: We performed an ablation where we forced the trained model to predict $T1 \rightarrow T6$ trajectories using the *wrong* (null) environmental vectors. This successfully proved that the model's accuracy is driven by it learning true environmental specificity, not just memorizing generic average dynamics.
* **Linear Markov Baseline (`German_linear_markov_evaluation.py`)**: We compared our Latent Assembly results against a simplistic Linear Softmax Markov model that completely ignores the environmental context to quantify the predictive benefit of modeling specific substrate/pressure conditions.
