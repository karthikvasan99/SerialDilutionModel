"""
Dedicated standalone script: plot_german_theta_simplex.py
Analyzes and visualizes whether the learned strain affinities (Thetas) lie on a simplex:
- Panel A: 3D Trait Space in (theta_MeOH, theta_TMA, theta_H2:CO2) with the fitted 2D affine hyperplane (R^2 = 96.0%).
- Panel B: Intrinsic 2D Simplex Facet (in-plane biplot projection looking orthogonal to the hyperplane).
- Panel C: Barycentric 2-Simplex (Delta^2) showing relative metabolic budget allocation across H2:CO2, MeOH, and TMA.
"""

import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Paper', 'figures')
POSTER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Poster', 'figures')

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(POSTER_FIG_DIR, exist_ok=True)

# 1. Load Parameters and Affinities
params_path = os.path.join(PROJECT_ROOT, 'Data', 'German_champion_model_params.pt')
params = torch.load(params_path, weights_only=False)
Theta = params['Theta'].numpy()
R0_table = params['R0_table'].numpy()
strains_81 = params['strains_81']
N = len(strains_81)

df_aff = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'German_Contrast_Vector_Strain_Affinities.csv'))
proj_meoh = df_aff['Projected_Affinity_MeOH_5b'].values
proj_tma  = df_aff['Projected_Affinity_TMA_5b'].values

# Basal H2:CO2 direction and projections
r_basal = R0_table[0]
u_h2co2 = r_basal / np.linalg.norm(r_basal)
proj_h2co2 = Theta @ u_h2co2

# Key strain indices
target_idx   = 0   # Methanomicrobium sp.
pyr_idx      = 63  # Pyramidobacter sp. C12-8
vicugnae_idx = [i for i, s in enumerate(strains_81) if 'vicugnae' in s][0]
iso4_idx     = [i for i, s in enumerate(strains_81) if 'ISO4-H5' in s][0]
denitro_idx  = [i for i, s in enumerate(strains_81) if 'Denitrobacterium' in s][0]

# 2. Compute 3D Plane & Simplex Properties
coords = np.column_stack([proj_meoh, proj_tma, proj_h2co2])
pca_3d = PCA(n_components=3).fit(coords)
normal = pca_3d.components_[-1]
mean_c = coords.mean(axis=0)
d_val  = np.dot(normal, mean_c)

# Plane equation: normal[0]*MeOH + normal[1]*TMA + normal[2]*H2CO2 = d_val
# => H2CO2 = (d_val - normal[0]*MeOH - normal[1]*TMA) / normal[2]
c_meoh  = - normal[0] / normal[2]
c_tma   = - normal[1] / normal[2]
c_const = d_val / normal[2]
var_in_plane = pca_3d.explained_variance_ratio_[:2].sum() * 100.0
var_ortho    = pca_3d.explained_variance_ratio_[2] * 100.0

print(f"Hyperplane Fit:")
print(f"  theta_H2CO2 = {c_meoh:.4f} * theta_MeOH + {c_tma:.4f} * theta_TMA + {c_const:.4f}")
print(f"  In-Plane Variance: {var_in_plane:.2f}% | Orthogonal Residual: {var_ortho:.2f}%")

# 3. Create 3-Panel Figure
fig = plt.figure(figsize=(20.5, 6.2), dpi=300)

# =============================================================================
# PANEL A: 3D Trait Space & 2D Affine Simplex Hyperplane
# =============================================================================
ax1 = fig.add_subplot(131, projection='3d')

# Plane surface mesh
x_grid = np.linspace(-0.7, 1.8, 14)
y_grid = np.linspace(-0.4, 1.6, 14)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
Z_mesh = c_meoh * X_mesh + c_tma * Y_mesh + c_const

ax1.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.18, color='#2c3e50', edgecolor='none', zorder=2)

# Scatter points
sc1 = ax1.scatter(
    proj_meoh, proj_tma, proj_h2co2,
    c=proj_h2co2, cmap='viridis',
    s=55, edgecolor='black', linewidth=0.7, alpha=0.88, zorder=5
)

# Highlight archetypes
ax1.scatter([proj_meoh[target_idx]], [proj_tma[target_idx]], [proj_h2co2[target_idx]],
            color='#e74c3c', edgecolor='black', linewidth=1.5, s=120, marker='D', zorder=10, label=r'$\it{Methanomicrobium\ sp.}$')
ax1.scatter([proj_meoh[vicugnae_idx]], [proj_tma[vicugnae_idx]], [proj_h2co2[vicugnae_idx]],
            color='#e67e22', edgecolor='black', linewidth=1.2, s=90, marker='s', zorder=9, label=r'$\it{Streptococcus\ vicugnae}$')
ax1.scatter([proj_meoh[iso4_idx]], [proj_tma[iso4_idx]], [proj_h2co2[iso4_idx]],
            color='#27ae60', edgecolor='black', linewidth=1.2, s=90, marker='^', zorder=9, label=r'ISO4-H5 (MeOH)')
ax1.scatter([proj_meoh[pyr_idx]], [proj_tma[pyr_idx]], [proj_h2co2[pyr_idx]],
            color='#2980b9', edgecolor='black', linewidth=1.2, s=90, marker='o', zorder=9, label=r'$\it{Pyramidobacter\ sp.}$')

ax1.set_xlabel(r'$\theta_i^{\mathrm{MeOH}}$', fontsize=10.0, fontweight='bold', labelpad=6)
ax1.set_ylabel(r'$\theta_i^{\mathrm{TMA}}$', fontsize=10.0, fontweight='bold', labelpad=6)
ax1.set_zlabel(r'$\theta_i^{\mathrm{H}_2:\mathrm{CO}_2}$', fontsize=10.0, fontweight='bold', labelpad=6)
ax1.set_title(r'$\mathbf{A}$  3D Trait Space & 2D Affine Simplex' + f'\n(In-Plane Variance: {var_in_plane:.1f}%, Residual: {var_ortho:.1f}%)', fontsize=11.2, pad=12, loc='left', fontweight='bold')
ax1.view_init(elev=20, azim=-55)
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=7.6, framealpha=0.92, facecolor='white', edgecolor='#cccccc')

# Plane formula annotation
eq_text = rf"$\theta_i^{{\mathrm{{H}}_2:\mathrm{{CO}}_2}} = {c_meoh:.2f}\,\theta_i^{{\mathrm{{MeOH}}}} + {c_tma:.2f}\,\theta_i^{{\mathrm{{TMA}}}} + {c_const:.2f}$" + f"\n($R^2 = {var_in_plane/100:.3f}$)"
ax1.text2D(0.04, 0.05, eq_text, transform=ax1.transAxes, fontsize=8.0, fontweight='bold', color='#2c3e50',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9f9', edgecolor='#bdc3c7', alpha=0.92, lw=0.8))

# =============================================================================
# PANEL B: Intrinsic 2D Simplex Facet (In-Plane Biplot Projection)
# =============================================================================
ax2 = fig.add_subplot(132)

coords_2d = pca_3d.transform(coords)[:, :2]
sc2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], c=proj_h2co2, cmap='viridis', s=65, edgecolor='black', linewidth=0.7, alpha=0.90, zorder=4)

# Convex hull
hull = ConvexHull(coords_2d)
for simplex in hull.simplices:
    ax2.plot(coords_2d[simplex, 0], coords_2d[simplex, 1], 'k--', lw=1.2, alpha=0.65, zorder=3)

# Project the 3 original coordinate axes (biplot rays)
scale_ray = 0.85
v_meoh_ray  = pca_3d.components_[:2, 0] * scale_ray
v_tma_ray   = pca_3d.components_[:2, 1] * scale_ray
v_h2co2_ray = pca_3d.components_[:2, 2] * scale_ray

origin = [0, 0]
ax2.annotate('', xy=v_meoh_ray, xytext=origin, arrowprops=dict(arrowstyle='->', color='#16a085', lw=2.2, shrinkA=0, shrinkB=0), zorder=6)
ax2.annotate('', xy=v_tma_ray, xytext=origin, arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.2, shrinkA=0, shrinkB=0), zorder=6)
ax2.annotate('', xy=v_h2co2_ray, xytext=origin, arrowprops=dict(arrowstyle='->', color='#1b365d', lw=2.2, shrinkA=0, shrinkB=0), zorder=6)

ax2.text(v_meoh_ray[0] + 0.05, v_meoh_ray[1], r'$\mathbf{v}_{\mathrm{MeOH}}$', color='#16a085', fontsize=10.0, fontweight='bold', va='center')
ax2.text(v_tma_ray[0] - 0.05, v_tma_ray[1] + 0.02, r'$\mathbf{v}_{\mathrm{TMA}}$', color='#8e44ad', fontsize=10.0, fontweight='bold', ha='right', va='bottom')
ax2.text(v_h2co2_ray[0] + 0.05, v_h2co2_ray[1] + 0.03, r'$\mathbf{u}_{\mathrm{H}_2:\mathrm{CO}_2}$', color='#1b365d', fontsize=10.0, fontweight='bold', va='bottom')

# Highlight archetypes
ax2.scatter(coords_2d[target_idx, 0], coords_2d[target_idx, 1], color='#e74c3c', edgecolor='black', linewidth=1.5, s=125, marker='D', zorder=10, label=r'$\it{Methanomicrobium\ sp.}$')
ax2.scatter(coords_2d[vicugnae_idx, 0], coords_2d[vicugnae_idx, 1], color='#e67e22', edgecolor='black', linewidth=1.2, s=95, marker='s', zorder=9, label=r'$\it{Streptococcus\ vicugnae}$')
ax2.scatter(coords_2d[iso4_idx, 0], coords_2d[iso4_idx, 1], color='#27ae60', edgecolor='black', linewidth=1.2, s=95, marker='^', zorder=9, label=r'ISO4-H5')
ax2.scatter(coords_2d[pyr_idx, 0], coords_2d[pyr_idx, 1], color='#2980b9', edgecolor='black', linewidth=1.2, s=95, marker='o', zorder=9, label=r'$\it{Pyramidobacter\ sp.}$')

ax2.set_xlim(-1.75, 1.45)
ax2.set_ylim(-0.82, 0.92)

ax2.set_xlabel(f'In-Plane Axis 1 (PC1: {pca_3d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10.0, fontweight='bold')
ax2.set_ylabel(f'In-Plane Axis 2 (PC2: {pca_3d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10.0, fontweight='bold')
ax2.set_title(r'$\mathbf{B}$  Intrinsic 2D Simplex Facet (Biplot View)' + f'\n(Convex Hull Encloses Multi-Substrate Archetypes)', fontsize=11.2, pad=12, loc='left', fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.45)
ax2.legend(loc='lower left', fontsize=7.6, framealpha=0.92, facecolor='white', edgecolor='#cccccc')

# =============================================================================
# PANEL C: Barycentric 2-Simplex (Delta^2) on Directional Projections
# phi_i(v) = max(0, Theta_i.v) / sum_v' max(0, Theta_i.v')
# =============================================================================
ax3 = fig.add_subplot(133)

# Non-negative directional allocation along the 3 actual axes:
r_h2   = np.maximum(0, proj_h2co2)
r_meoh = np.maximum(0, proj_meoh)
r_tma  = np.maximum(0, proj_tma)
r_sum  = r_h2 + r_meoh + r_tma

p_h2   = r_h2 / r_sum
p_meoh = r_meoh / r_sum
p_tma  = r_tma / r_sum

# Ternary projection:
# Bottom-left: u_H2:CO2 (p_h2)
# Bottom-right: v_MeOH (p_meoh)
# Top: v_TMA (p_tma)
h_tri = np.sqrt(3) / 2
x_tern = p_meoh + 0.5 * p_tma
y_tern = p_tma * h_tri

# Equilateral triangle
triangle_x = [0.0, 1.0, 0.5, 0.0]
triangle_y = [0.0, 0.0, h_tri, 0.0]
ax3.plot(triangle_x, triangle_y, 'k-', lw=2.0, zorder=2)

# Grid lines (20%, 40%, 60%, 80%)
for f in [0.2, 0.4, 0.6, 0.8]:
    # Constant TMA (horizontal lines)
    y_line = f * h_tri
    ax3.plot([0.5 * f, 1.0 - 0.5 * f], [y_line, y_line], color='#bdc3c7', linestyle=':', lw=0.8, zorder=1)
    # Constant MeOH
    ax3.plot([f, 0.5 * f], [0, f * h_tri], color='#bdc3c7', linestyle=':', lw=0.8, zorder=1)
    # Constant H2:CO2
    ax3.plot([1.0 - f, 1.0 - 0.5 * f], [0, f * h_tri], color='#bdc3c7', linestyle=':', lw=0.8, zorder=1)

sc3 = ax3.scatter(x_tern, y_tern, c=proj_h2co2, cmap='viridis', s=65, edgecolor='black', linewidth=0.7, alpha=0.90, zorder=5)

# Highlight archetypes
ax3.scatter(x_tern[target_idx], y_tern[target_idx], color='#e74c3c', edgecolor='black', linewidth=1.5, s=125, marker='D', zorder=10, label=r'$\it{Methanomicrobium\ sp.}$')
ax3.scatter(x_tern[vicugnae_idx], y_tern[vicugnae_idx], color='#e67e22', edgecolor='black', linewidth=1.2, s=95, marker='s', zorder=9, label=r'$\it{Streptococcus\ vicugnae}$')
ax3.scatter(x_tern[iso4_idx], y_tern[iso4_idx], color='#27ae60', edgecolor='black', linewidth=1.2, s=95, marker='^', zorder=9, label=r'ISO4-H5')
ax3.scatter(x_tern[pyr_idx], y_tern[pyr_idx], color='#2980b9', edgecolor='black', linewidth=1.2, s=95, marker='o', zorder=9, label=r'$\it{Pyramidobacter\ sp.}$')

# Vertex labels
ax3.text(-0.02, -0.04, r'$\mathbf{u}_{\mathrm{H}_2:\mathrm{CO}_2}$' + '\n(Basal Syngas)', ha='center', va='top', fontsize=9.8, fontweight='bold', color='#1b365d')
ax3.text(1.02, -0.04, r'$\mathbf{v}_{\mathrm{MeOH}}$' + '\n(Methanol Direction)', ha='center', va='top', fontsize=9.8, fontweight='bold', color='#16a085')
ax3.text(0.50, h_tri + 0.04, r'$\mathbf{v}_{\mathrm{TMA}}$' + '\n(TMA Direction)', ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#8e44ad')

# Percentage annotations along edges
ax3.text(0.50, -0.035, '50%', fontsize=7.2, color='#7f8c8d', ha='center', va='top')
ax3.text(0.25 - 0.03, h_tri/2, '50%', fontsize=7.2, color='#7f8c8d', ha='right', va='center')
ax3.text(0.75 + 0.03, h_tri/2, '50%', fontsize=7.2, color='#7f8c8d', ha='left', va='center')

ax3.set_xlim(-0.16, 1.16)
ax3.set_ylim(-0.14, 1.04)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title(r'$\mathbf{C}$  Direction-Specific Barycentric Simplex ($\Delta^2$)' + '\n' + r'($\phi_{i,\mathbf{v}} \propto \max(0, \mathbf{\Theta}_i^\top \mathbf{v})$)', fontsize=11.2, pad=12, loc='center', fontweight='bold')
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.6, framealpha=0.92, facecolor='white', edgecolor='#cccccc')

# Unified Colorbar
cbar_ax = fig.add_axes([0.93, 0.20, 0.012, 0.60])
cbar = fig.colorbar(sc3, cax=cbar_ax)
cbar.set_label(r'$\theta_i^{\mathrm{H}_2:\mathrm{CO}_2}$ (Basal Affinity)', fontsize=9.5, fontweight='bold')
cbar.ax.tick_params(labelsize=8.0)

fig.subplots_adjust(left=0.04, right=0.91, wspace=0.25, top=0.88, bottom=0.12)

out_poster   = os.path.join(POSTER_FIG_DIR, 'German_Theta_Simplex_Analysis.png')
out_paper    = os.path.join(PAPER_FIG_DIR, 'German_Theta_Simplex_Analysis.png')
out_artifact = os.path.join(ARTIFACT_DIR, 'German_Theta_Simplex_Analysis.png')

fig.savefig(out_poster, dpi=300, bbox_inches='tight')
fig.savefig(out_paper, dpi=300, bbox_inches='tight')
fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Successfully generated German_Theta_Simplex_Analysis.png in:")
print(f"  - {out_paper}")
print(f"  - {out_poster}")
print(f"  - {out_artifact}")
