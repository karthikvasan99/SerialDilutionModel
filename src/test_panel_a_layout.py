import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

fig = plt.figure(figsize=(8, 5), dpi=200)
gs_a = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.9], wspace=0.35, left=0.1, right=0.92, top=0.88, bottom=0.12)
ax1 = fig.add_subplot(gs_a[0, 0])
ax2 = fig.add_subplot(gs_a[0, 1])
ax3 = fig.add_subplot(gs_a[0, 2])

th_c = np.random.rand(50, 3)
th_s = np.random.rand(50, 3)
w = np.random.rand(3, 3)

im1 = ax1.imshow(th_c, aspect='auto', cmap='YlGnBu')
im2 = ax2.imshow(th_s, aspect='auto', cmap='YlGnBu')
im3 = ax3.imshow(w, aspect='equal', cmap='Oranges')

ax1.set_title(r"$\mathbf{\Theta}^{(c)}$ (Biofilm)", fontsize=11, fontweight='bold')
ax2.set_title(r"$\mathbf{\Theta}^{(s)}$ (Planktonic)", fontsize=11, fontweight='bold')
ax3.set_title(r"Cross-Feeding $\mathbf{W}$", fontsize=11, fontweight='bold')

for ax in [ax1, ax2]:
    ax.set_xticks(range(3))
    ax.set_xticklabels(['$R_1$', '$R_2$', '$R_3$'], fontsize=10, fontweight='bold')
    ax.set_yticks([])
ax1.set_ylabel('Strains (N=50)', fontsize=11, fontweight='bold')

ax3.set_xticks(range(3))
ax3.set_yticks(range(3))
ax3.set_xticklabels(['$R_1$', '$R_2$', '$R_3$'], fontsize=9.5, fontweight='bold')
ax3.set_yticklabels(['$R_1$', '$R_2$', '$R_3$'], fontsize=9.5, fontweight='bold')
ax3.set_xlabel('Product', fontsize=10, fontweight='bold')
ax3.set_ylabel('Substrate', fontsize=10, fontweight='bold')

fig.savefig('test_panel_a.png', dpi=200)
print("Saved test_panel_a.png successfully!")
