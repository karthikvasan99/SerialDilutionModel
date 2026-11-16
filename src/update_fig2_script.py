with open('src/plot_candidate_figure2_zeroshot.py', 'r') as f:
    code = f.read()

old_b = """    # PANEL B: CROSS-CONDITION UNIVERSALITY & BENCHMARK
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(-0.12, 1.12, 'b', transform=ax_b.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    cond_names = ['No Carrier\\n(Liquid)', 'Plain Agar\\n(Inert Bead)', 'Mucin Agar\\n(Mucosal Bead)']
    x = np.arange(len(cond_names))
    w = 0.125

    x_b1 = np.array([x[0] - 2.0*w, x[1] - 2.5*w, x[2] - 2.5*w])
    x_b2 = np.array([x[0] - 1.0*w, x[1] - 1.5*w, x[2] - 1.5*w])
    x_b3 = np.array([x[0] + 0.0*w, x[1] - 0.5*w, x[2] - 0.5*w])
    x_b4 = np.array([x[0] + 1.0*w, x[1] + 0.5*w, x[2] + 0.5*w])
    x_b5 = np.array([x[0] + 2.0*w, x[1] + 1.5*w, x[2] + 1.5*w])
    x_b6 = np.array([x[1] + 2.5*w, x[2] + 2.5*w])

    b1 = ax_b.bar(x_b1, true_r2, width=w, label='Universal CRM', color='#1B6CA8', edgecolor='#0F2942', lw=1.1, zorder=3)
    b2 = ax_b.bar(x_b2, now_r2, width=w, label=r'$\\mathbf{W} = \\mathbf{0}$ (No Byproducts)', color='#E08B53', edgecolor='#803010', lw=1.1, zorder=3)
    b3 = ax_b.bar(x_b3, shuff_r0_r2, width=w, yerr=shuff_r0_std, capsize=3, label=r'Scrambled $\\mathbf{R}_0$', color='#7A92A3', edgecolor='#2C3E50', lw=1.1, zorder=3)
    b4 = ax_b.bar(x_b4, persist_r2, width=w, label='Persistence Null', color='#F5A623', edgecolor='#B37400', lw=1.1, zorder=3)
    b5 = ax_b.bar(x_b5, shuff_th_r2, width=w, yerr=shuff_th_std, capsize=3, label=r'Shuffled $\\mathbf{\\Theta}$', color='#D9534F', edgecolor='#701010', lw=1.1, zorder=3)
    b6 = ax_b.bar(x_b6, shuff_adh_r2[1:], width=w, yerr=shuff_adh_std[1:], capsize=3, label='Shuffled Adhesion', color='#8E44AD', edgecolor='#4A154B', lw=1.1, zorder=3)

    ax_b.axhline(0, color='black', lw=1.0, linestyle='-', zorder=4)

    # Number labels on bars
    for bar in b1:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1B6CA8')
    for bar in b2:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#803010')
    for bar in b4:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#B37400')
    for i, bar in enumerate(b6):
        h = bar.get_height()
        err = shuff_adh_std[i + 1] if not np.isnan(shuff_adh_std[i+1]) else 0.0
        ax_b.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#4A154B')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cond_names, fontsize=11.5, fontweight='bold', color='#0F2942')
    ax_b.set_ylabel(r'Prediction Accuracy ($R^2$)', fontsize=12, fontweight='bold', labelpad=4)
    ax_b.set_ylim(-0.75, 1.18)
    ax_b.tick_params(axis='y', labelsize=10.5)
    ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=9.0, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')"""

new_b = """    # PANEL B: USER PROTOCOL (CROSS-MEDIA TRANSFER VS PERSISTENCE ON SAME FOOTING)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(-0.12, 1.12, 'b', transform=ax_b.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    cond_names = ['No Carrier\\n(Liquid Broth)', 'Plain Agar\\n(Inert Hydrogel)', 'Mucin Agar\\n(Mucosal Hydrogel)']
    x = np.arange(len(cond_names))
    w = 0.22

    # Evaluated on Passages P3..P6
    r2_full = [0.8572, 0.9237, 0.9284]
    r2_transfer = [-4.1083, 0.7744, 0.6965]
    r2_persist = [0.1098, 0.8046, 0.7451]
    r2_transfer_plot = [max(-0.65, v) for v in r2_transfer]

    offsets = [-w, 0, w]

    b1 = ax_b.bar(x + offsets[0], r2_full, width=w, label='Full CRM (Joint Calibration)', color='#1B6CA8', edgecolor='#0F2942', lw=1.1, zorder=3)
    b2 = ax_b.bar(x + offsets[1], r2_transfer_plot, width=w, label='Cross-Media Transfer (Relearn a & k_att)', color='#9B59B6', edgecolor='#5B2C6F', lw=1.1, zorder=3)
    b3 = ax_b.bar(x + offsets[2], r2_persist, width=w, label='Persistence Baseline (Same Footing: P2 Constant)', color='#F5A623', edgecolor='#B37400', lw=1.1, zorder=3)

    ax_b.axhline(0, color='black', lw=1.0, linestyle='-', zorder=4)

    for bar, val in zip(b1, r2_full):
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1B6CA8')

    for bar, val in zip(b2, r2_transfer):
        h = bar.get_height()
        if val < -0.5:
            ax_b.text(bar.get_x() + bar.get_width()/2., -0.60, f'{val:.2f}\\n(Sink Collapse)', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#7D3C98')
        else:
            ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#5B2C6F')

    for bar, val in zip(b3, r2_persist):
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#B37400')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cond_names, fontsize=11.5, fontweight='bold', color='#0F2942')
    ax_b.set_ylabel(r'Prediction Accuracy ($R^2$ on $P_3\\text{--}P_6$)', fontsize=12.5, fontweight='bold', labelpad=4)
    ax_b.set_ylim(-0.75, 1.20)
    ax_b.tick_params(axis='y', labelsize=10.5)
    ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8.0, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')"""

code_updated = code.replace(old_b, new_b).replace('Figure2_Pollard_Candidate_ZeroShot.png', 'Figure2_Pollard_UserProtocol_P3toP6.png')
with open('src/plot_candidate_figure2_userprotocol.py', 'w') as f:
    f.write(code_updated)
print("Updated src/plot_candidate_figure2_userprotocol.py")
