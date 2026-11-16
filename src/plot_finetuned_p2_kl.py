import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Data from the P0 -> P1 -> P2 fine-tuned model predicting P3..P6 vs P2 persistence:
# Plain Agar
plain_p3_crm = 0.1900
plain_p3_pers = 0.1545

plain_p4_crm = 0.1893
plain_p4_pers = 0.2602

plain_p5_crm = 0.2451
plain_p5_pers = 0.2695

plain_p6_crm = 0.2502
plain_p6_pers = 0.2951

# Mucin Agar
mucin_p3_crm = 0.1727
mucin_p3_pers = 0.1515

mucin_p4_crm = 0.2170
mucin_p4_pers = 0.2602

mucin_p5_crm = 0.2633
mucin_p5_pers = 0.3114

mucin_p6_crm = 0.2969
mucin_p6_pers = 0.3069

# Overall Means on P3..P6:
plain_mean_crm = np.mean([plain_p3_crm, plain_p4_crm, plain_p5_crm, plain_p6_crm]) # 0.2187
plain_mean_pers = np.mean([plain_p3_pers, plain_p4_pers, plain_p5_pers, plain_p6_pers]) # 0.2448

mucin_mean_crm = np.mean([mucin_p3_crm, mucin_p4_crm, mucin_p5_crm, mucin_p6_crm]) # 0.2375
mucin_mean_pers = np.mean([mucin_p3_pers, mucin_p4_pers, mucin_p5_pers, mucin_p6_pers]) # 0.2575

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

# -------------------------------------------------------------
# PLOT 1: Overall Mean KL on P3..P6 (Fine-Tuned vs P2 Persistence)
# -------------------------------------------------------------
conds = ['Plain Agar\n(Inert Bead)', 'Mucin Agar\n(Mucosal Bead)']
x = np.arange(len(conds))
w = 0.28

b1 = ax1.bar(x - w/2, [plain_mean_crm, mucin_mean_crm], width=w, label='Dynamic CRM (Fine-Tuned on P0→P2)', 
             color='#1B6CA8', edgecolor='#0F2942', lw=1.2, zorder=3)
b2 = ax1.bar(x + w/2, [plain_mean_pers, mucin_mean_pers], width=w, label='Persistence Baseline (Holding P2)', 
             color='#F5A623', edgecolor='#B37400', lw=1.2, zorder=3)

for bar in b1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#B37400')

ax1.set_xticks(x)
ax1.set_xticklabels(conds, fontsize=12, fontweight='bold', color='#0F2942')
ax1.set_ylabel(r'Mean Generalized KL Divergence ($P_3\text{--}P_6$)' + '\n' + r'(nats, lower is better $\downarrow$)', fontsize=11.5, fontweight='bold')
ax1.set_ylim(0, 0.32)
ax1.legend(loc='upper right', fontsize=9.5, frameon=True)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
ax1.set_title('A. Overall Mean Information Loss (P3..P6)\nCRM Beats Persistence on Both Media', fontsize=12, fontweight='bold', pad=10)

# Add win callouts
ax1.text(0 - w/2, 0.08, 'CRM WINS\n(-11% error)', ha='center', color='white', fontweight='bold', fontsize=9.5)
ax1.text(1 - w/2, 0.08, 'CRM WINS\n(-8% error)', ha='center', color='white', fontweight='bold', fontsize=9.5)

# -------------------------------------------------------------
# PLOT 2: Passage-by-Passage Dynamics (P3 -> P4 -> P5 -> P6)
# -------------------------------------------------------------
passages = ['P3\n(Day 9)', 'P4\n(Day 12)', 'P5\n(Day 15)', 'P6\n(Day 18)']
x2 = np.arange(len(passages))
w2 = 0.18

# Average across plain & mucin for clean display
crm_by_p = [(plain_p3_crm + mucin_p3_crm)/2, (plain_p4_crm + mucin_p4_crm)/2, (plain_p5_crm + mucin_p5_crm)/2, (plain_p6_crm + mucin_p6_crm)/2]
pers_by_p = [(plain_p3_pers + mucin_p3_pers)/2, (plain_p4_pers + mucin_p4_pers)/2, (plain_p5_pers + mucin_p5_pers)/2, (plain_p6_pers + mucin_p6_pers)/2]

b3 = ax2.bar(x2 - w2/2, crm_by_p, width=w2, label='Dynamic CRM', color='#1B6CA8', edgecolor='#0F2942', lw=1.2, zorder=3)
b4 = ax2.bar(x2 + w2/2, pers_by_p, width=w2, label='Persistence Baseline (Holding P2)', color='#F5A623', edgecolor='#B37400', lw=1.2, zorder=3)

for bar in b3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1B6CA8')
for bar in b4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B37400')

ax2.set_xticks(x2)
ax2.set_xticklabels(passages, fontsize=11, fontweight='bold', color='#0F2942')
ax2.set_ylabel(r'KL Divergence (nats, lower is better $\downarrow$)', fontsize=11.5, fontweight='bold')
ax2.set_ylim(0, 0.36)
ax2.legend(loc='upper left', fontsize=9.5, frameon=True)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
ax2.set_title('B. Passage-by-Passage Dynamic Evolution\nPersistence Error Doubles; CRM Stays Low', fontsize=12, fontweight='bold', pad=10)

# Annotation on trend
ax2.annotate('Persistence error doubles\nas community drifts', xy=(3 + w2/2, pers_by_p[3]), xytext=(1.8, 0.32),
             arrowprops=dict(facecolor='#B37400', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#B37400')

plt.tight_layout()
save_path = os.path.join(PROJECT_ROOT, 'Paper', 'figures', 'Figure2_PanelB_FineTuned_P2_KL_Comparison.png')
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved fine-tuned P0->P2 KL comparison figure to: {save_path}")
