import matplotlib.pyplot as plt
import numpy as np
import os

# Create Figures directory if it doesn't exist
os.makedirs('Figures', exist_ok=True)

# Data for Panel A: Empirical T6 Methanogen Abundance
osu_emp_5 = 52.32
osu_emp_30 = 49.80
brrs_emp_5 = 64.22
brrs_emp_30 = 69.33

# Data for Panel B: Trained on OSU + BRRS non-MeOH (The Breakthrough)
cv_pred_5 = 41.29
cv_pred_30 = 60.67

# Let's create a 1x2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ['5.0 bar', '30.0 bar']
x = np.arange(len(labels))
width = 0.35

# Panel A: Empirical True Abundance
ax1.bar(x - width/2, [osu_emp_5, osu_emp_30], width, label='OSU', color='#1f77b4')
ax1.bar(x + width/2, [brrs_emp_5, brrs_emp_30], width, label='BRRS', color='#ff7f0e')
ax1.set_ylabel('T6 Methanogen Abundance (%)', fontsize=12)
ax1.set_title('A. Empirical Reality (MeOH)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.legend(fontsize=11)
ax1.set_ylim(0, 80)
for i, v in enumerate([osu_emp_5, osu_emp_30]):
    ax1.text(i - width/2, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
for i, v in enumerate([brrs_emp_5, brrs_emp_30]):
    ax1.text(i + width/2, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

# Panel B: Trained on OSU + BRRS baselines
ax2.bar(x - width/2, [brrs_emp_5, brrs_emp_30], width, label='True BRRS', color='#ff7f0e', alpha=0.5)
ax2.bar(x + width/2, [cv_pred_5, cv_pred_30], width, label='Predicted BRRS', color='#2ca02c')
ax2.set_ylabel('T6 Methanogen Abundance (%)', fontsize=12)
ax2.set_title('B. Model Transfer Prediction', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=12)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 80)
for i, v in enumerate([brrs_emp_5, brrs_emp_30]):
    ax2.text(i - width/2, v + 2, f"{v:.1f}%", ha='center', color='gray')
for i, v in enumerate([cv_pred_5, cv_pred_30]):
    ax2.text(i + width/2, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig('Figures/MeOH_Transfer_Final_Message.png', dpi=300)
print("Plot saved to Figures/MeOH_Transfer_Final_Message.png")
