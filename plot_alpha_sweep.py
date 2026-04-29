import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "final_results/testing_winners"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("final_results/validation_sweep/sweep_results.csv")
hyb = df[df['feature_set'] == 'hybrid'].copy()
hyb['alpha'] = hyb['alpha'].astype(float)

fig, ax = plt.subplots(figsize=(7, 4.5))
markers = {'RandomForest': 'o', 'LogReg': 's', 'GaussianNB': '^'}
for clf in ['LogReg', 'RandomForest', 'GaussianNB']:
    sub = hyb[hyb['classifier'] == clf].sort_values('alpha')
    ax.plot(sub['alpha'], sub['val_macro_f1'], marker=markers[clf], label=clf, linewidth=2)
    best = sub.loc[sub['val_macro_f1'].idxmax()]
    ax.scatter([best['alpha']], [best['val_macro_f1']], s=140, facecolors='none',
               edgecolors='red', linewidths=1.8, zorder=5)

ax.set_xlabel(r"$\alpha$ (stylometric weight)")
ax.set_ylabel("Validation macro-F1")
ax.set_title(r"Hybrid blend: val macro-F1 vs $\alpha$")
ax.set_xticks([i/10 for i in range(11)])
ax.grid(alpha=0.3)
ax.legend(loc='lower center', ncol=3)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "hybrid_alpha_sweep.png")
plt.savefig(out_path, dpi=150)
print(f"saved {out_path}")
