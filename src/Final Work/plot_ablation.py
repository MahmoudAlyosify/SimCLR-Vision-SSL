import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def main():
    # Setup aesthetic for IEEE-style academic plots
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titlesize': 16,
        'figure.dpi': 300
    })

    # Data from Linear Probe Evaluation
    experiments = [
        "Exp 36\n(Pure Rotation)",
        "Exp 35\n(Weak Baseline)",
        "Exp 9\n(Crop + Blur)",
        "Exp 13\n(Crop+Flip+Blur)",
        "Exp 10\n(Crop + Cutout)"
    ]

    # Historical data (Without Color Jitter)
    acc_without_jitter = [34.40, 59.22, 63.01, 64.49, 66.27]
    
    # Placeholders for new run (With Color Jitter)
    acc_with_jitter = [0.0, 0.0, 0.0, 0.0, 0.0]  

    df = pd.DataFrame({
        'Experiment': experiments * 2,
        'Top-1 Accuracy (%)': acc_without_jitter + acc_with_jitter,
        'Condition': ['Base Augmentation\n(Without Color Jitter)'] * 5 + ['Base + Color Jitter\n(With Jitter)'] * 5
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Custom color palette (Blue for without jitter, Green for with jitter)
    palette = {"Base Augmentation\n(Without Color Jitter)": "#1f77b4", "Base + Color Jitter\n(With Jitter)": "#2ca02c"}
    
    barplot = sns.barplot(
        data=df, 
        x='Experiment', 
        y='Top-1 Accuracy (%)', 
        hue='Condition',
        palette=palette,
        edgecolor='black',
        linewidth=1.2,
        ax=ax
    )

    # Add data labels above the bars
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:  # Only label non-zero bars
            ax.annotate(f'{height:.2f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=11, fontweight='bold')

    # Formatting & Axis labels
    ax.set_title('Impact of Color Jitter on Shortcut Learning in SimCLR\n(Linear Probe Top-1 Test Accuracy)', pad=20, fontweight='bold')
    ax.set_ylabel('Top-1 Test Accuracy (%)', fontweight='bold', labelpad=15)
    ax.set_xlabel('Structural Augmentation Strategy', fontweight='bold', labelpad=15)
    ax.set_ylim(0, 100)
    
    # Customize Legend
    ax.legend(title='', loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9, edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    
    # Save outputs
    out_png = 'ablation_shortcut_learning.png'
    out_pdf = 'ablation_shortcut_learning.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Academic charts successfully saved to {out_png} and {out_pdf}")
    
if __name__ == "__main__":
    main()
