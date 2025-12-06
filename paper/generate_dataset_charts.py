#!/usr/bin/env python
"""
Generate Dataset Overview Figures and Tables

Creates bar charts and summary tables for dataset overview.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Dataset statistics
DATASETS = {
    "SaTML CTF 2024": {"total": 300, "injections": 300, "safe": 0, "source": "IEEE SaTML"},
    "deepset": {"total": 662, "injections": 406, "safe": 256, "source": "HuggingFace"},
    "LLMail-Inject": {"total": 200, "injections": 200, "safe": 0, "source": "Microsoft"},
    "NotInject": {"total": 1500, "injections": 0, "safe": 1500, "source": "Expanded"},
    "Adversarial": {"total": 594, "injections": 594, "safe": 0, "source": "Synthetic"},
    "Multi-Lang": {"total": 72, "injections": 72, "safe": 0, "source": "Synthetic"},
}


def plot_dataset_composition(output_path: str):
    """Generate dataset composition bar chart."""
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    names = list(DATASETS.keys())
    injections = [DATASETS[n]["injections"] for n in names]
    safe = [DATASETS[n]["safe"] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, injections, width, label='Injections', color='#ef4444')
    bars2 = ax.bar(x + width/2, safe, width, label='Safe', color='#22c55e')
    
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Dataset Composition', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_dataset_table(output_path: str):
    """Generate dataset summary LaTeX table."""
    latex = r"""\begin{table}[h]
\centering
\caption{Dataset Summary}
\label{tab:datasets}
\begin{tabular}{lrrrll}
\toprule
\textbf{Dataset} & \textbf{Total} & \textbf{Inj.} & \textbf{Safe} & \textbf{Source} & \textbf{Purpose} \\
\midrule
"""
    
    purposes = {
        "SaTML CTF 2024": "Adaptive attacks",
        "deepset": "General benchmark",
        "LLMail-Inject": "Indirect attacks",
        "NotInject": "Over-defense eval",
        "Adversarial": "Robustness test",
        "Multi-Lang": "Cross-lingual test",
    }
    
    for name, stats in DATASETS.items():
        purpose = purposes.get(name, "-")
        latex += f"{name} & {stats['total']:,} & {stats['injections']:,} & {stats['safe']:,} & {stats['source']} & {purpose} \\\\\n"
    
    # Total row
    total = sum(d["total"] for d in DATASETS.values())
    total_inj = sum(d["injections"] for d in DATASETS.values())
    total_safe = sum(d["safe"] for d in DATASETS.values())
    latex += r"\midrule" + "\n"
    latex += f"\\textbf{{Total}} & \\textbf{{{total:,}}} & \\textbf{{{total_inj:,}}} & \\textbf{{{total_safe:,}}} & - & - \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Dataset Overview Figures")
    print("=" * 60)
    
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path("paper/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures and tables
    plot_dataset_composition(str(figures_dir / "dataset_composition.png"))
    generate_dataset_table(str(tables_dir / "dataset_summary.tex"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    total = sum(d["total"] for d in DATASETS.values())
    total_inj = sum(d["injections"] for d in DATASETS.values())
    print(f"Total samples: {total:,}")
    print(f"Injections: {total_inj:,} ({total_inj/total:.1%})")
    print(f"Safe: {total - total_inj:,} ({(total - total_inj)/total:.1%})")
    
    print(f"\n✅ Figures saved to: {figures_dir}")
    print(f"✅ Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
