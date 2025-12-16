
import json

def generate_per_dataset_metrics_table():
    with open("results/latest_benchmark.json") as f:
        data = json.load(f)

    latex = r"""\begin{table}[H]
\centering
\caption{Per-Dataset Detection Metrics}
\label{tab:per-dataset-metrics}
\begin{tabular}{lccccc}
\toprule
\textbf{Dataset} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{FPR} \\
\midrule
"""

    datasets = data["datasets"]
    for name, metrics in datasets.items():
        if name == "Overall":
            continue

        accuracy = f"{metrics['accuracy'] * 100:.1f}\%"
        precision = f"{metrics['precision'] * 100:.1f}\%" if metrics['precision'] > 0 else "N/A"
        recall = f"{metrics['recall'] * 100:.1f}\%" if metrics['recall'] > 0 else "N/A"
        f1 = f"{metrics['f1'] * 100:.1f}\%" if metrics['f1'] > 0 else "N/A"
        fpr = f"{metrics['fpr'] * 100:.1f}\%"

        # The names in the json file are different from the paper
        if name == "deepset_benign":
            name = "deepset (benign)"
        elif name == "deepset_injections":
            name = "deepset (injections)"
        
        latex += f"{name} & {accuracy} & {precision} & {recall} & {f1} & {fpr} \\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open("paper/tables/per_dataset_metrics.tex", "w") as f:
        f.write(latex)
    print("Saved: paper/tables/per_dataset_metrics.tex")

if __name__ == "__main__":
    generate_per_dataset_metrics_table()
