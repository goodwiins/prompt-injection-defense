"""
Benchmark Reporter

Generate formatted reports in console, JSON, and Markdown formats.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import structlog

from .metrics import BenchmarkMetrics, BASELINE_METRICS, compare_to_baselines
from .runner import BenchmarkResults

logger = structlog.get_logger()


class BenchmarkReporter:
    """
    Generate benchmark reports in multiple formats.
    """
    
    # Target thresholds
    TARGETS = {
        "accuracy": 0.95,
        "max_fpr": 0.05,
        "max_over_defense": 0.05,
        "max_latency_p95": 100.0,
    }
    
    def __init__(self, results: BenchmarkResults):
        """
        Initialize reporter with benchmark results.
        
        Args:
            results: BenchmarkResults from benchmark run
        """
        self.results = results
    
    def print_console(self, show_baselines: bool = True) -> None:
        """
        Print formatted report to console.
        
        Args:
            show_baselines: Whether to show baseline comparisons
        """
        self._print_header()
        self._print_summary_table()
        self._print_overall_metrics()
        self._print_target_status()
        
        if show_baselines:
            self._print_baseline_comparison()
    
    def _print_header(self) -> None:
        """Print report header."""
        print("\n" + "═" * 80)
        print("                     BENCHMARK RESULTS SUMMARY")
        print("═" * 80)
        print(f"Detector: {self.results.metadata.get('detector_type', 'N/A')}")
        print(f"Threshold: {self.results.metadata.get('threshold', 0.5)}")
        print(f"Timestamp: {self.results.metadata.get('timestamp', 'N/A')}")
        print("─" * 80)
    
    def _print_summary_table(self) -> None:
        """Print dataset summary table."""
        # Header
        print(f"\n{'Dataset':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>8} {'FPR':>8} {'Lat(P95)':>10}")
        print("─" * 80)
        
        # Data rows
        for name, metrics in self.results.results.items():
            acc = f"{metrics.accuracy:.1%}"
            prec = f"{metrics.precision:.1%}" if metrics.precision > 0 else "N/A"
            rec = f"{metrics.recall:.1%}" if metrics.recall > 0 else "N/A"
            f1 = f"{metrics.f1_score:.1%}" if metrics.f1_score > 0 else "N/A"
            fpr = f"{metrics.false_positive_rate:.1%}"
            lat = f"{metrics.latency_p95:.1f}ms"
            
            # Highlight NotInject for over-defense
            if "notinject" in name.lower():
                name_display = f"{name} (OD)"
            else:
                name_display = name
            
            print(f"{name_display:<25} {acc:>10} {prec:>10} {rec:>10} "
                  f"{f1:>8} {fpr:>8} {lat:>10}")
        
        print("─" * 80)
    
    def _print_overall_metrics(self) -> None:
        """Print overall aggregated metrics."""
        print(f"\n{'OVERALL':<25} {self.results.overall_accuracy:>10.1%} "
              f"{'':>10} {'':>10} {'':>8} {self.results.overall_fpr:>8.1%}")
        
        if self.results.over_defense_rate is not None:
            print(f"Over-Defense Rate: {self.results.over_defense_rate:.1%}")
    
    def _print_target_status(self) -> None:
        """Print status of target thresholds."""
        print("\n" + "─" * 80)
        print("TARGET STATUS:")
        
        # Accuracy target
        acc_met = self.results.overall_accuracy >= self.TARGETS["accuracy"]
        status = "✅" if acc_met else "❌"
        print(f"  {status} Accuracy >= {self.TARGETS['accuracy']:.0%}: "
              f"{self.results.overall_accuracy:.1%}")
        
        # FPR target
        fpr_met = self.results.overall_fpr <= self.TARGETS["max_fpr"]
        status = "✅" if fpr_met else "❌"
        print(f"  {status} FPR <= {self.TARGETS['max_fpr']:.0%}: "
              f"{self.results.overall_fpr:.1%}")
        
        # Over-defense target
        if self.results.over_defense_rate is not None:
            od_met = self.results.over_defense_rate <= self.TARGETS["max_over_defense"]
            status = "✅" if od_met else "❌"
            print(f"  {status} Over-Defense Rate <= {self.TARGETS['max_over_defense']:.0%}: "
                  f"{self.results.over_defense_rate:.1%}")
        
        # Latency target
        max_lat = max(
            m.latency_p95 for m in self.results.results.values()
        ) if self.results.results else 0
        lat_met = max_lat <= self.TARGETS["max_latency_p95"]
        status = "✅" if lat_met else "❌"
        print(f"  {status} Latency P95 < {self.TARGETS['max_latency_p95']:.0f}ms: "
              f"{max_lat:.1f}ms")
    
    def _print_baseline_comparison(self) -> None:
        """Print comparison against industry baselines."""
        print("\n" + "─" * 80)
        print("BASELINE COMPARISONS:")
        
        # Use overall metrics for comparison
        # Use overall metrics for comparison
        overall_metrics = BenchmarkMetrics(
            accuracy=self.results.overall_accuracy,
            false_positive_rate=self.results.overall_fpr,
            false_negative_rate=self.results.overall_fnr,
            precision=self.results.overall_precision,
            recall=self.results.overall_recall,
            f1_score=self.results.overall_f1,
            latency_p50=max(
                m.latency_p50 for m in self.results.results.values()
            ) if self.results.results else 0
        )
        
        comparisons = compare_to_baselines(overall_metrics)
        
        for baseline_name, comparison in comparisons.items():
            improvements = comparison.get("improvements", {})
            if not improvements:
                continue
            
            print(f"\n  vs {baseline_name}:")
            for metric, data in improvements.items():
                if data["is_better"]:
                    arrow = "↑" if data["improvement_pct"] > 0 else "↓"
                    color = "better"
                else:
                    arrow = "↓" if data["improvement_pct"] < 0 else "↑"
                    color = "worse"
                
                print(f"    {metric}: {data['current']:.4f} vs {data['baseline']:.4f} "
                      f"({arrow} {abs(data['improvement_pct']):.1f}%)")
    
    def to_markdown(self) -> str:
        """
        Generate Markdown report.
        
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append("# Benchmark Results\n")
        lines.append(f"**Detector:** {self.results.metadata.get('detector_type', 'N/A')}")
        lines.append(f"**Threshold:** {self.results.metadata.get('threshold', 0.5)}")
        lines.append(f"**Timestamp:** {self.results.metadata.get('timestamp', 'N/A')}\n")
        
        # Summary table
        lines.append("## Dataset Results\n")
        lines.append("| Dataset | Accuracy | Precision | Recall | F1 | FPR | Latency (P95) |")
        lines.append("|---------|----------|-----------|--------|----|-----|---------------|")
        
        for name, metrics in self.results.results.items():
            lines.append(
                f"| {name} | {metrics.accuracy:.1%} | {metrics.precision:.1%} | "
                f"{metrics.recall:.1%} | {metrics.f1_score:.1%} | "
                f"{metrics.false_positive_rate:.1%} | {metrics.latency_p95:.1f}ms |"
            )
        
        # Overall
        lines.append(f"\n**Overall Accuracy:** {self.results.overall_accuracy:.1%}")
        lines.append(f"**Overall FPR:** {self.results.overall_fpr:.1%}")
        
        if self.results.over_defense_rate is not None:
            lines.append(f"**Over-Defense Rate:** {self.results.over_defense_rate:.1%}")
        
        # Target status
        lines.append("\n## Target Status\n")
        
        acc_met = self.results.overall_accuracy >= self.TARGETS["accuracy"]
        lines.append(f"- {'✅' if acc_met else '❌'} Accuracy ≥ {self.TARGETS['accuracy']:.0%}")
        
        fpr_met = self.results.overall_fpr <= self.TARGETS["max_fpr"]
        lines.append(f"- {'✅' if fpr_met else '❌'} FPR ≤ {self.TARGETS['max_fpr']:.0%}")
        
        if self.results.over_defense_rate is not None:
            od_met = self.results.over_defense_rate <= self.TARGETS["max_over_defense"]
            lines.append(f"- {'✅' if od_met else '❌'} Over-Defense Rate ≤ {self.TARGETS['max_over_defense']:.0%}")
        
        return "\n".join(lines)
    
    def to_json(self, indent: int = 2) -> str:
        """
        Generate JSON report.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON formatted string
        """
        return self.results.to_json(indent=indent)
    
    def save(self, path: str, format: str = "auto") -> None:
        """
        Save report to file.
        
        Args:
            path: Output file path
            format: Output format ('json', 'markdown', 'auto')
        """
        path = Path(path)
        
        if format == "auto":
            if path.suffix == ".json":
                format = "json"
            elif path.suffix in [".md", ".markdown"]:
                format = "markdown"
            else:
                format = "json"
        
        if format == "json":
            content = self.to_json()
        elif format == "markdown":
            content = self.to_markdown()
        else:
            raise ValueError(f"Unknown format: {format}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        
        logger.info("Report saved", path=str(path), format=format)


def print_quick_summary(results: BenchmarkResults) -> None:
    """
    Print a quick one-line summary of results.
    
    Args:
        results: BenchmarkResults to summarize
    """
    print(f"Accuracy: {results.overall_accuracy:.1%} | "
          f"FPR: {results.overall_fpr:.1%} | "
          f"Datasets: {len(results.results)}")
