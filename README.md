# Multi-Agent LLM Prompt Injection Defense Framework

A comprehensive defense system achieving **97.6% accuracy** with **1.8% over-defense** against prompt injection attacks in multi-agent LLM systems using **Balanced Intent Training (BIT)**.

## 🎯 Key Results (Paper-Aligned Benchmark)

| Dataset           | Samples   | Accuracy  | Recall | FPR      | P95 Latency |
| ----------------- | --------- | --------- | ------ | -------- | ----------- |
| SaTML CTF 2024    | 300       | **98.7%** | 98.7%  | 0.0%     | 4.2ms       |
| deepset (attacks) | 203       | 92.6%     | 92.6%  | 0.0%     | 3.8ms       |
| NotInject HF      | 339       | 98.2%     | N/A    | **1.8%** | 1.8ms       |
| LLMail-Inject     | 200       | **100%**  | 100%   | 0.0%     | 3.5ms       |
| **Overall**       | **1,042** | **97.6%** | -      | **1.8%** | ~3ms        |

### Target Status

- ✅ **Accuracy ≥ 95%**: 97.6%
- ✅ **FPR ≤ 5%**: 1.8%
- ✅ **Latency P95 < 100ms**: 4.2ms

## 🏆 Baseline Comparison

| Model               | Accuracy  | FPR      | Latency  |
| ------------------- | --------- | -------- | -------- |
| **BIT (Ours)**      | **97.6%** | **1.8%** | **~3ms** |
| Lakera Guard        | 87.9%     | 5.7%     | 66ms     |
| ProtectAI           | 90.0%     | -        | 500ms    |
| Glean AI            | 97.8%     | 3.0%     | -        |
| HuggingFace DeBERTa | 90.0%     | 10.0%    | 48ms     |

**25x faster than Lakera Guard with 11% better accuracy!**

## 🛡️ Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DETECTION LAYER                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Pattern   │  │  Embedding   │  │ Behavioral │ │
│  │  Detector   │  │  Classifier  │  │  Monitor   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│               COORDINATION LAYER                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Guard Agent │  │  PeerGuard   │  │   OVON     │ │
│  │             │  │  Validator   │  │  Protocol  │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│                 RESPONSE LAYER                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Circuit    │  │    Alert     │  │ Quarantine │ │
│  │  Breaker    │  │ Correlation  │  │  Manager   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Train BIT model
python train_bit_model.py

# Run paper-aligned benchmark
python -m benchmarks.run_benchmark --paper --threshold 0.764
```

```python
from src.detection.embedding_classifier import EmbeddingClassifier

detector = EmbeddingClassifier()
detector.load_model("models/bit_xgboost_model.json")

result = detector.predict(["Ignore all previous instructions"])
# Output: [1]  (1 = injection detected)
```

## 📊 Comprehensive Evaluation

### Adversarial Robustness (92.1%)

| Technique        | Detection Rate |
| ---------------- | -------------- |
| base64 encoding  | 100%           |
| word splitting   | 100%           |
| leetspeak        | 89%            |
| homoglyphs       | 89%            |
| zero-width chars | 89%            |

### Cross-Model Generalization (GPT-4: 89.5%)

| Attack Type        | Detection |
| ------------------ | --------- |
| DAN/Jailbreak      | 100%      |
| Prompt Extraction  | 100%      |
| Indirect Injection | 100%      |
| Social Engineering | 50%       |

### Multi-Language Detection (61% overall)

| Language | Detection |
| -------- | --------- |
| Arabic   | 100%      |
| Russian  | 100%      |
| Korean   | 100%      |
| Chinese  | 78%       |
| Japanese | 75%       |

## 📁 Project Structure

```
├── src/
│   ├── detection/
│   │   ├── embedding_classifier.py  # XGBoost + embeddings
│   │   ├── ensemble.py              # Multi-model ensemble
│   │   └── patterns.py              # 10-category regex
│   ├── coordination/
│   │   ├── guard_agent.py           # Main orchestration
│   │   ├── peerguard.py             # Mutual validation
│   │   └── ovon_protocol.py         # OVON messaging
│   └── response/
│       ├── circuit_breaker.py       # Tiered alerts
│       └── quarantine.py            # Agent isolation
├── scripts/
│   ├── run_baselines.py             # Baseline comparison
│   ├── adversarial_eval.py          # Adversarial testing
│   ├── cross_model_gpt4.py          # GPT-4 evaluation
│   ├── calculate_tivs.py            # TIVS metric
│   ├── statistical_analysis.py      # CIs, McNemar's test
│   ├── error_analysis.py            # Failure categorization
│   ├── multilang_attacks.py         # Multi-language dataset
│   ├── generate_dashboard.py        # HTML dashboard
│   └── interpretability.py          # Model explainability
├── benchmarks/
│   ├── run_benchmark.py             # Main benchmark runner
│   └── baselines/                   # TF-IDF, HuggingFace
├── paper/                           # Academic paper assets
│   ├── figures/                     # Generated charts (9 PNG)
│   ├── tables/                      # LaTeX tables (4 TEX)
│   └── generate_*.py                # Figure generation scripts
├── results/                         # All evaluation results
├── dashboard.html                   # Interactive visualization
└── docs/
    └── PROJECT_FEEDBACK_REPORT.md   # Academic feedback
```

## 📊 Paper Figures

Generate publication-ready figures:

```bash
# ROC and PR curves (AUC = 0.9985)
python paper/generate_roc_curves.py

# Ablation study charts
python paper/generate_ablation_charts.py

# Latency analysis (CDF, boxplot)
python paper/generate_latency_charts.py

# MOF over-defense analysis
python paper/generate_mof_charts.py

# Dataset composition
python paper/generate_dataset_charts.py
```

### Generated Assets

| Figure                      | Description                  |
| --------------------------- | ---------------------------- |
| `roc_deepset.png`           | ROC curve (AUC=0.9985)       |
| `pr_deepset.png`            | Precision-Recall curve       |
| `ablation_accuracy.png`     | Accuracy/F1 by configuration |
| `ablation_errors.png`       | FPR/FNR comparison           |
| `latency_cdf.png`           | Latency CDF (P50=1.8ms)      |
| `overdefense_threshold.png` | MOF vs no-MOF FPR            |
| `dataset_composition.png`   | Sample distribution          |

| Table                     | Description         |
| ------------------------- | ------------------- |
| `ablation_table.tex`      | Ablation metrics    |
| `baseline_comparison.tex` | Industry comparison |
| `mof_ablation.tex`        | MOF impact          |
| `dataset_summary.tex`     | Dataset overview    |

## 🔬 Run Evaluations

```bash
# Paper-aligned benchmark (recommended, 1,042 samples)
python -m benchmarks.run_benchmark --paper --threshold 0.764

# Full benchmark on all datasets
python -m benchmarks.run_benchmark --all

# Quick benchmark (100 samples per dataset)
python -m benchmarks.run_benchmark --quick

# Specific datasets
python -m benchmarks.run_benchmark --datasets satml deepset_injections notinject_hf llmail

# Baseline comparison
python scripts/run_baselines.py

# Statistical analysis (95% CI, McNemar's)
python scripts/statistical_analysis.py

# Generate dashboard
python scripts/generate_dashboard.py
open dashboard.html
```

## 📈 TIVS (Total Injection Vulnerability Score)

```
TIVS = (ISR × 0.4) + (POF × 0.2) + (FPR × 0.25) - (PSR × 0.15)
```

| System         | TIVS        | Status     |
| -------------- | ----------- | ---------- |
| **MOF (Ours)** | **-0.1065** | Best       |
| ProtectAI      | -0.0700     |            |
| Lakera Guard   | -0.0597     |            |
| HuggingFace    | +0.2250     | Vulnerable |

## 📚 Research Foundations

- **Balanced Intent Training (BIT)** - Novel training strategy for over-defense mitigation
- **LLM Tagging** (Lee & Tiwari, ICLR 2025)
- **InjecGuard** MOF training strategy (Liang et al., ACL 2025)
- **NotInject** over-defense benchmark (Liang et al., 2024)
- **BrowseSafe** HTML modality analysis (Perplexity, 2025)
- **OVON Protocol** for agent messaging

## 📄 Citation

```bibtex
@software{bit_prompt_injection_defense_2025,
  title={Multi-Agent LLM Prompt Injection Defense with Balanced Intent Training},
  author={Abdel El Bikha, Jennifer Marrero},
  year={2025},
  url={https://github.com/goodwiins/prompt-injection-defense}
}
```

## 📜 License

MIT License
