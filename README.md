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

## 🧬 The Science Behind BIT (Balanced Intent Training)

### What is BIT?

**BIT solves the "over-defense problem"**: Traditional classifiers learn lexical shortcuts (e.g., "if contains 'ignore' → injection") rather than semantic intent. This causes **high false positive rates** (86% in baseline models) on benign prompts containing trigger words.

**The BIT mechanism:**
1. **Dataset Composition (40/40/20 split)**:
   - 40% injection attacks
   - 40% safe/benign prompts
   - **20% benign-trigger samples** (safe prompts containing injection-like keywords)

2. **Weighted Loss Optimization**:
   ```python
   # Standard training (w=1.0 for all samples)
   loss = sum(error(y_true, y_pred))

   # BIT training (w=2.0 for benign-trigger samples)
   loss = sum(
       1.0 * error(injection_samples) +
       1.0 * error(safe_samples) +
       2.0 * error(benign_trigger_samples)  # ← THE KEY
   )
   ```

3. **Result**: Model learns to distinguish **malicious intent** from **keyword presence**.

---

### Why It Works: The Proof

We have **three independent lines of evidence** that the weighted loss mechanism (not just data composition) drives improvement:

#### **Evidence 1: Inverse Weighting Experiment** ⭐⭐⭐

Train 3 models on **identical 40/40/20 data**, varying only the weight:

| Weight (w) | NotInject FPR | Interpretation |
|------------|---------------|----------------|
| w=2.0 (Full BIT) | **1.8%** | Upweight benign-triggers → best |
| w=1.0 (Uniform) | 12.4% | No weighting → worse |
| w=0.5 (Inverse) | 18.7% | Downweight benign-triggers → worst |

**Monotonic relationship proves directionality matters!** If improvement was just from "adding benign-trigger samples," all 3 would perform equally.

#### **Evidence 2: Architecture Independence** (Table 7)

| Trigger Word | DeBERTa FPR | XGBoost w/o BIT | XGBoost with BIT |
|--------------|-------------|-----------------|------------------|
| "ignore"     | 89.2% | 94.1% | **0%** ✅ |
| "bypass"     | 92.1% | 95.8% | **0%** ✅ |
| "jailbreak"  | 97.3% | 98.2% | **0%** ✅ |

**Proves it's the mechanism, not the model:** XGBoost without BIT suffers from keyword bias just like DeBERTa. Only BIT-trained models achieve 0% FPR.

#### **Evidence 3: Statistical Significance**

McNemar's test: **χ²=36.2, p<0.001** (n=339)

The 10.6pp FPR improvement (1.8% vs 12.4%) is **statistically significant**, not random noise.

---

### How to Use BIT

**Step 1: Prepare your dataset with benign-trigger samples**
```python
from train_bit_model import generate_synthetic_benign_triggers

# Generate samples like:
# "Please ignore my previous typo"
# "Take the bypass to avoid traffic"
# "iPhone jailbreak tutorial"
benign_triggers = generate_synthetic_benign_triggers(n_samples=1000)
```

**Step 2: Apply 40/40/20 balancing**
```python
# Balance dataset
n_total = 5000
injections = sample(all_injections, int(n_total * 0.4))    # 2000
safe = sample(all_safe, int(n_total * 0.4))                # 2000
benign_trigger = sample(benign_triggers, int(n_total * 0.2)) # 1000
```

**Step 3: Train with weighted loss**
```python
from src.detection.embedding_classifier import EmbeddingClassifier

# Assign weights
weights = [
    2.0 if sample_type == "benign_trigger" else 1.0
    for sample_type in types
]

# Train XGBoost with sample weights
classifier = EmbeddingClassifier(use_xgboost=True)
classifier.train_on_dataset(
    texts,
    labels,
    sample_weights=weights  # ← THE MAGIC
)
```

**Step 4: Optimize threshold for 98% recall**
```python
from train_bit_model import optimize_threshold

threshold = optimize_threshold(
    classifier,
    X_val,
    y_val,
    target_recall=0.98
)
# Result: threshold=0.764
```

---

### Reproducing Paper Results

**Run the complete BIT training pipeline:**
```bash
# Train BIT model from scratch (~10 minutes)
python train_bit_model.py

# Run inverse weighting proof experiment
python run_inverse_weight_experiment.py

# Verify statistical significance
python run_statistical_tests.py

# Evaluate on NotInject benchmark
python -m benchmarks.run_benchmark --paper --threshold 0.764
```

**Interactive notebook:** See `paper_walkthrough.ipynb` for step-by-step demonstration.

---

### Where Else Can BIT Be Applied?

The BIT mechanism is broadly applicable to any classification problem where models learn **lexical shortcuts** instead of **semantic intent**:

| Domain | Problem | BIT Solution |
|--------|---------|--------------|
| **Content Moderation** | Flags "Let's discuss racism" as toxic | Upweight educational content with toxic keywords |
| **Spam Detection** | Flags "Here's your invoice" as phishing | Upweight legitimate emails with trigger words |
| **Medical Diagnosis** | "Headache" → assumes brain tumor | Upweight benign conditions with severe symptoms |
| **Hate Speech Detection** | Flags reclaimed slurs, quotes | Upweight in-group usage, educational context |
| **Code Vulnerability** | Flags all `eval()` as vulnerable | Upweight safe usages in sandboxed contexts |
| **Resume Screening** | Over-optimizes for keyword matches | Upweight diverse backgrounds with fewer buzzwords |

**The pattern:** Whenever "the right answer for the wrong reason" is common, BIT forces learning of true decision boundaries.

---

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
