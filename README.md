# Multi-Agent LLM Prompt Injection Defense Framework

A comprehensive defense system achieving **97.3% accuracy** with **0.1% false positive rate** against prompt injection attacks in multi-agent LLM systems using **Injection-Aware MPNet**.

**[Read the Paper (PDF)](paper/paper.pdf)**

## Key Results

| Dataset           | Samples   | Accuracy  | Recall  | FPR      | F1 Score |
| ----------------- | --------- | --------- | ------- | -------- | -------- |
| SaTML CTF 2024    | 500       | 98.8%     | 98.8%   | 0.0%     | 99.4%    |
| deepset (full)    | 546       | **100%**  | 100%    | 0.0%     | 100%     |
| LLMail-Inject     | 500       | **100%**  | 100%    | 0.0%     | 100%     |
| NotInject (local) | 247       | **100%**  | N/A     | **0.0%** | N/A      |
| NotInject (HF)    | 339       | **100%**  | N/A     | **0.0%** | N/A      |
| BrowseSafe        | 500       | 98.2%     | 96.7%   | 0.5%     | 98.0%    |
| TensorTrust       | 500       | 83.2%     | 83.2%   | 0.0%     | 90.9%    |
| **Overall**       | **3,132** | **97.3%** | -       | **0.1%** | -        |

### Performance Highlights

- **0.0% FPR on NotInject**: Perfect over-defense mitigation
- **100% recall** on deepset and LLMail benchmarks
- **P50 latency: 35.5ms** | P95 latency: 43.3ms

## Baseline Comparison

| System                         | Accuracy  | FPR      | Latency  | Open Source |
| ------------------------------ | --------- | -------- | -------- | ----------- |
| **Injection-Aware MPNet (Ours)** | **97.3%** | **0.1%** | **35.5ms** | Yes       |
| Glean AI                       | 97.8%     | 3.0%     | -        | No          |
| ProtectAI                      | 90.0%     | -        | 500ms    | No          |
| HuggingFace DeBERTa            | 90.0%     | 10.0%    | 48ms     | No          |
| Lakera Guard                   | 87.9%     | 5.7%     | 66ms     | No          |

## Three-Layer Architecture

```
+-----------------------------------------------------+
|                  DETECTION LAYER                    |
|  +-----------+  +------------+  +----------------+  |
|  |  Pattern  |  | Embedding  |  |   Behavioral   |  |
|  | Detector  |  | Classifier |  |    Monitor     |  |
|  +-----------+  +------------+  +----------------+  |
+-----------------------------------------------------+
                         |
+-----------------------------------------------------+
|               COORDINATION LAYER                    |
|  +-----------+  +------------+  +----------------+  |
|  |   Guard   |  | PeerGuard  |  |      OVON      |  |
|  |   Agent   |  | Validator  |  |    Protocol    |  |
|  +-----------+  +------------+  +----------------+  |
+-----------------------------------------------------+
                         |
+-----------------------------------------------------+
|                 RESPONSE LAYER                      |
|  +-----------+  +------------+  +----------------+  |
|  |  Circuit  |  |   Alert    |  |   Quarantine   |  |
|  |  Breaker  |  |Correlation |  |    Manager     |  |
|  +-----------+  +------------+  +----------------+  |
+-----------------------------------------------------+
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark with pre-trained model
python -m benchmarks.run_benchmark --paper --model models/injection_aware_mpnet_classifier.json --threshold 0.764

# Train your own model
python finetune_mpnet_embeddings.py
```

### Basic Usage

```python
from src.detection.embedding_classifier import EmbeddingClassifier

detector = EmbeddingClassifier()
detector.load_model("models/injection_aware_mpnet_classifier.json")

result = detector.predict(["Ignore all previous instructions"])
# Output: [1]  (1 = injection detected)

result = detector.predict(["How do I ignore files in git?"])
# Output: [0]  (0 = safe, correctly handles benign-trigger)
```

## Injection-Aware MPNet: How It Works

### The Over-Defense Problem

Traditional classifiers learn lexical shortcuts (e.g., "if contains 'ignore' -> injection") rather than semantic intent. This causes high false positive rates on benign prompts containing trigger words like:
- "How do I **ignore** files in .gitignore?"
- "Explain the **bypass** mechanism in this circuit"
- "What is a **system** call in operating systems?"

### Our Solution: Contrastive Learning

We fine-tune `all-mpnet-base-v2` using contrastive learning on carefully curated sentence pairs:

**Training Pair Types:**
| Pair Type | Count | Purpose |
|-----------|-------|---------|
| Injection -- Injection | 2,000 | Cluster attacks together |
| Benign -- Benign | 2,000 | Cluster safe prompts together |
| Injection -- Benign | 2,000 | Separate attacks from safe |
| **Injection -- Benign-Trigger** | **4,000** | Key: separate attacks from trigger words |
| Benign-Trigger -- Benign-Trigger | 2,000 | Cluster benign-triggers together |
| Benign-Trigger -- Benign | 2,000 | Keep benign-triggers near safe |

**Key Insight:** We double the injection--benign-trigger pairs (4,000 vs 2,000) to emphasize distinguishing malicious intent from keyword presence.

### Training Pipeline

```python
# 1. Load base model
model = SentenceTransformer('all-mpnet-base-v2')

# 2. Create contrastive pairs with labels
# label=1.0 for similar pairs, label=0.0 for dissimilar pairs
train_examples = [
    InputExample(texts=[injection, benign_trigger], label=0.0),  # dissimilar
    InputExample(texts=[benign, benign], label=1.0),             # similar
    ...
]

# 3. Fine-tune with CosineSimilarityLoss
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=4)

# 4. Train XGBoost classifier on embeddings
embeddings = model.encode(texts)
classifier = XGBClassifier(max_depth=6, n_estimators=100)
classifier.fit(embeddings, labels)
```

## Project Structure

```
prompt-injection-defense/
├── src/
│   ├── detection/
│   │   ├── embedding_classifier.py  # XGBoost + embeddings
│   │   ├── ensemble.py              # Multi-model ensemble
│   │   └── patterns.py              # Regex pattern detector
│   ├── coordination/
│   │   ├── guard_agent.py           # Main orchestration
│   │   ├── peerguard.py             # Mutual validation
│   │   └── ovon_protocol.py         # OVON messaging
│   └── response/
│       ├── circuit_breaker.py       # Tiered alerts
│       └── quarantine.py            # Agent isolation
├── models/
│   ├── injection_aware_mpnet_classifier.json    # Main model
│   └── injection_aware_mpnet/                   # Fine-tuned embeddings
├── benchmarks/
│   ├── run_benchmark.py             # Main benchmark runner
│   ├── benchmark_datasets.py        # Dataset loaders
│   └── baselines/                   # TF-IDF, HuggingFace
├── training/
│   └── finetune_mpnet_embeddings.py # Model training script
├── paper/
│   ├── paper.tex                    # Academic paper
│   ├── figures/                     # Generated charts
│   └── tables/                      # LaTeX tables
├── data/                            # Benchmark datasets
├── notebooks/                       # Jupyter notebooks
└── docs/                            # Documentation
```

## Run Benchmarks

```bash
# Full benchmark on all datasets (3,132 samples)
python -m benchmarks.run_benchmark --paper --model models/injection_aware_mpnet_classifier.json --threshold 0.764

# Quick benchmark (100 samples per dataset)
python -m benchmarks.run_benchmark --quick

# Specific datasets
python -m benchmarks.run_benchmark --datasets satml deepset_injections notinject_hf llmail

# Generate paper tables
python -m benchmarks.run_benchmark --paper --model models/injection_aware_mpnet_classifier.json --threshold 0.764 --output-tables
```

## Datasets

| Dataset | Total | Injections | Safe | Source | Purpose |
|---------|-------|------------|------|--------|---------|
| SaTML CTF 2024 | 500 | 500 | 0 | IEEE SaTML | Adaptive attacks |
| deepset (full) | 546 | 203 | 343 | HuggingFace | General benchmark |
| LLMail-Inject | 500 | 500 | 0 | Microsoft | Indirect attacks |
| NotInject (local) | 247 | 0 | 247 | Synthetic | Over-defense eval |
| NotInject (HF) | 339 | 0 | 339 | HuggingFace | Over-defense eval |
| BrowseSafe | 500 | 227 | 273 | Perplexity | HTML-embedded attacks |
| TensorTrust | 500 | 500 | 0 | ETH Zurich | Adversarial attacks |
| **Total** | **3,132** | **1,930** | **1,202** | - | - |

## Research Foundations

- **Injection-Aware MPNet** - Contrastive learning for over-defense mitigation
- **InjecGuard** MOF training strategy (Liang et al., ACL 2025)
- **NotInject** over-defense benchmark (Liang et al., 2024)
- **BrowseSafe** HTML modality analysis (Perplexity, 2025)
- **TensorTrust** adversarial attacks (ETH Zurich)
- **OVON Protocol** for agent messaging

## Citation

```bibtex
@software{injection_aware_mpnet_2025,
  title={Multi-Agent LLM Prompt Injection Defense with Injection-Aware MPNet},
  author={Abdelghafour El Bikha},
  year={2025},
  institution={John Jay College},
  url={https://github.com/goodwiins/prompt-injection-defense}
}
```

## License

MIT License
