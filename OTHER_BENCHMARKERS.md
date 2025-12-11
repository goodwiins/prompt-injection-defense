# Other Prompt Injection Benchmarks and Tools

This document provides an overview of other benchmarkers and tools available for prompt injection detection, both within this codebase and in the broader research community.

## 📊 Benchmarks in This Codebase

### 1. Built-in Datasets
- **SaTML CTF 2024** - 136K+ samples from adversarial competition
- **deepset/prompt-injections** - 546 samples of diverse injection attempts
- **NotInject** - 339 samples testing over-defense (benign with trigger words)
- **LLMail-Inject** - 370K+ email-based indirect injection scenarios
- **BrowseSafe-Bench** - 14K+ HTML-embedded attacks for browser agents
- **AgentDojo** - 97 multi-agent workflow injection scenarios
- **TensorTrust** - 126K+ human-generated adversarial examples

### 2. Baseline Implementations

#### HuggingFace Classifier
```python
from benchmarks.baselines.hf_classifier import HuggingFaceBaseline

# Pre-trained models available:
- protectai/deberta-v3-base-prompt-injection
- deepset/roberta-base-sentiment
- microsoft/deberta-v3-base

baseline = HuggingFaceBaseline("protectai/deberta-v3-base-prompt-injection")
predictions = baseline.predict(texts)
```

#### TF-IDF SVM
```python
from benchmarks.baselines.tfidf_svm import TfidfSVMBaseline

baseline = TfidfSVMBaseline()
baseline.train(train_texts, train_labels)
predictions = baseline.predict(test_texts)
```

## 🏆 External Benchmarks

### 1. AgentDojo (NeurIPS 2024)
- **Source**: https://github.com/ethz-spylab/agentdojo
- **Samples**: 97 scenarios across multiple domains
- **Domains**: Banking, Slack, Travel, Workspace
- **Focus**: End-to-end attack success in multi-agent workflows
- **Evaluation**: ASR (Attack Success Rate), ALR (Attack Leakage Rate)

### 2. TensorTrust
- **Source**: https://github.com/tensortrust/tensortrust-data
- **Samples**: 126K+ human-generated adversarial examples
- **Method**: Crowdsourced red team attempts
- **Features**: Diverse attack strategies, real-world prompts
- **Evaluation**: Precision, Recall, F1, human evaluation

### 3. JailbreakHub
- **Source**: https://github.com/JailbreakHub/chatjailbreakhub
- **Samples**: 100K+ jailbreak attempts
- **Categories**: Roleplay, Instruction following, Refusal
- **Focus**: LLM safety boundaries
- **Evaluation**: Success rate, diversity of attacks

### 4. InjexBench
- **Source**: https://github.com/Veritas-Institute/InjexBench
- **Samples**: 1K+ injection scenarios
- **Types**: Direct, indirect, multi-turn attacks
- **Languages**: English, Chinese, Japanese, etc.
- **Evaluation**: Multi-lingual detection performance

### 5. Prompt Injection Benchmark (BREACH)
- **Source**: https://github.com/jianzhnie/Prompt-Injection-Benchmark
- **Samples**: 500+ curated injections
- **Categories**: Prompt leaking, context hijacking, token smuggling
- **Focus**: Advanced injection techniques
- **Evaluation**: Detection resistance, effectiveness

## 🛡️ Commercial Solutions (for comparison)

### 1. Lakera Guard
- **Accuracy**: 87.91%
- **Latency**: 66ms (P50)
- **FPR**: 5.70%
- **Deployment**: Cloud API
- **Strength**: Real-time protection, low false positives

### 2. ProtectAI LLM Guard
- **Accuracy**: 90.00%
- **Latency**: 500ms (P50)
- **Features**: OWASP Top 10 coverage
- **Deployment**: Cloud/on-premise
- **Strength**: Comprehensive security scanning

### 3. ActiveFence
- **F1 Score**: 85.70%
- **FPR**: 5.40%
- **Specialization**: Content moderation
- **Deployment**: Enterprise API
- **Strength**: Contextual understanding

### 4. Glean AI
- **Accuracy**: 97.80%
- **FPR**: 3.00%
- **Focus**: Enterprise search safety
- **Deployment**: Cloud service
- **Strength**: High accuracy, low false positives

### 5. PromptArmor
- **FPR**: 0.56%
- **FNR**: 0.13%
- **Specialization**: Lightweight detection
- **Deployment**: Edge/cloud
- **Strength**: Minimal over-blocking

## 📈 Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Minimizing false positives
- **Recall**: Detecting all attacks (most important)
- **F1-Score**: Harmonic mean of precision and recall
- **FPR**: False Positive Rate (over-defense)
- **Latency**: Response time (P50, P95, P99)

### Specialized Metrics
- **ASR**: Attack Success Rate (AgentDojo)
- **TIVS**: Total Injection Vulnerability Score
- **Over-defense Rate**: False alarms on benign trigger prompts
- **Coverage**: Percentage of attack types detected
- **Robustness**: Performance against unseen attacks

## 🚀 Running Benchmarks

### In This Codebase
```bash
# Paper-aligned benchmark (recommended)
python -m benchmarks.run_benchmark --paper

# All datasets
python -m benchmarks.run_benchmark --all

# Quick test
python -m benchmarks.run_benchmark --quick --samples 100

# Specific datasets
python -m benchmarks.run_benchmark --datasets satml deepset browsesafe

# Include baseline comparisons
python -m benchmarks.run_benchmark --paper --no-baselines
```

### External Benchmarks
```bash
# AgentDojo
cd benchmarks/external/agentdojo
python examples/pipeline.py

# TensorTrust (via HuggingFace)
python -c "from datasets import load_dataset; ds = load_dataset('tensortrust/tensortrust-data')"

# Custom evaluation
python benchmarks/evaluate_custom.py --model-path models/bit_xgboost_model.json
```

## 📝 Best Practices

1. **Use Multiple Datasets**: No single benchmark captures all attack types
2. **Report FPR**: Critical for practical deployment (over-defense)
3. **Include Latency**: Real-time applications need sub-100ms response
4. **Test Edge Cases**: Benign prompts with trigger words
5. **Statistical Significance**: Use confidence intervals for small datasets
6. **Reproducibility**: Document random seeds, data splits, thresholds

## 🔗 Resources

- [Papers with Code](https://paperswithcode.com/task/prompt-injection)
- [HuggingFace Hub - Text Classification](https://huggingface.co/models?pipeline=text-classification&language=en&sort=trending)
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-models/)
- [LLM Security GitHub](https://github.com/topics/llm-security)

## 💡 Key Insights

1. **BIT Mechanism**: Achieves 98.6% accuracy with 2.7% FPR using balanced training
2. **Trade-offs**: Higher recall often means higher FPR - balance is key
3. **Domain Specific**: Email attacks differ from web-based attacks
4. **Evolution**: Attack techniques constantly evolve - need continuous evaluation
5. **Practical**: Low latency (<10ms) enables real-time protection