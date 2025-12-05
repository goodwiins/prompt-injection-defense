# Research Plan: Multi-Agent LLM Prompt Injection Defense

## Project Status

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Accuracy | 97.8% | ≥95% | ✅ Achieved |
| False Positive Rate | 5.4% | ≤5% | ⚠️ Close |
| Over-Defense Rate | 9.7% | ≤5% | 🔄 In Progress |
| Latency P95 | 4.3ms | <100ms | ✅ Achieved |

---

## Research Contributions

### Completed Contributions

1. **Multi-Layer Defense Architecture**
   - Detection, Coordination, and Response layers
   - Clean separation of concerns

2. **Cascade Ensemble Detection**
   - Fast path (MiniLM): 3-8ms
   - Deep path (MPNet): 100-200ms, triggered only on uncertainty
   - 25x faster than Lakera Guard

3. **MOF (Mitigating Over-Defense) Strategy**
   - Reduced over-defense from 86% → 9.7%
   - Synthetic NotInject samples with trigger words

4. **Research Paper Implementations**
   - LLM Tagging (Lee & Tiwari, ICLR 2025)
   - PeerGuard mutual validation (96% TPR)
   - OVON Protocol for secure messaging
   - Behavioral anomaly detection
   - Alert correlation (Galileo AI 2024)

---

## Planned Improvements

### Priority 1: Reduce Over-Defense Rate (9.7% → ≤5%)

- [ ] Expand NotInject dataset with more benign samples
- [ ] Implement contrastive learning on trigger-word samples
- [ ] Add calibration techniques (temperature scaling, Platt scaling)
- [ ] Use ensemble disagreement as uncertainty signal
- [ ] Fine-tune fast path threshold (currently 0.5)

### Priority 2: Ablation Studies

| Experiment | Purpose |
|------------|---------|
| Full system vs no pattern detection | Pattern detector contribution |
| Full system vs no ensemble | Ensemble contribution |
| Full system vs no cascade | Cascade routing benefit |
| Full system vs no MOF training | MOF strategy impact |
| Fast path only | Speed vs accuracy tradeoff |
| Deep path only | Maximum accuracy baseline |

Script to create: `scripts/run_ablations.py`

### Priority 3: Additional Baselines

- [ ] Rebuff (open source)
- [ ] LLM-Guard (open source)
- [ ] HuggingFace prompt injection classifier
- [ ] TF-IDF + SVM (simple baseline)
- [ ] Zero-shot LLM detection (GPT-4 as judge)
- [ ] DistilBERT fine-tuned baseline

### Priority 4: Adversarial Robustness

Test against:
- [ ] GCG (Greedy Coordinate Gradient) attacks
- [ ] AutoDAN automated jailbreaks
- [ ] Prompt injection CTF winning entries
- [ ] Obfuscation techniques (leetspeak, unicode, base64)
- [ ] Multi-turn conversation attacks

### Priority 5: Cross-Model Generalization

Evaluate detection on outputs from:
- [ ] GPT-4 / GPT-4o
- [ ] Claude 3.5 Sonnet
- [ ] Llama 3 70B
- [ ] Gemini Pro
- [ ] Mistral Large

---

## Experiments Tracking

### Experiment 1: Over-Defense Reduction

**Hypothesis**: Expanding NotInject dataset with domain-specific benign samples will reduce FPR.

**Setup**:
- Current NotInject: 339 samples
- Target: 1000+ samples
- Categories: coding help, system administration, security education

**Metrics**: FPR on NotInject, overall accuracy maintained

**Status**: Not started

---

### Experiment 2: Threshold Optimization

**Hypothesis**: Optimal thresholds differ per dataset/domain.

**Setup**:
- Sweep fast_threshold: [0.3, 0.4, 0.5, 0.6, 0.7]
- Sweep deep_threshold: [0.75, 0.80, 0.85, 0.90, 0.95]
- Generate ROC curves per dataset

**Metrics**: AUC-ROC, F1 at optimal threshold

**Status**: Not started

---

### Experiment 3: Ablation Study

**Hypothesis**: Each component contributes measurably to overall performance.

**Setup**:
```
Configurations to test:
1. Full system (baseline)
2. Pattern detection only
3. Embedding classifier only
4. Ensemble without cascade
5. Fast path only
6. Without MOF training data
```

**Metrics**: Accuracy, Precision, Recall, F1, FPR, Latency

**Status**: Not started

---

## Dataset Summary

| Dataset | Size | Type | Use |
|---------|------|------|-----|
| deepset/prompt-injections | 662 | Binary | Primary benchmark |
| SaTML CTF 2024 | 137k+ | Multi-turn | Adversarial attacks |
| LLMail-Inject | 208k | Indirect | Email-based attacks |
| NotInject | 339 | Over-defense | False positive testing |
| imoxto/prompt_injection_cleaned | 535k | Large-scale | Training data |

---

## Publication Roadmap

### Target Venues

| Venue | Deadline | Fit |
|-------|----------|-----|
| USENIX Security | Feb 2025 | Security focus |
| ACL 2025 | Jan 2025 | NLP focus |
| AAAI 2025 | Aug 2024 | Applied ML |
| NeurIPS 2025 | May 2025 | ML focus |

### Paper Outline

1. **Introduction**
   - Prompt injection threat in multi-agent systems
   - Limitations of existing defenses

2. **Related Work**
   - Single-agent defenses
   - Multi-agent security
   - Embedding-based detection

3. **Methodology**
   - Multi-layer architecture
   - Cascade ensemble detection
   - MOF strategy for over-defense

4. **Experiments**
   - Benchmark results
   - Ablation studies
   - Adversarial evaluation
   - Cross-model generalization

5. **Discussion**
   - Limitations
   - Future work

6. **Conclusion**

---

## Code TODOs

### High Priority
- [ ] `scripts/run_ablations.py` - Ablation study runner
- [ ] `scripts/adversarial_eval.py` - Adversarial attack testing
- [ ] `benchmarks/baselines/` - Additional baseline implementations
- [ ] `notebooks/error_analysis.ipynb` - Failure mode categorization

### Medium Priority
- [ ] Add confidence intervals to benchmark results
- [ ] Add cross-validation support
- [ ] Statistical significance tests (McNemar's)
- [ ] Latency vs accuracy Pareto curves

### Low Priority
- [ ] Multi-language attack dataset
- [ ] Visualization dashboard for results
- [ ] Model interpretability (attention visualization)

---

## Notes

### Key Findings

1. **Pattern detection catches obvious attacks** but misses semantic variations
2. **Embedding classifier generalizes well** but has higher latency
3. **Cascade routing** achieves best speed/accuracy tradeoff
4. **Over-defense** is the main remaining challenge

### Open Questions

1. How do we handle multi-turn conversation context?
2. Can we detect indirect prompt injection (data poisoning)?
3. What's the theoretical detection limit?
4. How do adversarial training techniques affect robustness?

---

## References

1. Lee & Tiwari. "Prompt Infection: LLM-to-LLM Prompt Injection" (ICLR 2025)
2. "PeerGuard: Mutual Reasoning Defense Against Prompt Poisoning" (2024)
3. "InjecGuard: Mitigating Over-Defense in Prompt Injection Detection" (2024)
4. "Cross-LLM Behavioral Backdoor Detection" (NeurIPS 2024)
5. Galileo AI. "Alert Correlation for LLM Security" (2024)
6. Open Voice Network. "OVON Protocol Specification" (2024)

---

*Last updated: 2025-12-05*
