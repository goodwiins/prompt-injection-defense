# all-mpnet-base-v2 Model Assessment

**Model**: all-mpnet-base-v2 (768-dim embeddings, 109M parameters)
**Training**: BIT methodology (40% safe, 40% injection, 20% benign-triggers)
**Date**: 2025-12-15

---

## Executive Summary

The all-mpnet-base-v2 model represents **significant improvement** over all-MiniLM-L6-v2 (85.4% vs ~30% recall at comparable thresholds) but **cannot simultaneously achieve industry-standard recall (>90%) and acceptable FPR (<5%)**.

**Key Finding**: The model exhibits a fundamental recall-FPR tradeoff that cannot be resolved through threshold tuning alone.

---

## Performance Analysis

### Threshold Sweep Results (1,885 samples: 930 attacks, 955 safe)

| Threshold | Recall | FPR | Precision | F1 | Accuracy | Use Case |
|-----------|--------|-----|-----------|----|----|----------|
| 0.700 | **85.4%** | 12.4% | 87.1% | 86.2% | 86.5% | **High-security (Recommended)** |
| 0.764 | ~78% | 9.5% | ~85% | ~83% | ~82% | Balanced (suboptimal) |
| 0.850 | 67.7% | **4.8%** | 93.2% | 78.5% | 81.6% | Low-risk only |
| 0.900 | 55.1% | 1.6% | 97.2% | 70.3% | 77.0% | Ultra-conservative (not recommended) |

**Overall AUC**: 0.9414

---

## Critical Assessment

### ❌ **Cannot Meet Production Security Standards**

Industry standards for production security systems:
- **Consumer applications**: >85% recall
- **Enterprise security**: >90% recall
- **High-security contexts**: >95% recall
- **Acceptable FPR**: <5%

**This model cannot achieve both metrics simultaneously.**

### Real-World Impact Example

```
For a system receiving 100 prompt injection attacks per day:

θ=0.700 (recommended):
├─ Detected: 85 attacks ✅
├─ Missed: 15 attacks ❌
├─ False alarms: ~12 per 100 benign prompts
└─ Result: 15 breaches/day, manageable false positive load

θ=0.850 (NOT recommended):
├─ Detected: 68 attacks ✅
├─ Missed: 32 attacks ❌ ⚠️ UNACCEPTABLE
├─ False alarms: ~5 per 100 benign prompts
└─ Result: 32 breaches/day (2.1x worse attack rate)
```

**Choosing θ=0.850 doubles your breach rate to save 7% FPR.**

---

## Deployment Recommendations

### ✅ **Recommended: High-Security Configuration (θ=0.700)**

```python
# Configuration
classifier = EmbeddingClassifier(
    model_name='all-mpnet-base-v2',
    threshold=0.700  # Prioritize catching attacks
)

# Implementation with human review
def detect_with_review(prompt):
    result = classifier.predict_proba([prompt])[0, 1]

    if result >= 0.85:
        return "BLOCK"        # High confidence attack
    elif result >= 0.70:
        return "REVIEW"       # Medium confidence - human review
    else:
        return "ALLOW"        # Likely benign
```

**Rationale**:
- **85.4% recall** catches most attacks (acceptable for production)
- **12.4% FPR** requires human review pipeline but is manageable
- **In security, missing attacks is worse than false alarms**

**Requirements**:
- Human review capacity for ~12% of prompts
- Clear escalation procedures for flagged prompts
- Regular audit of false positives to tune system

---

### ⚠️ **Alternative: Low-Risk Applications Only (θ=0.850)**

```python
# Only for non-sensitive applications
classifier.threshold = 0.850
```

**Acceptable for**:
- Educational chatbots
- Public content generation
- Non-sensitive creative tools

**NOT acceptable for**:
- Financial services
- Healthcare applications
- PII/sensitive data handling
- Enterprise security

**Warning**: This configuration **misses 1 in 3 attacks** (32% missed attack rate).

---

## Comparison with Baselines

| System | Recall | FPR | Assessment |
|--------|--------|-----|------------|
| **Lakera Guard** | ~88% | 5.7% | Better recall, acceptable FPR |
| **PromptArmor** | ~99% | 0.56% | Significantly better on both metrics |
| **Glean AI** | ~97% | 3.0% | Much better overall |
| **all-mpnet-base-v2 (θ=0.700)** | **85.4%** | **12.4%** | Competitive recall, high FPR |
| **all-mpnet-base-v2 (θ=0.850)** | **67.7%** | **4.8%** | Unacceptable recall |

**Conclusion**: This model is **not yet competitive** with commercial solutions but shows promise with proper deployment strategy and further optimization.

---

## Dataset-Specific Performance (θ=0.700)

| Dataset | Samples | Recall | FPR | Notes |
|---------|---------|--------|-----|-------|
| SaTML CTF 2024 | 300 | ~96% | 0.0% | Excellent on adversarial attacks |
| LLMail-Inject | 200 | ~85% | 0.0% | Good on email injections |
| BrowseSafe | 500 | ~75% | 2.2% | Moderate on HTML attacks |
| deepset | 546 | ~74% | ~25% | Known labeling issues affect FPR |
| NotInject | 339 | 0% | 0.0% | Perfect - no false positives on trigger words |

---

## Known Limitations

1. **Cannot achieve >90% recall with <5% FPR** through threshold tuning alone
2. **12.4% FPR** at recommended threshold requires human review infrastructure
3. **Embeddings not specialized** for prompt injection detection
4. **Lower performance on HTML-based attacks** (BrowseSafe: 75% recall)
5. **High FPR on deepset** (partially due to dataset labeling issues)

---

## Next Steps for Improvement

### Immediate (Production Deployment)

1. **Deploy with θ=0.700** and human review pipeline
2. **Implement tiered response**:
   - confidence ≥ 0.85: Automatic block
   - confidence 0.70-0.85: Human review
   - confidence < 0.70: Allow
3. **Monitor and tune** based on real-world false positive patterns

### Short-term (2-4 weeks)

**Fine-tune embeddings** on injection-specific data:

```python
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('all-mpnet-base-v2')

# Create injection-specific training pairs
train_examples = create_contrastive_pairs(
    injections=injection_samples,
    benign=safe_samples,
    benign_triggers=notinject_samples
)

# Fine-tune with contrastive learning
train_loss = losses.ContrastiveLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100,
    output_path='models/injection_aware_mpnet'
)
```

**Expected improvement**: 85% → 92-95% recall at same FPR

### Medium-term (1-2 months)

1. **Ensemble methods**: Combine with complementary detectors
2. **Active learning**: Retrain on production false positives
3. **Domain-specific models**: Separate models for HTML, email, text injections
4. **Adversarial training**: Augment training data with GCG/AutoDAN attacks

---

## Honest Conclusion

The all-mpnet-base-v2 model shows **significant improvement** over smaller models but:

✅ **Strengths**:
- 85.4% recall (acceptable for production with caveats)
- Excellent on adversarial attacks (96% on SaTML)
- Perfect on over-defense testing (0% FPR on NotInject)

❌ **Weaknesses**:
- Cannot achieve both >90% recall and <5% FPR
- Requires 12.4% FPR to get acceptable recall
- Not competitive with commercial solutions
- Needs human review infrastructure

**Recommendation**: Deploy with **θ=0.700 + human review** for high-security applications, or **fine-tune embeddings** before claiming production-ready status for autonomous deployment.

---

**Status**: ⚠️ **Production-ready with human review only**
**Not recommended**: Autonomous deployment without supervision
