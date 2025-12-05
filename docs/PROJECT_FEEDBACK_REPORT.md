# Project Feedback Report: Detecting and Preventing Prompt Injection Attacks in Multi-Agent LLM Systems

**Group Members:** Abdel El Bikha, Jennifer Marrero

---

## Executive Summary

This project proposes a well-structured three-layer defense framework to address prompt injection attacks in multi-agent LLM systems—a timely and increasingly critical research area. The proposal demonstrates strong alignment with current literature, particularly the emerging threat of "prompt infection" (self-replicating malicious prompts across interconnected agents). The target metrics of ≥95% detection accuracy with ≤5% false positives and <1% injection success rate are ambitious but grounded in recent benchmarks showing state-of-the-art systems achieving 87-99% accuracy depending on dataset and conditions. This feedback provides detailed recommendations for strengthening the methodology, dataset construction, evaluation framework, and architectural design.

---

## 1. Alignment with Current Literature and Research Landscape

### 1.1 Threat Model Validation

The project correctly identifies multi-agent systems as an amplified attack vector. Recent research at ICLR 2025 formally introduced "Prompt Infection"—where malicious prompts self-replicate across interconnected agents like a computer virus—demonstrating that multi-agent systems are highly susceptible even when agents do not directly share communications. The proposed focus on trust boundary exploitation is validated by research showing that adversaries can exploit transitive trust chains for lateral movement and data exfiltration. A 2025 study on the "Trust-Vulnerability Paradox" reveals that the very trust mechanisms established to improve coordination simultaneously enlarge attack surfaces.

### 1.2 Attack Taxonomy Consideration

The project should incorporate the comprehensive attack taxonomy from recent literature, which categorizes prompt injections into ten types:

| Attack Category                 | Description                                          | Detection Complexity |
| ------------------------------- | ---------------------------------------------------- | -------------------- |
| Direct Override                 | Explicit instructions to disregard previous commands | Low                  |
| Authority Assertions            | Claims of special privileges to bypass constraints   | Medium               |
| Hidden/Obfuscated Commands      | Encoded text, HTML comments, whitespace manipulation | High                 |
| Role-Play Overrides             | Persona adoption to relax restrictions               | Medium               |
| Logical Traps                   | Paradoxes forcing policy violations                  | High                 |
| Multi-Step Injections           | Gradual, incremental prompt sequences                | Very High            |
| Conflicting Instructions        | Opposing directives testing priority resolution      | Medium               |
| HTML/Markdown Embeds            | Structured markup with nested directives             | Medium               |
| Hybrid (Legitimate + Injection) | Blended genuine and malicious content                | High                 |
| Social Engineering              | Emotional appeals and urgency claims                 | Medium               |

This taxonomy should inform both the detection layer design and the dataset construction strategy.

---

## 2. Three-Layer Framework: Detailed Analysis and Recommendations

### 2.1 Detection Layer

**Semantic Embedding Analysis**

The proposal to leverage semantic embeddings is well-supported by recent research. A 2024 study demonstrated that embedding-based ML classifiers using OpenAI's `text-embedding-3-small`, `gte-large`, and `all-MiniLM-L6-v2` models can effectively distinguish malicious from benign prompts. Random Forest and XGBoost classifiers built on these embeddings outperformed encoder-only neural network approaches in terms of AUC and precision.

**Recommendation:** Consider a multi-embedding ensemble approach:

- **Fast path:** Lightweight embeddings (e.g., `all-MiniLM-L6-v2`) for real-time screening
- **Deep path:** Larger embeddings (e.g., OpenAI `text-embedding-3-small`) for flagged inputs
- **Specialized path:** Domain-specific fine-tuned models (e.g., `protectai/deberta-v3-base-prompt-injection`)

**Attention Mechanism Monitoring**

The "Attention Tracker" approach represents cutting-edge research, achieving up to 10% AUROC improvement over existing methods without requiring additional LLM inference. This technique identifies "important heads" in the attention mechanism that are most vulnerable to the "distraction effect"—where attention shifts from original instructions to injected commands.

**Implementation Consideration:** The attention-based detection requires access to model internals, which may not be available for all LLM deployments. Consider a hybrid approach:

- Attention tracking for open-weight models (Llama, Mistral)
- Behavior-based detection for API-only models (GPT-4, Claude)

**Behavioral Pattern Recognition**

Research on cross-LLM behavioral backdoor detection reveals a critical insight: single-model detectors cannot protect multi-LLM deployments, with a 43.4% generalization gap observed when detectors trained on one LLM are applied to another. However, model-aware training achieves 90.6% universal accuracy across heterogeneous LLM ecosystems.

**Recommendation:** Implement behavioral baselines for each agent type and monitor for:

- Sudden output distribution shifts
- Anomalous tool invocation patterns
- Unexpected inter-agent communication patterns
- Reasoning chain inconsistencies

### 2.2 Coordination Layer

The multi-agent defense architecture aligns with state-of-the-art approaches. The "AutoDefense" framework demonstrates that a three-agent defense agency (intention analyzer, prompt analyzer, judge) using LLaMA-2-13B effectively reduces jailbreak attack success rates while maintaining low false positives.

**Recommended Agent Roles:**

| Agent           | Function                                                           | Model Recommendation         |
| --------------- | ------------------------------------------------------------------ | ---------------------------- |
| Preprocessor    | Input normalization, encoding detection, delimiter standardization | Rule-based + lightweight ML  |
| Guard/Sanitizer | Injection marker detection, payload neutralization                 | Meta Llama 3.1 or DeBERTa-v3 |
| Policy Enforcer | Compliance verification, metadata tagging                          | Meta Llama 3.1               |
| KPI Evaluator   | Real-time metric computation                                       | Specialized classifier       |

**Inter-Agent Communication Protocol**

The OVON (Open Voice Network) framework provides a structured approach for inter-agent communication via JSON messages containing both utterance text and metadata "whisper fields". This enables transparent documentation of detection actions and supports auditability.

**LLM Tagging Defense**

The "LLM Tagging" mechanism proposed by Lee & Tiwari (2024) tags each message with its source agent identity, significantly mitigating infection spread across agents. This should be integrated into the coordination layer architecture.

### 2.3 Response-Filtering Layer

**Real-Time Anomaly Detection**

Recent research provides comprehensive guidance on real-time anomaly detection in multi-agent systems. Key recommendations include:

- **Tiered Alert System:** Low-confidence anomalies generate informational alerts; high-confidence, high-impact anomalies trigger immediate notifications
- **Alert Correlation:** Group related alerts from multiple detection systems to provide holistic visibility
- **Circuit Breakers:** Disable automation if too many responses trigger within short periods

**Agent Quarantine Mechanisms**

The proposal's quarantine concept is validated by enterprise security practices. When a system is quarantined, all incoming and outgoing traffic is blocked except for secure connections to monitoring infrastructure. For multi-agent LLM systems, this translates to:

1. **Immediate isolation:** Compromised agent is disconnected from inter-agent communication
2. **Restricted operation:** Agent continues in diagnostic mode with logging enabled
3. **Controlled reintegration:** After verification, agent rejoins with enhanced monitoring

**PeerGuard Defense Mechanism**

Consider incorporating the "PeerGuard" mutual reasoning defense, which uses agent-to-agent reasoning to detect illogical behavior indicative of prompt-based poisoning. Experimental evaluations show true positive rates up to 96% and false positive rates below 10% across GPT-4o and Llama 3 models.

---

## 3. Dataset Construction Strategy

### 3.1 Existing Public Datasets

| Dataset                         | Size                          | Description                                | Source      |
| ------------------------------- | ----------------------------- | ------------------------------------------ | ----------- |
| deepset/prompt-injections       | 662 samples                   | First public prompt injection dataset      | HuggingFace |
| SaTML CTF 2024                  | 137k+ multi-turn chats        | Adaptive attack conversations              | IEEE SaTML  |
| LLMail-Inject                   | 208,095 unique prompts        | Indirect prompt injection attacks          | Microsoft   |
| imoxto/prompt_injection_cleaned | 535,105 prompts               | Comprehensive malicious/benign dataset     | HuggingFace |
| INJECAGENT                      | Tool-integrated agent attacks | First benchmark for indirect IPI on agents | ACL 2024    |
| NotInject                       | 339 benign samples            | Over-defense evaluation dataset            | Research    |

**Recommendation:** Construct a composite dataset of ~500,000 samples:

- 70% benign multi-agent communication logs (generate synthetically using multiple LLMs)
- 20% known injection attacks (from public datasets)
- 10% engineered multi-agent-specific attacks (custom generation)

### 3.2 Multi-Agent-Specific Data Generation

The project should generate attack scenarios specific to multi-agent architectures:

1. **Infection propagation attacks:** Prompts designed to self-replicate across agents
2. **Trust boundary exploitation:** Attacks leveraging inter-agent trust relationships
3. **Cross-agent privilege escalation:** Exploiting one agent's permissions to affect others
4. **Tool call injection:** Malicious instructions targeting agent tool invocations

Research shows that coding and testing phase agents pose significantly higher security risks than design phase agents, with IMBIA achieving attack success rates of 93%, 45%, and 71% across ChatDev, MetaGPT, and AgentVerse frameworks respectively.

### 3.3 Synthetic Data Considerations

When generating synthetic attack data, use the taxonomy from Section 2.1 with 50 prompts per category (500 total for each generation round). Apply GPT-4 or similar models for generation but validate outputs manually to ensure quality and diversity.

---

## 4. Evaluation Framework Refinement

### 4.1 Proposed Metrics

The project's proposed metrics (ROC-AUC, F1, detection latency, injection success rate) are appropriate but should be expanded:

| Metric                          | Definition                          | Target | Justification                  |
| ------------------------------- | ----------------------------------- | ------ | ------------------------------ |
| ROC-AUC                         | Area under ROC curve                | ≥0.95  | Standard classification metric |
| F1 Score                        | Harmonic mean of precision/recall   | ≥0.90  | Balanced performance measure   |
| False Positive Rate (FPR)       | Benign inputs flagged as attacks    | ≤5%    | Current SOTA: 0.2-6%           |
| False Negative Rate (FNR)       | Attacks missed                      | ≤1%    | Critical for security          |
| Detection Latency               | Time from input to classification   | ≤100ms | Real-time requirement          |
| Injection Success Rate (ISR)    | Attacks bypassing all defenses      | <1%    | Ambitious but achievable       |
| Policy Override Frequency (POF) | Policy breaches due to injections   | <2%    | Novel metric from literature   |
| Prompt Sanitization Rate (PSR)  | Injections successfully neutralized | ≥98%   | Effectiveness measure          |

### 4.2 Composite Vulnerability Score

Consider adopting the Total Injection Vulnerability Score (TIVS) framework:

$$\text{TIVS} = \frac{(\text{ISR} \cdot w_1) + (\text{POF} \cdot w_2) - (\text{PSR} \cdot w_3) - (\text{CCS} \cdot w_4)}{N_A \cdot (w_1 + w_2 + w_3 + w_4)}$$

Where $N_A$ is the number of agents and weights are typically set equal (0.25 each). Lower (more negative) TIVS indicates better mitigation.

### 4.3 Benchmark Comparisons

Evaluate against established baselines:

| Solution            | Reported Accuracy      | FPR      | Latency | Source      |
| ------------------- | ---------------------- | -------- | ------- | ----------- |
| Lakera Guard        | 87.91%                 | 5.7%     | 0.066s  | Commercial  |
| ProtectAI LLM Guard | ~90%                   | Variable | ~0.5s   | Open-source |
| ActiveFence         | F1: 0.857              | 5.4%     | N/A     | Commercial  |
| Glean AI            | 97.8%                  | 3.0%     | N/A     | Commercial  |
| PromptArmor         | FPR: 0.56%, FNR: 0.13% | <1%      | N/A     | Research    |

---

## 5. Technical Implementation Recommendations

### 5.1 Model Selection

**For Classification:**

- **DeBERTa-v3:** State-of-the-art for text classification, ~3,000x faster inference than LLM-based detection
- **ModernBERT:** 8192 token context length, improved downstream performance
- **DistilBERT:** Lightweight option for edge deployment

**For Agent Implementation:**

- **Meta Llama 3.1:** Recommended for Guard/Sanitizer and Policy Enforcer agents
- **GPT-4o-mini:** For coordinator agents requiring instruction hierarchy support

### 5.2 Addressing Over-Defense

A critical challenge is over-defense—falsely flagging benign inputs containing trigger words common in injections. The "InjecGuard" approach with Mitigating Over-defense for Free (MOF) training strategy significantly reduces trigger word bias.

**Recommendation:** Include the NotInject dataset (339 benign samples enriched with trigger words) in validation to measure over-defense rates.

### 5.3 Multi-Modal Considerations

Recent research highlights that multimodal AI introduces unique prompt injection risks through cross-modal attacks (e.g., instructions hidden in images). If the multi-agent system processes multiple modalities, incorporate:

- Image-text alignment verification
- Cross-modal consistency checking
- Modality-specific sanitization

---

## 6. Hypothesis Evaluation

The hypothesis of ≥95% detection accuracy with ≤5% false positives while reducing successful injections to <1% is:

**Achievable but requires careful implementation:**

- Current SOTA achieves 87-99% accuracy depending on dataset complexity
- False positive rates of 0.2-6% have been demonstrated
- Reducing injection success to <1% requires multi-layered defense (single defenses typically achieve 60-90% reduction)

**Key Success Factors:**

1. Ensemble approach combining multiple detection methods
2. Multi-agent coordination with specialized roles
3. Real-time response mechanisms with quarantine capabilities
4. Comprehensive training data covering diverse attack types

---

## 7. Open-Source Implementation Considerations

For academic and enterprise adoption, consider:

1. **Modular Architecture:** Each layer should be independently deployable
2. **Framework Agnostic:** Support integration with AutoGen, LangChain, CrewAI
3. **API Compatibility:** OVON-compliant JSON messaging for interoperability
4. **Documentation:** Include attack simulation tools for red-teaming
5. **Benchmarking Suite:** Standardized evaluation scripts against public datasets

The SaTML LLM CTF platform is open-sourced and provides a foundation for competition-style evaluation.

---

## 8. Suggested Timeline and Milestones

| Phase                                | Duration | Deliverables                                         |
| ------------------------------------ | -------- | ---------------------------------------------------- |
| Literature Review & Dataset Curation | 3 weeks  | Annotated bibliography, composite dataset            |
| Detection Layer Development          | 4 weeks  | Embedding classifiers, attention tracker integration |
| Coordination Layer Implementation    | 3 weeks  | Multi-agent pipeline with OVON messaging             |
| Response Layer & Quarantine          | 2 weeks  | Automated response workflows                         |
| Integration & Evaluation             | 3 weeks  | End-to-end system, benchmark results                 |
| Documentation & Open-Source Release  | 2 weeks  | Code repository, technical documentation             |

---

## 9. Conclusion

This project addresses a critical and timely challenge in AI security. The three-layer framework is well-conceived and aligns with current research trends. Key recommendations for strengthening the proposal include:

1. Expand the attack taxonomy to cover all ten categories identified in recent literature
2. Implement attention-based detection alongside embedding classifiers
3. Incorporate LLM tagging and PeerGuard mutual reasoning mechanisms
4. Use composite datasets from multiple public sources plus custom multi-agent scenarios
5. Adopt the TIVS composite metric alongside standard evaluation measures
6. Address over-defense through specialized training strategies

The target metrics are ambitious but achievable with proper implementation. This work has the potential to establish meaningful benchmarks for securing next-generation multi-agent AI systems.

---

## References

1. Lee & Tiwari. "Prompt Infection: LLM-to-LLM Prompt Injection" (ICLR 2025)
2. WithSecure Labs. "Detecting Prompt Injection: BERT-based Classifier"
3. AIM Multiple. "Security of AI Agents"
4. "Prompt Hacking in LLMs 2024-2025"
5. Salesforce. "Prompt Injection Detection"
6. "Multi-Agent Security" (ScienceDirect)
7. "Agent Hijacking" (arXiv)
8. LearnPrompting. "Prompt Injection"
9. Fujitsu. "Multi-AI Agent Security"
10. Snyk Labs. "Agent Hijacking"
11. Microsoft Developer Blog. "Protecting Against Indirect Injection Attacks"
12. ACM. "Prompt Injection in LLMs"
13. Simon Willison. "New Prompt Injection Papers"
14. "Multi-Agent LLM Security Framework" (arXiv)
15. CrowdStrike. "Secure AI-Generated Code with Multiple Self-Learning AI Agents"

_Full reference list includes 117 sources from ICLR 2025, ACL 2024, NeurIPS 2024, IEEE SaTML 2024, and industry reports from Microsoft, Anthropic, and leading AI security firms._

---

_Last updated: 2025-12-05_
