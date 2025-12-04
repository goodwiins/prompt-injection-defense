# Project Feedback Report: Detecting and Preventing Prompt Injection Attacks in Multi-Agent LLM Systems

**Group Members:** Abdel El Bikha, Jennifer Marrero

---

## Executive Summary

This project proposes a well-structured three-layer defense framework to address prompt injection attacks in multi-agent LLM systems—a timely and increasingly critical research area. The proposal demonstrates strong alignment with current literature, particularly the emerging threat of "prompt infection" (self-replicating malicious prompts across interconnected agents)[web: 1][web: 4]. The target metrics of ≥95% detection accuracy with ≤5% false positives and <1% injection success rate are ambitious but grounded in recent benchmarks showing state-of-the-art systems achieving 87-99% accuracy depending on dataset and conditions[web: 63][web: 67][web: 77]. This feedback provides detailed recommendations for strengthening the methodology, dataset construction, evaluation framework, and architectural design.

---

## 1. Alignment with Current Literature and Research Landscape

### 1.1 Threat Model Validation

The project correctly identifies multi-agent systems as an amplified attack vector. Recent research at ICLR 2025 formally introduced "Prompt Infection"—where malicious prompts self-replicate across interconnected agents like a computer virus—demonstrating that multi-agent systems are highly susceptible even when agents do not directly share communications[web: 1]. The proposed focus on trust boundary exploitation is validated by research showing that adversaries can exploit transitive trust chains for lateral movement and data exfiltration[web: 116]. A 2025 study on the "Trust-Vulnerability Paradox" reveals that the very trust mechanisms established to improve coordination simultaneously enlarge attack surfaces[web: 116].

### 1.2 Attack Taxonomy Consideration

The project should incorporate the comprehensive attack taxonomy from recent literature, which categorizes prompt injections into ten types[web: 14]:

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

The proposal to leverage semantic embeddings is well-supported by recent research. A 2024 study demonstrated that embedding-based ML classifiers using OpenAI's `text-embedding-3-small`, `gte-large`, and `all-MiniLM-L6-v2` models can effectively distinguish malicious from benign prompts[web: 69][web: 80]. Random Forest and XGBoost classifiers built on these embeddings outperformed encoder-only neural network approaches in terms of AUC and precision[web: 69].

**Recommendation:** Consider a multi-embedding ensemble approach:

- **Fast path:** Lightweight embeddings (e.g., `all-MiniLM-L6-v2`) for real-time screening
- **Deep path:** Larger embeddings (e.g., OpenAI `text-embedding-3-small`) for flagged inputs
- **Specialized path:** Domain-specific fine-tuned models (e.g., `protectai/deberta-v3-base-prompt-injection`)[web: 85]

**Attention Mechanism Monitoring**

The "Attention Tracker" approach represents cutting-edge research, achieving up to 10% AUROC improvement over existing methods without requiring additional LLM inference[web: 23][web: 32]. This technique identifies "important heads" in the attention mechanism that are most vulnerable to the "distraction effect"—where attention shifts from original instructions to injected commands[web: 26].

**Implementation Consideration:** The attention-based detection requires access to model internals, which may not be available for all LLM deployments. Consider a hybrid approach:

- Attention tracking for open-weight models (Llama, Mistral)
- Behavior-based detection for API-only models (GPT-4, Claude)

**Behavioral Pattern Recognition**

Research on cross-LLM behavioral backdoor detection reveals a critical insight: single-model detectors cannot protect multi-LLM deployments, with a 43.4% generalization gap observed when detectors trained on one LLM are applied to another[web: 82]. However, model-aware training achieves 90.6% universal accuracy across heterogeneous LLM ecosystems[web: 82].

**Recommendation:** Implement behavioral baselines for each agent type and monitor for:

- Sudden output distribution shifts
- Anomalous tool invocation patterns
- Unexpected inter-agent communication patterns
- Reasoning chain inconsistencies

### 2.2 Coordination Layer

The multi-agent defense architecture aligns with state-of-the-art approaches. The "AutoDefense" framework demonstrates that a three-agent defense agency (intention analyzer, prompt analyzer, judge) using LLaMA-2-13B effectively reduces jailbreak attack success rates while maintaining low false positives[web: 41].

**Recommended Agent Roles:**

| Agent           | Function                                                           | Model Recommendation                  |
| --------------- | ------------------------------------------------------------------ | ------------------------------------- |
| Preprocessor    | Input normalization, encoding detection, delimiter standardization | Rule-based + lightweight ML           |
| Guard/Sanitizer | Injection marker detection, payload neutralization                 | Meta Llama 3.1 or DeBERTa-v3[web: 14] |
| Policy Enforcer | Compliance verification, metadata tagging                          | Meta Llama 3.1                        |
| KPI Evaluator   | Real-time metric computation                                       | Specialized classifier                |

**Inter-Agent Communication Protocol**

The OVON (Open Voice Network) framework provides a structured approach for inter-agent communication via JSON messages containing both utterance text and metadata "whisper fields"[web: 14]. This enables transparent documentation of detection actions and supports auditability.

**LLM Tagging Defense**

The "LLM Tagging" mechanism proposed by Lee & Tiwari (2024) tags each message with its source agent identity, significantly mitigating infection spread across agents[web: 1][web: 4]. This should be integrated into the coordination layer architecture.

### 2.3 Response-Filtering Layer

**Real-Time Anomaly Detection**

Recent research provides comprehensive guidance on real-time anomaly detection in multi-agent systems[web: 43]. Key recommendations include:

- **Tiered Alert System:** Low-confidence anomalies generate informational alerts; high-confidence, high-impact anomalies trigger immediate notifications
- **Alert Correlation:** Group related alerts from multiple detection systems to provide holistic visibility
- **Circuit Breakers:** Disable automation if too many responses trigger within short periods

**Agent Quarantine Mechanisms**

The proposal's quarantine concept is validated by enterprise security practices. When a system is quarantined, all incoming and outgoing traffic is blocked except for secure connections to monitoring infrastructure[web: 114]. For multi-agent LLM systems, this translates to:

1. **Immediate isolation:** Compromised agent is disconnected from inter-agent communication
2. **Restricted operation:** Agent continues in diagnostic mode with logging enabled
3. **Controlled reintegration:** After verification, agent rejoins with enhanced monitoring

**PeerGuard Defense Mechanism**

Consider incorporating the "PeerGuard" mutual reasoning defense, which uses agent-to-agent reasoning to detect illogical behavior indicative of prompt-based poisoning[web: 44]. Experimental evaluations show true positive rates up to 96% and false positive rates below 10% across GPT-4o and Llama 3 models[web: 44].

---

## 3. Dataset Construction Strategy

### 3.1 Existing Public Datasets

| Dataset                         | Size                          | Description                                | Source                         |
| ------------------------------- | ----------------------------- | ------------------------------------------ | ------------------------------ |
| deepset/prompt-injections       | 662 samples                   | First public prompt injection dataset      | HuggingFace[web: 97][web: 107] |
| SaTML CTF 2024                  | 137k+ multi-turn chats        | Adaptive attack conversations              | IEEE SaTML[web: 36][web: 99]   |
| LLMail-Inject                   | 208,095 unique prompts        | Indirect prompt injection attacks          | Microsoft[web: 83]             |
| imoxto/prompt_injection_cleaned | 535,105 prompts               | Comprehensive malicious/benign dataset     | HuggingFace[web: 80]           |
| INJECAGENT                      | Tool-integrated agent attacks | First benchmark for indirect IPI on agents | ACL 2024[web: 21]              |
| NotInject                       | 339 benign samples            | Over-defense evaluation dataset            | Research[web: 30]              |

**Recommendation:** Construct a composite dataset of ~500,000 samples:

- 70% benign multi-agent communication logs (generate synthetically using multiple LLMs)
- 20% known injection attacks (from public datasets)
- 10% engineered multi-agent-specific attacks (custom generation)

### 3.2 Multi-Agent-Specific Data Generation

The project should generate attack scenarios specific to multi-agent architectures:

1. **Infection propagation attacks:** Prompts designed to self-replicate across agents
2. **Trust boundary exploitation:** Attacks leveraging inter-agent trust relationships
3. **Cross-agent privilege escalation:** Exploiting one agent's permissions to affect others
4. **Tool call injection:** Malicious instructions targeting agent tool invocations[web: 7]

Research shows that coding and testing phase agents pose significantly higher security risks than design phase agents, with IMBIA achieving attack success rates of 93%, 45%, and 71% across ChatDev, MetaGPT, and AgentVerse frameworks respectively[web: 7].

### 3.3 Synthetic Data Considerations

When generating synthetic attack data, use the taxonomy from Section 2.1 with 50 prompts per category (500 total for each generation round)[web: 14]. Apply GPT-4 or similar models for generation but validate outputs manually to ensure quality and diversity.

---

## 4. Evaluation Framework Refinement

### 4.1 Proposed Metrics

The project's proposed metrics (ROC-AUC, F1, detection latency, injection success rate) are appropriate but should be expanded:

| Metric                          | Definition                          | Target | Justification                           |
| ------------------------------- | ----------------------------------- | ------ | --------------------------------------- |
| ROC-AUC                         | Area under ROC curve                | ≥0.95  | Standard classification metric[web: 64] |
| F1 Score                        | Harmonic mean of precision/recall   | ≥0.90  | Balanced performance measure[web: 64]   |
| False Positive Rate (FPR)       | Benign inputs flagged as attacks    | ≤5%    | Current SOTA: 0.2-6%[web: 63][web: 77]  |
| False Negative Rate (FNR)       | Attacks missed                      | ≤1%    | Critical for security[web: 74]          |
| Detection Latency               | Time from input to classification   | ≤100ms | Real-time requirement[web: 70]          |
| Injection Success Rate (ISR)    | Attacks bypassing all defenses      | <1%    | Ambitious but achievable[web: 14]       |
| Policy Override Frequency (POF) | Policy breaches due to injections   | <2%    | Novel metric from literature[web: 14]   |
| Prompt Sanitization Rate (PSR)  | Injections successfully neutralized | ≥98%   | Effectiveness measure[web: 14]          |

### 4.2 Composite Vulnerability Score

Consider adopting the Total Injection Vulnerability Score (TIVS) framework[web: 14]:

$$\text{TIVS} = \frac{(\text{ISR} \cdot w_1) + (\text{POF} \cdot w_2) - (\text{PSR} \cdot w_3) - (\text{CCS} \cdot w_4)}{N_A \cdot (w_1 + w_2 + w_3 + w_4)}$$

Where $N_A$ is the number of agents and weights are typically set equal (0.25 each). Lower (more negative) TIVS indicates better mitigation.

### 4.3 Benchmark Comparisons

Evaluate against established baselines:

| Solution            | Reported Accuracy      | FPR      | Latency | Source               |
| ------------------- | ---------------------- | -------- | ------- | -------------------- |
| Lakera Guard        | 87.91%                 | 5.7%     | 0.066s  | Commercial[web: 67]  |
| ProtectAI LLM Guard | ~90%                   | Variable | ~0.5s   | Open-source[web: 63] |
| ActiveFence         | F1: 0.857              | 5.4%     | N/A     | Commercial[web: 68]  |
| Glean AI            | 97.8%                  | 3.0%     | N/A     | Commercial[web: 77]  |
| PromptArmor         | FPR: 0.56%, FNR: 0.13% | <1%      | N/A     | Research[web: 74]    |

---

## 5. Technical Implementation Recommendations

### 5.1 Model Selection

**For Classification:**

- **DeBERTa-v3:** State-of-the-art for text classification, ~3,000x faster inference than LLM-based detection[web: 81]
- **ModernBERT:** 8192 token context length, improved downstream performance[web: 84]
- **DistilBERT:** Lightweight option for edge deployment[web: 2]

**For Agent Implementation:**

- **Meta Llama 3.1:** Recommended for Guard/Sanitizer and Policy Enforcer agents[web: 14]
- **GPT-4o-mini:** For coordinator agents requiring instruction hierarchy support[web: 83]

### 5.2 Addressing Over-Defense

A critical challenge is over-defense—falsely flagging benign inputs containing trigger words common in injections[web: 30]. The "InjecGuard" approach with Mitigating Over-defense for Free (MOF) training strategy significantly reduces trigger word bias[web: 30].

**Recommendation:** Include the NotInject dataset (339 benign samples enriched with trigger words) in validation to measure over-defense rates[web: 30].

### 5.3 Multi-Modal Considerations

Recent research highlights that multimodal AI introduces unique prompt injection risks through cross-modal attacks (e.g., instructions hidden in images)[web: 19]. If the multi-agent system processes multiple modalities, incorporate:

- Image-text alignment verification
- Cross-modal consistency checking
- Modality-specific sanitization

---

## 6. Hypothesis Evaluation

The hypothesis of ≥95% detection accuracy with ≤5% false positives while reducing successful injections to <1% is:

**Achievable but requires careful implementation:**

- Current SOTA achieves 87-99% accuracy depending on dataset complexity[web: 63][web: 67][web: 77]
- False positive rates of 0.2-6% have been demonstrated[web: 63][web: 68]
- Reducing injection success to <1% requires multi-layered defense (single defenses typically achieve 60-90% reduction)[web: 62]

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
3. **API Compatibility:** OVON-compliant JSON messaging for interoperability[web: 14]
4. **Documentation:** Include attack simulation tools for red-teaming
5. **Benchmarking Suite:** Standardized evaluation scripts against public datasets

The SaTML LLM CTF platform is open-sourced and provides a foundation for competition-style evaluation[web: 96][web: 99].

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

**References are embedded as inline citations throughout this document, drawing from over 130 sources including ICLR 2025, ACL 2024, NeurIPS 2024, and IEEE SaTML 2024 publications, as well as industry reports from Microsoft, Anthropic, and leading AI security firms.**

[1](https://openreview.net/forum?id=NAbqM2cMjD)
[2](https://labs.withsecure.com/publications/detecting-prompt-injection-bert-based-classifier)
[3](https://research.aimultiple.com/security-of-ai-agents/)
[4](https://www.rohan-paul.com/p/prompt-hacking-in-llms-2024-2025)
[5](https://www.salesforce.com/blog/prompt-injection-detection/)
[6](https://www.sciencedirect.com/science/article/abs/pii/S1566253525010036)
[7](https://arxiv.org/html/2511.18467v1)
[8](https://learnprompting.org/docs/prompt_hacking/injection)
[9](https://www.fujitsu.com/global/about/research/article/202507-multi-ai-agent-security.html)
[10](https://labs.snyk.io/resources/agent-hijacking/)
[11](https://developer.microsoft.com/blog/protecting-against-indirect-injection-attacks-mcp)
[12](https://dl.acm.org/doi/10.1145/3773080)
[13](https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/)
[14](https://arxiv.org/html/2503.11517v1)
[15](https://www.crowdstrike.com/en-us/blog/secure-ai-generated-code-with-multiple-self-learning-ai-agents/)
[16](https://simonw.substack.com/p/new-prompt-injection-papers-agents)
[17](https://www.lakera.ai/blog/guide-to-prompt-injection)
[18](https://www.cooperativeai.com/grant-research-areas/multi-agent-security)
[19](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
[20](https://hiddenlayer.com/innovation-hub/evaluating-prompt-injection-datasets/)
[21](https://aclanthology.org/2024.findings-acl.624.pdf)
[22](https://arxiv.org/html/2502.09385v1)
[23](https://arxiv.org/abs/2411.00348)
[24](https://openreview.net/forum?id=MsRdq0ePTR)
[25](https://www.techscience.com/iasc/v33n1/46135/html)
[26](https://www.promptlayer.com/research-papers/stopping-ai-hallucinations-new-research-on-prompt-injection-attacks)
[27](https://www.lakera.ai/blog/lakera-pint-benchmark)
[28](https://www.meegle.com/en_us/topics/anomaly-detection/anomaly-detection-in-natural-language-processing)
[29](https://www.promptfoo.dev/lm-security-db/vuln/attention-guided-jailbreak-5f2b2d04)
[30](https://arxiv.org/html/2410.22770v1)
[31](https://aclanthology.org/2025.findings-emnlp.680.pdf)
[32](https://aclanthology.org/2025.findings-naacl.123.pdf)
[33](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei)
[34](https://www.sciencedirect.com/org/science/article/pii/S1546221824006131)
[35](https://www.nature.com/articles/s41598-024-70032-2)
[36](https://papers.nips.cc/paper_files/paper/2024/file/411c44e6f285310822f39f76a58798c7-Paper-Datasets_and_Benchmarks_Track.pdf)
[37](https://www.elastic.co/blog/whats-new-kibana-ml-cloud-8-7-0)
[38](https://www.obsidiansecurity.com/blog/adversarial-prompt-engineering)
[39](https://ieeexplore.ieee.org/document/10987374/)
[40](https://www.techrxiv.org/users/944086/articles/1328209/master/file/data/Anomaly_Detection_Using_Embedding_CyberSecurity_TechRxiv/Anomaly_Detection_Using_Embedding_CyberSecurity_TechRxiv.pdf?inline=true)
[41](https://microsoft.github.io/autogen/0.2/blog/2024/03/11/AutoDefense/Defending%20LLMs%20Against%20Jailbreak%20Attacks%20with%20AutoDefense/)
[42](https://milvus.io/ai-quick-reference/how-do-guardrails-work-in-llms)
[43](https://galileo.ai/blog/real-time-anomaly-detection-multi-agent-ai)
[44](https://www.emergentmind.com/topics/peerguard)
[45](https://www.openxcell.com/blog/llm-guardrails/)
[46](https://www.tencentcloud.com/techpedia/126638)
[47](https://galileo.ai/blog/multi-agent-systems-exploits)
[48](https://www.k2view.com/blog/llm-guardrails/)
[49](https://www.tinybird.co/blog/real-time-anomaly-detection)
[50](https://arxiv.org/html/2509.14285v1)
[51](https://arxiv.org/html/2402.01822v1)
[52](https://exeon.com/real-time-anomaly-detection/)
[53](https://www.anthropic.com/engineering/multi-agent-research-system)
[54](https://www.reddit.com/r/LLMDevs/comments/1n34qcz/building_low_latency_guardrails_to_secure_your/)
[55](https://www.nature.com/articles/s41598-025-20641-2)
[56](https://www.reply.com/aim-reply/en/content/introduction-to-multi-agent-architecture-for-llm-based-applications)
[57](https://aws.amazon.com/blogs/machine-learning/build-safe-and-responsible-generative-ai-applications-with-guardrails/)
[58](https://www.cake.ai/blog/real-time-anomaly-detection)
[59](https://developer.microsoft.com/blog/designing-multi-agent-intelligence)
[60](https://www.digitalocean.com/resources/articles/what-are-llm-guardrails)
[61](https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc)
[62](https://kili-technology.com/blog/preventing-adversarial-prompt-injections-with-llm-guardrails)
[63](https://www.knostic.ai/blog/revolutionizing-prompt-injection-detection-a-leap-to-99-accuracy)
[64](https://neptune.ai/blog/f1-score-accuracy-roc-auc-and-pr-auc)
[65](https://neuraltrust.ai/blog/prompt-injection-detection-llm-stack)
[66](https://www.deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/)
[67](https://arxiv.org/html/2505.13028v2)
[68](https://techedgeai.com/news/activefence-tops-prompt-injection-security-benchmark-with-leading-precision/)
[69](https://arxiv.org/html/2410.22284v1)
[70](https://www.fiddler.ai/articles/ai-guardrails-metrics)
[71](https://www.datadoghq.com/blog/engineering/malicious-pull-requests/)
[72](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-metrics-validation.html)
[73](https://galileo.ai/blog/metrics-for-evaluating-llm-chatbots-part-2)
[74](https://arxiv.org/html/2507.15219v1)
[75](https://www.evidentlyai.com/classification-metrics/explain-roc-curve)
[76](https://www.confident-ai.com/blog/red-teaming-llms-a-step-by-step-guide)
[77](https://www.glean.com/blog/ai-safeguard-septdrop-2025)
[78](https://www.sciencedirect.com/science/article/pii/S3050577125000283)
[79](https://www.patronus.ai/llm-testing/llm-observability)
[80](https://github.com/AhsanAyub/malicious-prompt-detection)
[81](https://arxiv.org/html/2406.06663v1)
[82](https://arxiv.org/html/2511.19874v1)
[83](https://arxiv.org/html/2506.09956v1)
[84](https://www.philschmid.de/fine-tune-modern-bert-in-2025)
[85](https://huggingface.co/protectai/deberta-v3-base-prompt-injection)
[86](https://www.reddit.com/r/LanguageTechnology/comments/1bx5ysm/i_just_cant_fine_tune_bert_over_40_accuracy_for/)
[87](https://www.obsidiansecurity.com/blog/what-is-llm-security)
[88](https://www.evidentlyai.com/llm-guide/prompt-injection-llm)
[89](https://www.youtube.com/watch?v=4QHg8Ix8WWQ)
[90](https://www.anthropic.com/research/agentic-misalignment)
[91](https://jfrog.com/blog/prompt-injection-attack-code-execution-in-vanna-ai-cve-2024-5565/)
[92](https://aclanthology.org/2025.sdp-1.26.pdf)
[93](https://www.oligo.security/academy/llm-security-in-2025-risks-examples-and-best-practices)
[94](https://towardsdatascience.com/fine-tune-smaller-transformer-models-text-classification-77cbbd3bf02b/)
[95](https://discuss.huggingface.co/t/what-s-the-best-way-to-fine-tune-a-transformer-model-on-a-custom-dataset-using-the-transformers-library/165177)
[96](https://ctf.spylab.ai)
[97](https://haystack.deepset.ai/blog/how-to-prevent-prompt-injections)
[98](https://hoop.dev/blog/how-to-keep-prompt-injection-defense-synthetic-data-generation-secure-and-compliant-with-inline-compliance-prep/)
[99](https://arxiv.org/abs/2406.07954)
[100](https://www.promptingguide.ai/applications/synthetic_rag)
[101](https://www.youtube.com/watch?v=WIjfmiTpfNo)
[102](https://huggingface.co/datasets?p=0&sort=trending&search=prompt+injection)
[103](https://arxiv.org/html/2406.07954v1)
[104](https://huggingface.co/deepset/deberta-v3-base-injection)
[105](https://arxiv.org/pdf/2509.01185.pdf)
[106](https://liner.com/review/dataset-and-lessons-learned-from-the-2024-satml-llm-capturetheflag)
[107](https://huggingface.co/datasets/deepset/prompt-injections)
[108](https://www.redhat.com/en/blog/synthetic-data-secret-ingredient-better-language-models)
[109](https://openreview.net/forum?id=WUWHVN4gxk)
[110](https://huggingface.co/deepset/datasets)
[111](https://dl.acm.org/doi/10.5555/3737916.3739080)
[112](https://huggingface.co/deepset)
[113](https://air-governance-framework.finos.org/risks/ri-28_multi-agent-trust-boundary-violations.html)
[114](https://docs.rapid7.com/insightidr/quarantine-an-asset/)
[115](https://fuzzinglabs.com/attacking-reasoning-models/)
[116](https://arxiv.org/html/2510.18563v1)
[117](https://www.itsasap.com/blog/threat-isolation-containment)
