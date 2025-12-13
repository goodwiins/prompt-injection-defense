#!/usr/bin/env python3
"""
Append latest benchmark results to bit_demonstration.ipynb.
"""

import json

BENCHMARK_RESULTS = """════════════════════════════════════════════════════════════════════════════════
                     BENCHMARK RESULTS SUMMARY
════════════════════════════════════════════════════════════════════════════════
Detector: embedding_classifier
Threshold: 0.5
Timestamp: 2025-12-12 16:52:02
────────────────────────────────────────────────────────────────────────────────

Dataset                     Accuracy  Precision     Recall       F1      FPR   Lat(P95)
────────────────────────────────────────────────────────────────────────────────
satml                          94.8%     100.0%      94.8%    97.3%     0.0%      3.2ms
deepset                        66.3%      52.6%      95.6%    67.8%    51.0%      5.3ms
deepset_injections             95.6%     100.0%      95.6%    97.7%     0.0%      5.4ms
notinject (OD)                 98.0%        N/A        N/A      N/A     2.0%      2.9ms
notinject_hf (OD)              95.9%        N/A        N/A      N/A     4.1%      2.2ms
llmail                         99.1%     100.0%      99.1%    99.5%     0.0%      3.7ms
browsesafe                     49.8%      49.7%      99.0%    66.2%    98.4%      4.3ms
────────────────────────────────────────────────────────────────────────────────

OVERALL                        97.6%                                   72.6%
Over-Defense Rate: 2.0%

────────────────────────────────────────────────────────────────────────────────
TARGET STATUS:
  ✅ Accuracy >= 95%: 97.6%
  ❌ FPR <= 5%: 72.6%
  ✅ Over-Defense Rate <= 5%: 2.0%
  ✅ Latency P95 < 100ms: 5.4ms

────────────────────────────────────────────────────────────────────────────────
BASELINE COMPARISONS:

  vs Lakera Guard:
    accuracy: 0.9755 vs 0.8791 (↑ 11.0%)
    fpr: 0.7257 vs 0.0570 (↓ 1173.1%)
    latency_p50: 3.5033 vs 66.0000 (↑ 94.7%)

  vs ProtectAI LLM Guard:
    accuracy: 0.9755 vs 0.9000 (↑ 8.4%)
    latency_p50: 3.5033 vs 500.0000 (↑ 99.3%)

  vs ActiveFence:
    f1_score: 0.9876 vs 0.8570 (↑ 15.2%)
    fpr: 0.7257 vs 0.0540 (↓ 1243.8%)

  vs Glean AI:
    accuracy: 0.9755 vs 0.9780 (↓ 0.3%)
    fpr: 0.7257 vs 0.0300 (↓ 2318.9%)

  vs PromptArmor:
    fpr: 0.7257 vs 0.0056 (↓ 12858.5%)
    fnr: 0.0207 vs 0.0013 (↓ 1490.0%)
"""

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source.strip().split('\n')]
    }

def main():
    notebook_path = "bit_demonstration.ipynb"
    print(f"📖 Loading {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Create new cell
    header = "## 10. Verified Benchmark Results\n\nThe following are the actual results from a full benchmark run on the production model (2025-12-12):"
    content = f"```text\n{BENCHMARK_RESULTS}\n```"
    
    final_cell_source = header + "\n\n" + content
    
    new_cell = create_markdown_cell(final_cell_source)
    
    # Check if section 10 already exists to avoid duplicates
    existing_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if '## 10. Verified Benchmark Results' in src:
                existing_idx = i
                break
    
    if existing_idx is not None:
        print(f"✅ Updating existing Section 10 at cell {existing_idx}...")
        notebook['cells'][existing_idx] = new_cell
    else:
        print("✅ Appending new Section 10 at the end...")
        notebook['cells'].append(new_cell)
    
    print(f"💾 Saving notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        
    print("✅ Benchmark results added successfully!")

if __name__ == "__main__":
    main()
