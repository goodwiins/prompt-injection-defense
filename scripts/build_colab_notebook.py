#!/usr/bin/env python3
"""Build a self-contained Colab notebook that reproduces the leak-fixed retrain on GPU.

The notebook clones the upstream repo, overwrites the two changed files with the
exact patched versions from THIS working tree (embedded via %%writefile), runs the
zero-leak gate, fine-tunes on GPU, and pushes the model to the HF Hub.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
finetune_src = (ROOT / "training/finetune_mpnet_embeddings.py").read_text()
verify_src = (ROOT / "scripts/verify_no_leak.py").read_text()

UPSTREAM = "https://github.com/goodwiins/prompt-injection-defense"
HF_REPO = "goodwiinz/injection-aware-mpnet-leakfix"


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md(f"""# Injection-Aware MPNet — Leakage-Fixed Retrain (GPU)

Self-contained. **Runtime → Change runtime type → GPU (T4 is fine)** before running.

This notebook:
1. Clones the upstream repo
2. Overwrites `training/finetune_mpnet_embeddings.py` + adds `scripts/verify_no_leak.py`
   with the **leak-fixed** versions (eval-holdout subtraction + disjoint validation split)
3. Runs the **zero-leak gate** (must print `PASS`)
4. Fine-tunes the encoder + XGBoost on GPU (~15-40 min)
5. Pushes the model to `{HF_REPO}` on the HF Hub

The fix removes all evaluation texts from the training pool (≈39% of the original
pool was eval data, incl. 100% of NotInject-HF). Expect **lower, honest** numbers."""),

    md("## 1. Clone repo + install deps"),
    code(f"!git clone {UPSTREAM} repo\n"
         "%cd repo\n"
         "!pip install -q -r requirements.txt\n"
         "# browsesafe loader needs bs4; not in requirements.txt\n"
         "!pip install -q beautifulsoup4 huggingface_hub"),

    md("## 2. Apply the leak fix (overwrite the two files)"),
    code("%%writefile training/finetune_mpnet_embeddings.py\n" + finetune_src),
    code("%%writefile scripts/verify_no_leak.py\n" + verify_src),

    md("## 3. Zero-leak gate — must print `PASS: zero overlap`"),
    code("!PYTHONPATH=. python scripts/verify_no_leak.py"),

    md("""## 4. (Optional) speed up on GPU
The original config is `BATCH_SIZE=4, EPOCHS=4` (batch 4 was only to avoid OOM on a
small GPU). On a T4/A100 you can safely raise the batch size to cut wall-clock — but
for numbers directly comparable to the paper's config, **leave it at 4**. Uncomment to bump."""),
    code("# import re, pathlib\n"
         "# p = pathlib.Path('training/finetune_mpnet_embeddings.py')\n"
         "# s = p.read_text()\n"
         "# s = s.replace('BATCH_SIZE = 4  # Reduced from 16 to avoid OOM', 'BATCH_SIZE = 16')\n"
         "# p.write_text(s)\n"
         "# print('BATCH_SIZE bumped to 16')"),

    md("## 5. Fine-tune encoder + XGBoost (GPU)"),
    code("import torch\n"
         "assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > GPU'\n"
         "print('GPU:', torch.cuda.get_device_name(0))\n"
         "!PYTHONPATH=. python training/finetune_mpnet_embeddings.py"),

    md(f"""## 6. Push the leak-fixed model to the HF Hub
You need an HF **write** token: https://huggingface.co/settings/tokens
Pushes the fine-tuned encoder + the XGBoost classifier JSON (+ metadata) to `{HF_REPO}`."""),
    code("from huggingface_hub import login, HfApi, create_repo\n"
         "from getpass import getpass\n"
         "login(token=getpass('HF write token: '))\n"
         f"repo_id = '{HF_REPO}'\n"
         "create_repo(repo_id, repo_type='model', exist_ok=True)\n"
         "api = HfApi()\n"
         "# encoder (SentenceTransformer folder)\n"
         "api.upload_folder(folder_path='models/injection_aware_mpnet', repo_id=repo_id,\n"
         "                  commit_message='leak-fixed encoder')\n"
         "# XGBoost classifier + metadata\n"
         "for f in ['models/injection_aware_mpnet_classifier.json',\n"
         "          'models/injection_aware_mpnet_classifier_metadata.json']:\n"
         "    api.upload_file(path_or_fileobj=f, path_in_repo=f.split('/')[-1],\n"
         "                    repo_id=repo_id, commit_message='leak-fixed classifier')\n"
         "print('Pushed to', repo_id)"),

    md(f"""## Done
Tell me when this finishes — I'll pull `{HF_REPO}` back, re-benchmark, and update the paper
with the honest numbers + Threats to Validity."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = ROOT / "notebooks/colab_retrain_leakfix.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({len(cells)} cells)")
