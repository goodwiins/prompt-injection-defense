#!/usr/bin/env python3
"""Embed reviewed code and existing local examples in a portable Colab notebook."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    import nbformat
    from benchmarks.benchmark_datasets import load_notinject_dataset

    generator_path = ROOT / "training/train_injection_aware_classifier.py"
    tree = ast.parse(generator_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_trigger_benign_samples")
    from typing import List, Tuple
    scope = dict(load_notinject_dataset=load_notinject_dataset, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(generator_path), "exec"), scope)
    texts, labels = scope["generate_trigger_benign_samples"]()
    local = load_notinject_dataset()
    heldout = set(local.texts)
    rows = [dict(source="local-synthetic-historical", split="all", index=i, text=t, label=0, kind="trigger", fixed="test") for i,t in enumerate(local.texts)]
    for i, (text, label) in enumerate(zip(texts, labels)):
        if text in heldout:
            continue
        if text.startswith("When learning "):
            family, split = "education", "validation"
        elif text.startswith("In daily life"):
            family, split = "daily", "test"
        elif text.startswith("For this recipe"):
            family, split = "recipe", "train"
        elif text.startswith("The manual says"):
            family, split = "manual", "train"
        elif text.startswith("In "):
            family, split = "technical", "train"
        else:
            raise ValueError(f"Unrecognized template at index {i}")
        rows.append(dict(source="synthetic-additional-templates", split="all", index=i, text=text, label=label, kind="trigger", group=f"template:{family}", fixed=split))
    cache_path = ROOT / "data/tensortrust_cache.json"
    cache = json.loads(cache_path.read_text())
    rows.extend(dict(source="tensortrust-local-cache", split="cached", index=i, text=t, label=1, fixed="test") for i,t in enumerate(cache["texts"]))
    supplement = dict(rows=rows, provenance={
        "generator_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        "local_loader_sha256": hashlib.sha256((ROOT / "benchmarks/benchmark_datasets.py").read_bytes()).hexdigest(),
        "tensortrust_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "tensortrust_caveat": "existing 200-row cache; upstream revision and sampling provenance unknown; exploratory external slice only",
        "template_split": "technical/recipe/manual train; education validation; daily test; all historical local templates test"})
    code = (ROOT / "training/retrain_audited.py").read_text()
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = dict(colab=dict(name="Audited MPNet retraining.ipynb"), kernelspec=dict(name="python3", display_name="Python 3"), accelerator="GPU")
    runner = """import os, subprocess

def run_audit(*extra):
    command = ["python", "-u", "/content/retrain_audited.py", "--output", "/content/audit_run", "--supplement", "/content/audit_inputs.json", *extra]
    env = dict(os.environ, USE_TF="0", USE_FLAX="0", WANDB_MODE="disabled")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError("Audit run failed; inspect the output above")
"""
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# Audited MPNet retraining\n\nSingle-seed audit experiment: frozen public revisions, common data splits for both training stages, validation-only threshold selection, base/uniform/doubled-pair comparison. Four epochs and 14,000 pairs per epoch for each trained encoder.\n\nSelect a **T4 GPU** runtime. Run preparation first and inspect its counts. This is capped exploratory evidence, not full benchmark replication or a semantic near-duplicate guarantee. Existing TensorTrust cache provenance is limited. No credentials, model publishing, or Drive mounting are needed. Download the evidence archive before disconnecting the runtime."),
        nbformat.v4.new_code_cell("%pip uninstall -y torchvision torchaudio librosa\n%pip install -q torch==2.9.1 sentence-transformers==5.1.2 transformers==4.57.3 datasets==4.4.1 xgboost==3.1.2 scikit-learn==1.7.2 numpy==2.3.5 scipy==1.16.3 beautifulsoup4==4.14.2 accelerate==1.11.0"),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\n" + f"Path('/content/retrain_audited.py').write_text({code!r})\n" + f"Path('/content/audit_inputs.json').write_text({json.dumps(supplement, ensure_ascii=False)!r})\n" + "print('Reviewed script and source examples written.')"),
        nbformat.v4.new_code_cell(runner + '\nrun_audit("--prepare-only")'),
        nbformat.v4.new_markdown_cell("## Train and evaluate\nThe next cell fits a base-encoder classifier, then uniform and doubled-pair encoders/classifiers. It can take hours. Test scores never choose thresholds. Runtime outputs and model files remain in `/content/audit_run`. A failed source or split aborts the run."),
        nbformat.v4.new_code_cell("run_audit()"),
        nbformat.v4.new_code_cell("import json\nfrom pathlib import Path\nfor variant in ('base', 'uniform', 'double'):\n    report = Path('/content/audit_run') / variant / 'metrics.json'\n    assert report.exists(), f'{variant} is incomplete'\n    print(report.read_text())"),
        nbformat.v4.new_code_cell("import tarfile\nfrom pathlib import Path\nfrom google.colab import files\n# Small evidence archive; checkpoints stay available for a separate download.\nwith tarfile.open('/content/audit_evidence.tar.gz', 'w:gz') as archive:\n    for path in Path('/content/audit_run').rglob('*'):\n        if path.is_file() and 'encoder' not in path.parts:\n            archive.add(path, arcname=str(path.relative_to('/content')))\n    for name in ('retrain_audited.py', 'audit_inputs.json'):\n        archive.add('/content/' + name, arcname=name)\nfiles.download('/content/audit_evidence.tar.gz')")
    ]
    nbformat.validate(notebook)
    destination = ROOT / "notebooks/colab_retrain_audited.ipynb"
    nbformat.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    build()
