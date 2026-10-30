#!/usr/bin/env python3
"""Self-contained Colab experiment; run with --prepare-only to freeze inputs first.

This is a single-seed, capped audit experiment, not a replication of historical
scores. Exact text/token duplicates and known template/team groups are isolated.
Semantic near-duplicates and unknown competition provenance remain limitations.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import unicodedata

SEED = 42
MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
SOURCES = {
    "deepset/prompt-injections": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
    "ethz-spylab/ctf-satml24": "11426babe61ee16b0efe7b55e0430359e0fc4235",
    "microsoft/llmail-inject-challenge": "1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
    "perplexity-ai/browsesafe-bench": "b506fb5bc7fd4472c8738055a67a0ef6406afdc9",
    "leolee99/NotInject": "847ae76cf8fea5ed325429e569ae8cfef022d2e0",
}
CONFIG = dict(seed=SEED, base_model=MODEL, base_revision=MODEL_REVISION,
              max_seq_length=384, epochs=4, batch_size=4, learning_rate=5e-5,
              warmup_steps=100, pairs_per_epoch=14000,
              threshold_rule="maximum validation balanced accuracy; highest threshold breaks ties",
              limitations=["one seed", "capped source prefixes, not full benchmarks",
                           "no semantic near-duplicate or unknown-provenance guarantee",
                           "text-only detection, not downstream attack success"])


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def freeze(rows):
    """Union duplicates/groups before assigning any training membership."""
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("text_hash", "token_hash", "group"):
            signature = (key, row[key])
            if signature in seen:
                parents[root(i)] = root(seen[signature])
            seen[signature] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row)
    exclusions, kept = [], []
    # Conflicting labels for identical model inputs cannot be resolved by ordering.
    labels = defaultdict(set)
    for row in rows:
        for key in ("text_hash", "token_hash"):
            labels[(key, row[key])].add(row["label"])
    for members in components.values():
        group_id = min(r["group"] for r in members)
        fixed = {r["fixed_split"] for r in members if r.get("fixed_split")}
        # External test wins; no member of its connected group reaches training.
        split = "test" if "test" in fixed else "validation" if "validation" in fixed else None
        if split is None:
            bucket = int(digest(f"{SEED}:{group_id}")[:8], 16) % 10
            split = "train" if bucket < 6 else "validation" if bucket < 8 else "test"
        if fixed == {"train"}:
            split = "train"
        unique_text, unique_tokens = set(), set()
        for row in sorted(members, key=lambda r: (r.get("fixed_split") != "test", r["id"])):
            reason = None
            if any(len(labels[(key, row[key])]) > 1 for key in ("text_hash", "token_hash")):
                reason = "conflicting labels for identical text or model input"
            elif row["text_hash"] in unique_text or row["token_hash"] in unique_tokens:
                reason = "duplicate text or truncated model input"
            if reason:
                exclusions.append(dict(id=row["id"], reason=reason))
                continue
            unique_text.add(row["text_hash"])
            unique_tokens.add(row["token_hash"])
            kept.append(dict(row, split=split, component=group_id))
    assert_isolated(kept)
    return kept, exclusions


def assert_isolated(rows):
    for key in ("id", "text_hash", "token_hash", "group", "component"):
        membership = {}
        for row in rows:
            previous = membership.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split {key}: {row['id']}")
    for split in ("train", "validation", "test"):
        kinds = {r["kind"] for r in rows if r["split"] == split}
        if kinds != {"injection", "safe", "trigger"}:
            raise ValueError(f"Missing class/kind in {split}: {kinds}")


def collect(supplement, tokenizer):
    from bs4 import BeautifulSoup
    from datasets import load_dataset

    rows, rejected = [], []

    def add(source, split, index, text, label, kind=None, group=None, fixed=None):
        row_id = f"{source}/{split}/{index}"
        if text is None or (isinstance(text, str) and not text.strip()):
            rejected.append(dict(id=row_id, reason="empty input text"))
            return
        if not isinstance(text, str) or label not in (0, 1):
            raise ValueError(f"Invalid input schema: {row_id}")
        text_hash = digest(normalized(text))
        tokens = tokenizer(text, truncation=True, max_length=384)["input_ids"]
        rows.append(dict(id=row_id, source=source, source_split=split, text=text,
                         label=label, kind=kind or ("injection" if label else "safe"),
                         text_hash=text_hash, token_hash=digest(json.dumps(tokens)),
                         group=group or text_hash, fixed_split=fixed))

    def dataset(source, split, config=None):
        print(f"Loading {source}/{split}", flush=True)
        return load_dataset(source, config, split=split, revision=SOURCES[source], streaming=True)

    source = "deepset/prompt-injections"
    for split in ("train", "test"):
        for i, row in enumerate(dataset(source, split)):
            add(source, split, i, row["text"], row["label"], fixed="test" if split == "test" else None)
    source = "perplexity-ai/browsesafe-bench"
    for split, cap in (("train", 2000), ("test", 600)):
        for i, row in enumerate(dataset(source, split)):
            if i >= cap:
                break
            html = row["content"]
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            if len(text) < 50 and len(html) > 100:
                text = html[:500] + " [TEXT]: " + text
            if row["label"] not in ("yes", "no"):
                raise ValueError("Unexpected BrowseSafe label")
            add(source, split, i, text, int(row["label"] == "yes"), fixed="test" if split == "test" else None)
    source = "ethz-spylab/ctf-satml24"
    accepted = 0
    for i, row in enumerate(dataset(source, "attack", "interaction_chats")):
        history = row["history"]
        if history and history[0]["role"] == "user" and history[0]["content"].strip():
            add(source, "attack", i, history[0]["content"], 1)
            accepted += 1
            if accepted == 1500:
                break
    if not accepted:
        raise ValueError("No SaTML user inputs")
    source = "microsoft/llmail-inject-challenge"
    for i, row in enumerate(dataset(source, "Phase1")):
        if i == 500:
            break
        # The attack input is body; never stringify objectives, output, or metadata.
        add(source, "Phase1", i, row["body"], 1,
            group=f"llmail-team:{row['team_id']}" if row.get("team_id") else None)
    source = "leolee99/NotInject"
    for split in ("NotInject_one", "NotInject_two", "NotInject_three"):
        count = 0
        for i, row in enumerate(dataset(source, split)):
            add(source, split, i, row["prompt"], 0, kind="trigger", fixed="test")
            count += 1
        if count != 113:
            raise ValueError(f"Expected 113 official NotInject rows, got {count}")
    for row in supplement["rows"]:
        add(**row)
    expected = set(SOURCES) | {r["source"] for r in supplement["rows"]}
    if {r["source"] for r in rows} != expected:
        raise ValueError("A required data source is empty")
    return rows, rejected


def make_pairs(rows, variant, count=14000):
    rng = random.Random(SEED)
    pools = {kind: [r["id"] for r in rows if r["kind"] == kind] for kind in ("injection", "safe", "trigger")}
    if any(len(p) < 2 for p in pools.values()):
        raise ValueError("At least two unique inputs per pair kind are required")
    kinds = [("injection", "injection", 1.), ("safe", "safe", 1.), ("trigger", "trigger", 1.),
             ("injection", "safe", 0.), ("injection", "trigger", 0.), ("safe", "trigger", .8)]
    weights = [1, 1, 1, 1, 2 if variant == "double" else 1, 1]
    sizes = [count * w // sum(weights) for w in weights]
    for i in range(count - sum(sizes)):
        sizes[i] += 1
    pairs = []
    for (a, b, target), n in zip(kinds, sizes):
        for _ in range(n):
            left, right = rng.sample(pools[a], 2) if a == b else (rng.choice(pools[a]), rng.choice(pools[b]))
            pairs.append(dict(left=left, right=right, target=target, kind=f"{a}/{b}"))
    rng.shuffle(pairs)
    return pairs


def choose_threshold(labels, scores):
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both validation classes")
    candidates = sorted(set([0., 1.0000001] + list(map(float, scores))))
    def objective(threshold):
        tp = sum(y == 1 and p >= threshold for y, p in zip(labels, scores))
        tn = sum(y == 0 and p < threshold for y, p in zip(labels, scores))
        return ((tp / sum(labels) + tn / (len(labels) - sum(labels))) / 2, threshold)
    return max(candidates, key=objective)


def metric(successes, total):
    if not total:
        return None
    p, z = successes / total, 1.959963984540054
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    half = z*((p*(1-p)/total + z*z/(4*total*total))**.5)/d
    return dict(value=p, nominal_wilson_95=[max(0., center-half), min(1., center+half)])


def metrics(rows):
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in rows)
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in rows)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in rows)
    return dict(n=len(rows), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=metric(tp+tn, len(rows)), recall=metric(tp, tp+fn), fpr=metric(fp, fp+tn))


def run(output):
    import importlib.metadata
    import platform
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from transformers import set_seed
    from xgboost import XGBClassifier

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before training")
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert_isolated(rows)
    parts = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    by_id = {r["id"]: r for r in rows}
    environment = dict(python=platform.python_version(), device=torch.cuda.get_device_name(),
                       packages={p: importlib.metadata.version(p) for p in ("torch", "sentence-transformers", "transformers", "datasets", "xgboost", "scikit-learn", "numpy")})
    dump(output / "environment.json", environment)
    for variant in ("base", "uniform", "double"):
        target = output / variant
        if (target / "metrics.json").exists():
            print(f"Already complete: {variant}", flush=True)
            continue
        target.mkdir(exist_ok=True)
        set_seed(SEED)
        encoder = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cuda")
        encoder.max_seq_length = CONFIG["max_seq_length"]
        if variant != "base":
            pairs = make_pairs(parts["train"], variant)
            jsonl(target / "training_pairs.jsonl", pairs)
            # Verify the persisted pair artifact, not a freshly subtracted holdout set.
            pairs = [json.loads(line) for line in (target / "training_pairs.jsonl").read_text().splitlines()]
            assert all(by_id[p[k]]["split"] == "train" for p in pairs for k in ("left", "right"))
            examples = [InputExample(texts=[by_id[p["left"]]["text"], by_id[p["right"]]["text"]], label=p["target"]) for p in pairs]
            loader = DataLoader(examples, shuffle=True, batch_size=CONFIG["batch_size"])
            encoder.fit(train_objectives=[(loader, losses.CosineSimilarityLoss(encoder))],
                        epochs=CONFIG["epochs"], warmup_steps=CONFIG["warmup_steps"],
                        optimizer_params={"lr": CONFIG["learning_rate"]},
                        output_path=str(target / "encoder"), show_progress_bar=True)
        vectors = {s: encoder.encode([r["text"] for r in rs], batch_size=32, show_progress_bar=True, normalize_embeddings=True) for s, rs in parts.items() if s != "test"}
        clf = XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=6, min_child_weight=1,
                            subsample=.8, colsample_bytree=.8, gamma=.1, reg_alpha=.01,
                            reg_lambda=1, eval_metric="auc", tree_method="hist", n_jobs=2, random_state=SEED)
        clf.fit(vectors["train"], [r["label"] for r in parts["train"]])
        clf.save_model(target / "classifier.json")
        val_scores = clf.predict_proba(vectors["validation"])[:, 1]
        threshold = choose_threshold([r["label"] for r in parts["validation"]], val_scores)
        dump(target / "threshold.json", dict(threshold=threshold, rule=CONFIG["threshold_rule"],
                                             validation_ids=[r["id"] for r in parts["validation"]]))
        jsonl(target / "validation_predictions.jsonl", [dict(id=r["id"], label=r["label"], score=float(p)) for r, p in zip(parts["validation"], val_scores)])
        # Test encoding/decisions occur only after the threshold file is written.
        test_vectors = encoder.encode([r["text"] for r in parts["test"]], batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        scores = clf.predict_proba(test_vectors)[:, 1]
        predictions = [dict(id=r["id"], source=r["source"], label=r["label"], score=float(p), prediction=int(p >= threshold)) for r, p in zip(parts["test"], scores)]
        jsonl(target / "test_predictions.jsonl", predictions)
        np.save(target / "test_embeddings.npy", test_vectors)
        timed_rows = sorted(parts["test"], key=lambda r: digest(r["id"]))[:150]
        timings = []
        for i, row in enumerate(timed_rows[:10] + timed_rows):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            clf.predict_proba(encoder.encode([row["text"]], normalize_embeddings=True, show_progress_bar=False))
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if i >= 10:
                timings.append(dict(id=row["id"], milliseconds=elapsed))
        jsonl(target / "single_request_latency.jsonl", timings)
        centroids = {k: test_vectors[[i for i,r in enumerate(parts["test"]) if r["kind"] == k]].mean(axis=0) for k in ("injection", "safe", "trigger")}
        distance = lambda a,b: float(np.linalg.norm(centroids[a] - centroids[b]))
        denominator = distance("trigger", "injection")
        report = dict(variant=variant, threshold=threshold, pooled=metrics(predictions),
                      per_source={s: metrics([r for r in predictions if r["source"] == s]) for s in sorted({r["source"] for r in predictions})},
                      latency=dict(n=len(timings), p50_ms=float(np.percentile([r["milliseconds"] for r in timings], 50)), p95_ms=float(np.percentile([r["milliseconds"] for r in timings], 95)), scope="one input: tokenization + GPU encoder + CPU XGBoost; excludes prior HTML preprocessing"),
                      centroid_ratio_trigger_safe_over_trigger_injection=distance("trigger", "safe") / denominator if denominator else None,
                      limitations=CONFIG["limitations"] + ["Wilson intervals are nominal; related inputs are not independent"])
        dump(target / "metrics.json", report)
        print(json.dumps(report), flush=True)
        del encoder, clf, vectors, test_vectors
        torch.cuda.empty_cache()
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file() and p.name != "sha256.json"}
    dump(output / "sha256.json", hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("audit_run"))
    parser.add_argument("--supplement", type=Path, default=Path("audit_inputs.json"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / "manifest.jsonl").exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        supplement = json.loads(args.supplement.read_text())
        raw, rejected = collect(supplement, tokenizer)
        rows, exclusions = freeze(raw)
        dump(args.output / "protocol.json", dict(CONFIG, sources=SOURCES, supplement_provenance=supplement["provenance"], script_sha256=digest(Path(__file__).read_text())))
        jsonl(args.output / "source_rows.jsonl", raw)
        jsonl(args.output / "source_rejections.jsonl", rejected)
        jsonl(args.output / "exclusions.jsonl", exclusions)
        jsonl(args.output / "manifest.jsonl", rows)
        dump(args.output / "manifest_sha256.json", digest((args.output / "manifest.jsonl").read_text()))
        dump(args.output / "split_counts.json", dict(raw=len(raw), rejected=len(rejected), retained=len(rows), excluded=len(exclusions), counts=dict(Counter(f"{r['split']}:{r['source']}:{r['kind']}" for r in rows))))
    protocol = json.loads((args.output / "protocol.json").read_text())
    if protocol["script_sha256"] != digest(Path(__file__).read_text()):
        raise ValueError("Script changed after preparation; use a fresh output directory")
    expected_manifest = json.loads((args.output / "manifest_sha256.json").read_text())
    if expected_manifest != digest((args.output / "manifest.jsonl").read_text()):
        raise ValueError("Frozen manifest changed after preparation")
    print((args.output / "split_counts.json").read_text(), flush=True)
    if not args.prepare_only:
        run(args.output)


if __name__ == "__main__":
    main()
