#!/usr/bin/env python3
"""Update the threshold to optimal value."""

import json

# Update both metadata files with optimal threshold 0.1
for filepath in [
    'models/bit_xgboost_balanced_v2_classifier_metadata.json',
    'models/bit_xgboost_balanced_v2_metadata.json'
]:
    with open(filepath, 'r') as f:
        metadata = json.load(f)

    print(f"Current threshold in {filepath}: {metadata.get('threshold', 'N/A')}")

    # Update to optimal threshold found
    metadata['threshold'] = 0.1

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Updated threshold to 0.1 in {filepath}")

print("\nRun evaluation again with: python run_eval_balanced_v2.py")