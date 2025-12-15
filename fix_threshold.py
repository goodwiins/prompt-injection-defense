#!/usr/bin/env python3
"""Fix the threshold for the balanced model."""

import json

# Load the metadata
with open('models/bit_xgboost_balanced_v2_classifier_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Current threshold: {metadata['threshold']:.4f}")

# Update to a more reasonable threshold (0.5 gives balanced predictions)
new_threshold = 0.5
metadata['threshold'] = new_threshold

# Save back
with open('models/bit_xgboost_balanced_v2_classifier_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Updated threshold to: {new_threshold}")

print("\nRun evaluation again with: python run_eval_balanced_v2.py")