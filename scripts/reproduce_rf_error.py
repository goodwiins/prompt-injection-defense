
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

print("🚀 Testing RandomForestClassifier...")

try:
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # Initialize classifier
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': -1,
        'random_state': 42
    }
    clf = RandomForestClassifier(**rf_params)
    
    # Train
    print("Training...")
    clf.fit(X, y)
    print("Training complete.")
    
    # Predict
    print("Predicting...")
    prob = clf.predict_proba(X)
    print(f"Prediction shape: {prob.shape}")
    
    # Check estimators_
    if hasattr(clf, 'estimators_'):
        print(f"Estimators count: {len(clf.estimators_)}")
    else:
        print("❌ 'estimators_' attribute missing!")

    print("✅ Test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
