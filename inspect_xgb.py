
import inspect
import xgboost as xgb

print(f"XGBoost version: {xgb.__version__}")
print("XGBClassifier.fit signature:")
print(inspect.signature(xgb.XGBClassifier.fit))
