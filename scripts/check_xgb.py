
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
try:
    from xgboost.callback import EarlyStopping
    print("EarlyStopping callback available")
except ImportError:
    print("EarlyStopping callback NOT available")
