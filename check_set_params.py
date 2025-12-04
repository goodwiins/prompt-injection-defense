
import xgboost as xgb
from xgboost.callback import EarlyStopping

try:
    clf = xgb.XGBClassifier()
    callbacks = [EarlyStopping(rounds=10)]
    clf.set_params(callbacks=callbacks)
    print("set_params(callbacks=...) succeeded")
    print(f"Callbacks param: {clf.get_params().get('callbacks')}")
except Exception as e:
    print(f"set_params failed: {e}")
