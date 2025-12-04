
import sys
import os

# Add the project to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.utils.evaluation import TIVSEvaluator
    print("✅ TIVSEvaluator imported successfully!")
    evaluator = TIVSEvaluator()
    print("✅ TIVSEvaluator instantiated successfully!")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
