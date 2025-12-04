"""
Add this to the top of your notebook to fix imports.
Run this cell first before importing from src modules.
"""
import sys
import os

# Get the directory where the notebook is located
notebook_dir = os.getcwd()

# Add the project root to Python path (where src/ directory is)
if notebook_dir not in sys.path:
    sys.path.insert(0, notebook_dir)

print(f"✅ Added {notebook_dir} to Python path")
print(f"📂 Current working directory: {os.getcwd()}")
print(f"🔍 Python path includes:")
for path in sys.path[:3]:
    print(f"   - {path}")
