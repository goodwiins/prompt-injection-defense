
import json
import os

notebook_path = "bit_demonstration from Colab.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The code we want to inject
new_code = [
    "import sys\n",
    "import os\n",
    "import importlib\n",
    "import benchmarks.runner\n",
    "import benchmarks.reporter\n",
    "import benchmarks.metrics\n",
    "\n",
    "# Reload modules to ensure we have the latest baseline comparison logic\n",
    "importlib.reload(benchmarks.metrics)\n",
    "importlib.reload(benchmarks.runner)\n",
    "importlib.reload(benchmarks.reporter)\n",
    "\n",
    "from benchmarks.runner import create_runner_from_model\n",
    "from benchmarks.reporter import BenchmarkReporter\n",
    "\n",
    "# Load the BIT model\n",
    "model_path = \"models/bit_xgboost_model.json\"\n",
    "if not os.path.exists(model_path):\n",
    "    print(f\"⚠️ Model not found at {model_path}. Please ensure you have trained the model.\")\n",
    "else:\n",
    "    print(f\"Loading model from {model_path}...\")\n",
    "    runner = create_runner_from_model(model_path, model_type=\"embedding_classifier\")\n",
    "    \n",
    "    # Run benchmark (limit to 100 samples per dataset for demonstration speed)\n",
    "    # Set samples_per_dataset=None to run full evaluation\n",
    "    print(\"Running benchmark (100 samples/dataset)...\")\n",
    "    results = runner.run_quick(samples_per_dataset=100, verbose=True)\n",
    "    \n",
    "    # Print full report with Baseline Comparisons\n",
    "    print(\"\\nGENERATING REPORT...\\n\")\n",
    "    reporter = BenchmarkReporter(results)\n",
    "    reporter.print_console()\n"
]

# Find the cell to replace
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        # Look for the triggering command
        if any("!python -m benchmarks.run_benchmark" in line for line in source) and any("--paper" in line for line in source):
            print("Found target cell. updating...")
            cell["source"] = new_code
            # clear outputs to avoid confusion with old outputs
            cell["outputs"] = []
            cell["execution_count"] = None
            updated = True
            break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {notebook_path}")
else:
    print("Could not find the target cell to update.")
