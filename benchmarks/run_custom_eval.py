
import os
import sys
import logging

# 1. Configure Environment with User Credentials
os.environ["AZURE_OPENAI_API_KEY"] = "7awsW9wUrULKH9CLPkeh1oeaeeGmKvIotw8HSYjVtjMcIgj0NqyLJQQJ99BIACYeBjFXJ3w3AAABACOG5UHM"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://goodwiinzapi.cognitiveservices.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"
# Override the model name used by the client to match the Azure Deployment Name
os.environ["MODEL_OVERRIDE"] = "gpt-4o"
# Dummy key to pass AgentDojo's initial check
os.environ["OPENAI_API_KEY"] = "dummy"

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())
# Ensure agentdojo src is in path
sys.path.append(os.path.join(os.getcwd(), 'benchmarks/external/agentdojo/src'))

# 2. Import Wrapper (Patches AgentPipeline)
import benchmarks.external.agentdojo_wrapper

# 3. Import Benchmark Script
from agentdojo.scripts import benchmark

# 4. Patch Click Options to allow 'bit_guard'
# This modifies the click Command object in place
defense_option = next((p for p in benchmark.main.params if p.name == "defense"), None)
if defense_option:
    if hasattr(defense_option.type, "choices"):
        print(f"Patching defense choices. Original: {defense_option.type.choices}")
        # Convert tuple to list to modify
        import click
        current_choices = list(defense_option.type.choices)
        if "bit_guard" not in current_choices:
            current_choices.append("bit_guard")
            defense_option.type.choices = current_choices
        print(f"Patched defense choices. New: {defense_option.type.choices}")

if __name__ == "__main__":
    print("Starting Custom AgentDojo Eval...")
    # Invoke the click command
    # We pass arguments via sys.argv or explicitly if we called callback. 
    # But calling main() parses sys.argv by default.
    benchmark.main()
