
import os
import sys
import logging

# 1. Configure Environment for Local Execution
# Clear Azure keys to avoid confusion (though LocalLLM shouldn't use them)
if "AZURE_OPENAI_API_KEY" in os.environ:
    del os.environ["AZURE_OPENAI_API_KEY"]
if "AZURE_OPENAI_ENDPOINT" in os.environ:
    del os.environ["AZURE_OPENAI_ENDPOINT"]

# Set Local LLM Port to 8001 (assuming 8000 is taken by API)
if "LOCAL_LLM_PORT" not in os.environ:
    os.environ["LOCAL_LLM_PORT"] = "8001"

# Dummy key for OpenAI checks
os.environ["OPENAI_API_KEY"] = "dummy"

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())
# Ensure agentdojo src is in path
sys.path.append(os.path.join(os.getcwd(), 'benchmarks/external/agentdojo/src'))

# 2. Import Wrapper (Patches AgentPipeline)
try:
    import benchmarks.external.agentdojo_wrapper
except ImportError as e:
    print(f"Failed to import wrapper: {e}")
    sys.exit(1)

# 3. Import Benchmark Script
from agentdojo.scripts import benchmark

# 4. Patch Click Options to allow 'bit_guard'
# This modifies the click Command object in place
defense_option = next((p for p in benchmark.main.params if p.name == "defense"), None)
if defense_option:
    if hasattr(defense_option.type, "choices"):
        print(f"Patching defense choices. Original: {defense_option.type.choices}")
        import click
        # Convert tuple to list to modify
        current_choices = list(defense_option.type.choices)
        if "bit_guard" not in current_choices:
            current_choices.append("bit_guard")
            defense_option.type.choices = current_choices
        print(f"Patched defense choices. New: {defense_option.type.choices}")

if __name__ == "__main__":
    print(f"Starting Local AgentDojo Eval (Port: {os.environ['LOCAL_LLM_PORT']})...")
    print("Ensure you have an OpenAI-compatible server running (e.g., vLLM or llama-cpp-python).")
    benchmark.main()
