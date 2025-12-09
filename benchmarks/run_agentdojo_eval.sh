#!/bin/bash
# Script to run AgentDojo evaluation with BIT Guard (Azure OpenAI)

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/benchmarks/external/agentdojo/src

echo "Running AgentDojo Benchmark with BIT Guard (Azure)..."
echo "Model: gpt-4o (Azure Deployment)"
echo "Defense: bit_guard"

# We use the custom runner which sets up credentials and patches arguments
python benchmarks/run_custom_eval.py \
    --suite workspace \
    --user-task user_task_0 \
    --attack direct \
    --model GPT_4O_2024_05_13 \
    --defense bit_guard \
    --logdir results/agentdojo_runs

