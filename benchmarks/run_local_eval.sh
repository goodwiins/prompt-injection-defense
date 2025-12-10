#!/bin/bash
# Script to run AgentDojo evaluation with BIT Guard (Local Model)
# Usage: ./benchmarks/run_local_eval.sh [PORT]

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/benchmarks/external/agentdojo/src

PORT="${1:-1234}"
export LOCAL_LLM_PORT="$PORT"

echo "Running AgentDojo Benchmark with BIT Guard (Local Model on port $PORT)..."
echo "Model: local"
echo "Defense: bit_guard"

# We use the local runner
python benchmarks/run_local_runner.py \
    --suite workspace \
    --user-task user_task_0 \
    --attack direct \
    --model LOCAL \
    --defense bit_guard \
    --logdir results/agentdojo_runs/local
