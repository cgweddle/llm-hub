#!/usr/bin/env sh
set -eu

if [ -n "${FLOW_RUNNER_LLM_CONFIG:-}" ]; then
  python -m src.tasks.materialize_llm_config
fi

unset FLOW_RUNNER_LLM_CONFIG

exec python -m src.tasks.run_flow
