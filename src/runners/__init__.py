"""Flow execution runners.

`flow_runner` generates an orchestrator from a flow's graph_config (via
`src.factories.flow_to_script_factory`), execs it, and supplies the runtime
helpers the generated code calls (`_run_stage`, `_resolve_node_input`).
"""
