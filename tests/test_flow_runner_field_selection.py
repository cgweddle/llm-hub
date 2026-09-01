"""
Unit tests for _build_node_input's expanded-output field selection.

An edge from an expanded dict output into a generic whole-input target
(agent node, or tool without named param handles) stores {out_field: ""}.
The agent/trigger branches must feed the selected field, and the tool
branch must merge the selected value instead of creating a "" kwarg.
Also pins the pre-existing behaviors the refactor must not change.
"""
from src.runners.flow_runner import FlowRunContext, _build_node_input


UPSTREAM = {"text": "body", "status": 200}


def make_ctx(nodes, edges, state, tool_schemas=None, entry="F"):
    ctx = FlowRunContext(
        flow_id=1, user_id=1,
        graph_config={"nodes": nodes, "edges": edges,
                      "entry_point": entry, "exit_points": []},
        llm_config={}, agent_llms={}, conda_env=None, root_execution_id=1,
        entry_point=entry, functions_by_node={}, tool_schemas=tool_schemas or {},
        agents_by_node={}, agent_graphs={}, sequence_by_node={},
    )
    ctx.state.update(state)
    return ctx


def agent_input(mapping):
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "AG": {"node_type": "agent", "id": 1, "name": "AG"}},
        edges=[{"from_node": "F", "to_node": "AG", "mapping": mapping}],
        state={"F": UPSTREAM},
    )
    return _build_node_input(ctx, "AG", None)


def test_agent_gets_selected_field():
    assert agent_input({"text": ""}) == "body"


def test_agent_selected_missing_field_falls_back_to_whole():
    assert '"status": 200' in agent_input({"nope": ""})


def test_agent_without_mapping_still_gets_whole_output():
    result = agent_input(None)
    assert '"text": "body"' in result and '"status": 200' in result


def test_trigger_gets_selected_field():
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "T": {"node_type": "trigger", "id": 0, "name": "T"}},
        edges=[{"from_node": "F", "to_node": "T", "mapping": {"status": ""}}],
        state={"F": UPSTREAM},
    )
    assert _build_node_input(ctx, "T", None) == 200


def tool_input(mapping, input_values=None, schema=None):
    node = {"node_type": "tool", "id": 2, "name": "C"}
    if input_values:
        node["input_values"] = input_values
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"}, "C": node},
        edges=[{"from_node": "F", "to_node": "C", "mapping": mapping}],
        state={"F": UPSTREAM},
        tool_schemas={"C": schema or {"properties": {"a": {"type": "str"},
                                                     "b": {"type": "int"}}}},
    )
    return _build_node_input(ctx, "C", None)


def test_tool_generic_input_autowraps_selected_field():
    assert tool_input({"text": ""}) == {"a": "body"}


def test_tool_generic_input_respects_typed_in_values():
    assert tool_input({"text": ""}, input_values={"a": "x"}) == {"a": "x", "b": "body"}


def test_tool_generic_input_merges_selected_dict_field():
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "C": {"node_type": "tool", "id": 2, "name": "C"}},
        edges=[{"from_node": "F", "to_node": "C", "mapping": {"inner": ""}}],
        state={"F": {"inner": {"a": 1, "b": 2}}},
        tool_schemas={"C": {"properties": {"a": {}, "b": {}}}},
    )
    assert _build_node_input(ctx, "C", None) == {"a": 1, "b": 2}


# ─── Regressions: pre-existing branches must be unchanged ────────────────────

def test_tool_field_to_param_mapping_unchanged():
    assert tool_input({"text": "a"}) == {"a": "body"}


def test_tool_passthrough_merge_unchanged():
    assert tool_input(None) == UPSTREAM


def test_tool_autowrap_nondict_unchanged():
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "C": {"node_type": "tool", "id": 2, "name": "C"}},
        edges=[{"from_node": "F", "to_node": "C", "mapping": None}],
        state={"F": "plain text"},
        tool_schemas={"C": {"properties": {"a": {}, "b": {}}}},
    )
    assert _build_node_input(ctx, "C", None) == {"a": "plain text"}


def test_tool_nothing_to_map_into_returns_upstream():
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "C": {"node_type": "tool", "id": 2, "name": "C",
                     "input_values": {"a": 1, "b": 2}}},
        edges=[{"from_node": "F", "to_node": "C", "mapping": None}],
        state={"F": "plain text"},
        tool_schemas={"C": {"properties": {"a": {}, "b": {}}}},
    )
    assert _build_node_input(ctx, "C", None) == "plain text"
