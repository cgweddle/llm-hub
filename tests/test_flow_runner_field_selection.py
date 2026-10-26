"""
Unit tests for _build_node_input's expanded-output field selection AND the
named-inputs contract for agent targets (claude_plans/named-agent-inputs.md,
plan step 5).

An edge from an expanded dict output into a generic whole-input target
(agent node, or tool without named param handles) stores {out_field: ""}.
The trigger branch still just feeds the selected scalar value, and the tool
branch still merges the selected value instead of creating a "" kwarg —
unchanged.

The AGENT branch changed: it now loops over ALL incoming edges (not just
incoming[0]) and returns a dict of named inputs, `{target_input_name:
_stringify_for_agent(selected_upstream)}`. An edge's `mapping` is the
tool-edge shape `{out_field: target_input_name}`; `""` (or no mapping at
all) targets the legacy `"input"` name — the one-time read for edges that
predate named inputs (decision 5), not a runtime fallback.

Also pins the pre-existing behaviors the refactor must not change (tool /
trigger branches stay scalar / dict-of-kwargs, never wrapped in a named
dict).
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
    """Single edge F->AG with the given mapping (tool-edge shape:
    {out_field: target_input_name}; "" targets the legacy "input" name)."""
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "AG": {"node_type": "agent", "id": 1, "name": "AG"}},
        edges=[{"from_node": "F", "to_node": "AG", "mapping": mapping}],
        state={"F": UPSTREAM},
    )
    return _build_node_input(ctx, "AG", None)


def test_agent_gets_selected_field():
    """mapping value "" targets the legacy "input" name — still a dict."""
    assert agent_input({"text": ""}) == {"input": "body"}


def test_agent_mapped_field_delivers_to_named_input():
    """A non-empty mapping value is the agent's NAMED input, not "input"."""
    assert agent_input({"text": "draft"}) == {"draft": "body"}


def test_agent_selected_missing_field_falls_back_to_whole():
    result = agent_input({"nope": ""})
    assert '"status": 200' in result["input"]


def test_agent_without_mapping_still_gets_whole_output():
    result = agent_input(None)
    assert '"text": "body"' in result["input"] and '"status": 200' in result["input"]


def test_agent_multiple_edges_deliver_separate_named_inputs():
    """The agent branch loops over ALL incoming edges — not just
    incoming[0] — building one dict with one entry per edge, instead of
    the old "last writer wins" (or outright-ignored) behavior."""
    ctx = make_ctx(
        nodes={"F": {"node_type": "tool", "id": 1, "name": "F"},
               "G": {"node_type": "tool", "id": 2, "name": "G"},
               "AG": {"node_type": "agent", "id": 1, "name": "AG"}},
        edges=[
            {"from_node": "F", "to_node": "AG", "mapping": {"text": "draft"}},
            {"from_node": "G", "to_node": "AG", "mapping": {"": "notes"}},
        ],
        state={"F": UPSTREAM, "G": "a note"},
    )
    assert _build_node_input(ctx, "AG", None) == {"draft": "body", "notes": "a note"}


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
