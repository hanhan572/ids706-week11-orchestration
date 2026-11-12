"""Lab LangGraph skeleton readiness + Airflow student DAG smoke checks."""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from lab.langgraph.graph import (
    REQUIRED_NODES,
    SimpleLoopState,
    assess_graph_ready,
    partition_drawable_edges,
    scan_declared_edges,
    scan_declared_nodes,
)

REPO = Path(__file__).resolve().parents[2]


def test_scan_declared_nodes_partial() -> None:
    src = '''
def build_graph():
    g = StateGraph(SimpleLoopState)
    g.add_node("data_scientist", node_data_scientist)
    g.add_edge(START, "data_scientist")
    g.add_node("hypothesis_generator", node_hypothesis)
    return g.compile()
'''
    assert scan_declared_nodes(src) == ["data_scientist", "hypothesis_generator"]


def test_scan_declared_nodes_empty() -> None:
    assert scan_declared_nodes("def build_graph():\n    raise NotImplementedError\n") == []


def test_scan_declared_edges_ignores_undeclared_spine() -> None:
    src = '''
def build_graph():
    g.add_node("data_scientist", node_data_scientist)
    g.add_edge(START, "data_scientist")
    g.add_node("hypothesis_generator", node_hypothesis)
    return g.compile()
'''
    edges = {(e["source"], e["target"]) for e in scan_declared_edges(src)}
    assert ("__start__", "data_scientist") in edges
    assert ("data_scientist", "hypothesis_generator") not in edges


def test_partition_drops_unreachable_edge() -> None:
    nodes = ["data_scientist", "hypothesis_generator"]
    edges = [
        {"source": "__start__", "target": "data_scientist"},
        {"source": "hypothesis_generator", "target": "data_scientist"},
    ]
    ok, problems = partition_drawable_edges(nodes, edges)
    assert ok == [{"source": "__start__", "target": "data_scientist"}]
    assert any("unreachable from START" in p for p in problems)


def test_assess_ready_when_required_nodes_present() -> None:
    def _noop(state: SimpleLoopState) -> dict:
        return {}

    g = StateGraph(SimpleLoopState)
    for name in REQUIRED_NODES:
        g.add_node(name, _noop)
    g.add_edge(START, "data_scientist")
    g.add_edge("data_scientist", "hypothesis_generator")
    g.add_edge("hypothesis_generator", "ml_engineer")
    g.add_edge("ml_engineer", "judge")
    g.add_edge("judge", "memory_writer")
    g.add_edge("memory_writer", "context_builder")
    g.add_edge("context_builder", "holdout_eval")
    g.add_edge("holdout_eval", END)
    app = g.compile()

    status = assess_graph_ready(app)
    assert status["ready"] is True
    assert status["missing_nodes"] == []
    assert set(REQUIRED_NODES).issubset(set(status["nodes"]))


def test_student_airflow_dag_is_unwired_skeleton() -> None:
    path = REPO / "lab" / "airflow" / "dags" / "lab_baseline_prep.py"
    text = path.read_text()
    assert 'dag_id="lab_baseline_prep"' in text
    assert "# left = validate_customers()" in text
    assert "raise NotImplementedError" in text
    # Wiring must stay commented so Airflow Graph stays empty for students.
    assert "\n    left = validate_customers()\n" not in text
