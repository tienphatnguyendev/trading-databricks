"""Tests for the MAS E2E deep-dive research notebook + its helpers."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from notebooks.qa.utils import _extract_headline


class TestExtractHeadline:
    def test_strips_standard_prefix(self) -> None:
        request = "Can you summarize the market impact of the following news: BTC rallies past 120k"
        assert _extract_headline(request) == "BTC rallies past 120k"

    def test_handles_extra_whitespace_after_colon(self) -> None:
        request = "Can you summarize the market impact of the following news:   Fed cuts rates"
        assert _extract_headline(request) == "Fed cuts rates"

    def test_fallback_to_full_request_when_prefix_missing(self) -> None:
        request = "Completely unrelated phrasing about ETH"
        assert _extract_headline(request) == "Completely unrelated phrasing about ETH"

    def test_empty_string_returns_empty(self) -> None:
        assert _extract_headline("") == ""

    def test_none_input_returns_empty(self) -> None:
        # Defensive against null Spark rows
        assert _extract_headline(None) == ""


# ---------------------------------------------------------------------------
# Task 2 — Dual-run guards
# ---------------------------------------------------------------------------
from notebooks.qa.utils import _display, _get_dbutils  # noqa: E402


class TestDualRunGuards:
    def test_display_falls_back_to_print_when_builtin_absent(self, capsys: pytest.CaptureFixture) -> None:
        # In a uv run python context, `display` is not a builtin.
        _display("hello-mas")
        captured = capsys.readouterr()
        assert "hello-mas" in captured.out

    def test_get_dbutils_returns_none_outside_databricks(self) -> None:
        # uv run python has no dbutils — helper must return None, not raise.
        assert _get_dbutils() is None


# ---------------------------------------------------------------------------
# Task 3 — Preflight probes
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock, patch  # noqa: E402

from notebooks.qa.utils import _endpoint_ready, _table_has_rows, _vs_index_exists  # noqa: E402


class TestPreflightProbes:
    def test_endpoint_ready_returns_true_when_state_is_ready(self) -> None:
        fake_client = MagicMock()
        fake_client.serving_endpoints.get.return_value.state.ready = "READY"
        with patch("notebooks.qa.utils.WorkspaceClient", return_value=fake_client):
            ok, detail = _endpoint_ready("databricks-claude-sonnet")
        assert ok is True
        assert "READY" in detail

    def test_endpoint_ready_returns_false_when_not_ready(self) -> None:
        fake_client = MagicMock()
        fake_client.serving_endpoints.get.return_value.state.ready = "NOT_READY"
        with patch("notebooks.qa.utils.WorkspaceClient", return_value=fake_client):
            ok, detail = _endpoint_ready("databricks-claude-sonnet")
        assert ok is False

    def test_endpoint_ready_catches_sdk_exception(self) -> None:
        with patch("notebooks.qa.utils.WorkspaceClient", side_effect=RuntimeError("network")):
            ok, detail = _endpoint_ready("databricks-claude-sonnet")
        assert ok is False
        assert "network" in detail

    def test_table_has_rows_true_when_count_positive(self) -> None:
        fake_spark = MagicMock()
        fake_spark.table.return_value.limit.return_value.count.return_value = 1
        with patch("notebooks.qa.utils.get_spark", return_value=fake_spark):
            ok, detail = _table_has_rows("catalog.schema.table")
        assert ok is True

    def test_table_has_rows_false_on_empty(self) -> None:
        fake_spark = MagicMock()
        fake_spark.table.return_value.limit.return_value.count.return_value = 0
        with patch("notebooks.qa.utils.get_spark", return_value=fake_spark):
            ok, detail = _table_has_rows("catalog.schema.table")
        assert ok is False

    def test_table_has_rows_catches_exception(self) -> None:
        with patch("notebooks.qa.utils.get_spark", side_effect=RuntimeError("no spark")):
            ok, detail = _table_has_rows("catalog.schema.table")
        assert ok is False
        assert "no spark" in detail

    def test_vs_index_exists_true_when_describe_succeeds(self) -> None:
        fake_vsc = MagicMock()
        fake_vsc.get_index.return_value.describe.return_value = {"status": {"ready": True}}
        with patch("notebooks.qa.utils.VectorSearchClient", return_value=fake_vsc):
            ok, detail = _vs_index_exists(endpoint_name="mas-few-shot", index_name="c.s.i")
        assert ok is True

    def test_vs_index_exists_false_on_exception(self) -> None:
        with patch("notebooks.qa.utils.VectorSearchClient", side_effect=RuntimeError("offline")):
            ok, detail = _vs_index_exists(endpoint_name="mas-few-shot", index_name="c.s.i")
        assert ok is False
        assert "offline" in detail


# ---------------------------------------------------------------------------
# Task 4 — Stratified sample
# ---------------------------------------------------------------------------
import polars as pl  # noqa: E402

from notebooks.qa.utils import _stratified_sample  # noqa: E402


class TestStratifiedSample:
    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame({
            "story_id": [str(i) for i in range(20)],
            "_bucket": ["a"] * 10 + ["b"] * 10,
            "request": [f"r{i}" for i in range(20)],
        })

    def test_returns_requested_total_size(self) -> None:
        df = self._frame()
        out = _stratified_sample(df, n=8, by="_bucket", seed=42)
        assert out.height == 8

    def test_distributes_across_buckets(self) -> None:
        df = self._frame()
        out = _stratified_sample(df, n=8, by="_bucket", seed=42)
        counts = out.group_by("_bucket").len().sort("_bucket")
        # stratified: both buckets must be represented
        assert counts.height == 2
        assert counts["len"].min() >= 1

    def test_seed_is_deterministic(self) -> None:
        df = self._frame()
        a = _stratified_sample(df, n=8, by="_bucket", seed=42)
        b = _stratified_sample(df, n=8, by="_bucket", seed=42)
        assert a["story_id"].to_list() == b["story_id"].to_list()

    def test_n_larger_than_frame_returns_full_frame(self) -> None:
        df = self._frame()
        out = _stratified_sample(df, n=50, by="_bucket", seed=42)
        assert out.height == df.height


# ---------------------------------------------------------------------------
# Task 5 — build_instrumented_mas_graph
# ---------------------------------------------------------------------------
class TestBuildInstrumentedMasGraph:
    def test_compiles_with_five_nodes(self) -> None:
        from notebooks.qa.utils import build_instrumented_mas_graph
        from trading_databricks.agents.config import MASConfig

        compiled = build_instrumented_mas_graph(MASConfig())
        mermaid = compiled.get_graph().draw_mermaid()
        for node in (
            "context_retriever",
            "few_shot_retriever",
            "uncertainty_router",
            "causality_extractor",
            "mechanism_validator",
        ):
            assert node in mermaid, f"{node} missing from compiled graph topology"

    def test_entry_and_finish_points(self) -> None:
        from notebooks.qa.utils import build_instrumented_mas_graph
        from trading_databricks.agents.config import MASConfig

        compiled = build_instrumented_mas_graph(MASConfig())
        g = compiled.get_graph()
        # LangGraph exposes nodes dict + edges list on get_graph() result
        assert "context_retriever" in g.nodes
        assert "mechanism_validator" in g.nodes


# ---------------------------------------------------------------------------
# Task 6 — wire_graph_environment
# ---------------------------------------------------------------------------
class TestWireGraphEnvironment:
    def test_offline_mode_replaces_query_vs_index(self) -> None:
        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents.config import MASConfig
        import importlib
        fsr_mod = importlib.import_module("trading_databricks.agents.nodes.few_shot_retriever")

        config = MASConfig()
        original = fsr_mod.query_vs_index
        try:
            with wire_graph_environment(config, offline=True):
                # Placeholder must be replaced with a callable that does not raise
                result = fsr_mod.query_vs_index(
                    index_name="x",
                    columns=[],
                    query_text="Fed raises rates",
                    num_results=5,
                    filters_json="{}"
                )
                assert isinstance(result, list)
        finally:
            assert fsr_mod.query_vs_index is original, "wire_graph_environment must restore on exit"

    def test_offline_mode_never_patches_mechanism_validator(self) -> None:
        """Regression guard: _mechanism_validator_node must remain the real function."""
        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents import graph as agents_graph
        from trading_databricks.agents.config import MASConfig

        real = agents_graph._mechanism_validator_node
        with wire_graph_environment(MASConfig(), offline=True):
            assert agents_graph._mechanism_validator_node is real

    def test_offline_mode_predict_returns_deterministic_llm_shape(self) -> None:
        """Offline mock predict returns a taxonomy-compliant JSON response."""
        import json
        import mlflow.deployments

        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents.config import MASConfig

        config = MASConfig()
        with wire_graph_environment(config, offline=True):
            client = mlflow.deployments.get_deploy_client("databricks")
            resp = client.predict(
                endpoint=config.llm_endpoint,
                inputs={"messages": [{"role": "user", "content": "x"}], "temperature": 0.0, "n": 1, "max_tokens": 100},
            )
            assert "choices" in resp
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            assert "treatment_class" in parsed
            assert "mechanism" in parsed
            assert "expected_direction" in parsed

    def test_offline_mode_predict_returns_embedding_shape(self) -> None:
        import mlflow.deployments

        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents.config import MASConfig

        config = MASConfig()
        with wire_graph_environment(config, offline=True):
            client = mlflow.deployments.get_deploy_client("databricks")
            resp = client.predict(
                endpoint=config.validator.embedding_model,
                inputs={"input": ["hello"]},
            )
            assert "data" in resp
            assert "embedding" in resp["data"][0]
            assert isinstance(resp["data"][0]["embedding"], list)


# ---------------------------------------------------------------------------
# Task 7 — Notebook structural tests
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = Path("notebooks/qa/mas/e2e_mas_deep_dive_research.py")


def _notebook_text() -> str:
    assert NOTEBOOK_PATH.exists(), f"notebook not found at {NOTEBOOK_PATH}"
    return NOTEBOOK_PATH.read_text()


class TestNotebookStructure:
    def test_first_line_is_databricks_source_marker(self) -> None:
        content = _notebook_text()
        assert content.splitlines()[0] == "# Databricks notebook source"

    def test_has_setup_section(self) -> None:
        content = _notebook_text()
        assert "## §1 Setup" in content
        assert "MAS_RESEARCH_OFFLINE" in content
        assert "MAS_RESEARCH_SAMPLE_SIZE" in content
        assert "mlflow_component_run" in content
        assert "mlflow.langchain.autolog()" in content

    def test_notebook_is_valid_python(self) -> None:
        """The file must parse as Python — MAGIC lines are comments, COMMAND markers are comments."""
        content = _notebook_text()
        ast.parse(content)

    # Task 8
    def test_has_topology_diagram_section(self) -> None:
        content = _notebook_text()
        assert "## §2 System topology diagram" in content
        assert "build_instrumented_mas_graph(config)" in content
        assert "draw_mermaid" in content
        assert 'mlflow.log_text(mermaid, "topology/graph.mmd")' in content

    # Task 9
    def test_has_preflight_section(self) -> None:
        content = _notebook_text()
        assert "## §3 Pre-flight checks" in content
        assert "preflight_rows" in content
        assert "_endpoint_ready(config.llm_endpoint)" in content
        assert "_endpoint_ready(config.validator.embedding_model)" in content
        assert "_table_has_rows" in content
        assert "_vs_index_exists" in content
        assert "cold_start_mode" in content
        assert "accepted_chain_count" in content
        # hard-fail enforcement
        assert 'raise RuntimeError(f"preflight critical check failed' in content

    # Task 10
    def test_has_dataset_ingestion_section(self) -> None:
        content = _notebook_text()
        assert "## §4 Dataset ingestion" in content
        assert "smart-trading-dev.others.mas_training_dataset" in content
        assert "expected_route" in content
        assert "_extract_headline" in content
        assert "_stratified_sample" in content
        assert "DeltaDatasetSource" in content
        assert "mlflow.log_input" in content
        assert "eval_dataset" in content
        assert "negative_control_dataset" in content

    # Task 11
    def test_has_graph_wireup_section(self) -> None:
        content = _notebook_text()
        assert "## §5 Graph wiring" in content
        assert "wire_graph_environment(config, offline=OFFLINE)" in content

    # Task 12
    def test_has_predict_fn_and_scorers_section(self) -> None:
        content = _notebook_text()
        assert "## §6 predict_fn + custom scorers" in content
        assert "def predict_fn(" in content
        assert "from notebooks.qa.mas._scorer_helpers import" in content
        assert "@scorer" in content
        assert "def schema_conformance(" in content
        assert "def taxonomy_whitelist_compliance(" in content
        assert "make_routing_legality_scorer" in content
        assert "def evidence_grounding_substring(" in content
        assert "def extraction_path_label(" in content
        assert "make_judge" in content
        assert 'name="groundedness"' in content
        assert "load_whitelist" in content

    # Task 13
    def test_has_evaluate_call_section(self) -> None:
        content = _notebook_text()
        assert "## §7 mlflow.genai.evaluate()" in content
        assert "mlflow.genai.evaluate(" in content
        assert "data=eval_dataset" in content
        assert "predict_fn=predict_fn" in content
        assert "scorers=[" in content
        assert "schema_conformance" in content
        assert "groundedness_judge" in content
        assert "negative_control" in content

    # Task 14
    def test_has_trace_analysis_section(self) -> None:
        content = _notebook_text()
        assert "## §8 Trace-derived analysis" in content
        assert "mlflow.search_traces" in content
        assert 'extract_fields=["spans"]' in content
        assert "node_latency_stats" in content
        assert "span_metrics_raw" in content
        # no fabricated latency numbers
        assert "np.random" not in content
        assert "random.random()" not in content
        assert "time.time() - t0) * 10" not in content

    # Task 15
    def test_has_execution_overlay_section(self) -> None:
        content = _notebook_text()
        assert "## §9 Execution-overlay" in content
        assert "topology_overlay" in content
        assert "log_plotly" in content

    # Task 16
    def test_has_scorer_aggregation_section(self) -> None:
        content = _notebook_text()
        assert "## §10 Scorer result aggregation" in content
        assert "eval_metrics_table" in content
        assert "build_path_distribution" in content
        assert "build_label_distribution" in content

    # Task 17
    def test_has_autopsy_sandbox_section(self) -> None:
        content = _notebook_text()
        assert "## §11 Failure autopsy sandbox" in content
        assert "def render_autopsy(" in content
        assert "def replay_single(" in content
        assert "_get_dbutils()" in content
        assert "dbu.widgets.text" in content
        assert "nested=True" in content

    # Task 18
    def test_has_feedback_stubs_section(self) -> None:
        content = _notebook_text()
        assert "## §12 Human feedback hooks" in content
        assert "mlflow.log_feedback" in content
        assert "AssessmentSource" in content
        assert "sme_verdict" in content

    # Task 19
    def test_has_assertions_and_summary_section(self) -> None:
        content = _notebook_text()
        assert "## §13 Sanity assertions" in content
        assert "assert eval_result" in content
        assert "## §14 Summary" in content
        assert "graph_env_ctx.__exit__(None, None, None)" in content

    def test_no_fabricated_latency_code_anywhere(self) -> None:
        """Regression guard for the 99_e2e_graph.py bug we are fixing."""
        content = _notebook_text()
        assert "10.0 + (time.time() - t0) * 10" not in content
        assert "np.random.uniform" not in content
        assert "random.random()" not in content

    def test_no_mock_mechanism_validator(self) -> None:
        """Regression guard: _mechanism_validator_node must never be MagicMock-patched in this notebook."""
        content = _notebook_text()
        assert 'patch("trading_databricks.agents.graph._mechanism_validator_node")' not in content
        assert "MagicMock" not in content


# ---------------------------------------------------------------------------
# Task 12 — Scorer helper unit tests
# ---------------------------------------------------------------------------
class TestScorerHelpers:
    def test_check_schema_conformance_all_valid(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 0.8,
            "trigger_event": "Fed raises rates",
            "evidence_spans": ["Fed raises rates 75 bps"],
        }
        assert check_schema_conformance(extraction) == 1.0

    def test_check_schema_conformance_missing_field(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 0.8,
            # missing trigger_event
            "evidence_spans": ["x"],
        }
        result = check_schema_conformance(extraction)
        assert 0.0 <= result < 1.0

    def test_check_schema_conformance_none_returns_zero(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        assert check_schema_conformance(None) == 0.0

    def test_check_schema_conformance_invalid_direction(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "sideways",  # invalid
            "confidence": 0.8,
            "trigger_event": "Fed raises rates",
            "evidence_spans": ["x"],
        }
        result = check_schema_conformance(extraction)
        assert result < 1.0

    def test_check_schema_conformance_bad_confidence(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 1.7,  # out of [0,1]
            "trigger_event": "x",
            "evidence_spans": ["x"],
        }
        assert check_schema_conformance(extraction) < 1.0

    def test_check_taxonomy_whitelist_compliance_valid(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        whitelist = {("Macro", "liquidity_contraction"): ["bearish", "volatile"]}
        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
        }
        assert check_taxonomy_whitelist_compliance(extraction, whitelist) is True

    def test_check_taxonomy_whitelist_compliance_wrong_direction(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        whitelist = {("Macro", "liquidity_contraction"): ["bearish"]}
        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bullish",
        }
        assert check_taxonomy_whitelist_compliance(extraction, whitelist) is False

    def test_check_taxonomy_whitelist_compliance_unknown_pair(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        whitelist = {("Macro", "liquidity_contraction"): ["bearish"]}
        extraction = {
            "treatment_class": "Unknown.x",
            "mechanism": "mystery_mechanism",
            "expected_direction": "bearish",
        }
        assert check_taxonomy_whitelist_compliance(extraction, whitelist) is False

    def test_check_taxonomy_whitelist_compliance_none_extraction(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        assert check_taxonomy_whitelist_compliance(None, {}) is False

    def test_check_routing_legality_cold_start_only_careful_is_legal(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_routing_legality

        assert check_routing_legality("careful", cold_start_mode=True,
                                      top_similarity=0.0, n_examples=5) is True
        assert check_routing_legality("fast", cold_start_mode=True,
                                      top_similarity=0.95, n_examples=5) is False

    def test_check_routing_legality_warm_start_rules(self) -> None:
        from notebooks.qa.mas._scorer_helpers import check_routing_legality

        # fast: high sim, >= 2 examples
        assert check_routing_legality("fast", cold_start_mode=False,
                                      top_similarity=0.9, n_examples=3) is True
        # fast illegal when sim too low
        assert check_routing_legality("fast", cold_start_mode=False,
                                      top_similarity=0.7, n_examples=3) is False
        # careful legal when sim low
        assert check_routing_legality("careful", cold_start_mode=False,
                                      top_similarity=0.3, n_examples=3) is True
        # standard in the middle zone
        assert check_routing_legality("standard", cold_start_mode=False,
                                      top_similarity=0.7, n_examples=3) is True

    def test_evidence_grounding_ratio(self) -> None:
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        body = "The Fed raised rates by 75 bps today, citing inflation."
        spans = ["Fed raised rates by 75 bps", "citing inflation"]
        assert evidence_grounding_ratio(spans, body) == 1.0

    def test_evidence_grounding_ratio_partial(self) -> None:
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        body = "The Fed raised rates by 75 bps today."
        spans = ["Fed raised rates by 75 bps", "imagined text that is not there"]
        assert evidence_grounding_ratio(spans, body) == 0.5

    def test_evidence_grounding_ratio_no_spans_is_zero(self) -> None:
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        assert evidence_grounding_ratio([], "body") == 0.0

    def test_evidence_grounding_ratio_casefold_match(self) -> None:
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        body = "The FED RAISED rates by 75 bps."
        spans = ["fed raised"]
        assert evidence_grounding_ratio(spans, body) == 1.0
