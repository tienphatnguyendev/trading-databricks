"""Tests for trading_databricks.agents.schemas StructType constants."""
from __future__ import annotations

from pyspark.sql.types import ArrayType, DecimalType, StringType

from trading_databricks.agents.schemas import (
    CAUSAL_FEATURES_SCHEMA,
    CONFOUNDING_AUDIT_SCHEMA,
    DAGNESS_AUDIT_SCHEMA,
    MECHANISM_WHITELIST_SCHEMA,
    TREATMENT_TAXONOMY_SCHEMA,
)


def test_all_schemas_importable():
    assert CAUSAL_FEATURES_SCHEMA is not None
    assert TREATMENT_TAXONOMY_SCHEMA is not None
    assert MECHANISM_WHITELIST_SCHEMA is not None
    assert DAGNESS_AUDIT_SCHEMA is not None
    assert CONFOUNDING_AUDIT_SCHEMA is not None


def test_causal_features_schema_field_count():
    assert len(CAUSAL_FEATURES_SCHEMA.fields) == 23


def test_causal_features_non_nullable_fields():
    non_nullable = [f.name for f in CAUSAL_FEATURES_SCHEMA.fields if not f.nullable]
    assert set(non_nullable) == {
        "story_id",
        "published_at",
        "is_impactful",
        "agent_model_version",
        "processed_at",
    }


def test_dollar_bar_close_is_decimal_type():
    field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "dollar_bar_close")
    assert isinstance(field.dataType, DecimalType)
    assert field.dataType == DecimalType(18, 8)
    assert field.nullable is True


def test_evidence_spans_is_array_type():
    field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "evidence_spans")
    assert isinstance(field.dataType, ArrayType)
    assert field.dataType == ArrayType(StringType())


def test_extraction_path_field_exists():
    field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "extraction_path")
    assert isinstance(field.dataType, StringType)
    assert field.nullable is True


def test_evidence_relevance_score_field_exists():
    from pyspark.sql.types import DoubleType

    field = next(
        f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "evidence_relevance_score"
    )
    assert isinstance(field.dataType, DoubleType)
    assert field.nullable is True


def test_confounding_cluster_id_field_exists():
    field = next(
        f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "confounding_cluster_id"
    )
    assert isinstance(field.dataType, StringType)
    assert field.nullable is True


def test_treatment_taxonomy_schema_field_count():
    assert len(TREATMENT_TAXONOMY_SCHEMA.fields) == 4


def test_treatment_taxonomy_all_non_nullable():
    nullable_fields = [f.name for f in TREATMENT_TAXONOMY_SCHEMA.fields if f.nullable]
    assert nullable_fields == []


def test_mechanism_whitelist_schema_valid_directions_array():
    field = next(
        f for f in MECHANISM_WHITELIST_SCHEMA.fields if f.name == "valid_directions"
    )
    assert isinstance(field.dataType, ArrayType)
    assert field.dataType == ArrayType(StringType())
    assert field.nullable is False


def test_dagness_audit_schema_field_count():
    assert len(DAGNESS_AUDIT_SCHEMA.fields) == 8


def test_confounding_audit_schema_field_count():
    assert len(CONFOUNDING_AUDIT_SCHEMA.fields) == 10
