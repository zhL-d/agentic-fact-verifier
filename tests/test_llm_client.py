"""Unit tests for llm_client.py's schema-building helper."""

from pydantic import BaseModel, Field

from agentic_fact_verifier.llm_client import schema_response_format


class Inner(BaseModel):
    value: int


class Outer(BaseModel):
    items: list[Inner]
    label: str
    optional_notes: list[str] = Field(default_factory=list)


def _find_all_object_nodes(node):
    """Recursively yield every dict node with type == "object"."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for v in node.values():
            yield from _find_all_object_nodes(v)
    elif isinstance(node, list):
        for item in node:
            yield from _find_all_object_nodes(item)


def test_response_format_has_correct_top_level_shape():
    rf = schema_response_format(Outer, "Outer")
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Outer"
    assert rf["json_schema"]["strict"] is True
    assert "schema" in rf["json_schema"]


def test_every_object_node_gets_additional_properties_false():
    rf = schema_response_format(Outer, "Outer")
    schema = rf["json_schema"]["schema"]
    object_nodes = list(_find_all_object_nodes(schema))
    assert len(object_nodes) >= 2
    assert all(node["additionalProperties"] is False for node in object_nodes)


def test_every_property_is_marked_required_even_with_a_pydantic_default():
    rf = schema_response_format(Outer, "Outer")
    schema = rf["json_schema"]["schema"]
    object_nodes = list(_find_all_object_nodes(schema))
    for node in object_nodes:
        assert set(node["required"]) == set(node["properties"].keys()), (
            f"node {node.get('title', node)} has properties not listed in required"
        )
    assert "optional_notes" in schema["required"]
