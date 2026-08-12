from pathlib import Path

from audit_pid2graph_open100_v8 import graphml_schema


def test_graphml_ids_are_not_tag_references() -> None:
    fixture = Path("outputs/test_runtime_v8_unit/pid2graph_schema_fixture.graphml")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
          <key id="d0" for="node" attr.name="label" attr.type="string"/>
          <key id="d1" for="node" attr.name="xmin" attr.type="double"/>
          <graph edgedefault="undirected">
            <node id="P-101"><data key="d0">valve</data><data key="d1">1.5</data></node>
            <node id="N2"><data key="d0">pump</data><data key="d1">2.5</data></node>
            <edge source="P-101" target="N2"/>
          </graph>
        </graphml>
        """,
        encoding="utf-8",
    )
    schema = graphml_schema(fixture)
    assert schema["node_count"] == 2
    assert schema["edge_count"] == 1
    assert schema["explicit_tag_reference_candidates"] == []
