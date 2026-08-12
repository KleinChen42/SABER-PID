from xml.etree import ElementTree as ET

from prepare_dexpi_external_v8 import (
    normalize_tag,
    prefix_map,
    structured_tag_candidates,
    tags_in_text,
    visible_structured_tags,
)


def test_normalize_hierarchical_tag_variants() -> None:
    assert normalize_tag("PI 4712.01") == "pi4712-01"
    assert normalize_tag("PI-4712-01") == "pi4712-01"
    assert normalize_tag("P4712") == "p4712"
    assert tags_in_text("P4712, PI 4712.01") == {"p4712", "pi4712-01"}


def test_structured_candidate_requires_graphic_confirmation() -> None:
    root = ET.fromstring(
        """
        <PlantModel>
          <Equipment>
            <GenericAttributes>
              <GenericAttribute Name="TagNamePrefixAssignmentClass" Value="P" />
              <GenericAttribute Name="TagNameSequenceNumberAssignmentClass" Value="4712" />
            </GenericAttributes>
            <Label><Text String="P4712" /></Label>
          </Equipment>
          <Equipment>
            <GenericAttributes>
              <GenericAttribute Name="TagNamePrefixAssignmentClass" Value="T" />
              <GenericAttribute Name="TagNameSequenceNumberAssignmentClass" Value="9999" />
            </GenericAttributes>
          </Equipment>
        </PlantModel>
        """
    )
    assert structured_tag_candidates(root) == {"p4712", "t9999"}
    visible, diagnostics = visible_structured_tags(root)
    assert visible == {"p4712"}
    assert diagnostics["visible_structured_tag_count"] == 1
    assert prefix_map(visible) == {"p": ["p4712"]}


def test_legacy_complete_function_and_number() -> None:
    root = ET.fromstring(
        """
        <PlantModel>
          <InstrumentationLoopFunction>
            <GenericAttributes>
              <GenericAttribute Name="Complete function" Value="P" />
              <GenericAttribute Name="TagName" Value="001" />
            </GenericAttributes>
            <Presentation><Text String="P 001" /></Presentation>
          </InstrumentationLoopFunction>
        </PlantModel>
        """
    )
    visible, _ = visible_structured_tags(root)
    assert visible == {"p001"}
