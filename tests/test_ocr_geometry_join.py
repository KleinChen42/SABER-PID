from score_editorial_extension_experiments_v4 import geometry_joined_ocr_prediction


def line(text: str, left: float, top: float, right: float, bottom: float) -> dict:
    return {
        "text": text,
        "box": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def test_geometry_join_stacks_short_class_fragment_and_suffix() -> None:
    row = {
        "candidate_prefix": "RO",
        "candidate_tags": ["ro-10", "ro-99999"],
        "ocr_lines": [
            line("RO-10", 100, 100, 180, 130),
            line("364", 112, 135, 168, 165),
            line("RO-99999", 400, 100, 520, 130),
            line("777", 410, 135, 470, 165),
        ],
    }
    derived = geometry_joined_ocr_prediction(row)
    assert derived["candidate_tags"] == ["ro-10 364", "ro-99999"]
    assert derived["postprocess"] == "reference-free vertical prefix/suffix join v1"


def test_geometry_join_does_not_pair_distant_number() -> None:
    row = {
        "candidate_prefix": "STA",
        "candidate_tags": ["sta"],
        "ocr_lines": [
            line("STA", 100, 100, 180, 130),
            line("786", 600, 135, 660, 165),
        ],
    }
    assert geometry_joined_ocr_prediction(row)["candidate_tags"] == ["sta"]
