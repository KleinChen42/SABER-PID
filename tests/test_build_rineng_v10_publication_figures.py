import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.build_rineng_v10_publication_figures import audit_text_layout  # noqa: E402


def test_layout_audit_accepts_separated_text():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_axis_off()
    ax.text(0.1, 0.8, "alpha")
    ax.text(0.8, 0.2, "beta")
    report = audit_text_layout(fig, "separated")
    plt.close(fig)
    assert report["collisions"] == []
    assert report["clipped"] == []


def test_layout_audit_rejects_overlapping_text():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_axis_off()
    ax.text(0.5, 0.5, "same place", ha="center")
    ax.text(0.5, 0.5, "same place", ha="center")
    with pytest.raises(ValueError, match="text layout audit failed"):
        audit_text_layout(fig, "overlap")
    plt.close(fig)
