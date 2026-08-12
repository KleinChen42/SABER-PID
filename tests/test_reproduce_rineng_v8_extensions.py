from pathlib import Path

from reproduce_rineng_v8_extensions import reproduction_commands


def test_reproduction_chain_is_inference_free_and_complete() -> None:
    commands = reproduction_commands(Path("/repo"), "python")
    scripts = [command[1] for command in commands]
    assert scripts == [
        "scripts/build_cost_sensitive_operating_modes_v8.py",
        "scripts/score_rineng_v8_extensions.py",
        "scripts/score_dexpi_external_v8.py",
        "scripts/validate_rineng_v8_extensions.py",
        "scripts/build_rineng_v8_extension_figures.py",
        "scripts/build_rineng_v8_tables.py",
    ]
    assert not any("run_qwen" in script or "run_internvl" in script for script in scripts)
