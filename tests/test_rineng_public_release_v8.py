from build_rineng_public_release_v8 import STEM, VERSION, ZIP_TIME


def test_v8_release_identity_is_frozen() -> None:
    assert VERSION == "saber-pid-rineng-v8"
    assert STEM == "saber_pid_rineng_v8_public_release"
    assert ZIP_TIME == (2026, 8, 12, 0, 0, 0)
