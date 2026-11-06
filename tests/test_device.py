from src.device import available_devices


def test_cpu_is_available() -> None:
    assert "cpu" in available_devices()
