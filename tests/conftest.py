import pytest

from paper_radar.s2 import reset_client


@pytest.fixture(autouse=True)
def _reset_s2_client():
    reset_client()
    yield
    reset_client()
