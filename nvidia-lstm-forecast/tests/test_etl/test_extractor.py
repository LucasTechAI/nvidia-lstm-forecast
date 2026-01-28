import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_extractor():
    assert True

[tool.poetry.dependencies]
pytest = "^7.0"