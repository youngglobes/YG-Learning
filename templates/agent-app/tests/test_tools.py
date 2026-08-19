"""Tool tests. No model, no API key, no cost - these are plain functions."""

from agent_app.tools import get_temperature


def test_known_city():
    out = get_temperature.invoke({"city": "Chennai"})
    assert "32C" in out


def test_case_insensitive():
    assert get_temperature.invoke({"city": "CHENNAI"}) == get_temperature.invoke({"city": "chennai"})


def test_unknown_city_returns_recoverable_error():
    """Module 2: errors are returned, not raised, and list valid options."""
    out = get_temperature.invoke({"city": "Atlantis"})
    assert out.lower().startswith("error")
    assert "chennai" in out.lower()          # tells the model how to recover


def test_parameters_are_documented():
    """Module 2: parse_docstring=True must put descriptions in the schema."""
    assert "description" in get_temperature.args["city"], (
        "Parameter descriptions missing - did you forget @tool(parse_docstring=True)?"
    )
