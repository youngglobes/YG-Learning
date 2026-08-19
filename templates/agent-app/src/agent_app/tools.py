"""Tool definitions.

Module 2 rules, applied:
  - parse_docstring=True, so parameter descriptions reach the schema
  - the docstring says WHEN to call, and when not to
  - errors are returned as strings the model can recover from, never raised
"""

from __future__ import annotations

from langchain.tools import tool

# Replace this with something real.
_CITIES: dict[str, int] = {"chennai": 32, "mumbai": 29, "delhi": 38}


@tool(parse_docstring=True)
def get_temperature(city: str) -> str:
    """Get the current temperature in Celsius for an Indian city.

    Call this when the user asks about weather, temperature, or how hot a
    place is. Do not call it for greetings or general conversation.

    Args:
        city: City name, e.g. Chennai. Case-insensitive.
    """
    temp = _CITIES.get(city.lower())
    if temp is None:
        # Written for the model to act on, not for a human log reader.
        return (
            f"Error: no data for '{city}'. "
            f"Available cities: {', '.join(sorted(_CITIES))}."
        )
    return f"{temp}C in {city.title()}"


ALL_TOOLS = [get_temperature]
