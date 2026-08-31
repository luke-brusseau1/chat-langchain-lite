"""Consistency checks for the canned docs data in agent/tools.py."""

import re

from agent.tools import CONCEPTS_DB, SETUP_GUIDES_DB

# The installation guide is the human-readable copy of the same fact the
# per-concept "min_python" fields carry; both are quoted to users verbatim.
_GUIDE_MIN_PYTHON = re.compile(
    r"Minimum supported Python is (?P<core>\d+\.\d+) for langchain/langgraph"
    r" and (?P<langsmith>\d+\.\d+) for langsmith"
)

_CORE_CONCEPTS = ("langchain", "langgraph")


def _guide_versions() -> dict[str, str]:
    match = _GUIDE_MIN_PYTHON.search(SETUP_GUIDES_DB["installation"])
    assert match is not None, "installation guide no longer states a minimum Python"
    return match.groupdict()


def test_core_concepts_min_python_matches_installation_guide():
    expected = f"{_guide_versions()['core']}+"
    for name in _CORE_CONCEPTS:
        assert CONCEPTS_DB[name]["min_python"] == expected, (
            f"CONCEPTS_DB['{name}']['min_python'] is "
            f"{CONCEPTS_DB[name]['min_python']!r}, but the installation guide "
            f"says {expected!r}"
        )


def test_langsmith_min_python_matches_installation_guide():
    expected = f"{_guide_versions()['langsmith']}+"
    assert CONCEPTS_DB["langsmith"]["min_python"] == expected


if __name__ == "__main__":
    test_core_concepts_min_python_matches_installation_guide()
    test_langsmith_min_python_matches_installation_guide()
    print("ok")
