# tests/test_no_lookup_in_other_agents.py
import pytest
from crew.crew_setup import retrieval_agent, composer_agent


def test_verify_principle_of_least_autonomy():
    """Confirms tools are strictly isolated from unauthorized agents, preventing runtime hijacking."""
    # Verify the policy retrieval agent possesses zero database access visibility privileges
    assert len(retrieval_agent.tools) == 0 or retrieval_agent.tools is None

    # Verify the composer agent possesses zero database access visibility privileges
    assert len(composer_agent.tools) == 0 or composer_agent.tools is None
    print("\n>>> SUCCESS: Principle of Least Autonomy confirmed. Retrieval and Composer agents are tool-isolated.")


if __name__ == "__main__":
    test_verify_principle_of_least_autonomy()
