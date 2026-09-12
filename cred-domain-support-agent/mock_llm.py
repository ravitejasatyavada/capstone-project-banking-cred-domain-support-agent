# mock_llm.py
import json
import re
from typing import Any, List, Dict
from crewai.llms.base_llm import BaseLLM
from pydantic import Field


class MockCrewILLM(BaseLLM):
    """ReAct-safe Mock LLM extending CrewAI BaseLLM to eliminate outbound networking."""
    model: str = Field(default="mock-model")

    def call(self, messages: List[Dict[str, Any]], tools: Any = None, **kwargs: Any) -> str:
        prompt_text = " ".join([m.get("content", "") for m in messages])

        # Guard against recursive loops by processing tool queries directly
        if "Action:" in prompt_text:
            # ReAct parser step handling arguments by structural design pattern matches
            if "rag_lookup" in prompt_text or "context" in prompt_text.lower():
                return 'Thought: I need to query the knowledge base.\nAction: rag_lookup\nAction Input: {"query": "eligibility criteria"}\n'
            if "check_loan_application_status" in prompt_text or "REC-" in prompt_text:
                match = re.search(r'REC-\d+', prompt_text)
                rec_id = match.group(0) if match else "REC-1001"
                return f'Thought: I need to query the database.\nAction: check_loan_application_status\nAction Input: {{"record_id": "{rec_id}"}}\n'

        # Parse intent for final evaluations
        if "REC-" in prompt_text:
            return 'Thought: I have the lookup response.\nFinal Answer: {"status": "Under Review", "loan_amount_inr": 2500000, "escalation_score": 0.45, "summary": "The application status is under review with moderate escalation tracking."}'

        return 'Thought: Structuring final response.\nFinal Answer: {"status": "Success", "loan_amount_inr": 0, "escalation_score": 0.1, "summary": "Based on official guidelines, requirements are satisfied."}'


class MockAutoGenClient:
    """Mock client for AutoGen agents supporting schema-driven structural generation."""

    def __init__(self, config):
        self.config = config

    def create(self, params):
        class MockChoice:
            class MockMessage:
                def __init__(self):
                    # Emulate output structured verification models cleanly
                    self.content = json.dumps({
                        "approved": True,
                        "final_answer": "The application status is Verified and Approved based on corporate parameters.",
                        "reason": "Draft perfectly conforms to established compliance frameworks."
                    })
                    self.function_call = None
                    self.tool_calls = None

            def __init__(self):
                self.message = self.MockMessage()
                self.finish_reason = "stop"

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
                self.model = "mock-autogen"

        return MockResponse()
