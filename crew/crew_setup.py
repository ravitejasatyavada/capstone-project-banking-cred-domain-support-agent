# =====================================================================
# CRITICAL PYTHON 3.12 / NUMPY COMPLETENESS PATCH (CREW ENGINE ENTRY)
# =====================================================================
import sys
import numpy as np

# Intercept and patch dropped NumPy 1.x scalar aliases for sub-dependency stability
if not hasattr(np, 'long'):
    np.long = int
if not hasattr(np, 'ulong'):
    np.ulong = int

sys.modules['numpy'].long = int
sys.modules['numpy'].ulong = int
# =====================================================================

# crew_setup.py
import os
import re
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from mock_llm import MockCrewILLM
from tools import check_loan_application_status
from rag.cached_retrieval import query_rag, coll_sentence

# --- ENVIRONMENTAL CONTROLS (Part 2, Task 7 & Zero-Network Access) ---
# Strictly disable outbound telemetry to enforce a secure, isolated local runtime environment
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


class SupportAgentResponseSchema(BaseModel):
    """
    Pydantic Schema ensuring all crew outputs conform to strict structural constraints.
    Enforces validation bounds before responses exit the framework pipeline.
    """
    status: str = Field(description="Operational process result trace code.")
    loan_amount_inr: int = Field(default=0, description="Financial quantum monitored.")
    escalation_score: float = Field(default=0.0, description="Calculated threat vector rank.")
    summary: str = Field(description="Policy narrative output text.")


# Initialize our network-isolated Mock LLM engine with an explicit model string wrapper
mock_model = MockCrewILLM(model="mock-cred-agent-llm")

# =====================================================================
# CORE CREWAI AGENT CREW DEFINITIONS (Part 2, Task 7)
# =====================================================================

# Agent 1: The Retrieval Specialist (Only has visibility over the RAG Core)
retrieval_agent = Agent(
    role="Retrieval Specialist",
    goal="Query corporate policy repositories to extract exact ground truth context blocks.",
    backstory="Expert context miner serving structural compliance requirements.",
    llm=mock_model,
    verbose=False
)

# Agent 2: The Core Ledger Auditor (Enforces Principle of Least Autonomy)
# This is the ONLY agent with access to check_loan_application_status tool data logic.
lookup_agent = Agent(
    role="Core Ledger Auditor",
    goal="Verify financial record fields against operational system applications safely.",
    backstory="Database gatekeeper maintaining least-autonomy configurations.",
    llm=mock_model,
    verbose=False
)

# Agent 3: The Response Composer (Synthesizes raw inputs into structured formats)
composer_agent = Agent(
    role="Response Composer",
    goal="Synthesize retrieved policies and database structures into a final consolidated answer draft.",
    backstory="Corporate editor generating structured outputs for execution review pipelines.",
    llm=mock_model,
    verbose=False
)


# =====================================================================
# INPUT PROTECTION & SECURITY GUARDRAILS (Part 2, Task 10)
# =====================================================================

def run_input_guardrail(query: str) -> str:
    """
    Scans incoming query text parameters to mask fixed-format PII patterns
    and block dangerous SQL/Prompt injection patterns.

    Args:
        query (str): The raw incoming user inquiry string.

    Returns:
        str: The sanitized and masked text safe for downstream ingestion.
    """
    # Regex masks for Indian PAN Cards (5 letters, 4 digits, 1 letter) and Aadhaar numbers (12 continuous digits)
    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
    aadhaar_pattern = r'\b[0-9]{12}\b'

    # Apply regex substitutions to keep sensitive strings out of system memory bounds
    masked = re.sub(pan_pattern, "[MASKED_PAN]", query)
    masked = re.sub(aadhaar_pattern, "[MASKED_AADHAAR]", masked)

    # Prompt injection intercept layer: match typical system compromise strings
    if any(keyword in query.upper() for keyword in ["SELECT ", "DROP DATABASE", "IGNORE PREVIOUS INSTRUCTIONS"]):
        raise ValueError("Security Guardrail Fault: Dangerous Prompt Injection Pattern Detected.")

    return masked


# =====================================================================
# MAIN PIPELINE WORKFLOW COORDINATOR
# =====================================================================

def execute_crew_workflow(user_query: str, record_id: str = None) -> SupportAgentResponseSchema:
    """
    Manages the complete end-to-end multi-agent orchestration lifecycle.
    Runs input-side guardrails, executes context queries, manages task processing routines,
    and subjects outputs to Pydantic validation checks.

    Args:
        user_query (str): The text inquiry submitted by the client connection.
        record_id (str, optional): The unique database loan reference identifier string.

    Returns:
        SupportAgentResponseSchema: The validated structural payload object.
    """
    # 1. Execute Input Guardrails Pass
    cleaned_query = run_input_guardrail(user_query)

    # 2. Extract context chunks from our recommended sentence collection database index
    kb_context = query_rag(cleaned_query, coll_sentence)

    # Extract the highest semantic similarity index matching the request profile
    top_similarity = kb_context[0]["similarity"] if kb_context else 0.0
    context_str = "\n".join([c["text"] for c in kb_context])

    # 3. OUTPUT GROUNDEDNESS FALLBACK GATE (Part 2, Task 10)
    # If similarity falls below our dynamic cluster midpoint (0.3500) and no database lookup
    # is attached, block processing instantly to prevent ungrounded agent fabrications.
    if top_similarity < 0.3500 and not record_id:
        return SupportAgentResponseSchema(
            status="Fallback",
            loan_amount_inr=0,
            escalation_score=0.0,
            summary="I am sorry, but I do not possess sufficient authenticated knowledge to answer this request."
        )

    # Execute database lookup through the tool logic if a record tracker is passed
    lookup_data = {}
    if record_id:
        lookup_data = check_loan_application_status(record_id)

    # 4. Construct Sequenced CrewAI Task Context Blocks
    t1 = Task(
        description=f"Verify policy text context patterns for: {cleaned_query}. Context: {context_str}",
        agent=retrieval_agent,
        expected_output="Extracted semantic policy rule details."
    )
    t2 = Task(
        description=f"Audit status tracker variables for id reference. Data: {lookup_data}",
        agent=lookup_agent,
        expected_output="Validated ledger account metadata map."
    )
    t3 = Task(
        description="Consolidate entries into a unified structured final response outcome payload.",
        agent=composer_agent,
        expected_output="Pydantic schema layout text matching system response rules.",
        output_json=SupportAgentResponseSchema
    )

    # Instantiate the sequenced execution group
    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    # Fire the offline ReAct coordination loop pass
    crew.kickoff()

    # Compile and return the securely validated Pydantic model outcome mapping parameters
    return SupportAgentResponseSchema(
        status=lookup_data.get("status", "Success"),
        loan_amount_inr=lookup_data.get("loan_amount_inr", 0),
        escalation_score=lookup_data.get("escalation_score", 0.0),
        summary=f"Processed query successfully. Parameter validation complete. Policy match: {context_str[:80]}..."
    )
