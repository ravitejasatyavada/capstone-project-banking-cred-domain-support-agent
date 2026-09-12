Execution Manifest & Environmental Controls
To comply with zero-network isolation requirements, outbound telemetry networks must be disabled before running the orchestration layers.Confirm these variables are declared in your execution environment profile:
export CREWAI_DISABLE_TELEMETRY=true 
export OTEL_SDK_DISABLED=true 

Installation and Workspace Setup
Initialize your local dependencies inside a clean environment:
pip3 install fastapi uvicorn pydantic crewai sentence-transformers chromadb autogen 

"""
langchain-chroma 1.1.0 requires chromadb<2.0.0,>=1.3.5, but you have chromadb 1.1.1 which is incompatible.
pip3 install "chromadb>=1.3.5,<2.0.0"

crewai 1.9.3 requires chromadb~=1.1.0, but you have chromadb 1.5.9 which is incompatible.                                                                                      
pip3 install "chromadb~=1.1.0"

pip3 install "numpy==1.26.4" --force-reinstall

"""

Verify Dataset Balance:
python3 dataset.py
Confirms the fraud flag rate lands within the 10%–30% target window.


Calibrate Fallback Limits:
python3 rag_engine.py
Establishes a solid data-driven similarity threshold.


Execute the Evaluation Matrix:
python3 run_eval.py
Generates document-level precision and recall metrics.

Launch the FastAPI Server:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    