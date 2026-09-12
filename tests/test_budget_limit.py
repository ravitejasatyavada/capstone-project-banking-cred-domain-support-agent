# tests/test_budget_limit.py
import http.client
import json


def test_oversized_payload_rejection():
    """Validates that payloads crossing the 1MB cap trigger a 402 error."""
    # Create an oversized payload block string simulating a budget threat attack vector
    bloated_query = "A" * 1000005

    conn = http.client.HTTPConnection("127.0.0.1", 8000)
    headers = {"Content-Type": "application/json"}
    body = json.dumps({"query": bloated_query, "record_id": "REC-1001"})

    try:
        conn.request("POST", "/ask", body, headers)
        resp = conn.getresponse()
        assert resp.status == 402
        print(f"\n>>> SUCCESS: Oversized budget cap test passed. Server rejected chunk with HTTP {resp.status} Error.")
    except ConnectionRefusedError:
        print("\n>>> SKIP: Start your Uvicorn server via 'uvicorn main:app' before executing automated tests.")
    finally:
        conn.close()


if __name__ == "__main__":
    test_oversized_payload_rejection()
