# tests/test_rate_limiter.py
import http.client
import json


def run_rate_limiter_load_test() -> None:
    """
    Fires 35 continuous automated requests in a tight loop to trigger
    the FastAPI rate-limiting middleware guardrail.
    """
    print("====================================================================")
    print("LAUNCHING RATE-LIMITER MIDDLEWARE STRESS SUITE")
    print("====================================================================")
    print("Target Destination: POST http://127.0.0")
    print("Configured Boundary Limits: 30 calls max / 60-second window\n")

    # Connect over local host parameters (Adjust port to 8080 if 8000 is still locked!)
    host = "127.0.0.1"
    port = 8000
    endpoint = "/ask"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "query": "Verify compliance parameters for account auditing tracks.",
        "record_id": "REC-1001"
    }

    body = json.dumps(payload)
    success_count = 0
    blocked_count = 0

    conn = http.client.HTTPConnection(host, port)

    print("Firing 35 sequential requests to breach the 30-call ceiling...")
    for idx in range(1, 36):
        try:
            conn.request("POST", endpoint, body, headers)
            response = conn.getresponse()
            status = response.status
            response.read()  # Flush stream cleanly to release socket track

            if status == 200:
                success_count += 1
                if idx in (1, 2, 3, 15, 30):
                    print(f"  [Call {idx:02d}] HTTP {status} -> Allowed (Pipeline/Cache Entry)")
            elif status == 429:
                blocked_count += 1
                print(f"  [Call {idx:02d}] HTTP {status} -> BLOCKED BY SECURITY SHIELD 🛡️")
            else:
                print(f"  [Call {idx:02d}] Unexpected HTTP Status Code: {status}")

        except Exception as exc:
            print(f"  [Call {idx:02d}] Connection Break: {str(exc)}")
            print("Ensure your Uvicorn server is running in the other tab before firing!")
            return

    conn.close()

    print("\n====================================================================")
    print("STRESS SUITE BENCHMARK MATRIX SUMMARY")
    print("====================================================================")
    print(f"Total Transactions Transmitted : 35")
    print(f"Total Transactions Allowed (200): {success_count} (Under quota limit)")
    print(f"Total Transactions Blocked (429): {blocked_count} (Security verified)")
    print("====================================================================")

    if blocked_count > 0:
        print(">>> SUCCESS: AI Middleware Governance Layer passed security review criteria.")
    else:
        print(">>> CRITICAL: Security gate didn't trip. Re-verify runtime connection rules.")


if __name__ == "__main__":
    run_rate_limiter_load_test()
