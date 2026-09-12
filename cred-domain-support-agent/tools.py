# tools.py
from dataset import generate_dataset

dataset_records = generate_dataset()


def check_loan_application_status(record_id: str) -> dict:
    """Retrieves loan records and generates an escalation score from data distributions."""
    record = next((r for r in dataset_records if r["record_id"] == record_id), None)
    if not record:
        return {"error": "Record not found", "escalation_score": 1.0}

    # Formula: normalized recency (days/30) * 0.4 + (0.6 if fraud flagged else 0.0)
    recency_factor = record["days_since_created"] / 30.0
    fraud_factor = 0.6 if record["flagged_for_fraud_review"] else 0.0
    escalation_score = min((recency_factor * 0.4) + fraud_factor, 1.0)

    return {
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "escalation_score": round(escalation_score, 4),
        "flagged_for_fraud": record["flagged_for_fraud_review"]
    }
