import app as app_module
from fastapi.testclient import TestClient


PUBLIC_ENTITY_FIELDS = {
    "text",
    "type",
    "status",
    "evidence_text",
    "body_site",
    "duration",
    "severity",
    "value",
}
INTERNAL_FIELDS = {
    "test_classification",
    "kg_finding_text",
    "extraction_metadata",
}


def test_process_text_serializes_compatible_public_response(monkeypatch):
    class StubAgent:
        def __init__(self):
            pass

        def process(self, text):
            return {
                "clinical_text": text,
                "diseases": ["pneumonia"],
                "symptoms": ["pulse", "elevated esr"],
                "medications": ["metformin"],
                "tests": ["elevated esr"],
                "entities": [
                    {
                        "text": "pulse",
                        "type": "symptom",
                        "status": "present",
                        "evidence_text": "pulse was 112/min",
                        "body_site": None,
                        "duration": "today",
                        "severity": "mild",
                        "value": "112/min",
                    },
                    {
                        "text": "fever",
                        "type": "symptom",
                        "status": "absent",
                        "evidence_text": "denies fever",
                        "body_site": None,
                        "duration": None,
                        "severity": None,
                        "value": None,
                    },
                    {
                        "text": "cough",
                        "type": "symptom",
                        "status": "possible",
                        "evidence_text": "possible cough",
                        "body_site": None,
                        "duration": None,
                        "severity": None,
                        "value": None,
                    },
                    {
                        "text": "elevated ESR",
                        "type": "test",
                        "status": "present",
                        "evidence_text": "ESR is elevated",
                        "body_site": None,
                        "duration": None,
                        "severity": "elevated",
                        "value": "elevated",
                        "test_classification": "finding",
                        "kg_finding_text": "elevated ESR",
                        "extraction_metadata": {"source": "model"},
                    },
                ],
                "clinical_events": [],
                "demographics": {
                    "age": 56,
                    "age_unit": "years",
                    "sex": "male",
                    "pregnancy_status": None,
                    "gestational_age": None,
                },
                "removed_by_grounding_check": 0,
                "model": "openai/gpt-oss-120b",
                "status": "success",
                "error": None,
                "test_classification": "finding",
                "kg_finding_text": "must not be serialized",
                "extraction_metadata": {"internal": True},
            }

    monkeypatch.setattr(app_module, "ClinicalTextProcessingAgent", StubAgent)
    response = TestClient(app_module.app).post(
        "/process-text",
        json={"text": "Pulse was 112/min; denies fever; possible cough; ESR is elevated."},
    )

    assert response.status_code == 200
    body = response.json()
    required_keys = {
        "clinical_text",
        "diseases",
        "symptoms",
        "medications",
        "tests",
        "entities",
    }
    assert required_keys <= body.keys()
    assert body["clinical_text"] == (
        "Pulse was 112/min; denies fever; possible cough; ESR is elevated."
    )
    for key in ("diseases", "symptoms", "medications", "tests"):
        assert isinstance(body[key], list)
        assert all(isinstance(value, str) for value in body[key])

    assert isinstance(body["entities"], list)
    for entity in body["entities"]:
        assert PUBLIC_ENTITY_FIELDS <= entity.keys()
        assert isinstance(entity["text"], str)
        assert entity["type"] in {"disease", "symptom", "medication", "test"}
        assert entity["status"] in {"present", "absent", "possible"}
        assert isinstance(entity["evidence_text"], str)
        for key in ("body_site", "duration", "severity", "value"):
            assert entity[key] is None or isinstance(entity[key], str)
    assert body["entities"][0]["value"] == "112/min"
    assert body["entities"][1]["status"] == "absent"
    assert body["entities"][2]["status"] == "possible"
    assert not INTERNAL_FIELDS.intersection(body)
    assert all(not INTERNAL_FIELDS.intersection(entity) for entity in body["entities"])

    assert isinstance(body["clinical_events"], list)
    assert isinstance(body["demographics"], dict)
    assert isinstance(body["demographics"]["age"], int)
    assert isinstance(body["demographics"]["age_unit"], str)
    assert isinstance(body["demographics"]["sex"], str)
    assert body["demographics"]["pregnancy_status"] is None
    assert body["demographics"]["gestational_age"] is None
    assert isinstance(body["removed_by_grounding_check"], int)
    assert isinstance(body["model"], str)
    assert body["status"] == "success"
    assert body["error"] is None


def test_process_text_rejects_invalid_request_shape():
    response = TestClient(app_module.app).post("/process-text", json={})

    assert response.status_code == 422
