from testing.run_determinism import _error_category, _first_difference


def test_determinism_comparison_ignores_only_object_key_order():
    first = {"entities": [{"text": "cough", "status": "present"}], "symptoms": ["cough"]}
    second = {"symptoms": ["cough"], "entities": [{"status": "present", "text": "cough"}]}

    assert _first_difference(first, second) is None


def test_determinism_comparison_preserves_clinical_list_order():
    difference = _first_difference(
        {"symptoms": ["cough", "fever"]},
        {"symptoms": ["fever", "cough"]},
    )

    assert difference == {
        "path": "$.symptoms[0]",
        "first": "cough",
        "second": "fever",
    }


def test_determinism_comparison_reports_status_changes():
    first = {
        "symptoms": ["cough", "fever"],
        "entities": [{"text": "cough", "status": "present", "evidence_text": "cough"}],
        "removed_by_grounding_check": 0,
    }
    second = {
        "symptoms": ["fever", "cough"],
        "entities": [{"text": "cough", "status": "possible", "evidence_text": "cough"}],
        "removed_by_grounding_check": 0,
    }

    difference = _first_difference(first, second)

    assert difference == {
        "path": "$.entities[0].status",
        "first": "present",
        "second": "possible",
    }


def test_provider_failures_are_not_reported_as_model_nondeterminism():
    assert _error_category("Groq connection error") == (
        "transient_api_or_transport_error"
    )
    assert _error_category("Groq request timed out") == (
        "transient_api_or_transport_error"
    )
    assert _error_category("The model returned invalid JSON") == (
        "model_output_or_validation_error"
    )
