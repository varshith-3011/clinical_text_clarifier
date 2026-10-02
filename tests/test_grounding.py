from grounding import is_grounded


def test_exact_evidence_passes():
    assert is_grounded("denies chest pain", "Patient denies   chest pain.")


def test_case_differences_pass():
    assert is_grounded("DENIES CHEST PAIN", "Patient denies chest pain.")


def test_extra_whitespace_differences_pass():
    assert is_grounded("denies   chest   pain", "Patient denies   chest pain.")


def test_evidence_not_present_fails():
    assert not is_grounded("pneumonia", "Patient has cough.")


def test_empty_evidence_fails_safely():
    assert not is_grounded("", "Patient has cough.")
