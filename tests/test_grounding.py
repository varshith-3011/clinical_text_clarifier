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


def test_short_evidence_does_not_match_inside_a_word():
    assert not is_grounded("HT", "The patient is overweight.")


def test_short_evidence_matches_as_a_standalone_token():
    assert is_grounded("HT", "History of HT is documented.")
