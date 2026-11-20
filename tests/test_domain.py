from diagnotech.domain import classify


def test_in_domain_arabic():
    result = classify("ما هي أعراض الربو وكيف يتم علاجه؟")
    assert result.label == "in_domain"
    assert not result.is_emergency


def test_in_domain_english():
    assert classify("What are the symptoms of pneumonia?").label == "in_domain"


def test_emergency_wins():
    result = classify("أنا مش قادر اتنفس وعندي ألم شديد في الصدر")
    assert result.label == "emergency"
    assert result.is_emergency


def test_out_of_scope_non_medical():
    assert classify("ما هي عاصمة فرنسا؟").label == "out_of_scope"
    assert classify("من فاز بكأس العالم؟").label == "out_of_scope"


def test_out_of_scope_other_body_system():
    # Medical but not respiratory: knee pain has a medical context term
    # (pain/ألم) yet no respiratory anchor.
    assert classify("عندي ألم في ركبتي، ما هو أفضل علاج؟").label == "out_of_scope"


def test_greeting():
    assert classify("السلام عليكم").label == "greeting"
