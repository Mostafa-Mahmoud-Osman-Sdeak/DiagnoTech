from diagnotech.arabic import arabic_ratio, contains_arabic, normalize_arabic, normalize_for_match


def test_alef_unification():
    assert normalize_arabic("أإآ") == "ااا"


def test_tatweel_and_diacritics_removed():
    assert normalize_arabic("الربـــو") == "الربو"
    assert normalize_arabic("الرَّبْو") == "الربو"


def test_match_normalization_unifies_letters_and_case():
    assert normalize_for_match("الأَسْئِلَة!") == normalize_for_match("الاسئله")
    assert normalize_for_match("Asthma") == "asthma"


def test_contains_arabic_and_ratio():
    assert contains_arabic("ما هي أعراض الربو؟")
    assert not contains_arabic("What is asthma?")
    assert arabic_ratio("ربو asthma") > 0.3
    assert arabic_ratio("") == 0.0
