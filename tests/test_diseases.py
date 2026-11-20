from diagnotech.diseases import detect_aspects, detect_diseases, disease_matches


def test_detects_asthma_and_attack_priority():
    assert detect_diseases("ما هي أعراض الربو؟") == ["Asthma"]
    assert detect_diseases("ازاي أمنع نوبات الربو؟") == ["Asthma attack", "Asthma"]


def test_detects_various_diseases():
    assert "Tuberculosis" in detect_diseases("ما هي أعراض السل؟")
    assert "COPD" in detect_diseases("ما أسباب الانسداد الرئوي المزمن؟")
    assert "Pulmonary embolism" in detect_diseases("ما أعراض الجلطة الرئوية؟")
    assert "Acute sinusitis" in detect_diseases("أسباب التهاب الجيوب الأنفية الحاد؟")


def test_no_detection_for_unrelated():
    assert detect_diseases("ما هي عاصمة فرنسا؟") == []


def test_fuzzy_detects_typo_in_long_disease_name():
    # ز typed as س (1 edit) and a two-edit typo; long words allow 2 edits.
    assert "Influenza (flu)" in detect_diseases("ما هي أعراض الانفلونسا؟")
    assert "Influenza (flu)" in detect_diseases("ما هي أعراض الانفلنسا؟")


def test_fuzzy_detects_single_typo_in_medium_word():
    # خانوق with ق typed as ك (1 edit).
    assert "Croup" in detect_diseases("ما هي أعراض الخانوك؟")


def test_fuzzy_stays_silent_for_unrelated_and_short_words():
    assert detect_diseases("ما هي عاصمة فرنسا؟") == []
    # "حل" is 1 edit from the short lexicon word "سل", but short words are
    # never fuzzy-matched, so this must not detect Tuberculosis.
    assert detect_diseases("ما هو حل المسألة؟") == []


def test_detect_aspects():
    assert "symptoms signs" in detect_aspects("ما هي أعراض الربو؟")
    assert "prevention prevented prevent" in detect_aspects("ازاي أقدر أمنع نوبات الربو؟")
    assert "causes risk factors" in detect_aspects("ما أسباب المرض؟")
    assert detect_aspects("ما هي عاصمة فرنسا؟") == []


def test_disease_matches_family_and_case():
    assert disease_matches(["Asthma"], "Childhood asthma")
    assert disease_matches(["Cystic fibrosis"], "Cystic Fibrosis")
    assert not disease_matches(["Asthma"], "Pneumonia")
    assert not disease_matches([], "Asthma")
