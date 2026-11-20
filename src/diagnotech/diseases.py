"""Arabic -> English disease-name anchoring for retrieval.

Cross-lingual embeddings match the *aspect* of an Arabic question
(symptoms, causes, prevention) well, but the Arabic disease name itself
often anchors weakly to the English documents. This lexicon detects the
disease in the Arabic question and supplies its English name so the
retriever can expand the query and boost chunks from that disease.

Keys are written in normalized Arabic (see arabic.normalize_for_match).
Matching prefers longer Arabic phrases, so "نوبات الربو" resolves to
Asthma attack rather than Asthma.
"""
from __future__ import annotations

import re

from .arabic import normalize_for_match

# (arabic terms, canonical English disease names as used in the data)
_LEXICON: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("نوبات الربو", "ازمه الربو", "نوبه الربو", "هجمه الربو"), ("Asthma attack",)),
    (("ربو الاطفال",), ("Childhood asthma",)),
    (("ربو", "الربو", "حساسيه الصدر"), ("Asthma",)),
    (("التهاب الجيوب الانفيه المزمن",), ("Chronic sinusitis",)),
    (("التهاب الجيوب الانفيه", "جيوب انفيه"), ("Acute sinusitis", "Chronic sinusitis")),
    (("انسداد رئوي مزمن", "الانسداد الرئوي المزمن", "مرض الرئه الانسدادي المزمن"), ("COPD",)),
    (("سل", "السل", "الدرن"), ("Tuberculosis",)),
    (("سرطان الرئه",), ("Lung cancer",)),
    (("انقطاع النفس اثناء النوم", "توقف التنفس اثناء النوم", "انقطاع التنفس اثناء النوم"), ("Sleep apnea",)),
    (("انفلونزا الطيور",), ("Bird flu (avian influenza)",)),
    (("انفلونزا", "الانفلونزا", "نزله الانفلونزا"), ("Influenza (flu)",)),
    (("سعال ديكي", "السعال الديكي"), ("Whooping cough",)),
    (("تليف كيسي", "التليف الكيسي"), ("Cystic fibrosis",)),
    (("انتفاخ الرئه", "نفاخ رئوي", "الامفيزيما"), ("Emphysema",)),
    (("التهاب القصيبات", "التهاب القصيبات الهوائيه"), ("Bronchiolitis",)),
    (("خانوق", "الخانوق"), ("Croup",)),
    (("كورونا", "كوفيد", "كوفيد 19"), ("Coronavirus disease 2019 (COVID-19)",)),
    (("التهاب الحنجره",), ("Laryngitis",)),
    (("جلطه رئويه", "جلطه في الرئه", "انصمام رئوي", "انسداد الشريان الرئوي"), ("Pulmonary embolism",)),
    (("ساركويد", "الساركويد"), ("Sarcoidosis",)),
    (("ورم المتوسطه", "ميزوثليوما"), ("Mesothelioma",)),
    (("استرواح الصدر", "هواء في الصدر"), ("Pneumothorax",)),
    (("انخماص الرئه", "انكماش الرئه"), ("Atelectasis",)),
    (("انصباب جنبي", "ميه حول الرئه", "سوائل حول الرئه"), ("Pleural effusion",)),
    (("ميه في الرئه", "وذمه رئويه", "احتقان الرئه"), ("Pulmonary edema",)),
    (("تليف رئوي", "التليف الرئوي"), ("Pulmonary fibrosis",)),
    (("ارتفاع ضغط الشريان الرئوي", "فرط ضغط الدم الرئوي"), ("Pulmonary hypertension",)),
    (("الفيروس المخلوي التنفسي", "فيروس ار اس في"), ("Respiratory syncytial virus (RSV)",)),
    (("التهاب القصبه الهوائيه",), ("Tracheitis",)),
    (("خراج الرئه",), ("Abscess of the lung",)),
    (("الم الصدر", "الم في الصدر", "وجع الصدر"), ("Chest pain",)),
    (("كحه مزمنه", "سعال مزمن"), ("Chronic cough",)),
    (("مرض الرئه الخلالي", "امراض الرئه الخلاليه"), ("Interstitial lung disease",)),
    (("اسبست", "الاسبستوس"), ("Asbestosis",)),
    (("متلازمه الضائقه التنفسيه", "ضائقه تنفسيه حاده"), ("Acute respiratory distress syndrome (ARDS)",)),
    (("متلازمه هانتا", "فيروس هانتا"), ("Hantavirus pulmonary syndrome",)),
]

def _strip_al(text: str) -> str:
    """Remove the definite article prefix (ال) from each word.

    Arabic writers attach ال to every word of a phrase ("الجلطة الرئوية")
    while lexicon entries are naturally written bare ("جلطه رئويه").
    Stripping both sides makes phrase matching robust to that variation.
    """
    return " ".join(
        word[2:] if word.startswith("ال") and len(word) > 3 else word
        for word in text.split())


# Pre-normalized (with ال stripped), longest Arabic phrase first.
_NORMALIZED: list[tuple[str, tuple[str, ...]]] = sorted(
    ((_strip_al(normalize_for_match(term)), names)
     for terms, names in _LEXICON for term in terms),
    key=lambda item: -len(item[0]),
)

_ACUTE_TOKENS = {"حاد", "حاده"}
_CHRONIC_TOKENS = {"مزمن", "مزمنه"}

# Single-word lexicon terms, candidates for the fuzzy fallback below.
_SINGLE_WORD_TERMS: list[tuple[str, tuple[str, ...]]] = [
    (term, names) for term, names in _NORMALIZED if term and " " not in term
]


def _levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                 # deletion
                current[j - 1] + 1,              # insertion
                previous[j - 1] + (ca != cb),    # substitution
            ))
        previous = current
    return previous[-1]


def detect_diseases(text: str) -> list[str]:
    """Return canonical English disease names detected in Arabic text."""
    normalized = _strip_al(normalize_for_match(text or ""))
    found: list[str] = []
    for term, names in _NORMALIZED:
        if term and term in normalized:
            for name in names:
                if name not in found:
                    found.append(name)
    if not found:
        # Conservative fuzzy fallback, only when exact matching found
        # nothing: tolerate a small typo in a single-word disease name
        # (e.g. "انفلونسا" for "انفلونزا"). Short terms are excluded —
        # a 1-edit change to a 2-3 letter word alters its meaning —
        # and the allowed distance stays at 1 (2 for long words).
        # Tokens are cleaned of attached punctuation first: the Arabic
        # question mark ؟ survives normalize_for_match (it sits inside
        # the Arabic Unicode block) and would inflate every distance.
        tokens = [t for t in (re.sub(r"[^\w]", "", tok)
                              for tok in normalized.split()) if t]
        for term, names in _SINGLE_WORD_TERMS:
            if len(term) < 4:
                continue
            max_dist = 2 if len(term) >= 7 else 1
            if any(abs(len(token) - len(term)) <= max_dist
                   and _levenshtein(token, term) <= max_dist
                   for token in tokens):
                for name in names:
                    if name not in found:
                        found.append(name)
    # Acute/chronic disambiguation for sinusitis, which the bare lexicon
    # term maps to both variants.
    if "Acute sinusitis" in found and "Chronic sinusitis" in found:
        tokens = set(normalized.split())
        if tokens & _ACUTE_TOKENS and not tokens & _CHRONIC_TOKENS:
            found.remove("Chronic sinusitis")
        elif tokens & _CHRONIC_TOKENS and not tokens & _ACUTE_TOKENS:
            found.remove("Acute sinusitis")
    return found


# Aspect anchoring: Arabic aspect words -> English aspect terms used in
# the records' questions (symptoms / causes / prevention / ...). Appended
# to the expanded query for the same reason as disease anchoring.
_ASPECT_LEXICON: list[tuple[tuple[str, ...], str]] = [
    (("اعراض", "عرض"), "symptoms signs"),
    (("اسباب", "سبب", "عوامل خطر", "عوامل الخطر"), "causes risk factors"),
    (("وقايه", "امنع", "اتجنب", "تجنب", "يمنع", "منع"), "prevention prevented prevent"),
    (("علاج", "يتعالج", "تتعالج", "معالجه"), "treatment treated"),
    (("تشخيص", "فحوصات", "تحاليل", "اشعه"), "diagnosis tests examination"),
    (("مضاعفات",), "complications"),
    (("علامات تحذير", "تحذير", "انذار"), "warning signs"),
]


def detect_aspects(text: str) -> list[str]:
    """Return English aspect terms detected in an Arabic question."""
    normalized = normalize_for_match(text or "")
    found: list[str] = []
    for terms, english in _ASPECT_LEXICON:
        if any(normalize_for_match(term) in normalized for term in terms):
            if english not in found:
                found.append(english)
    return found


def disease_matches(detected: list[str], disease_name: str) -> bool:
    """True when a chunk's disease matches a detected disease, allowing
    family matches (Asthma matches Childhood asthma) and case variants."""
    if not detected or not disease_name:
        return False
    target = disease_name.lower()
    return any(name.lower() in target or target in name.lower() for name in detected)
