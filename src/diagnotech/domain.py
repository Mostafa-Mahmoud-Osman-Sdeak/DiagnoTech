"""Domain classification and emergency detection (spec sections 3 and 8).

A transparent lexicon scorer decides whether a question belongs to the
respiratory / chest domain before any retrieval happens. Emergency
patterns are checked first and always win, so life-threatening symptoms
trigger escalation guidance even when phrased unusually.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .arabic import normalize_for_match

# Respiratory and chest terms (Arabic + English). Matching is done on
# normalized text, so Arabic entries are written in normalized form
# (alef unified, ة as ه, ى as ي).
RESPIRATORY_TERMS: tuple[str, ...] = (
    # Arabic - organs and conditions
    "ربو", "حساسيه الصدر", "التهاب رئوي", "التهاب الرئه", "رئه", "رئوي", "رئتين",
    "قصبات", "قصبه هوائيه", "شعب هوائيه", "التهاب الشعب", "التهاب القصبات",
    "انسداد رئوي مزمن", "انتفاخ الرئه", "سل", "درن", "جيوب انفيه", "التهاب الجيوب",
    "حنجره", "التهاب الحنجره", "بلعوم", "التهاب البلعوم", "لوز", "التهاب اللوزتين",
    "غشاء الجنب", "انصباب جنبي", "استرواح الصدر", "تليف رئوي", "ساركويد",
    "توسع القصبات", "انقطاع النفس اثناء النوم", "توقف التنفس", "صفير", "ازيز",
    "سعال", "كحه", "بلغم", "ضيق نفس", "ضيق تنفس", "نهجان", "الم صدر", "الم في الصدر",
    "نزيف من الانف", "رعاف", "انسداد الانف", "احتقان الانف", "عطس", "شخير",
    "جهاز التنفس", "التنفس", "تنفسي", "الصدر", "صدري", "حجاب حاجز",
    "انفلونزا", "نزله برد", "برد", "كورونا", "كوفيد", "فيروس تنفسي",
    "حساسيه الانف", "حساسيه", "منظار الشعب", "وظائف الرئه", "قياس التنفس",
    "اشعه الصدر", "بخاخ", "استنشاق", "اكسجين", "تبخيره", "جلسه بخار",
    # English
    "asthma", "pneumonia", "bronchitis", "bronchial", "copd", "emphysema",
    "lung", "pulmonary", "respiratory", "sinusitis", "sinus", "tuberculosis",
    "pleural", "pleurisy", "pneumothorax", "trachea", "larynx", "pharynx",
    "tonsillitis", "cough", "wheeze", "wheezing", "dyspnea", "breathlessness",
    "shortness of breath", "chest pain", "chest", "apnea", "influenza", "flu",
    "covid", "sputum", "phlegm", "spirometry", "inhaler", "nebulizer",
    "rhinitis", "laryngitis", "pharyngitis", "atelectasis", "sarcoidosis",
    "cystic fibrosis", "bronchiolitis", "hemoptysis",
)

# Category-supporting medical terms: not respiratory by themselves, but
# typical of the supported categories (tests, medication safety, ...).
MEDICAL_CONTEXT_TERMS: tuple[str, ...] = (
    "دواء", "ادويه", "جرعه", "مضاد حيوي", "كورتيزون", "بخاخه", "تحليل", "فحص",
    "اشعه", "مقطعيه", "رنين", "منظار", "خزعه", "لقاح", "تطعيم", "وقايه",
    "اعراض", "تشخيص", "علاج", "طبيب", "مستشفي", "طوارئ", "حراره", "سخونيه",
    "medication", "medicine", "dose", "dosage", "antibiotic", "vaccine",
    "diagnosis", "treatment", "symptom", "test", "scan", "doctor",
)

EMERGENCY_PATTERNS: tuple[str, ...] = (
    "لا استطيع التنفس", "مش قادر اتنفس", "لا اقدر اتنفس", "بختنق", "اختناق",
    "ضيق نفس شديد", "الم شديد في الصدر", "الم قوي في الصدر", "دم مع الكحه",
    "بكح دم", "خروج دم مع السعال", "شفايفي زرقا", "شفاه زرقاء", "ازرقاق",
    "فقدت الوعي", "اغماء", "دوخه شديده مع ضيق نفس", "نبض سريع مع ضيق نفس",
    "cannot breathe", "can't breathe", "choking", "coughing blood",
    "severe chest pain", "lips are blue", "fainted",
)

GREETING_PATTERNS: tuple[str, ...] = (
    "سلام", "السلام عليكم", "اهلا", "مرحبا", "هاي", "صباح الخير", "مساء الخير",
    "hello", "hi", "hey", "شكرا", "متشكر",
)

OUT_OF_SCOPE_MESSAGE = (
    "عذراً، هذا السؤال خارج نطاق تخصص هذا المساعد. "
    "أستطيع مساعدتك فقط في الموضوعات المتعلقة بالجهاز التنفسي والصدر، "
    "مثل الربو والالتهاب الرئوي والحساسية الصدرية والسعال وضيق التنفس والفحوصات والأدوية المتعلقة بها."
)

GREETING_MESSAGE = (
    "أهلاً بك في DiagnoTech، مساعدك الطبي المتخصص في أمراض الجهاز التنفسي والصدر. "
    "اسألني عن الأعراض أو الأسباب أو الوقاية أو الفحوصات أو سلامة الأدوية المتعلقة بالتنفس والصدر."
)

EMERGENCY_MESSAGE = (
    "تنبيه مهم: الأعراض التي تصفها قد تكون خطيرة وتستدعي تدخلاً عاجلاً. "
    "اتصل بالإسعاف (123 في مصر) أو توجّه فوراً إلى أقرب قسم طوارئ، ولا تعتمد على هذا المساعد في الحالات الطارئة."
)


@dataclass
class DomainResult:
    label: str  # in_domain | out_of_scope | greeting | emergency
    is_emergency: bool = False
    score: int = 0
    matched_terms: list[str] = field(default_factory=list)


def classify(text: str) -> DomainResult:
    """Classify a user question before retrieval."""
    normalized = normalize_for_match(text or "")
    if not normalized:
        return DomainResult(label="out_of_scope")

    for pattern in EMERGENCY_PATTERNS:
        if normalize_for_match(pattern) in normalized:
            return DomainResult(label="emergency", is_emergency=True, score=99,
                                matched_terms=[pattern])

    words = normalized.split()
    if len(words) <= 3 and any(normalize_for_match(g) in normalized for g in GREETING_PATTERNS):
        return DomainResult(label="greeting")

    matched = [t for t in RESPIRATORY_TERMS if normalize_for_match(t) in normalized]
    if matched:
        return DomainResult(label="in_domain", score=len(matched), matched_terms=matched)

    # A clearly medical question with no respiratory anchor is treated as
    # out of scope for this specialized assistant (spec section 3).
    medical = [t for t in MEDICAL_CONTEXT_TERMS if normalize_for_match(t) in normalized]
    if medical:
        return DomainResult(label="out_of_scope", score=0, matched_terms=medical)
    return DomainResult(label="out_of_scope")
