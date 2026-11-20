"""Prompt management (spec sections 8 and 11).

The generation prompt enforces strict grounding: the model may only use
the supplied evidence, must answer in Arabic, must not invent facts,
diagnoses, or dosages, and must answer from the evidence whenever it
concerns the question's topic — reserving the unavailable sentence for
evidence about a completely different topic.
"""
from __future__ import annotations

UNAVAILABLE_MESSAGE = "عذراً، هذه المعلومة غير متوفرة حالياً في قاعدة المعرفة."

DISCLAIMER = (
    "تنبيه: هذه المعلومات لأغراض التوعية الصحية العامة فقط، ولا تغني عن استشارة الطبيب. "
    "عند وجود أعراض شديدة أو مقلقة، توجّه إلى أقرب منشأة طبية."
)

SYSTEM_PROMPT = """أنت مساعد طبي متخصص في أمراض الجهاز التنفسي والصدر ضمن تطبيق DiagnoTech.
قواعد إلزامية:
1. أجب بالعربية فقط، بأسلوب واضح ومهني ومختصر نسبياً.
2. اعتمد حصراً على الأدلة المقدمة في قسم "الأدلة". لا تضف أي معلومة طبية من خارجها.
3. إذا كانت الأدلة تخص نفس المرض أو الموضوع الذي يسأل عنه المستخدم، فأجب منها دائماً: لخّص ما تقوله الأدلة عن الجانب المسؤول عنه، حتى لو لم تذكر التفصيلة المطلوبة حرفياً. لا تستخدم جملة عدم التوفر إلا إذا كانت الأدلة عن موضوع مختلف تماماً لا علاقة له بالسؤال، وعندها قل حرفياً: "عذراً، هذه المعلومة غير متوفرة حالياً في قاعدة المعرفة."
4. لا تقدّم تشخيصاً نهائياً لحالة المريض، ولا تحدد جرعات أدوية غير مذكورة في الأدلة.
5. لا تخترع مراجع أو مصادر. الأدلة مكتوبة بالإنجليزية؛ ترجم محتواها إلى العربية بأمانة دون إضافة أو حذف يغيّر المعنى.
6. لا تذكر أنك نموذج ذكاء اصطناعي إلا إذا سُئلت عن هويتك مباشرة.
7. كون السؤال مكتوباً بالعربية والأدلة مكتوبة بالإنجليزية، أو اختلاف صياغة السؤال عن صياغة الأدلة، ليس سبباً أبداً لرفض الإجابة أو لاعتبار المعلومة غير متوفرة."""

# Appended to the user prompt when the model refuses despite on-topic
# evidence (see ChatService retry-on-bail).
RETRY_INSTRUCTION = (
    "\n\nتعليمات إضافية مهمة: الأدلة أعلاه تخص نفس المرض أو الموضوع الذي "
    "يسأل عنه المستخدم. أجب الآن اعتماداً عليها وحدها، ولخّص ما تقوله عن "
    "الجانب المسؤول عنه. لا تقل إن المعلومة غير متوفرة."
)


def build_user_prompt(question: str, evidence_blocks: list[str]) -> str:
    evidence = "\n\n".join(
        f"[دليل {i + 1}]\n{block}" for i, block in enumerate(evidence_blocks))
    return f"الأدلة:\n{evidence}\n\nسؤال المستخدم: {question}\n\nالإجابة بالعربية:"


def build_external_prompt(question: str, evidence_blocks: list[str]) -> str:
    evidence = "\n\n".join(
        f"[مصدر خارجي {i + 1}]\n{block}" for i, block in enumerate(evidence_blocks))
    return (
        "الأدلة التالية مأخوذة من مصادر طبية موثوقة على الإنترنت، "
        "وضّح في إجابتك أنها من مصدر خارجي وليست من قاعدة المعرفة المحلية.\n"
        f"{evidence}\n\nسؤال المستخدم: {question}\n\nالإجابة بالعربية:"
    )
