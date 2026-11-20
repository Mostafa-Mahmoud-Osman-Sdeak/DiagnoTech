from diagnotech.chat_service import ChatService
from diagnotech.config import Settings
from diagnotech.database import ConversationStore
from diagnotech.llm import LLMProvider
from diagnotech.prompts import UNAVAILABLE_MESSAGE
from diagnotech.retrieval import RetrievedEvidence


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply: str = "الربو مرض مزمن يسبب ضيقاً في التنفس."):
        self.reply = reply
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        assert "الأدلة" in user_prompt or "سؤال المستخدم" in user_prompt
        return self.reply


class FakeRetriever:
    def __init__(self, evidence: list[RetrievedEvidence]):
        self._evidence = evidence

    def retrieve(self, question: str, **kwargs):
        return self._evidence


def _service(tmp_path, evidence):
    settings = Settings()
    settings.external_research_enabled = False
    store = ConversationStore(tmp_path / "chat.db")
    llm = FakeLLM()
    service = ChatService(settings, store, FakeRetriever(evidence), llm, None)
    return service, store, llm


def _evidence() -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id="DT-1-0", record_id="DT-1", score=0.81,
        text="Question: What is asthma? Answer: Asthma is a chronic condition.",
        metadata={"source": "Mayo Clinic", "source_url": "https://example.org",
                  "disease_name": "Asthma", "category": "Symptoms"})


def test_known_answer_state_with_sources(tmp_path):
    service, store, llm = _service(tmp_path, [_evidence()])
    result = service.chat(None, "ما هو الربو؟")
    assert result["state"] == "known"
    assert result["sources"][0]["source"] == "Mayo Clinic"
    assert llm.calls == 1
    messages = store.get_messages(result["conversation_id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]


class BailOnceLLM(LLMProvider):
    """Refuses on the first call, answers properly on the retry."""

    name = "bail-once"

    def __init__(self):
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return UNAVAILABLE_MESSAGE
        assert "لا تقل إن المعلومة غير متوفرة" in user_prompt  # retry instruction
        return "الربو مرض مزمن يسبب ضيقاً في التنفس."


class BailAlwaysLLM(LLMProvider):
    name = "bail-always"

    def __init__(self):
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return UNAVAILABLE_MESSAGE


def test_retry_on_bail_recovers_known_answer(tmp_path):
    settings = Settings()
    settings.external_research_enabled = False
    store = ConversationStore(tmp_path / "chat.db")
    llm = BailOnceLLM()
    service = ChatService(settings, store, FakeRetriever([_evidence()]), llm, None)
    result = service.chat(None, "ما هو الربو؟")
    assert result["state"] == "known"
    assert "الربو مرض مزمن" in result["answer"]
    assert result["sources"][0]["source"] == "Mayo Clinic"
    assert llm.calls == 2  # initial attempt + one retry


def test_retry_on_bail_still_unavailable_when_model_insists(tmp_path):
    settings = Settings()
    settings.external_research_enabled = False
    store = ConversationStore(tmp_path / "chat.db")
    llm = BailAlwaysLLM()
    service = ChatService(settings, store, FakeRetriever([_evidence()]), llm, None)
    result = service.chat(None, "ما هو الربو؟")
    assert result["state"] == "unavailable"
    assert result["answer"] == UNAVAILABLE_MESSAGE
    assert llm.calls == 2  # retried exactly once, no loop


def test_unavailable_when_no_evidence(tmp_path):
    service, _, llm = _service(tmp_path, [])
    result = service.chat(None, "ما هو علاج الربو بالأعشاب النادرة؟")
    # in-domain (ربو) but no evidence and research disabled
    assert result["state"] == "unavailable"
    assert result["answer"] == UNAVAILABLE_MESSAGE
    assert llm.calls == 0


def test_out_of_scope_short_circuits(tmp_path):
    service, _, llm = _service(tmp_path, [_evidence()])
    result = service.chat(None, "ما هي عاصمة فرنسا؟")
    assert result["state"] == "out_of_scope"
    assert llm.calls == 0


def test_emergency_short_circuits(tmp_path):
    service, _, llm = _service(tmp_path, [_evidence()])
    result = service.chat(None, "مش قادر اتنفس وعندي ألم شديد في الصدر")
    assert result["state"] == "emergency"
    assert "123" in result["answer"]
    assert llm.calls == 0


def test_greeting(tmp_path):
    service, _, _ = _service(tmp_path, [])
    result = service.chat(None, "السلام عليكم")
    assert result["state"] == "greeting"
