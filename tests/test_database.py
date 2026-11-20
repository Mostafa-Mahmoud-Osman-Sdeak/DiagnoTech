from diagnotech.database import ConversationStore


def test_conversation_roundtrip_and_persistence(tmp_path):
    db = tmp_path / "test.db"
    store = ConversationStore(db)
    conv = store.create_conversation("سؤال عن الربو")
    store.add_message(conv["id"], "user", "ما هي أعراض الربو؟")
    store.add_message(conv["id"], "assistant", "الأعراض هي...", state="known",
                      sources=[{"source": "Mayo Clinic"}])
    store.close()

    reopened = ConversationStore(db)
    conversations = reopened.list_conversations()
    assert len(conversations) == 1
    messages = reopened.get_messages(conv["id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["state"] == "known"
    assert messages[1]["sources"][0]["source"] == "Mayo Clinic"
    assert reopened.count_messages(conv["id"]) == 2


def test_delete_conversation_cascades(tmp_path):
    store = ConversationStore(tmp_path / "test.db")
    conv = store.create_conversation()
    store.add_message(conv["id"], "user", "سؤال")
    assert store.delete_conversation(conv["id"]) is True
    assert store.get_conversation(conv["id"]) is None
    assert store.get_messages(conv["id"]) == []
    assert store.delete_conversation(conv["id"]) is False
