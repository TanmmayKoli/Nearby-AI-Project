"""Offline tests for the "Past conversations" transcript listing."""

import json
from datetime import datetime, timedelta, timezone

from agent.graph import save_transcript
from agent.history import list_conversations, load_conversation
from agent.state import LeadState

T0 = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)


def write(dir_, thread_id, *, source="app", minutes=0, category="plumbing", status="converted",
          opener="My water heater has been leaking all over the garage floor since yesterday"):
    dir_.mkdir(exist_ok=True)
    record = {
        "thread_id": thread_id, "source": source, "saved_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "messages": [{"role": "user", "content": opener}, {"role": "assistant", "content": "How soon?"}],
        "state": {"category": category, "status": status},
    }
    (dir_ / f"{thread_id}.json").write_text(json.dumps(record))


def test_lists_only_app_conversations_newest_first(tmp_path):
    write(tmp_path, "old", minutes=0)
    write(tmp_path, "new", minutes=30, category=None, status="in_progress", opener="Short one")
    write(tmp_path, "from-test", source="test", minutes=60)
    write(tmp_path, "untagged", source=None, minutes=90)  # transcripts saved before tagging
    (tmp_path / "broken.json").write_text("{not json")
    convos = list_conversations(tmp_path)
    assert [c["thread_id"] for c in convos] == ["new", "old"]
    newest, older = convos
    assert newest["category"] == "Unknown" and newest["status"] == "in_progress" and newest["opener"] == "Short one"
    assert older["category"] == "Plumbing" and older["saved_at"] == T0
    assert older["opener"].endswith("…") and len(older["opener"]) == 40


def test_limit_and_missing_dir(tmp_path):
    for i in range(5):
        write(tmp_path, f"t{i}", minutes=i)
    assert [c["thread_id"] for c in list_conversations(tmp_path, limit=2)] == ["t4", "t3"]
    assert list_conversations(tmp_path / "nope") == []


def test_save_transcript_records_source_and_load_roundtrip(tmp_path):
    save_transcript(LeadState(category="roofing"), "abc", tmp_path, source="app")
    record = load_conversation(tmp_path, "abc")
    assert record["source"] == "app" and record["state"]["category"] == "roofing"
    assert [c["thread_id"] for c in list_conversations(tmp_path)] == ["abc"]
    assert load_conversation(tmp_path, "missing") is None
