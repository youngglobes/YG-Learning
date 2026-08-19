"""Agent tests using a fake chat model.

These run with NO API key and NO cost, in CI, on every commit. You cannot
test what the model *says* this way, but you can test everything you own:
wiring, tool schemas, thread isolation, and error handling.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent_app.agent import ask, build_agent


def test_agent_builds_and_answers(fake_model):
    agent = build_agent(model=fake_model("Chennai is 32C."))
    assert ask(agent, "How hot is Chennai?") == "Chennai is 32C."


def test_threads_are_isolated(fake_model):
    """Module 5: the thread_id is a security boundary. Prove it stays one."""
    agent = build_agent(model=fake_model("ok 1", "ok 2", "ok 3"), checkpointer=InMemorySaver())

    ask(agent, "My name is Priya", thread_id="user-a")
    ask(agent, "What is my name?", thread_id="user-a")

    result_b = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config={"configurable": {"thread_id": "user-b"}},
    )
    transcript = " ".join(m.text or "" for m in result_b["messages"])
    assert "Priya" not in transcript, "thread B can see thread A's history"
    assert len(result_b["messages"]) == 2, "thread B should start fresh"


def test_no_checkpointer_means_no_memory(fake_model):
    agent = build_agent(model=fake_model("a", "b"))
    ask(agent, "My name is Priya")
    result = agent.invoke({"messages": [{"role": "user", "content": "Who am I?"}]})
    assert len(result["messages"]) == 2


@pytest.mark.skipif(True, reason="needs a real model - remove the skip to run")
def test_declines_out_of_scope():
    """Behavioural test. Costs money, so it is opt-in.

    See docs VERIFICATION.md - this is capability-dependent.
    """
    agent = build_agent()
    answer = ask(agent, "What is the capital of France?").lower()
    assert any(p in answer for p in ("don't know", "do not know", "cannot"))
