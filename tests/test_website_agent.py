"""
Tests for the LLM agent (website/agent.py) and the /api/ask endpoint,
using a fake chat client — no network involved.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tomli_w

from nightskycam_images.constants import (
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from nightskycam_images.db import populate
from nightskycam_images.website import create_app
from nightskycam_images.website.agent import (
    AgentUpstreamError,
    run_agent,
)

SYSTEM = "cam1"
DATE = "2025_06_01"
STEM_A = f"{SYSTEM}_{DATE}_20_00_00"
STEM_B = f"{SYSTEM}_{DATE}_21_00_00"


# ============================================================================
# Fake OpenAI-shaped client
# ============================================================================


def tool_call(name, arguments, call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=(
                arguments if isinstance(arguments, str) else json.dumps(arguments)
            ),
        ),
    )


def assistant(content=None, tool_calls=None):
    message = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls or None
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChatClient:
    """Mimics openai.OpenAI: .chat.completions.create(**kwargs)."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []  # kwargs of every create() call

        outer = self

        class _Completions:
            def create(self, **kwargs):
                # Snapshot: run_agent mutates its messages list afterwards.
                kwargs["messages"] = list(kwargs["messages"])
                outer.calls.append(kwargs)
                if not outer._script:
                    raise AssertionError("fake client script exhausted")
                item = outer._script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        self.chat = SimpleNamespace(completions=_Completions())


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def agent_db():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        date_dir = root / SYSTEM / DATE
        date_dir.mkdir(parents=True)
        thumb_dir = date_dir / THUMBNAIL_DIR_NAME
        thumb_dir.mkdir()
        for stem, weather in ((STEM_A, "clear"), (STEM_B, "rain")):
            img = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)
            cv2.imwrite(
                str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"),
                img[:20, :20],
            )
            with open(date_dir / f"{stem}.toml", "wb") as f:
                tomli_w.dump(
                    {
                        "process": "raw",
                        "weather": weather,
                        "cloud_cover": 10,
                        "classifiers": {"cloudy": 0.2},
                    },
                    f,
                )
        db_path = Path(tmp) / "test.db"
        populate(root, db_path, full=True)
        yield db_path


@pytest.fixture
def app(agent_db):
    application = create_app(agent_db, cache_dir=agent_db.parent / "cache")
    application.config["TESTING"] = True
    application.config["SAIA_MODEL"] = "fake-model"
    return application


USER = [{"role": "user", "content": "show me an image"}]


# ============================================================================
# Loop unit tests
# ============================================================================


def test_happy_path(agent_db):
    fake = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("count_images", {"filter": {}})]),
            assistant(
                tool_calls=[
                    tool_call(
                        "query_images",
                        {"filter": {"weather": ["clear"]}},
                    )
                ]
            ),
            assistant(
                tool_calls=[
                    tool_call(
                        "final_answer",
                        {"text": "Here you go.", "image_stems": [STEM_A]},
                    )
                ]
            ),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)

    assert answer.text == "Here you go."
    assert answer.image_stems == [STEM_A]
    assert answer.rounds == 3
    assert answer.truncated is False
    assert [s.tool for s in answer.steps] == ["count_images", "query_images"]

    # System prompt carries live vocabulary and the tool schemas are sent.
    first = fake.calls[0]
    system_prompt = first["messages"][0]["content"]
    assert SYSTEM in system_prompt
    assert "rain" in system_prompt
    assert "YYYY_MM_DD" in system_prompt
    assert "locations" in system_prompt  # deployment-site vocabulary line
    tool_names = {t["function"]["name"] for t in first["tools"]}
    assert "final_answer" in tool_names and "query_images" in tool_names
    query_schema = next(
        t for t in first["tools"] if t["function"]["name"] == "query_images"
    )
    assert (
        "locations"
        in query_schema["function"]["parameters"]["properties"]["filter"]["properties"]
    )


def test_filterspec_error_fed_back(agent_db):
    fake = FakeChatClient(
        [
            assistant(
                tool_calls=[
                    tool_call("count_images", {"filter": {"start_date": "junk"}})
                ]
            ),
            assistant(
                tool_calls=[
                    tool_call("final_answer", {"text": "ok", "image_stems": []})
                ]
            ),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.text == "ok"

    # The second call's messages must contain the validation error.
    tool_messages = [m for m in fake.calls[1]["messages"] if m.get("role") == "tool"]
    assert any("start_date" in m["content"] for m in tool_messages)
    assert "error" in answer.steps[0].summary


def test_unknown_tool_and_bad_arguments(agent_db):
    fake = FakeChatClient(
        [
            assistant(
                tool_calls=[
                    tool_call("delete_images", {}),
                    tool_call("count_images", "not json{", call_id="c2"),
                ]
            ),
            assistant(tool_calls=[tool_call("final_answer", {"text": "done"})]),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.text == "done"
    tool_messages = [m for m in fake.calls[1]["messages"] if m.get("role") == "tool"]
    assert any("unknown tool" in m["content"] for m in tool_messages)
    assert any("not valid JSON" in m["content"] for m in tool_messages)


def test_iteration_cap_forces_final_answer(agent_db):
    script = [
        assistant(tool_calls=[tool_call("count_images", {"filter": {}})])
        for _ in range(3)
    ]
    script.append(
        assistant(tool_calls=[tool_call("final_answer", {"text": "best effort"})])
    )
    fake = FakeChatClient(script)
    answer = run_agent(agent_db, fake, "fake-model", USER, max_rounds=3)

    assert answer.truncated is True
    assert answer.text == "best effort"
    assert answer.rounds == 3
    # The forced call pins tool_choice to final_answer.
    forced = fake.calls[-1]
    assert forced["tool_choice"]["function"]["name"] == "final_answer"
    assert [t["function"]["name"] for t in forced["tools"]] == ["final_answer"]


def test_hallucinated_stems_dropped(agent_db):
    fake = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("query_images", {"filter": {}})]),
            assistant(
                tool_calls=[
                    tool_call(
                        "final_answer",
                        {
                            "text": "images",
                            "image_stems": [
                                STEM_A,
                                "cam9_2099_01_01_00_00_00",
                                STEM_A,  # duplicate
                            ],
                        },
                    )
                ]
            ),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.image_stems == [STEM_A]


def test_plain_text_fallback(agent_db):
    fake = FakeChatClient([assistant(content="Just words.")])
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.text == "Just words."
    assert answer.image_stems == []
    assert answer.truncated is False


def test_stems_recovered_from_answer_text(agent_db):
    """Some models write stems into the text instead of the argument."""
    fake = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("query_images", {"filter": {}})]),
            assistant(
                tool_calls=[
                    tool_call(
                        "final_answer",
                        {
                            "text": (
                                "The most recent clear image is from "
                                "2025-06-01.\n\n"
                                f'image_stems: ["{STEM_A}"]'
                            ),
                            "image_stems": [],
                        },
                    )
                ]
            ),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.image_stems == [STEM_A]
    assert "image_stems" not in answer.text
    assert "most recent clear image" in answer.text


def test_stems_recovered_from_plain_text_fallback(agent_db):
    fake = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("query_images", {"filter": {}})]),
            assistant(content=f"Here it is: {STEM_B}"),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.image_stems == [STEM_B]


def test_explicit_stems_not_overridden_by_text(agent_db):
    """When the argument is used, text mentions must not add images."""
    fake = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("query_images", {"filter": {}})]),
            assistant(
                tool_calls=[
                    tool_call(
                        "final_answer",
                        {
                            "text": f"Compare with {STEM_B}.",
                            "image_stems": [STEM_A],
                        },
                    )
                ]
            ),
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", USER)
    assert answer.image_stems == [STEM_A]


def test_history_stems_accepted_and_truncated(agent_db):
    history = [{"role": "user", "content": f"question {i}"} for i in range(20)]
    history.append(
        {
            "role": "assistant",
            "content": "shown earlier",
            "image_stems": [STEM_B],
        }
    )
    history.append({"role": "user", "content": "show it again"})
    fake = FakeChatClient(
        [
            assistant(
                tool_calls=[
                    tool_call(
                        "final_answer",
                        {"text": "again", "image_stems": [STEM_B]},
                    )
                ]
            )
        ]
    )
    answer = run_agent(agent_db, fake, "fake-model", history)
    assert answer.image_stems == [STEM_B]
    # system + at most MAX_HISTORY_MESSAGES history entries
    assert len(fake.calls[0]["messages"]) <= 13


def test_upstream_error_mapped(agent_db):
    fake = FakeChatClient([ConnectionError("boom")])
    with pytest.raises(AgentUpstreamError, match="boom"):
        run_agent(agent_db, fake, "fake-model", USER)


def test_forced_finish_survives_upstream_error(agent_db):
    script = [
        assistant(tool_calls=[tool_call("count_images", {"filter": {}})]),
        ConnectionError("down"),
    ]
    fake = FakeChatClient(script)
    answer = run_agent(agent_db, fake, "fake-model", USER, max_rounds=1)
    assert answer.truncated is True
    assert "could not finish" in answer.text


# ============================================================================
# Endpoint tests
# ============================================================================


def _client_with(app, fake):
    app.config["SAIA_CLIENT"] = fake
    return app.test_client()


def test_api_ask_happy_path(app):
    fake = FakeChatClient(
        [
            assistant(tool_calls=[tool_call("query_images", {"filter": {}})]),
            assistant(
                tool_calls=[
                    tool_call(
                        "final_answer",
                        {"text": "One image.", "image_stems": [STEM_A]},
                    )
                ]
            ),
        ]
    )
    client = _client_with(app, fake)
    response = client.post("/api/ask", json={"messages": USER})
    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"]["text"] == "One image."
    (image,) = data["answer"]["images"]
    assert image["filename_stem"] == STEM_A
    assert image["urls"]["thumb"] == f"/thumb/{STEM_A}.jpeg"
    assert image["urls"]["page"] == f"/image/{STEM_A}"
    assert data["rounds"] == 2
    assert data["model"] == "fake-model"
    assert data["steps"][0]["tool"] == "query_images"


def test_api_ask_unconfigured_503(app):
    # No SAIA_CLIENT injected and no API key.
    app.config["SAIA_API_KEY"] = None
    client = app.test_client()
    response = client.post("/api/ask", json={"messages": USER})
    assert response.status_code == 503

    assert client.get("/ask").status_code == 404
    assert "Ask" not in client.get("/").get_data(as_text=True)


def test_ask_page_and_navbar_when_configured(app):
    client = _client_with(app, FakeChatClient([]))
    page = client.get("/ask")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "ask-form" in html and "ask.js" in html
    assert ">Ask<" in client.get("/").get_data(as_text=True)


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"messages": []},
        {"messages": "hello"},
        {"messages": [{"role": "system", "content": "x"}]},
        {"messages": [{"role": "user", "content": 5}]},
        {"messages": [{"role": "assistant", "content": "x"}]},  # last not user
        {"messages": [{"role": "user", "content": "x", "image_stems": [1]}]},
    ],
)
def test_api_ask_invalid_body_400(app, body):
    client = _client_with(app, FakeChatClient([]))
    response = client.post("/api/ask", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_api_ask_upstream_error_502(app):
    client = _client_with(app, FakeChatClient([ConnectionError("saia down")]))
    response = client.post("/api/ask", json={"messages": USER})
    assert response.status_code == 502
    assert "saia down" in response.get_json()["error"]
