import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.devices import parse_tokens
from worldwarpixel.sim.agents import Observation, RandomWalkPolicy
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.llm_policy import SOCIETY_TOOLS, LLMPolicy, user_prompt
from worldwarpixel.sim.run import Run, RunConfig
from worldwarpixel.sim.societies import MURMUR_WINDOW, Societies, SocietyError
from worldwarpixel.sim.tools import ToolError, TurnBudget


def test_membership_join_cooldown_and_flow() -> None:
    s = Societies(3, switch_cooldown=30)
    s.assign([1, 2, 3, 4, 5, 6])
    assert [s.size(i) for i in range(3)] == [2, 2, 2]
    m = s.join(1, 2, now=100)
    assert (m.src, m.dst, s.of(1)) == (0, 2, 2)
    with pytest.raises(SocietyError, match="already"):
        s.join(1, 2, now=101)
    with pytest.raises(SocietyError, match="recently"):
        s.join(1, 1, now=110)
    s.join(1, 1, now=131)
    assert s.flow(2, now=140) == (1, 1)
    with pytest.raises(SocietyError):
        s.join(2, 7, now=0)


def test_murmur_fades_and_names_are_cleaned() -> None:
    s = Societies(2, switch_cooldown=0)
    s.set_murmur(1, "the north is cold", "it is cold up north", now=10)
    assert s.murmur(1, now=10 + MURMUR_WINDOW - 1) == "the north is cold"
    assert s.murmur(1, now=10 + MURMUR_WINDOW + 1) is None
    assert s.rename(0, "  The   Red  ").name == "The Red"
    with pytest.raises(SocietyError):
        s.rename(0, "   ")


def test_board_digest_filters_by_society() -> None:
    b = Board()
    a = b.post_thread(1, "in a", "x", now=1, society=0)
    c = b.post_thread(2, "in b", "x", now=2, society=1)
    g = b.post_thread(3, "for all", "x", now=3)
    assert [d["thread_id"] for d in b.digest(society=0)] == [g.id, a.id]
    assert [d["thread_id"] for d in b.digest(society=1)] == [g.id, c.id]
    assert len(b.digest()) == 3 and b.society_of(c.id) == 1


def _run() -> Run:
    cfg = RunConfig(width=32, height=32, agents=6, cooldown=1, seed=3, societies=3)
    return Run("s", cfg, EventLog(), RandomWalkPolicy(0))


def test_run_tools_peek_join_rename_and_posts_land_in_own_society() -> None:
    from worldwarpixel.sim.run import _SocietyAccess
    from worldwarpixel.sim.tools import ToolSet

    run = _run()
    assert run.societies is not None
    agent = run.agents[0]
    home = run.societies.of(agent.id)
    other = (home + 1) % 3
    tools = ToolSet(
        run.canvas,
        run.board,
        agent.id,
        TurnBudget(posts=2, reads=1),
        lambda k, p: None,
        run._clock,
        _SocietyAccess(run, agent.id),
    )
    tid = tools.post_thread("hello", "first")["thread_id"]
    assert run.board.society_of(tid) == home
    assert run.society_threads(home)[0]["thread_id"] == tid
    assert run.society_threads(other) == []
    peek = tools.peek_society(other)
    assert peek["id"] == other and peek["threads"] == []
    with pytest.raises(ToolError):
        tools.peek_society(other)  # read budget spent
    assert tools.join_society(other)["society"] == other
    assert run.societies.of(agent.id) == other
    assert tools.name_society("Night Owls")["name"] == "Night Owls"
    assert run.society_summary()[other]["name"] == "Night Owls"
    kinds = [e.kind for e in run.log.read("s", limit=100)]
    assert "society.join" in kinds and "society.renamed" in kinds
    assert next(a for a in run.agent_list() if a["id"] == agent.id)["society"] == other


def test_prompt_states_societies_and_murmur_as_facts() -> None:
    run = _run()
    run.set_murmur(1, "people want more blue", "more blue please")
    obs = Observation(
        elapsed=0,
        budget=TurnBudget(),
        canvas_width=32,
        canvas_height=32,
        board=[],
        society={"id": 1, "name": "Society 2", "members": 2, "murmur": "people want more blue"},
        societies=[{"id": i, "name": f"Society {i + 1}", "members": 2} for i in range(3)],
    )
    text = user_prompt(run.agents[0], obs)
    assert 'You are in society 1 "Society 2"' in text
    assert '0 "Society 1" (2 people)' in text and "have been saying: people want more" in text
    policy = LLMPolicy(None, "m")  # type: ignore[arg-type]
    names = {t["function"]["name"] for t in policy._tools_for(obs)}
    assert {t["function"]["name"] for t in SOCIETY_TOOLS} <= names
    obs.society = None
    assert "join_society" not in {t["function"]["name"] for t in policy._tools_for(obs)}


def test_runs_without_societies_are_unchanged() -> None:
    run = Run(
        "n", RunConfig(width=32, height=32, agents=2, seed=1), EventLog(), RandomWalkPolicy(0)
    )
    assert run.societies is None and run.society_summary() == []
    with pytest.raises(ValueError):
        run.set_murmur(0, "x", "x")


def test_societies_run_end_to_end() -> None:
    async def main() -> None:
        cfg = RunConfig(width=32, height=32, agents=6, cooldown=0.01, jitter=0, seed=1, societies=2)
        run = Run("e", cfg, EventLog(), RandomWalkPolicy(seed=1, post_chance=1.0))
        await run.start()
        await asyncio.sleep(0.1)
        await run.stop()
        socs = {run.board.society_of(d["thread_id"]) for d in run.board.digest(limit=100)}  # type: ignore[arg-type]
        assert socs == {0, 1}

    asyncio.run(main())


def test_parse_tokens() -> None:
    assert parse_tokens("a:0, b:1,bad,c:x:2") == {"a": 0, "b": 1, "c:x": 2}


class FakeTranscriber:
    async def __call__(self, pcm: bytes, rate: int) -> str:
        return f"heard {len(pcm)} bytes at {rate}"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    monkeypatch.setenv("DEVICE_TOKENS", "tok-a:0,tok-b:1")
    get_settings.cache_clear()
    with TestClient(app) as c:
        c.app.state.transcriber = FakeTranscriber()  # type: ignore[attr-defined]

        async def echo(text: str) -> str:
            return f"rumour: {text}"

        c.app.state.murmurer = echo  # type: ignore[attr-defined]
        yield c
    get_settings.cache_clear()


SMALL = {"width": 32, "height": 16, "agents": 4, "cooldown": 5, "seed": 7, "policy": "random"}


def test_device_auth_state_say_and_canvas(client: TestClient) -> None:
    assert client.get("/devices/me").status_code == 401
    assert client.get("/devices/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    me = client.get("/devices/me", headers={"Authorization": "Bearer tok-b"}).json()
    assert me["society"] == 1 and "state" not in me
    hdr = {"Authorization": "Bearer tok-b"}
    assert client.post("/devices/say", json={"text": "hi"}, headers=hdr).status_code == 409

    run_id = client.post("/runs", json={**SMALL, "societies": 3}).json()["id"]
    said = client.post("/devices/say", json={"text": "  more   blue "}, headers=hdr).json()
    assert said == {"society": 1, "murmur": "rumour: more blue", "transcript": "more blue"}
    state = client.get("/devices/me?token=tok-b").json()["state"]
    assert state["run"] == run_id and state["murmur"] == "rumour: more blue"
    assert len(state["societies"]) == 3
    summary = client.get(f"/runs/{run_id}").json()["societies"]
    assert summary[1]["transcript"] == "more blue"

    thumb = client.get("/devices/canvas?w=8&h=4", headers=hdr)
    assert thumb.status_code == 200 and len(thumb.content) == 32


def test_device_voice_over_websocket(client: TestClient) -> None:
    client.post("/runs", json={**SMALL, "societies": 2})
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/devices/ws?token=bad"):
        pass
    with client.websocket_connect("/devices/ws", headers={"Authorization": "Bearer tok-a"}) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["society"] == 0
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"type": "ping"})
        ws.send_json({"type": "voice.start", "rate": 16000})
        ws.send_bytes(b"\0\0" * 1600)
        ws.send_json({"type": "voice.end"})
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "state" and msg["voice"]["status"] == "done":
                break
        assert msg["murmur"] == "rumour: heard 3200 bytes at 16000"
