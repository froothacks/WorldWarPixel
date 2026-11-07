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

    other = client.post("/devices/say?society=2", json={"text": "hey"}, headers=hdr).json()
    assert other["society"] == 2
    assert client.get("/devices/me?token=tok-a&society=2").json()["society"] == 2
    assert client.post("/devices/say?society=4", json={"text": "x"}, headers=hdr).status_code == 409

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
        ws.send_json({"type": "society.select", "society": 5})
        while (msg := ws.receive_json())["type"] != "error":
            pass
        ws.send_json({"type": "society.select", "society": 1})
        while (msg := ws.receive_json())["type"] != "society.selected":
            pass
        assert msg["society"] == 1
        state = ws.receive_json()
        assert state["type"] == "state" and state["society"]["id"] == 1


def test_agent_votes_are_throttled() -> None:
    from worldwarpixel.sim.tools import ToolSet

    now = [0.0]
    cfg = RunConfig(width=32, height=32, agents=2, cooldown=1, seed=1)
    run = Run("v", cfg, EventLog(), RandomWalkPolicy(0), clock=lambda: now[0])
    thread = run.board.post_thread(1, "t", "b", now=0.0)

    def tools() -> ToolSet:
        return ToolSet(
            run.canvas,
            run.board,
            0,
            TurnBudget(),
            lambda k, p: None,
            lambda: now[0],
            vote=lambda tid, up: run.agent_vote(0, tid, up),
        )

    turn = tools()
    assert turn.vote_thread(thread.id, up=True)["votes"] == 1
    with pytest.raises(ToolError, match="budget"):
        turn.vote_thread(thread.id)
    with pytest.raises(ToolError, match="recently"):
        tools().vote_thread(thread.id)
    now[0] = 601.0
    assert tools().vote_thread(thread.id, up=False)["votes"] == 0
    with pytest.raises(ToolError):
        tools().vote_thread(999)


def test_invites_contacts_moves_and_memory_reach_the_prompt() -> None:
    from worldwarpixel.sim.run import _SocietyAccess
    from worldwarpixel.sim.tools import ToolSet

    run = _run()
    socs = run.societies
    assert socs is not None
    a, b = run.agents[0], run.agents[1]
    assert socs.of(a.id) != socs.of(b.id)
    assert run.find_agent("@" + b.name.split()[0].upper()) in (b.id, None)
    assert run.find_agent(b.name) == b.id and run.find_agent("nobody at all") is None

    tools = ToolSet(
        run.canvas,
        run.board,
        a.id,
        TurnBudget(),
        lambda k, p: None,
        run._clock,
        _SocietyAccess(run, a.id),
    )
    assert tools.invite(b.name) == {"invited": b.name, "society": socs.of(a.id)}
    with pytest.raises(ToolError, match="budget"):
        tools.invite(b.name)
    facts = run._social_facts(b.id, socs.of(b.id), run._clock())
    assert facts["invites"] == [
        f'{a.name} invited you to {socs.of(a.id)} "{socs.get(socs.of(a.id)).name}"'
    ]
    assert facts["contacts"][0].startswith(a.name)

    _SocietyAccess(run, b.id).join(socs.of(a.id))
    assert socs.pending(b.id, run._clock()) == []
    moved = run._social_facts(a.id, socs.of(a.id), run._clock())["moves"]
    assert moved and moved[0].startswith(f"{b.name} arrived from")

    obs = Observation(
        elapsed=1,
        budget=TurnBudget(),
        canvas_width=32,
        canvas_height=32,
        board=[],
        society={**socs.get(0).to_dict(), "members": 2, "murmur": None, **facts},
        societies=[{**s.to_dict(), "members": 2} for s in socs.societies],
        memory=["5 s ago: I peeked at Blue"],
    )
    text = user_prompt(a, obs)
    assert "Your last turns:\n- 5 s ago: I peeked at Blue" in text
    assert "Invitations: " in text and "People you have talked with recently: " in text


def test_memory_keeps_actions_and_mentions_keep_hyphens() -> None:
    from worldwarpixel.sim.run import MENTION, _recall

    assert _recall("x" * 500 + " [posted thread 3; upvoted thread 2]").endswith(
        "x [posted thread 3; upvoted thread 2]"
    )
    assert MENTION.findall("hi @Min-jun, see @James's thread.") == ["Min-jun", "James"]
    socs = Societies(3, switch_cooldown=0)
    socs.assign([1, 2, 3])
    socs.invite(2, 1, 0.0)
    socs.invite(3, 1, 0.0)
    socs.join(1, 1, 1.0)
    assert [i.society for i in socs.pending(1, 2.0)] == [2]
