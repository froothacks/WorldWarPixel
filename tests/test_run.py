import asyncio

import pytest
from pydantic import ValidationError

from worldwarpixel.sim.agents import AgentSpec, RandomWalkPolicy
from worldwarpixel.sim.archive import restore_run
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import FRAME_PIXELS, PIXEL_STRUCT, RESYNC, Hub
from worldwarpixel.sim.run import Run, RunConfig

CONFIG = RunConfig(
    width=64,
    height=64,
    agents=5,
    cooldown=0.01,
    jitter=0,
    pixels_per_turn=4,
    seed=1,
).model_copy(update={"pixel_interval": 0})


def test_pixel_interval_has_a_50_ms_minimum_and_default() -> None:
    assert RunConfig().pixel_interval == 0.01
    with pytest.raises(ValidationError):
        RunConfig(pixel_interval=0.005)


async def _settle() -> None:
    await asyncio.sleep(0.1)


def test_agents_act_on_their_own_cooldowns() -> None:
    async def main() -> None:
        log = EventLog()
        run = Run("t", CONFIG, log, RandomWalkPolicy(seed=1, post_chance=1.0))
        run.hub.flush_interval = 0.02
        queue = run.hub.subscribe()
        await run.start()
        await _settle()
        assert run.state == "running"
        assert run.stats["turns"] >= 5
        assert run.stats["pixels"] == run.stats["turns"] * 4
        assert run.stats["posts"] >= 1
        kinds = {e.kind for e in log.read("t", limit=10_000)}
        assert {"run.created", "run.state", "pixel", "post", "agent.thought"} <= kinds

        run.pause()
        await _settle()
        turns = run.stats["turns"]
        await _settle()
        assert run.stats["turns"] == turns
        run.resume()
        await _settle()
        assert run.stats["turns"] > turns

        binary = [m for m in _drain(queue) if isinstance(m, bytes)]
        assert binary and all(b[0] == FRAME_PIXELS for b in binary)
        assert all((len(b) - 1) % PIXEL_STRUCT.size == 0 for b in binary)
        x, y, color, agent_id = PIXEL_STRUCT.unpack_from(binary[0], 1)
        assert 0 <= x < 64 and 0 <= y < 64 and 0 <= color < 16 and 1 <= agent_id <= 5

        await run.stop()
        assert run.summary()["state"] == "stopped"
        detail = run.agent_detail(1)
        assert detail is not None and detail["thoughts"]
        listed = run.agent_list()[0]
        assert listed["last_thought"] == detail["thoughts"][-1]
        assert listed["avatar"] is None
        assert all(a.color >= 2 for a in run.agents)
        assert run.agent_detail(99) is None

    asyncio.run(main())


def _drain(queue: asyncio.Queue[bytes | str]) -> list[bytes | str]:
    out: list[bytes | str] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def test_seeding_is_deterministic() -> None:
    a = Run("a", CONFIG, EventLog(), RandomWalkPolicy(0)).agents
    b = Run("b", CONFIG, EventLog(), RandomWalkPolicy(0)).agents
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert {x.id for x in a} == {1, 2, 3, 4, 5}


def test_slow_subscriber_gets_a_resync_marker_instead_of_a_hole() -> None:
    async def main() -> None:
        hub = Hub(queue_size=2)
        q = hub.subscribe()
        for _ in range(3):
            hub.push_pixel(1, 1, 1, 1)
            hub.flush()
        hub.publish("agent.status", agent_id=1, status="idle")
        assert q.get_nowait() == RESYNC
        assert isinstance(q.get_nowait(), str)
        assert q.empty()

    asyncio.run(main())


def test_pixels_drip_out_at_the_pixel_interval() -> None:
    async def main() -> None:
        cfg = CONFIG.model_copy(update={"agents": 1, "cooldown": 10.0, "pixel_interval": 0.05})
        run = Run("d", cfg, EventLog(), RandomWalkPolicy(seed=1, post_chance=0.0))
        run.rng.random = lambda: 0.0  # type: ignore[method-assign]  # no start-up delay
        await run.start()
        await asyncio.sleep(0.08)
        assert run.stats["turns"] == 1
        assert 0 < run.stats["pixels"] < 4, "a turn's pixels land one at a time"
        assert run.status[run.agents[0].id] == "painting"
        await asyncio.sleep(0.25)
        assert run.stats["pixels"] == 4
        assert run.status[run.agents[0].id] == "idle"
        run.god_set_pace(None, None, pixel_interval=0.0)
        await run.stop()

    asyncio.run(main())


def test_post_cooldown_allows_posts_on_turns_one_five_and_nine() -> None:
    class AlwaysReply:
        def __init__(self) -> None:
            self.turn = 0
            self.budgets: list[int] = []

        async def act(self, _agent: AgentSpec, tools: object, obs: object) -> str:
            self.turn += 1
            budget = obs.budget  # type: ignore[attr-defined]
            self.budgets.append(budget.posts)
            tools.reply(1, f"reply on turn {self.turn}")  # type: ignore[attr-defined]
            return "replied"

    async def main() -> None:
        agent = AgentSpec(1, "Priya Raman", "India", 2, 0, 0, 1.0)
        policy = AlwaysReply()
        config = CONFIG.model_copy(update={"agents": 1, "dispatch_interval": 0})
        run = Run("cooldown", config, EventLog(), policy, agents=[agent])
        run.board.post_thread(agent.id, "First thread", "Start here.", 0)
        for _ in range(9):
            await run._turn(agent, 1.0)
        replies = run.board.thread(1)[1:]
        assert [post.body for post in replies] == [
            "reply on turn 1",
            "reply on turn 5",
            "reply on turn 9",
        ]
        assert policy.budgets == [1, 0, 0, 0, 1, 0, 0, 0, 1]

    asyncio.run(main())


def test_zero_post_cooldown_keeps_posting_each_turn() -> None:
    class AlwaysReply:
        def __init__(self) -> None:
            self.turn = 0

        async def act(self, _agent: AgentSpec, tools: object, _obs: object) -> str:
            self.turn += 1
            tools.reply(1, f"reply on turn {self.turn}")  # type: ignore[attr-defined]
            return "replied"

    async def main() -> None:
        agent = AgentSpec(1, "Priya Raman", "India", 2, 0, 0, 1.0)
        config = CONFIG.model_copy(
            update={"agents": 1, "dispatch_interval": 0, "post_cooldown_turns": 0}
        )
        run = Run("no-cooldown", config, EventLog(), AlwaysReply(), agents=[agent])
        run.board.post_thread(agent.id, "First thread", "Start here.", 0)
        for _ in range(3):
            await run._turn(agent, 1.0)
        assert run.board.reply_count(1) == 3

    asyncio.run(main())


def test_feed_orders_announcements_dispatch_mentions_then_hot_threads() -> None:
    agents = [
        AgentSpec(1, "Priya Raman", "India", 2, 0, 0, 1.0),
        AgentSpec(2, "Kenji Sato", "Japan", 3, 1, 0, 1.0),
        AgentSpec(3, "Ana Silva", "Brazil", 4, 2, 0, 1.0),
    ]
    config = CONFIG.model_copy(update={"agents": 3, "dispatch_interval": 0})
    run = Run(
        "feed",
        config,
        EventLog(),
        RandomWalkPolicy(1),
        clock=lambda: 30.0,
        agents=agents,
    )
    run.god_announce("Announcement", "A shared update")
    for _ in range(50):
        run.world.observe_pixel(1, 2, 3, 4)
    run.dispatcher.tick(30.0)
    mention = run.board.post_thread(3, "Question", "Where are you?", 10.0)
    run.board.reply(2, mention.id, "@Priya can you help?", 20.0)
    hot = run.board.post_thread(2, "Busy discussion", "Let's coordinate", 11.0)
    run.board.reply(1, hot.id, "First reply", 12.0)
    run.board.reply(3, hot.id, "Second reply", 13.0)

    entries = run._feed(agents[0], 30.0)
    assert [entry.get("tag") for entry in entries] == [
        "announcement",
        "dispatch",
        "mention",
        None,
    ]
    assert entries[2]["thread_id"] == mention.id
    assert entries[3]["thread_id"] == hot.id
    assert run.agent_name(65534) == "World News"


def test_world_news_is_off_by_default_only_in_free_for_all() -> None:
    def interval(**update: object) -> float:
        cfg = CONFIG.model_copy(update=update)
        return Run("w", cfg, EventLog(), RandomWalkPolicy(0)).dispatcher.interval

    assert interval(policy="llm") == 0
    assert interval(policy="llm-guided") == 30 and interval(policy="random") == 30
    assert interval(policy="llm", dispatch_interval=15.0) == 15


def test_hackathon_roster_run_has_one_agent_per_attendee() -> None:
    config = RunConfig(width=64, height=64, agents=5, roster="hackathon", seed=1)
    log = EventLog()
    run = Run("h", config, log, RandomWalkPolicy(seed=1))
    assert len(run.agents) == run.config.agents == 86
    row = run.agent_list()[0]
    assert row["headline"] and row["country"] and row["affiliation"]
    stokes = next(r for r in run.agent_list() if r["name"] == "Matt Stokes")
    assert stokes["affiliation"] == "Waterloo"

    async def restore() -> Run | None:
        await run.start()
        await run.stop()
        return restore_run(log, run.id)

    restored = asyncio.run(restore())
    assert restored is not None
    avatar = next(agent for agent in run.agents if agent.persona_card.get("avatar"))
    restored_avatar = next(agent for agent in restored.agents if agent.name == avatar.name)
    assert restored_avatar.persona_card["avatar"] == avatar.persona_card["avatar"]
    row = next(row for row in restored.agent_list() if row["name"] == avatar.name)
    assert row["avatar"] == avatar.persona_card["avatar"]
