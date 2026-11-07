import asyncio

import pytest
from pydantic import ValidationError

from worldwarpixel.sim.agents import RandomWalkPolicy
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
    assert RunConfig().pixel_interval == 0.05
    with pytest.raises(ValidationError):
        RunConfig(pixel_interval=0.01)


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
