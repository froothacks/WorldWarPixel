import asyncio

from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import FRAME_PIXELS, PIXEL_STRUCT
from worldwarpixel.sim.run import Run, RunConfig

CONFIG = RunConfig(
    width=64, height=64, agents=5, cooldown=0.01, jitter=0, pixels_per_turn=4, seed=1
)


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
