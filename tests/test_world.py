from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.world import World


def _agent(i: int, name: str, country: str) -> AgentSpec:
    return AgentSpec(i, name, country, color=i, home_x=0, home_y=0, cooldown=1.0)


AGENTS = [
    _agent(1, "Priya Sharma", "India"),
    _agent(2, "Arjun Patel", "India"),
    _agent(3, "Wei Jun Zhang", "China"),
    _agent(4, "Emily Miller", "United States"),
    _agent(5, "Emily Garcia", "United States"),
]


def test_raids_count_cross_country_overpainting_only() -> None:
    world = World(AGENTS, Board())
    world.observe_pixel(1, 0)  # blank
    world.observe_pixel(1, 2)  # own country
    world.observe_pixel(1, 3)  # India paints over China
    world.observe_pixel(3, 1)  # China paints back
    world.observe_pixel(3, 1)
    world.observe_pixel(65535, 1)  # god is not a country
    snap = world.snapshot()
    india = next(n for n in snap["nodes"] if n["country"] == "India")
    assert india == {"country": "India", "agents": 2, "pixels": 3, "posts": 0}
    assert snap["edges"] == [{"a": "China", "b": "India", "talk": 0, "raids_ab": 2, "raids_ba": 1}]


def test_replies_and_mentions_become_talk_edges() -> None:
    board = Board()
    world = World(AGENTS, board)
    root = board.post_thread(3, "Hello from China", "Anyone want to trade?", now=1.0)
    world.observe_post(root.to_dict())
    reply = board.reply(1, root.id, "@Wei_Jun yes, and @Arjun come too", now=2.0)
    world.observe_post(reply.to_dict())
    snap = world.snapshot()
    china_india = next(e for e in snap["edges"] if {e["a"], e["b"]} == {"China", "India"})
    assert china_india["talk"] == 1  # reply + mention of the same person count once
    assert not [e for e in snap["edges"] if e["a"] == e["b"]]
    assert world.agent_talk[(1, 2)] == 1  # the @Arjun mention is same-country chatter
    assert next(n for n in snap["nodes"] if n["country"] == "China")["posts"] == 1


def test_ambiguous_first_name_prefers_thread_participant() -> None:
    board = Board()
    world = World(AGENTS, board)
    root = board.post_thread(5, "USA here", "Building a flag", now=1.0)
    world.observe_post(root.to_dict())
    ping = board.post_thread(3, "Hey", "@Emily want to ally?", now=2.0)
    world.observe_post(ping.to_dict())
    # Two Emilys, both in the US: the country edge exists even though the person is a guess.
    assert world.talk[(0, 2)] == 1
    reply = board.reply(3, root.id, "@Emily nice flag", now=3.0)
    world.observe_post(reply.to_dict())
    # Inside Emily Garcia's thread the mention is hers, not Emily Miller's.
    assert world.agent_talk[(3, 5)] == 1
    assert world.talk[(0, 2)] == 2


def test_mentions_in_thread_titles_count() -> None:
    board = Board()
    world = World(AGENTS, board)
    root = board.post_thread(1, "@Wei_Jun let's talk", "Border proposal below.", now=1.0)
    world.observe_post(root.to_dict())
    assert world.talk[(0, 1)] == 1
