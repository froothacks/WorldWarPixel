from worldwarpixel.sim.events import EventLog


def test_append_and_read_in_order() -> None:
    log = EventLog()
    a = log.append("r1", 1.0, "pixel", 3, {"x": 1, "y": 2, "color": 4})
    b = log.append("r1", 2.0, "post", 3, {"body": "hi"})
    log.append("r2", 3.0, "pixel", 1, {})
    events = log.read("r1")
    assert [e.id for e in events] == [a, b]
    assert events[0].payload == {"x": 1, "y": 2, "color": 4}
    assert events[0].agent_id == 3
    assert log.read("r1", after_id=a) == [events[1]]
    assert log.count("r1") == 2
    assert log.run_ids() == ["r1", "r2"]
    log.close()
