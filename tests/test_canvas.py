import pytest

from worldwarpixel.sim.canvas import PALETTE, Canvas, PixelWrite


def test_write_get_owner() -> None:
    c = Canvas(8, 4)
    c.write(PixelWrite(3, 2, 5, agent_id=7))
    assert c.get(3, 2) == 5
    assert c.owner(3, 2) == 7
    assert c.owner(0, 0) == 0
    assert len(c.snapshot()) == 32
    owners = c.owners_le()
    assert len(owners) == 64 and owners[2 * (2 * 8 + 3) : 2 * (2 * 8 + 3) + 2] == b"\x07\x00"


def test_view_is_hex_rows_clipped_to_canvas() -> None:
    c = Canvas(4, 3)
    c.write(PixelWrite(0, 0, 15, 1))
    c.write(PixelWrite(3, 2, 10, 1))
    assert c.view(0, 0, 4, 3) == ["f000", "0000", "000a"]
    assert c.view(2, 1, 10, 10) == ["00", "0a"]


def test_rejects_bad_coordinates_and_colors() -> None:
    c = Canvas(4, 4)
    with pytest.raises(ValueError):
        c.write(PixelWrite(4, 0, 1, 1))
    with pytest.raises(ValueError):
        c.write(PixelWrite(0, 0, len(PALETTE), 1))
    with pytest.raises(ValueError):
        Canvas(0, 4)
