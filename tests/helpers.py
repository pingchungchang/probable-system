"""Shared test fixtures. No torch, no network, no model download."""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SOS_DEBOUNCE", "0.05")  # tests must not wait 3 s


def png_bytes(r: int, g: int, b: int) -> bytes:
    """A valid 1x1 PNG, built by hand so Pillow is not needed to create one."""
    import binascii
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit truecolour
    raw = bytes([0, r, g, b])  # one scanline, filter type 0
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def text_png(lines: int = 26, width: int = 600) -> bytes:
    """A PNG whose pixel statistics look like text to the OCR gate.

    Alternating dark/light runs on every other row reproduce the dense
    small-scale contrast of glyphs without needing a font installed.
    """
    import binascii
    import zlib

    height = lines * 16
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type
        glyph_row = (y % 16) < 9
        for x in range(width):
            on = glyph_row and (x % 7) < 3 and (x // 7) % 3 != 2
            v = 20 if on else 240
            rows += bytes((v, v, v))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


class TempTree(unittest.TestCase):
    """Gives each test an isolated directory and database path."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / "sos.db"
        self.addCleanup(self._tmp.cleanup)

    def write_image(self, name: str, colour: tuple[int, int, int] | None = None) -> Path:
        """Distinct bytes per filename unless a colour is given explicitly.

        Deriving the default from the name matters: two fixtures with identical
        bytes legitimately share one embedding via the dedup gate, which silently
        breaks any test that assumes they rank independently.
        """
        if colour is None:
            h = hashlib.blake2b(name.encode(), digest_size=3).digest()
            colour = (h[0], h[1], h[2])
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png_bytes(*colour))
        return p

    def write_text(self, name: str, body: str = "hello") -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return p
