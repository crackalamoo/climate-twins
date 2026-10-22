"""Which characters the site's type has: a place name with a letter it lacks would mix in some other
font's (Ḩalwān, Cần Thơ), so places.py shows it by its ASCII name instead."""

import functools
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Libre Caslon Text's web fonts, split by unicode-range (static/fonts/build.sh). The regular, bold and
# italic cover the same characters, so the regular's two files stand for all three.
WEB_FONTS = [
    Path(__file__).resolve().parent / "static" / "fonts" / f"libre-caslon-text-{subset}-400-normal.woff2"
    for subset in ("latin", "latin-ext")
]


@functools.cache
def _drawn(ch: str, file: int) -> bytes:
    image = Image.new("L", (48, 48))
    font = ImageFont.truetype(str(WEB_FONTS[file]), 24)
    ImageDraw.Draw(image).text((8, 36), ch, font=font, fill=255, anchor="ls")
    return image.tobytes()


@functools.cache
def has(ch: str) -> bool:
    """Whether the web fonts have `ch`. Pillow can't ask FreeType's cmap, so: whether a file draws ch
    other than as .notdef (as it draws a private-use character, which they have none of)."""
    return ch.isspace() or any(_drawn(ch, f) != _drawn("", f) for f in range(len(WEB_FONTS)))


def covers(text: str) -> bool:
    return all(map(has, text))
