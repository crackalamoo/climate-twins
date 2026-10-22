"""The link-preview card: a 1200x630 PNG (Open Graph's size) of the query's heatmap on a world map,
its star and best matches, under the place's name, in the site's Atlas look.

The map is in the Equal Earth projection rather than the site's Web Mercator: the heatmap colours
shares of land by area, which an equal-area map shows at their true size (Mercator would blow up
Siberia and Canada), and the whole indexed world (Antarctica isn't) fits the card's 1.9:1 without
cropping, with room for the headline. Its curved edge also reads as a printed atlas's world plate.
"""

import functools
import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont, features

from . import page, tiles
from .boundaries import Boundaries
from .index import ClimateIndex, Query

WIDTH, HEIGHT = page.IMAGE_SIZE
HERE = Path(__file__).resolve().parent
CARD_FONT = HERE / "fonts" / "libre-caslon-text-regular.ttf"  # static/fonts/build.sh
PIN_FONT = HERE / "static" / "fonts" / "libre-caslon-text-latin-700-normal.woff2"  # digits only
FEATURES = ["kern", "liga", "onum"]  # old-style figures, as the panel sets text

log = logging.getLogger(__name__)

# The Atlas basemap's (static/atlas/style.json) land and water, on the panel's paper, in its ink.
PAPER = (247, 241, 227)
LAND = (242, 235, 219)
WATER = (197, 209, 211)
INK = (42, 36, 32)
MUTED = (122, 111, 95)  # style.css's --muted, for the kicker
EDGE = (156, 134, 112)  # the world's edge and borders: the basemap's country borders' brown

# The map: Equal Earth from MAP_NORTH (all of Greenland) to MAP_SOUTH (past Cape Horn and South
# Georgia), as large as fits MAP_BOX (left, top, right, bottom), below the headline: lower by a
# line where the headline takes two.
MAP_NORTH, MAP_SOUTH = 84.0, -57.0
MAP_BOX = (36, 132, WIDTH - 36, HEIGHT - 22)
EDGE_WIDTH = 1.25
# Country borders in the same brown, faint: the basemap's at low zoom (0.5 px at 60%).
BORDER_OPACITY = 0.6
# The land grid's cells are ~0.04 degrees, several to a card pixel: each pixel is sampled
# SUPERSAMPLE x SUPERSAMPLE times, for smooth coasts and borders and the heatmap's most similar cell.
SUPERSAMPLE = 3

# Type: the headline (the place) in Libre Caslon Text under a kicker (what the card shows of it) in
# letterspaced capitals. A long headline takes two lines, then shrinks (to HEADLINE_MIN_PX), then is
# cut short.
HEADLINE_PX, HEADLINE_MIN_PX = 58, 34
HEADLINE_BASELINE, HEADLINE_LEADING = 104, 62
KICKER_PX, KICKER_TRACKING = 22, 0.16  # tracking in ems
KICKER_BASELINE = 48
TEXT_LEFT = 52  # in line with the map's widest point

# Pins as the site draws them (web/map.ts), scaled for a card that's often shown ~500 px wide: the
# best PINS matches, discs of their rank's heatmap colour ringed in paper and numbered. Where they
# overlap, better ranks stack on top (the site merges them; on a card, a stack says "several here").
PINS = 5
PIN_RADIUS, PIN_RING, PIN_NUMBER_PX = 14.0, 2.5, 17
STAR_RADIUS, STAR_RING = 19.0, 2.5  # the query's ink star, ringed in paper
SPRITE_SCALE = 4  # pins and the star are drawn this much larger and scaled down, for smooth edges

# Equal Earth (Šavrič, Patterson & Jenny 2018).
_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796
_M = math.sqrt(3) / 2


def equal_earth(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Projected (x, y) of degrees (lat, lon), y up; x spans about +-2.71, y +-1.32."""
    t = np.arcsin(_M * np.sin(np.radians(lat)))
    t2 = t * t
    t6 = t2 * t2 * t2
    x = 2 * math.sqrt(3) * np.radians(lon) * np.cos(t) / (3 * (_A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)))
    y = t * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2))
    return x, y


def equal_earth_inverse(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Degrees (lat, lon) of projected (x, y); NaN outside the world."""
    t = np.asarray(y, dtype=np.float64) / _A1
    for _ in range(6):  # Newton's method; converges to float precision in 4-5 steps
        t2 = t * t
        t6 = t2 * t2 * t2
        f = t * (_A1 + _A2 * t2 + t6 * (_A3 + _A4 * t2)) - y
        t = t - f / (_A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2))
    t2 = t * t
    t6 = t2 * t2 * t2
    with np.errstate(invalid="ignore"):
        lat = np.degrees(np.arcsin(np.sin(t) / _M))
    lon = np.degrees(3 * x * (_A1 + 3 * _A2 * t2 + t6 * (7 * _A3 + 9 * _A4 * t2)) / (2 * math.sqrt(3) * np.cos(t)))
    outside = ~(np.abs(lon) <= 180)
    return np.where(outside, np.nan, lat), np.where(outside, np.nan, lon)


@functools.cache
def layout_engine() -> ImageFont.Layout:
    """Pillow's raqm layout (HarfBuzz: kerning, ligatures, old-style figures, marks) where it has it:
    raqm needs FriBiDi installed. Without, basic layout: none of those, but still a card."""
    if features.check_feature("raqm"):
        return ImageFont.Layout.RAQM
    log.warning("Pillow has no raqm (is FriBiDi installed?): preview cards are set without kerning or old-style figures")
    return ImageFont.Layout.BASIC


def font_features() -> dict:
    """draw.text's and getlength's OpenType features: only raqm takes them."""
    return {"features": FEATURES} if layout_engine() == ImageFont.Layout.RAQM else {}


@functools.cache
def face(px: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(CARD_FONT), px, layout_engine=layout_engine())


def tracked(draw: ImageDraw.ImageDraw, x: float, baseline: float, text: str, font: ImageFont.FreeTypeFont, tracking: float, fill) -> None:
    """Draw `text` letterspaced by `tracking` px (in place of its kerning)."""
    for ch in text:
        draw.text((x, baseline), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + tracking


def country_borders(boundaries: Boundaries, to_sheet, shape: tuple[int, int]) -> np.ndarray:
    """Where Natural Earth's admin-1 polygons of two countries meet on a sheet of `shape`, as a bool
    mask two samples wide; to_sheet(lat, lon) gives sheet positions. Each polygon is filled with its
    country's number, largest first so that enclaves (Lesotho) stay on top."""
    sheet = Image.new("I", shape[::-1], -1)
    draw = ImageDraw.Draw(sheet)
    countries = {c: i for i, c in enumerate(dict.fromkeys(boundaries.country_codes))}  # None: one more
    geometries = boundaries.tree.geometries
    for i in np.argsort(-shapely.area(geometries)):
        for polygon in shapely.get_parts(geometries[i]):
            lon, lat = shapely.get_coordinates(polygon.exterior).T
            draw.polygon(np.column_stack(to_sheet(lat, lon)).ravel().tolist(), fill=countries[boundaries.country_codes[i]])
    ids = np.asarray(sheet)
    across = (ids[:, 1:] != ids[:, :-1]) & (ids[:, 1:] >= 0) & (ids[:, :-1] >= 0)
    down = (ids[1:] != ids[:-1]) & (ids[1:] >= 0) & (ids[:-1] >= 0)
    edge = np.zeros(ids.shape, dtype=bool)
    edge[:, 1:] |= across
    edge[:, :-1] |= across
    edge[1:] |= down
    edge[:-1] |= down
    return edge


@dataclass(frozen=True)
class Plate:
    """What every card for one index shares: the paper, sea and land (`base`, (HEIGHT, WIDTH, 3)
    RGB); country borders, drawn over the heatmap (flat card pixels `border_pixels`, of opacity
    `border_alpha`); and the card pixels showing land (flat indices `pixels`): the positions of the
    land cells sampled in each (`cells`, pixel i's from starts[i]), where it is (`lat`, `lon`: a land
    sample's) and how much of it is land (`cover`, 0-1)."""

    base: np.ndarray
    border_pixels: np.ndarray
    border_alpha: np.ndarray
    pixels: np.ndarray
    cells: np.ndarray
    starts: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    cover: np.ndarray
    scale: float  # card pixels per projected unit
    origin: tuple[float, float]  # card position of projected (0, 0)

    @classmethod
    def build(cls, index: ClimateIndex, boundaries: Boundaries | None = None, lines: int = 1) -> "Plate":
        """The plate under a headline of `lines` lines."""
        left, top, right, bottom = MAP_BOX
        top += (lines - 1) * HEADLINE_LEADING
        x_max = float(equal_earth(0.0, 180.0)[0])
        y_north, y_south = float(equal_earth(MAP_NORTH, 0.0)[1]), float(equal_earth(MAP_SOUTH, 0.0)[1])
        scale = min((right - left) / (2 * x_max), (bottom - top) / (y_north - y_south))
        width, height = math.ceil(2 * x_max * scale), math.ceil((y_north - y_south) * scale)
        ox, oy = (left + right - width) // 2, (top + bottom - height) // 2  # the map's top-left
        s = SUPERSAMPLE

        def to_sheet(lat, lon):
            """The map's pixel positions at s x of (lat, lon)."""
            x, y = equal_earth(lat, lon)
            return (x + x_max) * scale * s, (y_north - y) * scale * s

        # Sample each of the map's pixels s x s times.
        x = (np.arange(width * s) + 0.5) / (s * scale) - x_max
        y = y_north - (np.arange(height * s) + 0.5) / (s * scale)
        lat, lon = equal_earth_inverse(*np.broadcast_arrays(x[None, :], y[:, None]))
        world = ~np.isnan(lat) & (lat <= MAP_NORTH) & (lat >= MAP_SOUTH)
        pos = np.full(lat.shape, -1, dtype=np.int64)
        pos[world] = index.positions_at(lat[world], lon[world])

        def samples(a: np.ndarray) -> np.ndarray:
            """(pixels, s*s): each of the map's pixels' samples, row by row."""
            return a.reshape(height, s, width, s).swapaxes(1, 2).reshape(height * width, s * s)

        def card_pixels(i: np.ndarray) -> np.ndarray:
            return (i // width + oy) * WIDTH + i % width + ox

        land = samples(pos >= 0)
        cover = land.mean(axis=1)
        inside = np.flatnonzero(cover)
        first = np.argmax(land[inside], axis=1)  # each land pixel's first land sample
        cells = samples(pos)[inside]
        border_alpha = np.zeros(width * height)
        if boundaries is not None:
            border_alpha = samples(country_borders(boundaries, to_sheet, lat.shape) & world).mean(axis=1) * BORDER_OPACITY
        bordered = np.flatnonzero(border_alpha)

        # Paper, the world in sea, land, the world's edge: drawn at s x, then averaged down.
        sheet = np.empty((*lat.shape, 3), dtype=np.uint8)
        sheet[:] = PAPER
        sheet[world] = WATER
        sheet[pos >= 0] = LAND
        image = Image.fromarray(sheet)
        meridian, parallel = np.linspace(MAP_NORTH, MAP_SOUTH, 91), np.linspace(180, -180, 91)
        east, west, north, south = np.full(91, 180), np.full(91, -180), np.full(91, MAP_NORTH), np.full(91, MAP_SOUTH)
        edge = np.column_stack(to_sheet(np.r_[meridian, south, meridian[::-1], north], np.r_[east, parallel, west, parallel[::-1]]))
        ImageDraw.Draw(image).line(edge.ravel().tolist() + edge[0].tolist(), fill=EDGE, width=round(EDGE_WIDTH * s), joint="curve")
        base = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        base[:] = PAPER
        base[oy : oy + height, ox : ox + width] = np.asarray(image.reduce(s))
        return cls(  # cells in 32 bits: ~1.2M of them, and positions fit
            base, card_pixels(bordered), border_alpha[bordered], card_pixels(inside), cells[cells >= 0].astype(np.int32),
            np.r_[0, np.cumsum(land[inside].sum(axis=1))[:-1]], samples(lat)[inside, first], samples(lon)[inside, first],
            cover[inside], scale, (ox + x_max * scale, oy + y_north * scale),
        )

    def project(self, lat: float, lon: float) -> tuple[float, float]:
        """Card position of (lat, lon)."""
        x, y = equal_earth(lat, lon)
        return self.origin[0] + float(x) * self.scale, self.origin[1] - float(y) * self.scale


def blend(card: np.ndarray, pixels: np.ndarray, rgb, alpha: np.ndarray) -> None:
    """Paint flat card `pixels` with `rgb` (one colour, or one per pixel) at `alpha` (0-1)."""
    flat = card.reshape(-1, 3)
    a = alpha[:, None]
    flat[pixels] = np.rint(flat[pixels] * (1 - a) + np.asarray(rgb) * a).astype(np.uint8)


def luminance(rgb) -> float:
    c = [v / 255 for v in rgb]
    r, g, b = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    """WCAG contrast ratio between two RGB colours."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def sprite(size: float, paint) -> Image.Image:
    """An RGBA image `size` px square, drawn by paint(draw, k) at SPRITE_SCALE (k) x and scaled down."""
    n = math.ceil(size) + 2
    big = Image.new("RGBA", (n * SPRITE_SCALE, n * SPRITE_SCALE), (*PAPER, 0))
    paint(ImageDraw.Draw(big), SPRITE_SCALE, n * SPRITE_SCALE / 2)
    return big.reduce(SPRITE_SCALE)


@functools.lru_cache(maxsize=256)  # a colour per similarity: rarely the same twice, but tiny
def pin(color: tuple[int, int, int], number: int) -> Image.Image:
    """A filled pin as web/map.ts draws ranks 1-5: a disc of the rank's colour ringed in paper, its
    number in paper where legible (4.5:1), else in ink."""
    ink = PAPER if contrast(color, PAPER) >= 4.5 else INK
    font = ImageFont.truetype(str(PIN_FONT), PIN_NUMBER_PX * SPRITE_SCALE)

    def paint(draw: ImageDraw.ImageDraw, k: int, c: float):
        r = (PIN_RADIUS + PIN_RING / 2) * k
        draw.ellipse((c - r, c - r, c + r, c + r), fill=color, outline=PAPER, width=round(PIN_RING * k))
        x0, y0, x1, y1 = draw.textbbox((0, 0), str(number), font=font, anchor="ls")
        draw.text((c - (x0 + x1) / 2, c - (y0 + y1) / 2), str(number), font=font, fill=ink, anchor="ls")

    return sprite(2 * (PIN_RADIUS + PIN_RING), paint)


@functools.cache
def star() -> Image.Image:
    """The query's star: ink, ringed in paper (the site's halo)."""

    def paint(draw: ImageDraw.ImageDraw, k: int, c: float):
        angles = np.radians(np.arange(10) * 36 - 90)
        radii = np.where(np.arange(10) % 2, 0.382, 1.0) * STAR_RADIUS * k
        points = list(zip((c + radii * np.cos(angles)).tolist(), (c + radii * np.sin(angles)).tolist()))
        draw.polygon(points, fill=PAPER)
        draw.line(points + points[:1], fill=PAPER, width=round(2 * STAR_RING * k), joint="curve")
        draw.polygon(points, fill=INK)

    return sprite(2 * (STAR_RADIUS + STAR_RING), paint)


def paste(card: Image.Image, image: Image.Image, x: float, y: float) -> None:
    """Paste `image` centred on card position (x, y)."""
    card.alpha_composite(image, (round(x - image.width / 2), round(y - image.height / 2)))


def set_headline(text: str) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """The headline's face and lines: one at HEADLINE_PX if it fits the card's width; else two, broken
    at the space that best balances them, shrunk to fit if need be; the second cut short as a last
    resort."""
    room, opts = WIDTH - 2 * TEXT_LEFT, font_features()
    font = face(HEADLINE_PX)
    if font.getlength(text, **opts) <= room or " " not in text:
        lines = [text]
    else:
        words = text.split(" ")
        splits = [[" ".join(words[:i]), " ".join(words[i:])] for i in range(1, len(words))]
        lines = min(splits, key=lambda ls: max(font.getlength(line, **opts) for line in ls))
    if (w := max(font.getlength(line, **opts) for line in lines)) > room:
        font = face(max(HEADLINE_MIN_PX, math.floor(HEADLINE_PX * room / w)))
    while font.getlength(lines[-1], **opts) > room:
        lines[-1] = lines[-1][:-2].rstrip() + "…"
    return font, lines


@functools.lru_cache(maxsize=2)  # cards with one headline line and with two
def plate(index: ClimateIndex, boundaries: Boundaries | None, lines: int) -> Plate:
    return Plate.build(index, boundaries, lines)


def render(
    index: ClimateIndex,
    boundaries: Boundaries | None,
    kicker: str,
    headline: str,
    q: Query | None = None,
    exclude_km: float = 0.0,
    center: tuple[float, float] | None = None,
    matches: list[tuple[float, float, float]] = (),
) -> bytes:
    """The card as PNG: `headline` under `kicker`, over the map with the heatmap of q (as tiles.shade
    colours it) centred on `center` (its star), and its best matches as (lat, lon, top %) pins, best
    first. `boundaries`: for country borders."""
    font, lines = set_headline(headline.replace("'", "’"))  # typeset apostrophes
    plate_ = plate(index, boundaries, len(lines))
    card = plate_.base.copy()
    rgba = None
    if q is not None:
        # A pixel spans several cells (~8 x 8 near the equator) and takes its most similar one, so
        # that a narrow strip of climate (a coast, a valley) shows as it does on the site zoomed in.
        d2 = np.minimum.reduceat(index.d2_at(q, plate_.cells), plate_.starts)
        rgba = tiles.shade(index, q, d2, plate_.lat, plate_.lon, exclude_km, center)
    if rgba is not None:
        blend(card, plate_.pixels, rgba[:, :3], rgba[:, 3] / 255.0 * plate_.cover)
    blend(card, plate_.border_pixels, EDGE, plate_.border_alpha)
    image = Image.fromarray(card).convert("RGBA")

    for rank, (lat, lon, top_pct) in reversed(list(enumerate(matches[:PINS], 1))):  # better ranks on top
        color = tuple(int(v) for v in tiles.colorize(np.array(top_pct))[:3])
        paste(image, pin(color, rank), *plate_.project(lat, lon))
    if center is not None:
        paste(image, star(), *plate_.project(*center))

    draw = ImageDraw.Draw(image)
    small, tracking = face(KICKER_PX), KICKER_TRACKING * KICKER_PX
    tracked(draw, TEXT_LEFT, KICKER_BASELINE, kicker.upper(), small, tracking, MUTED)
    for i, line in enumerate(lines):
        draw.text((TEXT_LEFT, HEADLINE_BASELINE + i * HEADLINE_LEADING), line, font=font, fill=INK, anchor="ls", **font_features())
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
