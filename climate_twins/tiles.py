"""The similarity heatmap as Web Mercator XYZ PNG tiles."""

import io
import math

import numpy as np
from PIL import Image

from .geo import great_circle_km
from .index import ClimateIndex, QueryDistances

TILE_SIZE = 256
MAX_ZOOM = 10  # the client overzooms past 6; beyond ~5 the 4.6 km cells are just upscaled

# (top_pct, (r, g, b, a)): "top X% most similar land". Interpolated in log10(top_pct);
# anything less similar than the first stop is transparent. Light, translucent yellow
# for the broad top 5% through to opaque deep purple for the rarest matches.
LEGEND_STOPS: list[tuple[float, tuple[int, int, int, int]]] = [
    (5.0, (255, 224, 120, 90)),
    (1.0, (253, 141, 60, 160)),
    (0.1, (215, 25, 50, 220)),
    (0.01, (84, 13, 110, 255)),
]

# Land inside the exclusion radius that would be colored is drawn grey instead: it is
# similar, but the search skips it and percentiles are relative to land beyond it.
EXCLUDED_RGB = (105, 110, 125)
EXCLUDED_ALPHA_SCALE = 0.7
EXCLUDED_LABEL = "Similar, but within the exclusion radius"

_STOP_LOG = np.log10([p for p, _ in LEGEND_STOPS])[::-1]  # ascending, for np.interp
_STOP_RGBA = np.array([c for _, c in LEGEND_STOPS], dtype=np.float64)[::-1]


def legend() -> dict:
    return {
        "stops": [{"top_pct": p, "rgba": list(c)} for p, c in LEGEND_STOPS],
        "excluded": {"rgba": [*EXCLUDED_RGB, round(255 * EXCLUDED_ALPHA_SCALE)], "label": EXCLUDED_LABEL},
    }


class TileRangeError(ValueError):
    """z/x/y outside the tile pyramid."""


def check_tile(z: int, x: int, y: int) -> None:
    if not 0 <= z <= MAX_ZOOM:
        raise TileRangeError(f"z must be in 0..{MAX_ZOOM}, got {z}")
    if not (0 <= x < 2**z and 0 <= y < 2**z):
        raise TileRangeError(f"x and y must be in 0..{2**z - 1} at z={z}, got x={x}, y={y}")


def pixel_centers(z: int, x: int, y: int, size: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) of each pixel centre of a tile: (size,) column vectors for each axis."""
    scale = size * 2**z
    u = (x * size + np.arange(size) + 0.5) / scale  # 0 at 180 W .. 1 at 180 E
    v = (y * size + np.arange(size) + 0.5) / scale  # 0 at the top (~85 N) .. 1 at the bottom
    lon = u * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * v))))
    return lat, lon


def colorize(top_pct: np.ndarray) -> np.ndarray:
    """(..., 4) uint8 RGBA for top-percent values; transparent beyond the first stop."""
    top = np.asarray(top_pct, dtype=np.float64)
    visible = top <= LEGEND_STOPS[0][0]
    t = np.log10(np.clip(top, LEGEND_STOPS[-1][0], LEGEND_STOPS[0][0]))
    rgba = np.stack([np.interp(t, _STOP_LOG, _STOP_RGBA[:, c]) for c in range(4)], axis=-1)
    rgba[~visible, 3] = 0.0
    return np.rint(rgba).astype(np.uint8)


def encode_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# Transparent pixels keep the faintest colour (alpha 0) so linear resampling at the
# edge of a visible area fades it out instead of darkening it.
_EMPTY_RGBA = colorize(np.full((TILE_SIZE, TILE_SIZE), 100.0))
EMPTY_PNG = encode_png(_EMPTY_RGBA)


def render_tile(
    index: ClimateIndex,
    qd: QueryDistances,
    z: int,
    x: int,
    y: int,
    exclude_km: float = 0.0,
    center: tuple[float, float] | None = None,
) -> bytes:
    """PNG of the heatmap for one query: each pixel takes the land cell under its centre.

    Percentiles are relative to land at least exclude_km from the query; colored land
    within exclude_km of `center` (default: the query cell) is drawn grey.
    """
    check_tile(z, x, y)
    lat, lon = pixel_centers(z, x, y)
    pos = index.positions_at(lat[:, None], lon[None, :])
    land = pos >= 0
    if not land.any():
        return EMPTY_PNG
    top = np.full(pos.shape, 100.0)
    top[land] = qd.top_pct(qd.d2[pos[land]], exclude_km)
    if not (top <= LEGEND_STOPS[0][0]).any():
        return EMPTY_PNG
    rgba = colorize(top)
    if exclude_km > 0:
        clat, clon = center if center is not None else (index.lat[qd.qi], index.lon[qd.qi])
        inside = land & (great_circle_km(clat, clon, lat[:, None], lon[None, :]) < exclude_km)
        rgba[inside, :3] = EXCLUDED_RGB
        rgba[inside, 3] = np.rint(rgba[inside, 3] * EXCLUDED_ALPHA_SCALE).astype(np.uint8)
    return encode_png(rgba)
