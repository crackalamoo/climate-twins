import io

import numpy as np
import pytest
from PIL import Image

from climate_twins import tiles
from climate_twins.geo import great_circle_km
from climate_twins.index import ClimateIndex

from .conftest import World

QUERY = (45.0, -99.0)
FAR = (31.0, 31.0)


def tile_of(lat: float, lon: float, z: int) -> tuple[int, int, int, int]:
    """(x, y, row, col) of the tile and pixel containing (lat, lon)."""
    n = 2**z * tiles.TILE_SIZE
    u = (lon + 180) / 360 * n
    v = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * n
    px, py = int(u), int(v)
    return px // 256, py // 256, py % 256, px % 256


def decode(png: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png))
    assert img.size == (256, 256) and img.mode == "RGBA"
    return np.asarray(img)


def test_pixel_centers_web_mercator():
    lat, lon = tiles.pixel_centers(0, 0, 0)
    assert lon[0] == pytest.approx(-180 + 360 / 512) and lon[-1] == pytest.approx(180 - 360 / 512)
    assert lat[0] == pytest.approx(85.0511, abs=0.35) and lat[0] < 85.0511
    assert lat[127] > 0 > lat[128] and lat[127] == pytest.approx(-lat[128])
    assert np.all(np.diff(lat) < 0)  # rows run north to south
    # z=1, x=1, y=0 is the north-east quadrant.
    lat, lon = tiles.pixel_centers(1, 1, 0)
    assert lat.min() > 0 and lon.min() > 0
    # Round trip through the forward projection.
    for la, lo, z in [(31.0, 31.0, 5), (-33.9, 18.4, 6), (60.1, -149.9, 3)]:
        x, y, r, c = tile_of(la, lo, z)
        plat, plon = tiles.pixel_centers(z, x, y)
        assert abs(plon[c] - lo) <= 360 / 2 ** (z + 8)
        assert abs(plat[r] - la) <= 360 / 2 ** (z + 8)


def test_positions_at_matches_scalar_lookup(base_index: ClimateIndex):
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-89.9, 89.9, 2000), rng.uniform(-180, 180, 2000)
    pos = base_index.positions_at(lat, lon)
    cells = base_index.cells.tolist()
    for la, lo, p in zip(lat, lon, pos):
        r, c = base_index.grid.row_col(la, lo)
        flat = r * base_index.grid.nlon + c
        assert p == (cells.index(flat) if flat in cells else -1)
    assert (pos >= 0).any() and (pos < 0).any()


def test_tile_lights_up_planted_twin(world: World):
    world.plant(world.pos(*FAR), world.pos(*QUERY))
    idx = world.index()
    q = idx.query(idx.locate(*QUERY)[0])
    z = 4
    x, y, r, c = tile_of(*FAR, z)
    rgba = decode(tiles.render_tile(idx, q, z, x, y))
    # The twin's cell is 2 degrees (~23 px at z=4) across and is the brightest thing around.
    assert rgba[r, c, 3] > 200
    assert rgba[r, c, 3] == rgba[..., 3].max()
    # Expected colour for every pixel, computed independently from the scalar grid lookup.
    lat, lon = tiles.pixel_centers(z, x, y)
    cells = {f: i for i, f in enumerate(idx.cells.tolist())}
    for i in range(0, 256, 15):
        for j in range(0, 256, 15):
            row, col = idx.grid.row_col(lat[i], lon[j])
            p = cells.get(row * idx.grid.nlon + col)
            want = tiles.colorize(np.array([q.top_pct(idx.d2_at(q, p)) if p is not None else 100.0]))[0]
            assert tuple(rgba[i, j]) == tuple(want), (i, j)


def test_tiles_measure_only_their_own_cells(world: World):
    """A tile computes distances for the cells under its pixels, the same as the full scan's,
    without the scan: interleaved visitors can't evict anything a tile depends on."""
    idx = world.index()
    q = idx.query(idx.locate(*QUERY)[0])
    for zxy in [(0, 0, 0), (2, 1, 1), (4, *tile_of(*FAR, 4)[:2])]:
        tiles.render_tile(idx, q, *zxy)
    assert len(idx.scan_cache) == 0
    lat, lon = tiles.pixel_centers(0, 0, 0)
    pos = idx.positions_at(lat[:, None], lon[None, :])
    pos = pos[pos >= 0]
    np.testing.assert_allclose(idx.d2_at(q, pos), idx.scan(q)[pos], rtol=1e-5, atol=1e-6)


def test_ocean_tile_is_transparent(base_index: ClimateIndex):
    q = base_index.query(base_index.locate(*QUERY)[0])
    x, y, _, _ = tile_of(-5.0, -140.0, 3)  # mid-Pacific: no land in the synthetic world
    assert tiles.render_tile(base_index, q, 3, x, y) == tiles.EMPTY_PNG
    assert (decode(tiles.EMPTY_PNG)[..., 3] == 0).all()


def test_only_top_percentile_band_is_visible(base_index: ClimateIndex):
    q = base_index.query(base_index.locate(*QUERY)[0])
    rgba = decode(tiles.render_tile(base_index, q, 0, 0, 0))
    lat, lon = tiles.pixel_centers(0, 0, 0)
    pos = base_index.positions_at(lat[:, None], lon[None, :])
    top = np.where(pos >= 0, q.top_pct(base_index.d2_at(q, pos)), 100.0)
    np.testing.assert_array_equal(rgba[..., 3] > 0, top <= tiles.LEGEND_STOPS[0][0])
    assert (rgba[..., 3] > 0).any()


def test_exclusion_greys_home_region_and_rescales_the_rest(base_index: ClimateIndex):
    """With exclude_km, colored land near the query is grey and land beyond it is colored
    by percentiles relative to that far land only."""
    q = base_index.query(base_index.locate(*QUERY)[0])
    ex = 1500.0
    rgba = decode(tiles.render_tile(base_index, q, 0, 0, 0, ex, center=QUERY))
    lat, lon = tiles.pixel_centers(0, 0, 0)
    pos = base_index.positions_at(lat[:, None], lon[None, :])
    top = np.where(pos >= 0, q.top_pct(base_index.d2_at(q, pos), ex), 100.0)
    inside = (pos >= 0) & (great_circle_km(*QUERY, lat[:, None], lon[None, :]) < ex)
    visible = rgba[..., 3] > 0
    np.testing.assert_array_equal(visible, top <= tiles.LEGEND_STOPS[0][0])
    grey = visible & inside
    assert grey.any() and (visible & ~inside).any()
    assert (rgba[grey][:, :3] == tiles.EXCLUDED_RGB).all()
    want = tiles.colorize(top)
    np.testing.assert_array_equal(rgba[visible & ~inside], want[visible & ~inside])


@pytest.mark.parametrize("zxy", [(-1, 0, 0), (tiles.MAX_ZOOM + 1, 0, 0), (2, 4, 0), (2, 0, 4), (2, -1, 0)])
def test_bad_tile_coordinates(base_index: ClimateIndex, zxy):
    q = base_index.query(base_index.locate(*QUERY)[0])
    with pytest.raises(tiles.TileRangeError):
        tiles.render_tile(base_index, q, *zxy)
