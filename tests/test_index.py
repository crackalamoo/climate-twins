import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

import climate_twins.index as index_mod
from climate_twins.features import FEATURES, MONTHS, RAW_VARS, shift_seasons
from climate_twins.geo import Grid, great_circle_km
from climate_twins.index import ClimateIndex, NoLandError

from .conftest import World

QUERY = (45.0, -99.0)  # interior of "North America"
FAR = (31.0, 31.0)  # "Eurafrica", ~10,000 km away
SOUTH = (-25.0, 135.0)  # "Australia"


def match_cells(result: dict) -> list[int]:
    """Positions (into idx.cells) of the matches; that is what cell["index"] holds."""
    return [m["cell"]["index"] for m in result["matches"]]


def match_latlon(result: dict) -> np.ndarray:
    return np.array([[m["cell"]["lat"], m["cell"]["lon"]] for m in result["matches"]]).reshape(-1, 2)


def brute_force_nms(
    idx: ClimateIndex, lat: float, lon: float, n: int, exclude_km: float, suppress_km: float, align: str
) -> list[int]:
    """Reference: float64 projected distances to every cell, greedy suppression over all of them
    for RESORT_FACTOR * n picks, then the n closest by full distance."""
    qi, _ = idx.locate(lat, lon)
    qvec = idx.vectors(qi)[0]
    variants = [qvec, shift_seasons(qvec)] if align == "auto" else [qvec]
    qp = idx.model.project(np.stack(variants)).astype(np.float64)
    pcs = idx.pcs.astype(np.float64)
    d2 = np.min([((pcs - q) ** 2).sum(axis=1) for q in qp], axis=0)
    every = np.arange(len(idx))
    lats, lons = idx.lat_of(every), idx.lon_of(every)
    picked: list[int] = []
    for i in np.argsort(d2, kind="stable"):
        if len(picked) == math.ceil(index_mod.RESORT_FACTOR * n):
            break
        if great_circle_km(lat, lon, lats[i], lons[i]) < exclude_km:
            continue
        if any(great_circle_km(lats[i], lons[i], lats[j], lons[j]) < suppress_km for j in picked):
            continue
        picked.append(int(i))
    if not picked:
        return []
    vectors = idx.vectors(np.array(picked)).astype(np.float64)
    full = np.min([((vectors - v) ** 2).sum(axis=1) for v in variants], axis=0)
    return sorted(picked, key=lambda i: (full[picked.index(i)], i))[:n]


@pytest.fixture(scope="module")
def fine_index(base_index: ClimateIndex) -> ClimateIndex:
    """TerraClimate-resolution (1/24 deg) grid with two tiny islands, one on the dateline."""
    grid = Grid(lat0=90 - 1 / 48, dlat=-1 / 24, nlat=4320, lon0=-180 + 1 / 48, dlon=1 / 24, nlon=8640)
    islands = []
    for (lat, lon), size in [((10.0, 20.0), 5), ((-30.0, 180.0), 6)]:
        r0, c0 = grid.row_col(lat, lon)
        for dr, dc in itertools.product(range(size), range(-size // 2, size // 2)):
            islands.append((r0 + dr) * grid.nlon + (c0 + dc) % grid.nlon)
    cells = np.unique(islands)
    n, k = len(cells), base_index.model.k
    raw = {v: np.zeros((n, MONTHS), dtype=np.int16) for v in RAW_VARS}
    return ClimateIndex(grid, base_index.codecs, base_index.model, cells, raw, np.zeros((n, k), np.float32), np.arange(n))


class TestLocate:
    def test_exact_cell(self, base_index: ClimateIndex):
        for i in [0, 17, len(base_index) // 2, len(base_index) - 1]:
            j, km = base_index.locate(float(base_index.lat_of(i)), float(base_index.lon_of(i)))
            assert j == i and km == pytest.approx(0.0, abs=1e-6)

    def test_point_inside_cell_snaps_to_it(self, base_index: ClimateIndex):
        i, km = base_index.locate(45.15, -98.9)
        assert (base_index.lat_of(i), base_index.lon_of(i)) == QUERY
        assert 0 < km < 25

    def test_offshore_snaps_to_nearest_coast(self, base_index: ClimateIndex):
        # 61 N is ocean on the 2-degree world; 59 N is the northern coast of "North America".
        with pytest.raises(NoLandError):
            base_index.locate(61.0, -99.0)
        i, km = base_index.locate(61.0, -99.0, max_km=300)
        assert (base_index.lat_of(i), base_index.lon_of(i)) == (59.0, -99.0)
        assert km == pytest.approx(2 * 111.195, rel=1e-3)

    def test_fine_grid_offshore_snaps_to_nearby_land(self, fine_index: ClimateIndex):
        lats = fine_index.lat_of(np.arange(len(fine_index)))
        north = lats.max()
        lon = float(fine_index.lon_of(np.argmax(lats)))
        i, km = fine_index.locate(north + 0.15, lon)  # ~3.6 cells north of the island
        assert fine_index.lat_of(i) == north and fine_index.lon_of(i) == lon
        assert km == pytest.approx(0.15 * 111.195, rel=1e-3)

    def test_fine_grid_snaps_across_dateline(self, fine_index: ClimateIndex):
        i, km = fine_index.locate(-30.05, -179.95)
        assert km < 25
        j, _ = fine_index.locate(-30.05, 180.05)
        assert i == j

    def test_far_at_sea_raises(self, base_index: ClimateIndex, fine_index: ClimateIndex):
        for idx, (lat, lon) in [(base_index, (-30.0, -150.0)), (fine_index, (10.0, 20.5)), (fine_index, (0.0, 0.0))]:
            with pytest.raises(NoLandError):
                idx.locate(lat, lon)


def test_planted_twin_is_found_first(world: World):
    world.plant(world.pos(*FAR), world.pos(*QUERY))
    idx = world.index()
    res = idx.similar(*QUERY, n=10)
    top = res["matches"][0]
    assert (top["cell"]["lat"], top["cell"]["lon"]) == FAR
    assert top["rank"] == 1
    assert top["distance"] == pytest.approx(0.0, abs=1e-3)
    assert top["seasons_shifted_6_months"] is False
    # Same climate; only the raw solar radiation differs, as the twin is at another latitude.
    radiation = "mean_solar_radiation_w_m2"
    assert {k: v for k, v in top["summary"].items() if k != radiation} == {
        k: v for k, v in res["query"]["summary"].items() if k != radiation
    }
    assert top["more_similar_than_pct_of_land"] > 99.0
    assert res["query"]["cell"]["lat"] == QUERY[0] and res["query"]["snapped_km"] == 0.0


def test_exclude_km_removes_nearby_identical_cells(world: World):
    q = world.pos(*QUERY)
    lat, lon = world.grid.lat_of(world.cells), world.grid.lon_of(world.cells)
    near = np.flatnonzero(great_circle_km(*QUERY, lat, lon) < 400)
    assert len(near) > 5
    for i in near:
        world.plant(int(i), q)
    idx = world.index()

    res = idx.similar(*QUERY, n=15, exclude_km=500, suppress_km=0)
    assert all(m["km_from_query"] >= 500 for m in res["matches"])
    assert not set(match_cells(res)) & set(near.tolist())

    # Without exclusion the query and its identical neighbours come first.
    res0 = idx.similar(*QUERY, n=len(near), exclude_km=0, suppress_km=0)
    assert set(match_cells(res0)) == set(near.tolist())
    assert all(m["distance"] == pytest.approx(0.0, abs=1e-3) for m in res0["matches"])


def test_suppress_km_spacing_and_sorted(base_index: ClimateIndex):
    suppress_km = 1500.0
    res = base_index.similar(*QUERY, n=30, exclude_km=500, suppress_km=suppress_km)
    assert len(res["matches"]) == 30
    ll = match_latlon(res)
    for a, b in itertools.combinations(range(len(ll)), 2):
        assert great_circle_km(*ll[a], *ll[b]) >= suppress_km - 1e-6
    d = [m["distance"] for m in res["matches"]]
    assert d == sorted(d)
    assert [m["rank"] for m in res["matches"]] == list(range(1, 31))


def test_southern_twin_needs_season_shift(world: World):
    twin, q = world.pos(*SOUTH), world.pos(*QUERY)
    world.plant(twin, q, shift_months=6)
    idx = world.index()

    auto = idx.similar(*QUERY, n=10, align="auto")
    top = auto["matches"][0]
    assert top["cell"]["index"] == twin
    assert top["seasons_shifted_6_months"] is True
    assert top["distance"] == pytest.approx(0.0, abs=1e-3)

    cal = idx.similar(*QUERY, n=10, align="calendar")
    assert cal["params"]["align"] == "calendar"
    assert not any(m["seasons_shifted_6_months"] for m in cal["matches"])
    cells = match_cells(cal)
    assert cells[0] != twin
    if twin in cells:
        assert cal["matches"][cells.index(twin)]["distance"] > cal["matches"][0]["distance"]


def test_exclusion_covering_the_globe_gives_no_matches(base_index: ClimateIndex):
    assert base_index.similar(*QUERY, n=5, exclude_km=20_000)["matches"] == []


@pytest.mark.parametrize("initial", [1, index_mod.INITIAL_CANDIDATES])
@pytest.mark.parametrize(
    "n, exclude_km, suppress_km, align",
    [(20, 500, 400, "auto"), (50, 0, 0, "calendar"), (40, 2000, 1200, "auto"), (200, 500, 2500, "auto")],
)
def test_pick_equals_brute_force_greedy_nms(
    monkeypatch, base_index: ClimateIndex, initial: int, n: int, exclude_km: float, suppress_km: float, align: str
):
    """Exercises candidate-pool growth (tiny initial pool) and exhaustion (n not reachable)."""
    monkeypatch.setattr(index_mod, "INITIAL_CANDIDATES", initial)
    for lat, lon in [QUERY, SOUTH, (55.0, 175.0)]:
        res = base_index.similar(lat, lon, n=n, exclude_km=exclude_km, suppress_km=suppress_km, align=align)
        assert match_cells(res) == brute_force_nms(base_index, lat, lon, n, exclude_km, suppress_km, align)


def test_difference_share_attributes_a_single_difference(world: World):
    """A twin differing only in precipitation attributes everything to ppt."""
    t, q = world.pos(*FAR), world.pos(*QUERY)
    world.plant(t, q)
    world.raw["ppt"][t] = world.raw["ppt"][q] + 30  # +3 mm every month
    share = world.index().similar(*QUERY, n=1)["matches"][0]
    assert share["cell"]["lat"] == FAR[0]
    assert share["difference_share"]["ppt"] == pytest.approx(1.0)


def test_difference_d2_is_the_unprojected_squared_distance(world: World):
    idx = world.index()
    res = idx.similar(*QUERY, n=5)
    qi, _ = idx.locate(*QUERY)
    q = idx.query(qi)
    for m in res["matches"]:
        d2 = m["difference_d2"]
        assert list(d2) == list(FEATURES)
        full = float(((idx.vectors(m["cell"]["index"])[0] - q.variants[int(m["seasons_shifted_6_months"])]) ** 2).sum())
        assert sum(d2.values()) == pytest.approx(full, rel=1e-3, abs=1e-3)
        total = sum(d2.values())
        for f in FEATURES:
            assert m["difference_share"][f] == pytest.approx(d2[f] / total, abs=2e-3)
        assert m["distance"] ** 2 == pytest.approx(full, rel=1e-3, abs=1e-6)


def test_distance_at_is_the_reported_distance(base_index: ClimateIndex):
    for align in ("auto", "calendar"):
        res = base_index.similar(*QUERY, n=10, align=align)
        q = base_index.query(base_index.locate(*QUERY)[0], align)
        got = base_index.distance_at(q, match_cells(res))
        np.testing.assert_allclose(got, [m["distance"] for m in res["matches"]], atol=1e-4)


def test_distances_to_snaps_points_like_queries(fine_index: ClimateIndex, monkeypatch):
    """Points in a land cell take it, a point just offshore snaps to the coast, one far at sea is NaN;
    each distance is the best of the query's alignments, as matches' are."""
    rng = np.random.default_rng(0)
    idx = fine_index
    vectors = idx.vectors(np.arange(len(idx))).astype(np.float64)
    q = SimpleNamespace(variants=rng.normal(size=(2, vectors.shape[1])).astype(np.float32))  # two alignments
    monkeypatch.setattr(idx, "query", lambda qi, align="auto": q)
    lats = idx.lat_of(np.arange(len(idx)))
    north, north_lon = float(lats.max()), float(idx.lon_of(np.argmax(lats)))
    inside = (float(lats[5]) + 0.01, float(idx.lon_of(5)) - 0.01)
    points = [inside, (north + 0.15, north_lon), (0.0, 0.0)]
    got = idx.distances_to(*inside, "auto", *zip(*points))
    assert np.isnan(got[2])
    for p, d in zip(points[:2], got[:2]):
        i, _ = idx.locate(*p)
        assert d == pytest.approx(np.sqrt(((vectors[i] - q.variants) ** 2).sum(axis=1).min()))


def test_describe_decodes_raw(base_index: ClimateIndex):
    d = base_index.describe(*QUERY)
    pos = d["cell"]["index"]
    tmax = base_index.raw["tmax"][pos] * 0.1 + 10.0
    np.testing.assert_allclose(d["monthly"]["tmax"], np.round(tmax, 1))
    assert d["summary"]["warmest_month_high_c"] == pytest.approx(tmax.max(), abs=0.05)


@pytest.mark.parametrize("initial", [1, index_mod.INITIAL_CANDIDATES])
def test_pick_with_exact_ties_equals_brute_force(monkeypatch, world: World, initial: int):
    """Many cells with identical climate (d2 ties) are visited in index order, as brute force does."""
    q = world.pos(*QUERY)
    lat = world.grid.lat_of(world.cells)
    # Same latitude as the query, so planted cells are bit-identical (exact d2 ties).
    for i in np.flatnonzero(lat == QUERY[0]):
        world.plant(int(i), q)
    idx = world.index()
    monkeypatch.setattr(index_mod, "INITIAL_CANDIDATES", initial)
    for exclude_km, suppress_km in [(0, 0), (500, 400)]:
        res = idx.similar(*QUERY, n=40, exclude_km=exclude_km, suppress_km=suppress_km, align="calendar")
        assert match_cells(res) == brute_force_nms(idx, *QUERY, 40, exclude_km, suppress_km, "calendar")


def test_scan_read_only_and_min_over_alignments(world: World):
    idx = world.index()
    qi = world.pos(*QUERY)
    q = idx.query(qi, "auto")
    d2 = idx.scan(q)
    assert not d2.flags.writeable and not q.reference().flags.writeable and not q.sample_d2.flags.writeable
    assert d2.dtype == q.sample_d2.dtype == idx.d2_at(q, [0, 1]).dtype == idx.pcs.dtype  # no float64 copies
    assert np.all(d2 <= idx.scan(idx.query(qi, "calendar")) + 1e-4)  # auto takes the min over alignments
    np.testing.assert_array_equal(q.reference(), np.sort(idx.d2_at(q, idx.sample)))
    np.testing.assert_allclose(q.sample_d2, d2[idx.sample], rtol=1e-5, atol=1e-6)


def test_only_the_latest_scan_is_kept(world: World):
    idx = world.index()
    for lat, lon in [QUERY, SOUTH, FAR]:
        idx.similar(lat, lon, n=2)
    assert len(idx.scan_cache) == index_mod.SCAN_CACHE_SIZE == 1
    assert len(idx.query_cache) == 3


def test_percentiles_are_relative_to_land_beyond_exclusion(world: World):
    """Excluding the query's surroundings removes them from the reference, so the same
    distance ranks as more exceptional; an exclusion covering everything falls back."""
    idx = world.index()
    qd = idx.query(world.pos(*QUERY))
    far = qd.sample_km >= 1500
    assert 0 < far.sum() < len(far)
    np.testing.assert_array_equal(qd.reference(1500), np.sort(qd.sample_d2[far]))
    d = float(np.median(qd.sample_d2))
    assert qd.top_pct(d, 1500) != qd.top_pct(d, 0)
    np.testing.assert_array_equal(qd.reference(1e9), qd.reference(0))


def test_similar_percentile_uses_exclusion(world: World):
    idx = world.index()
    for ex in (0, 1500):
        res = idx.similar(*QUERY, n=3, exclude_km=ex)
        q = idx.query(world.pos(*QUERY))
        for m in res["matches"]:
            want = 100.0 - float(q.top_pct(idx.scan(q)[m["cell"]["index"]], ex))
            assert m["more_similar_than_pct_of_land"] == pytest.approx(want, abs=1e-3)


def test_similar_identical_with_and_without_cache(world: World):
    world.plant(world.pos(*SOUTH), world.pos(*QUERY), shift_months=6)
    cold, warm = world.index(), world.index()
    for lat, lon in [QUERY, SOUTH]:
        warm.similar(lat, lon, n=5, detail=True)  # fill the cache
    for lat, lon in [QUERY, SOUTH]:
        for kw in [{}, {"align": "calendar"}, {"n": 50, "exclude_km": 0, "suppress_km": 0}]:
            assert warm.similar(lat, lon, detail=True, **kw) == cold.similar(lat, lon, detail=True, **kw)
            cold.query_cache = index_mod.QueryCache(index_mod.QUERY_CACHE_SIZE)
            cold.scan_cache = index_mod.QueryCache(index_mod.SCAN_CACHE_SIZE)


def test_query_cache_lru_eviction():
    cache = index_mod.QueryCache(2)
    calls: list[str] = []

    def compute(k):
        def f():
            calls.append(k)
            return k
        return f

    for k in ["a", "b", "c", "d", "a", "b"]:
        assert cache.get(k, compute(k)) == k
    assert len(cache) == 2
    n = len(calls)
    assert cache.get("b", compute("b")) == "b" and len(calls) == n


def test_query_cache_failed_compute_is_retried():
    cache = index_mod.QueryCache(2)

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cache.get("k", boom)
    assert cache.get("k", lambda: 7) == 7
