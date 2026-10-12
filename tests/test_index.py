import itertools
import threading
import time

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
    """Reference: float64 distances to every cell, then greedy suppression over all of them."""
    qi, _ = idx.locate(lat, lon)
    qvec = idx.vectors(qi)[0]
    variants = [qvec, shift_seasons(qvec)] if align == "auto" else [qvec]
    qp = idx.model.project(np.stack(variants)).astype(np.float64)
    pcs = idx.pcs.astype(np.float64)
    d2 = np.min([((pcs - q) ** 2).sum(axis=1) for q in qp], axis=0)
    picked: list[int] = []
    for i in np.argsort(d2, kind="stable"):
        if len(picked) == n:
            break
        if great_circle_km(lat, lon, idx.lat[i], idx.lon[i]) < exclude_km:
            continue
        if any(great_circle_km(idx.lat[i], idx.lon[i], idx.lat[j], idx.lon[j]) < suppress_km for j in picked):
            continue
        picked.append(int(i))
    return picked


# -- locate ---------------------------------------------------------------------


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
            j, km = base_index.locate(float(base_index.lat[i]), float(base_index.lon[i]))
            assert j == i and km == pytest.approx(0.0, abs=1e-6)

    def test_point_inside_cell_snaps_to_it(self, base_index: ClimateIndex):
        i, km = base_index.locate(45.15, -98.9)
        assert (base_index.lat[i], base_index.lon[i]) == QUERY
        assert 0 < km < 25

    def test_offshore_snaps_to_nearest_coast(self, base_index: ClimateIndex):
        # 61 N is ocean on the 2-degree world; 59 N is the northern coast of "North America".
        with pytest.raises(NoLandError):
            base_index.locate(61.0, -99.0)
        i, km = base_index.locate(61.0, -99.0, max_km=300)
        assert (base_index.lat[i], base_index.lon[i]) == (59.0, -99.0)
        assert km == pytest.approx(2 * 111.195, rel=1e-3)

    def test_fine_grid_offshore_snaps_to_nearby_land(self, fine_index: ClimateIndex):
        north = fine_index.lat.max()
        lon = float(fine_index.lon[np.argmax(fine_index.lat)])
        i, km = fine_index.locate(north + 0.15, lon)  # ~3.6 cells north of the island
        assert fine_index.lat[i] == north and fine_index.lon[i] == lon
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


# -- similar ---------------------------------------------------------------------


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


@pytest.mark.parametrize("suppress_km", [0.0, 400.0, 1500.0])
def test_suppress_km_spacing_and_sorted(base_index: ClimateIndex, suppress_km: float):
    res = base_index.similar(*QUERY, n=30, exclude_km=500, suppress_km=suppress_km)
    assert len(res["matches"]) == 30
    ll = match_latlon(res)
    for a, b in itertools.combinations(range(len(ll)), 2):
        assert great_circle_km(*ll[a], *ll[b]) >= suppress_km - 1e-6
    d = [m["distance"] for m in res["matches"]]
    assert d == sorted(d)
    assert [m["rank"] for m in res["matches"]] == list(range(1, 31))
    pct = [m["more_similar_than_pct_of_land"] for m in res["matches"]]
    assert pct == sorted(pct, reverse=True)


def test_southern_twin_needs_season_shift(world: World):
    s, q = world.pos(*SOUTH), world.pos(*QUERY)
    world.plant(s, q, shift_months=6)
    idx = world.index()
    s_cell = s

    auto = idx.similar(*QUERY, n=10, align="auto")
    top = auto["matches"][0]
    assert top["cell"]["index"] == s_cell
    assert top["seasons_shifted_6_months"] is True
    assert top["distance"] == pytest.approx(0.0, abs=1e-3)

    cal = idx.similar(*QUERY, n=10, align="calendar")
    assert cal["params"]["align"] == "calendar"
    assert not any(m["seasons_shifted_6_months"] for m in cal["matches"])
    cells = match_cells(cal)
    assert cells[0] != s_cell
    if s_cell in cells:
        assert cal["matches"][cells.index(s_cell)]["distance"] > cal["matches"][0]["distance"]


def test_fewer_results_than_requested(base_index: ClimateIndex):
    res = base_index.similar(*QUERY, n=200, exclude_km=500, suppress_km=3000)
    assert 0 < len(res["matches"]) < 200
    ll = match_latlon(res)
    for a, b in itertools.combinations(range(len(ll)), 2):
        assert great_circle_km(*ll[a], *ll[b]) >= 3000
    assert base_index.similar(*QUERY, n=5, exclude_km=20_000)["matches"] == []


@pytest.mark.parametrize("initial", [1, 4, 64, index_mod.INITIAL_CANDIDATES])
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


def test_difference_share(world: World):
    idx = world.index()
    res = idx.similar(*QUERY, n=10)
    for m in res["matches"]:
        share = m["difference_share"]
        assert list(share) == list(FEATURES)
        assert all(0 <= x <= 1 for x in share.values())
        assert sum(share.values()) == pytest.approx(1.0, abs=0.01)

    # A twin differing only in precipitation attributes everything to ppt.
    t, q = world.pos(*FAR), world.pos(*QUERY)
    world.plant(t, q)
    world.raw["ppt"][t] = world.raw["ppt"][q] + 30  # +3 mm every month
    share = world.index().similar(*QUERY, n=1)["matches"][0]
    assert share["cell"]["lat"] == FAR[0]
    assert share["difference_share"]["ppt"] == pytest.approx(1.0)


def test_describe_decodes_raw(base_index: ClimateIndex):
    d = base_index.describe(*QUERY)
    pos = d["cell"]["index"]
    tmax = base_index.raw["tmax"][pos] * 0.1 + 10.0
    np.testing.assert_allclose(d["monthly"]["tmax"], np.round(tmax, 1))
    assert d["summary"]["warmest_month_high_c"] == pytest.approx(tmax.max(), abs=0.05)


@pytest.mark.parametrize("initial", [1, 4, index_mod.INITIAL_CANDIDATES])
def test_pick_with_exact_ties_equals_brute_force(monkeypatch, world: World, initial: int):
    """Many cells with identical climate (d2 ties) are visited in index order, as brute force does."""
    q = world.pos(*QUERY)
    lat, lon = world.grid.lat_of(world.cells), world.grid.lon_of(world.cells)
    # Same latitude as the query, so planted cells are bit-identical (exact d2 ties).
    for i in np.flatnonzero(lat == QUERY[0]):
        world.plant(int(i), q)
    idx = world.index()
    monkeypatch.setattr(index_mod, "INITIAL_CANDIDATES", initial)
    for exclude_km, suppress_km in [(0, 0), (500, 400), (500, 0)]:
        res = idx.similar(*QUERY, n=40, exclude_km=exclude_km, suppress_km=suppress_km, align="calendar")
        assert match_cells(res) == brute_force_nms(idx, *QUERY, 40, exclude_km, suppress_km, "calendar")


# -- query cache ---------------------------------------------------------------------


def test_distances_read_only_and_min_over_alignments(world: World):
    idx = world.index()
    qi = world.pos(*QUERY)
    a = idx.distances(qi, "auto")
    assert not a.d2.flags.writeable and not a.reference().flags.writeable
    assert np.all(a.d2 <= idx.distances(qi, "calendar").d2 + 1e-4)  # auto takes the min over alignments
    np.testing.assert_array_equal(a.reference(), np.sort(a.d2[idx.sample]))


def test_percentiles_are_relative_to_land_beyond_exclusion(world: World):
    """Excluding the query's surroundings removes them from the reference, so the same
    distance ranks as more exceptional; an exclusion covering everything falls back."""
    idx = world.index()
    qd = idx.distances(world.pos(*QUERY))
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
        qd = idx.distances(world.pos(*QUERY))
        for m in res["matches"]:
            want = 100.0 - float(qd.top_pct(qd.d2[m["cell"]["index"]], ex))
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


def test_query_cache_concurrent_misses_compute_once():
    cache = index_mod.QueryCache(4)
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.05)
        return object()

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get("k", slow))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert all(r is results[0] for r in results)


def test_query_cache_failed_compute_is_retried():
    cache = index_mod.QueryCache(2)

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cache.get("k", boom)
    assert cache.get("k", lambda: 7) == 7
