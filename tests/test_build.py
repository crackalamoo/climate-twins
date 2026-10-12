"""End-to-end: TerraClimate-style netCDF -> build() -> ClimateIndex.load -> query."""

from pathlib import Path

import numpy as np
import pytest

from climate_twins.features import RAW_VARS
from climate_twins.geo import great_circle_km
from climate_twins.index import ClimateIndex

from .conftest import (
    NC_ANTARCTIC,
    NC_LAT,
    NC_LON,
    NC_MIN_LAT,
    NC_PARTIAL,
    NC_QUERY,
    NC_TWIN,
    nc_index,
    nc_land_mask,
)


@pytest.fixture(scope="module")
def loaded(built: tuple[Path, dict]) -> ClimateIndex:
    return ClimateIndex.load(built[0])


def test_grid_and_land_cells(loaded: ClimateIndex):
    g = loaded.grid
    assert (g.nlat, g.nlon) == (len(NC_LAT), len(NC_LON))
    assert g.lat0 == 85.0 and g.dlat == -10.0 and g.lon0 == -175.0 and g.dlon == 10.0

    land = nc_land_mask().ravel()
    expected = set(np.flatnonzero(land).tolist())
    expected.discard(int(np.ravel_multi_index(nc_index(*NC_PARTIAL), (g.nlat, g.nlon))))
    expected = {c for c in expected if g.lat_of(c) >= NC_MIN_LAT}
    assert np.all(np.diff(loaded.cells) > 0)
    assert set(loaded.cells.tolist()) == expected
    assert len(loaded) == loaded.meta["n_cells"] == len(expected)
    assert loaded.pcs.shape == (len(expected), loaded.model.k)
    assert np.all(np.isin(loaded.sample, np.arange(len(loaded))))

    # Fill in one month of one variable drops the cell; south of min_lat is dropped.
    for p in (NC_PARTIAL, NC_ANTARCTIC):
        with pytest.raises(LookupError):
            loaded.locate(*p)


def test_raw_values_round_trip(built: tuple[Path, dict], loaded: ClimateIndex):
    _, stored = built
    i, km = loaded.locate(*NC_QUERY)
    assert km == pytest.approx(0.0, abs=1e-6)
    r, c = nc_index(*NC_QUERY)
    for v in RAW_VARS:
        np.testing.assert_array_equal(loaded.raw[v][i], stored[v][:, r, c])
    phys = loaded.physical(i)
    np.testing.assert_allclose(phys["tmax"][0], stored["tmax"][:, r, c] * 0.1 + 10.0)


def test_pcs_match_in_memory_projection(loaded: ClimateIndex):
    idx = np.arange(len(loaded))
    np.testing.assert_allclose(loaded.model.project(loaded.vectors(idx)), loaded.pcs, atol=1e-5)


def test_query_finds_planted_twin(loaded: ClimateIndex):
    res = loaded.similar(*NC_QUERY, n=5)
    top = res["matches"][0]
    assert (top["cell"]["lat"], top["cell"]["lon"]) == NC_TWIN
    assert top["distance"] == pytest.approx(0.0, abs=1e-3)
    assert top["km_from_query"] == pytest.approx(round(float(great_circle_km(*NC_QUERY, *NC_TWIN))))
    assert 1 <= len(res["matches"]) <= 5


def test_rebuild_replaces_existing_index(tmp_path: Path, built: tuple[Path, dict]):
    """build() into an existing directory swaps it atomically and leaves no leftovers."""
    from climate_twins.build import build

    from .conftest import write_terraclimate

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_terraclimate(raw_dir)
    out = tmp_path / "processed"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    build(raw_dir, out, NC_MIN_LAT)
    assert not (out / "stale.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed", "raw"]
    np.testing.assert_array_equal(np.load(out / "cells.npy"), np.load(built[0] / "cells.npy"))


def test_land_cell_missing_in_later_variable_is_an_error(tmp_path: Path):
    """The land mask comes from the first variable; holes in later ones abort the build."""
    from climate_twins.build import build

    from .conftest import write_terraclimate

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_terraclimate(raw_dir, hole=("srad", *NC_QUERY))
    with pytest.raises(ValueError, match="srad: 1 land cells are missing data"):
        build(raw_dir, tmp_path / "processed", NC_MIN_LAT)
    assert not (tmp_path / "processed").exists()
