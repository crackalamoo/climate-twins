import numpy as np
import pytest

from climate_twins.geo import EARTH_RADIUS_KM, Grid, great_circle_km

from .conftest import GRID

KM_PER_DEG = 2 * np.pi * EARTH_RADIUS_KM / 360


def test_great_circle_antipodes():
    # arcsin is ill-conditioned near 1, so haversine loses ~0.1 m precision at antipodes.
    assert great_circle_km(0, 0, 0, 180) == pytest.approx(20015.1, abs=0.1)
    assert great_circle_km(10, 20, -10, -160) == pytest.approx(np.pi * EARTH_RADIUS_KM, abs=1e-2)
    assert great_circle_km(90, 0, -90, 0) == pytest.approx(np.pi * EARTH_RADIUS_KM, abs=1e-2)


class TestGrid:
    def test_round_trip_terraclimate_like_grid(self):
        """1/24 degree, latitude descending from the north, from coordinate arrays."""
        lat = 90 - (np.arange(4320) + 0.5) / 24
        lon = -180 + (np.arange(8640) + 0.5) / 24
        g = Grid.from_coords(lat, lon)
        assert g.dlat < 0 and g.dlon > 0 and g.cell_km == pytest.approx(KM_PER_DEG / 24)
        flat = np.array([0, 8639, 8640, 12_345_678, 4320 * 8640 - 1])
        for f, a, b in zip(flat, g.lat_of(flat), g.lon_of(flat)):
            r, c = g.row_col(a, b)
            assert r * g.nlon + c == f
            assert g.row_col(a + 0.4 / 24, b - 0.4 / 24) == (r, c)

    def test_row_col_clips_latitude_and_wraps_longitude(self):
        assert GRID.row_col(90, 0)[0] == 0
        assert GRID.row_col(-90, 0)[0] == GRID.nlat - 1
        assert GRID.row_col(0, -179.9)[1] == 0
        assert GRID.row_col(0, 179.9)[1] == GRID.nlon - 1
        assert GRID.row_col(0, 180.5) == GRID.row_col(0, -179.5)
        assert GRID.row_col(0, 541.0) == GRID.row_col(0, -179.0)

    def test_window_wider_than_globe_has_no_duplicates(self):
        w = GRID.window(row=45, col=7, row_radius=0, col_radius=GRID.nlon)
        assert sorted(w) == list(range(45 * GRID.nlon, 46 * GRID.nlon))
