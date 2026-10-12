import numpy as np
import pytest

from climate_twins.geo import EARTH_RADIUS_KM, Grid, great_circle_km

from .conftest import GRID

KM_PER_DEG = 2 * np.pi * EARTH_RADIUS_KM / 360  # ~111.195


class TestGreatCircle:
    def test_one_degree_of_longitude_shrinks_with_cos_lat(self):
        assert great_circle_km(60, 0, 60, 1) == pytest.approx(KM_PER_DEG * 0.5, rel=1e-3)

    def test_antipodes(self):
        assert great_circle_km(0, 0, 0, 180) == pytest.approx(20015.1, abs=0.1)
        # arcsin is ill-conditioned near 1, so haversine loses ~0.1 m precision at antipodes.
        assert great_circle_km(10, 20, -10, -160) == pytest.approx(np.pi * EARTH_RADIUS_KM, abs=1e-2)
        assert great_circle_km(90, 0, -90, 0) == pytest.approx(np.pi * EARTH_RADIUS_KM, abs=1e-2)

    def test_across_dateline(self):
        assert great_circle_km(0, 179.5, 0, -179.5) == pytest.approx(KM_PER_DEG, rel=1e-9)
        assert great_circle_km(0, 0, 0, 360) == pytest.approx(0.0, abs=1e-6)


class TestGrid:
    def test_round_trip_all_cells(self):
        flat = np.arange(GRID.nlat * GRID.nlon)
        lat, lon = GRID.lat_of(flat), GRID.lon_of(flat)
        back = np.array([r * GRID.nlon + c for r, c in (GRID.row_col(a, b) for a, b in zip(lat, lon))])
        np.testing.assert_array_equal(back, flat)

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
            # Nudging within the cell still lands on it.
            assert g.row_col(a + 0.4 / 24, b - 0.4 / 24) == (r, c)

    def test_row_col_clips_latitude_and_wraps_longitude(self):
        assert GRID.row_col(90, 0)[0] == 0
        assert GRID.row_col(-90, 0)[0] == GRID.nlat - 1
        assert GRID.row_col(0, -179.9)[1] == 0
        assert GRID.row_col(0, 179.9)[1] == GRID.nlon - 1
        assert GRID.row_col(0, 180.5) == GRID.row_col(0, -179.5)
        assert GRID.row_col(0, 541.0) == GRID.row_col(0, -179.0)

    def test_window_wraps_at_dateline(self):
        w = GRID.window(row=10, col=0, row_radius=1, col_radius=2)
        rows, cols = np.divmod(w, GRID.nlon)
        assert set(rows) == {9, 10, 11}
        assert set(cols) == {178, 179, 0, 1, 2}
        assert len(w) == len(set(w)) == 15
        lon = GRID.lon_of(w)
        assert np.all(great_circle_km(GRID.lat_of(10 * GRID.nlon), -179.0, GRID.lat_of(w), lon) < 600)

    def test_window_wider_than_globe_has_no_duplicates(self):
        w = GRID.window(row=45, col=7, row_radius=0, col_radius=GRID.nlon)
        assert sorted(w) == list(range(45 * GRID.nlon, 46 * GRID.nlon))

    def test_window_clips_rows_at_poles(self):
        w = GRID.window(row=0, col=5, row_radius=2, col_radius=0)
        np.testing.assert_array_equal(w, [5, 5 + GRID.nlon, 5 + 2 * GRID.nlon])

