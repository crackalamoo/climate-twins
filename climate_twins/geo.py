"""Great-circle geometry and the regular lat/lon grid TerraClimate is stored on."""

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance in km; broadcasts over array arguments."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def unit_vectors(lat, lon) -> np.ndarray:
    """(..., 3) points on the unit sphere; dot products give cos(central angle)."""
    p, l = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)], axis=-1)


def cos_of_km(km: float) -> float:
    """cos of the central angle spanning km; points >= km apart have dot <= this."""
    return float(np.cos(min(km / EARTH_RADIUS_KM, np.pi)))


@dataclass(frozen=True)
class Grid:
    """A global regular grid; row 0 is `lat0`, column 0 is `lon0`, longitudes wrap."""

    lat0: float
    dlat: float
    nlat: int
    lon0: float
    dlon: float
    nlon: int

    @classmethod
    def from_coords(cls, lat: np.ndarray, lon: np.ndarray) -> "Grid":
        return cls(
            lat0=float(lat[0]),
            dlat=float((lat[-1] - lat[0]) / (len(lat) - 1)),
            nlat=len(lat),
            lon0=float(lon[0]),
            dlon=float((lon[-1] - lon[0]) / (len(lon) - 1)),
            nlon=len(lon),
        )

    @property
    def cell_km(self) -> float:
        return abs(self.dlat) * np.pi / 180 * EARTH_RADIUS_KM

    def row_col(self, lat: float, lon: float) -> tuple[int, int]:
        row = int(np.clip(round((lat - self.lat0) / self.dlat), 0, self.nlat - 1))
        col = int(round((lon - self.lon0) / self.dlon)) % self.nlon
        return row, col

    def flat_of(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Flat index of the cell containing each point, or -1 beyond the grid's latitudes."""
        row = np.rint((np.asarray(lat, dtype=np.float64) - self.lat0) / self.dlat).astype(np.int64)
        col = np.rint((np.asarray(lon, dtype=np.float64) - self.lon0) / self.dlon).astype(np.int64) % self.nlon
        return np.where((row >= 0) & (row < self.nlat), row * self.nlon + col, -1)

    def lat_of(self, flat: np.ndarray) -> np.ndarray:
        return self.lat0 + (np.asarray(flat) // self.nlon) * self.dlat

    def lon_of(self, flat: np.ndarray) -> np.ndarray:
        return self.lon0 + (np.asarray(flat) % self.nlon) * self.dlon

    def window(self, row: int, col: int, row_radius: int, col_radius: int) -> np.ndarray:
        """Flat indices of the rectangle around (row, col), wrapping in longitude."""
        rows = np.arange(max(row - row_radius, 0), min(row + row_radius, self.nlat - 1) + 1)
        cols = np.unique(np.arange(col - col_radius, col + col_radius + 1) % self.nlon)
        return (rows[:, None] * self.nlon + cols[None, :]).ravel()

    def to_dict(self) -> dict:
        return dict(self.__dict__)
