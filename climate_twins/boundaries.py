"""Which country and region a point is in, from Natural Earth's admin-1 (state/province) polygons.

Reads ne_10m_admin_1_states_provinces.shp (and its .shx/.dbf) from
https://www.naturalearthdata.com/ (see README). Place labels use it to say where a point
itself is: nearby towns can't, where one side of a border is far more densely mapped than
the other (GeoNames has few towns in South Sudan and many in northern Uganda).

The polygons say where a point is; places.py names each one in GeoNames' terms, checking
Natural Earth's codes against GeoNames before using them: its divisions often differ from
GeoNames' (UK councils, Italian provinces) and some of its GeoNames admin1 codes are stale
(Svalbard as NO.21, now Trøndelag).
"""

import logging
import re
from pathlib import Path

import numpy as np
import shapefile
import shapely

log = logging.getLogger(__name__)

BOUNDARIES_FILE = "ne_10m_admin_1_states_provinces.shp"
# A point outside every polygon but this close to one (in degrees, ~25 km at the equator) is
# taken to be in it: a coastal cell's centre can fall just offshore of the coastline.
SNAP_DEG = 0.25
# The polygon's names, for checking its GeoNames admin1 code against the region GeoNames gives it.
NAME_FIELDS = ("name", "name_en", "name_alt")  # not gn_name: it is derived from the code
# An ISO 3166-2 region code: the country and up to three letters or digits ("CA-BC", "AU-NSW").
# Natural Earth also has placeholders ("AU-X02~", "-99"), which don't match.
ISO_3166_2 = re.compile(r"[A-Z]{2}-[A-Z0-9]{1,3}")


def _country_code(iso_a2: str, gn_a1_code: str) -> str | None:
    """The polygon's ISO country code. Its GeoNames admin1 code's prefix comes first, so a
    disputed region gets the country GeoNames files its towns under; then Natural Earth's own
    code; None for the few areas with neither (Somaliland, sovereign base areas)."""
    for code in (gn_a1_code.split(".")[0] if "." in gn_a1_code else "", iso_a2):
        if len(code) == 2 and code.isalpha():
            return code.upper()
    return None


def _points(lat, lon) -> np.ndarray:
    return shapely.points(((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0, lat)


class Boundaries:
    """Admin-1 polygons, each with Natural Earth's country code (None where it has none), its
    GeoNames admin1 code ("US.CA"; None where it has none), its names, and its ISO 3166-2 code
    ("US-CA"; None where it has none), which search uses as an abbreviation of the region."""

    def __init__(
        self,
        polygons: list,
        country_codes: list[str | None],
        admin1_codes: list[str | None],
        names: list[tuple[str, ...]],
        iso_codes: list[str | None] | None = None,
    ):
        iso_codes = iso_codes if iso_codes is not None else [None] * len(polygons)
        assert len(polygons) == len(country_codes) == len(admin1_codes) == len(names) == len(iso_codes)
        self.country_codes = country_codes
        self.admin1_codes = admin1_codes
        self.names = names
        self.iso_codes = iso_codes
        self.tree = shapely.STRtree(polygons)

    def __len__(self) -> int:
        return len(self.country_codes)

    @classmethod
    def load(cls, directory: Path) -> "Boundaries | None":
        """Boundaries from a Natural Earth directory, or None (with a warning) if missing."""
        path = directory / BOUNDARIES_FILE
        if not path.is_file():
            log.warning("%s missing: place labels won't say which country or region a point is in", path)
            return None
        polygons, country_codes, admin1_codes, names, iso_codes = [], [], [], [], []
        with shapefile.Reader(path) as reader:
            fields = {f[0] for f in reader.fields[1:]}
            name_fields = [f for f in NAME_FIELDS if f in fields]
            iso_field = ["iso_3166_2"] if "iso_3166_2" in fields else []
            for sr in reader.iterShapeRecords(fields=["iso_a2", "gn_a1_code", *iso_field, *name_fields]):
                if not sr.shape.points:
                    continue
                iso_a2, gn_a1_code = sr.record["iso_a2"] or "", sr.record["gn_a1_code"] or ""
                polygons.append(shapely.geometry.shape(sr.shape.__geo_interface__))
                country_codes.append(_country_code(iso_a2, gn_a1_code))
                admin1_codes.append(gn_a1_code if "." in gn_a1_code else None)
                spelled = (n for f in name_fields for n in (sr.record[f] or "").split("|"))
                names.append(tuple(dict.fromkeys(n.strip() for n in spelled if n.strip())))
                iso = (sr.record["iso_3166_2"] or "").strip() if iso_field else ""
                iso_codes.append(iso if ISO_3166_2.fullmatch(iso) else None)
        return cls(polygons, country_codes, admin1_codes, names, iso_codes)

    def polygon_at(self, lat: float, lon: float) -> int | None:
        """The polygon (lat, lon) is in, else the nearest within SNAP_DEG, else None (open sea)."""
        point = _points(lat, lon)
        hits = self.tree.query(point, predicate="within")
        if hits.size == 0:
            hits = self.tree.query_nearest(point, max_distance=SNAP_DEG)
        # Overlapping polygons (rare slivers) or equidistant ones: a deterministic choice.
        return int(np.min(hits)) if hits.size else None

    def polygons_containing(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """For each point, the polygon it's in (-1 if none; no snapping)."""
        none = len(self)
        out = np.full(len(lat), none, dtype=np.int64)
        points, polygons = self.tree.query(_points(lat, lon), predicate="within")
        np.minimum.at(out, points, polygons)  # in two polygons: the lower index, as in polygon_at
        out[out == none] = -1
        return out
