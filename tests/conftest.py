"""Synthetic climate worlds for testing, small enough to build in milliseconds.

Nothing here touches the real index in data/processed.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import shapefile

from climate_twins.boundaries import BOUNDARIES_FILE
from climate_twins.build import FILE_PATTERN, build
from climate_twins.features import (
    CLEARNESS_BLEND_W_M2,
    MONTHS,
    RAW_VARS,
    RawCodec,
    fit_feature_model,
    perceptual,
    physical,
    toa_radiation,
)
from climate_twins.geo import Grid
from climate_twins.index import ClimateIndex

# 2-degree global grid; cell centres sit on odd degrees (89, 87, ..., -89 and -179, ..., 179).
GRID = Grid(lat0=89.0, dlat=-2.0, nlat=90, lon0=-179.0, dlon=2.0, nlon=180)

# Non-trivial offsets so that tests catch any place that forgets to decode.
CODECS = {
    "tmax": RawCodec(scale=0.1, offset=10.0),
    "tmin": RawCodec(scale=0.1, offset=-10.0),
    "ppt": RawCodec(scale=0.1, offset=0.0),
    "vap": RawCodec(scale=0.001, offset=0.0),
    "srad": RawCodec(scale=0.1, offset=0.0),
    "ws": RawCodec(scale=0.01, offset=0.0),
}

# (lat_min, lat_max, lon_min, lon_max) boxes of land; the last two straddle the dateline.
LAND_BOXES = [
    (30, 60, -120, -70),  # "North America"
    (0, 50, 0, 60),  # "Eurafrica"
    (-40, -10, 120, 150),  # "Australia"
    (-50, -5, -75, -40),  # "South America"
    (50, 60, 170, 180),
    (50, 60, -180, -170),
]


def land_cells(grid: Grid, boxes=LAND_BOXES) -> np.ndarray:
    """Sorted flat indices of grid cells whose centre falls in any box."""
    flat = np.arange(grid.nlat * grid.nlon)
    lat, lon = grid.lat_of(flat), grid.lon_of(flat)
    land = np.zeros(flat.size, dtype=bool)
    for la0, la1, lo0, lo1 in boxes:
        land |= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    return flat[land]


def synthetic_physical(lat: np.ndarray, lon: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Plausible (n, 12) monthly normals in physical units for each (lat, lon).

    Smooth in space (so neighbours resemble each other), with hemisphere-dependent
    seasons and a little per-cell noise so that no two cells are exactly alike.
    """
    rng = np.random.default_rng(seed)
    lat = np.asarray(lat, dtype=np.float64)[:, None]
    lon = np.asarray(lon, dtype=np.float64)[:, None]
    n = lat.shape[0]
    m = np.arange(MONTHS)[None, :]
    season = np.cos(2 * np.pi * (m - 6) / MONTHS) * np.tanh(lat / 10)  # warm in July up north
    cont = 0.5 + 0.5 * np.sin(np.radians(2 * lon))  # 0 maritime .. 1 continental
    noise = lambda scale, shape=(n, MONTHS): rng.normal(0.0, scale, shape)  # noqa: E731

    tmean = 27 - 0.5 * np.abs(lat) + 0.35 * np.abs(lat) * (0.6 + 0.8 * cont) * season + noise(1.0, (n, 1))
    tmean = tmean + noise(0.3)
    dtr = 8 + 6 * cont + noise(0.5, (n, 1))
    tmax = tmean + dtr / 2
    tmin = tmean - dtr / 2
    td = tmin - (1 + 8 * cont) + noise(0.3)
    vap = 0.6112 * np.exp(17.625 * td / (243.04 + td))
    ppt = np.maximum(0.0, 90 * (1.1 - cont) * (1 + 0.7 * season * np.cos(np.radians(lon))) + noise(5.0))
    srad = np.maximum(0.0, 230 - 1.6 * np.abs(lat) + 1.3 * np.abs(lat) * season + noise(3.0))
    ws = np.maximum(0.0, 2.5 + 2 * (1 - cont) + 0.4 * season + noise(0.2))
    return {"tmax": tmax, "tmin": tmin, "ppt": ppt, "vap": vap, "srad": srad, "ws": ws}


def encode(phys: Mapping[str, np.ndarray], codecs: Mapping[str, RawCodec] = CODECS) -> dict[str, np.ndarray]:
    """Physical values -> stored int16, the inverse of RawCodec.decode (to within scale/2)."""
    return {
        v: np.round((phys[v] - codecs[v].offset) / codecs[v].scale).astype(np.int16) for v in RAW_VARS
    }


@dataclass
class World:
    """Land cells of a grid and their raw int16 (n, 12) climate arrays."""

    grid: Grid
    cells: np.ndarray
    raw: dict[str, np.ndarray]
    codecs: dict[str, RawCodec] = field(default_factory=lambda: dict(CODECS))

    def pos(self, lat: float, lon: float) -> int:
        """Position (row in raw / pcs) of the land cell centred at (lat, lon)."""
        r, c = self.grid.row_col(lat, lon)
        flat = r * self.grid.nlon + c
        i = int(np.searchsorted(self.cells, flat))
        assert i < len(self.cells) and self.cells[i] == flat, f"({lat}, {lon}) is not land"
        return i

    def plant(self, dst: int, src: int, shift_months: int = 0) -> None:
        """Give cell `dst` the climate of `src`, optionally rolled in months.

        Solar radiation is rescaled to dst's latitude so its sky clearness (the feature)
        matches src's; everything else is copied exactly."""
        src_lat, dst_lat = self.grid.lat_of(self.cells[[src, dst]])
        row = {v: self.raw[v][src] for v in RAW_VARS}
        for v, values in transplant(row, src_lat, dst_lat, self.codecs, shift_months).items():
            self.raw[v][dst] = values

    def index(self) -> ClimateIndex:
        return make_index(self.grid, self.cells, self.raw, self.codecs)


def transplant(
    row: Mapping[str, np.ndarray], src_lat: float, dst_lat: float, codecs: Mapping[str, RawCodec], shift_months: int = 0
) -> dict[str, np.ndarray]:
    """Stored (12,) values giving a cell at dst_lat the climate of `row` at src_lat.

    Everything is copied (rolled by shift_months) except solar radiation, which is
    solved for so that the clearness feature is identical at the new latitude."""
    out = {v: np.roll(np.asarray(row[v]), shift_months) for v in RAW_VARS}
    src = physical({v: np.asarray(row[v])[None] for v in RAW_VARS}, codecs, np.array([src_lat]))
    k = np.roll(src["clearness"][0], shift_months)
    # Invert clearness() = (srad + b*S/T) / (toa + b), where S = sum(srad), T = sum(toa).
    toa, b = toa_radiation(np.array([dst_lat]))[0], CLEARNESS_BLEND_W_M2
    total = (k * (toa + b)).sum() / (1 + MONTHS * b / toa.sum())
    srad = k * (toa + b) - b * total / toa.sum()
    c = codecs["srad"]
    out["srad"] = np.rint((srad - c.offset) / c.scale).astype(np.asarray(row["srad"]).dtype)
    return out


def make_index(
    grid: Grid, cells: np.ndarray, raw: Mapping[str, np.ndarray], codecs: Mapping[str, RawCodec] = CODECS
) -> ClimateIndex:
    """Fit a model on `raw` and assemble an in-memory ClimateIndex, like build() does."""
    lat = grid.lat_of(cells)
    model = fit_feature_model(raw, codecs, lat)
    pcs = model.project(model.vectors(perceptual(physical(raw, codecs, lat))))
    sample = np.arange(len(cells))
    return ClimateIndex(grid, codecs, model, cells, raw, pcs, sample)


def make_world(seed: int = 0) -> World:
    cells = land_cells(GRID)
    phys = synthetic_physical(GRID.lat_of(cells), GRID.lon_of(cells), seed)
    return World(GRID, cells, encode(phys))


@pytest.fixture
def world() -> World:
    """A fresh (mutable) synthetic world; plant twins in it, then call .index()."""
    return make_world()


@pytest.fixture(scope="session")
def base_index() -> ClimateIndex:
    """Index over the unmodified synthetic world. Treat as read-only."""
    return make_world().index()


# -- TerraClimate-like netCDF files for build() --------------------------------

NC_FILL = -32768
# Coarse 10-degree global grid, latitude descending from the north like TerraClimate.
NC_LAT = np.arange(85.0, -90.0, -10.0)
NC_LON = np.arange(-175.0, 180.0, 10.0)
NC_QUERY = (45.0, -95.0)
NC_TWIN = (25.0, 45.0)
NC_PARTIAL = (35.0, 5.0)  # land, but tmax (the first variable) misses a month -> not indexed
NC_ANTARCTIC = (-75.0, 5.0)  # land, but south of min_lat
NC_MIN_LAT = -60.0


def nc_land_mask() -> np.ndarray:
    lat, lon = np.meshgrid(NC_LAT, NC_LON, indexing="ij")
    return (
        ((lat >= 25) & (lat <= 65) & (lon >= -125) & (lon <= -65))
        | ((lat >= -5) & (lat <= 55) & (lon >= -5) & (lon <= 55))
        | ((lat >= -45) & (lat <= -15) & (lon >= 115) & (lon <= 155))
        | (lat <= -65)
    )


def nc_index(lat: float, lon: float) -> tuple[int, int]:
    return int(np.flatnonzero(NC_LAT == lat)[0]), int(np.flatnonzero(NC_LON == lon)[0])


def write_terraclimate(raw_dir: Path, hole: tuple[str, float, float] | None = None) -> dict[str, np.ndarray]:
    """Write six TerraClimate-style files; return the stored (12, nlat, nlon) int16 arrays.

    `hole` = (var, lat, lon) blanks one month of that variable at that cell (default:
    tmax at NC_PARTIAL).
    """
    hole = hole or ("tmax", *NC_PARTIAL)
    lat2, lon2 = np.meshgrid(NC_LAT, NC_LON, indexing="ij")
    phys = synthetic_physical(lat2.ravel(), lon2.ravel(), seed=1)
    raw = encode(phys)
    land = nc_land_mask()
    qi, ti = (np.ravel_multi_index(nc_index(*p), land.shape) for p in (NC_QUERY, NC_TWIN))
    # Plant a climate twin far from the query.
    twin = transplant({v: raw[v][qi] for v in RAW_VARS}, NC_QUERY[0], NC_TWIN[0], CODECS)
    stored = {}
    for v in RAW_VARS:
        r = raw[v].copy()
        r[ti] = twin[v]
        data = r.T.reshape(MONTHS, len(NC_LAT), len(NC_LON))
        data[:, ~land] = NC_FILL
        if v == hole[0]:
            data[3, *nc_index(*hole[1:])] = NC_FILL
        stored[v] = data
        with netCDF4.Dataset(raw_dir / FILE_PATTERN.format(var=v), "w") as ds:
            ds.createDimension("time", MONTHS)
            ds.createDimension("lat", len(NC_LAT))
            ds.createDimension("lon", len(NC_LON))
            ds.createVariable("lat", "f8", ("lat",))[:] = NC_LAT
            ds.createVariable("lon", "f8", ("lon",))[:] = NC_LON
            var = ds.createVariable(v, "i2", ("time", "lat", "lon"), fill_value=NC_FILL)
            var.scale_factor = CODECS[v].scale
            var.add_offset = CODECS[v].offset
            var.set_auto_maskandscale(False)
            var[:] = np.ascontiguousarray(data)
    return stored


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> tuple[Path, dict[str, np.ndarray]]:
    """(index directory written by build(), stored netCDF arrays it was built from)."""
    root = tmp_path_factory.mktemp("terraclimate")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    stored = write_terraclimate(raw_dir)
    out = root / "processed"
    build(raw_dir, out, NC_MIN_LAT)
    return out, stored


# -- GeoNames-style place files ---------------------------------------------------

# (name, asciiname, lat, lon, country code, admin1 code, population, feature code)
GEONAMES_ROWS = [
    ("São Paulo", "Sao Paulo", -23.55, -46.63, "BR", "27", 12_400_232, "PPLA"),
    ("São Luís", "Sao Luis", -2.53, -44.30, "BR", "13", 917_237, "PPLA"),
    ("Santos", "Santos", -23.96, -46.33, "BR", "27", 433_966, "PPL"),
    ("Saõ Ghost", "Sao Ghost", -23.50, -46.60, "BR", "27", 99_999_999, "PPLQ"),  # abandoned: skipped
    ("Porto", "Porto", 41.15, -8.61, "PT", "17", 249_633, "PPLA"),
    ("Porto Alegre", "Porto Alegre", -30.03, -51.23, "BR", "23", 1_488_252, "PPLA"),
    ("Zürich", "Zurich", 47.37, 8.55, "CH", "ZH", 341_730, "PPLA"),
    # A cluster for the label rule, around (45, -95) = NC_QUERY.
    ("Smallville", "Smallville", 45.05, -95.0, "US", "MN", 1_000, "PPL"),  # ~5.6 km N: too small
    ("Midtown", "Midtown", 45.0, -95.2, "US", "MN", 300_000, "PPL"),  # ~15.7 km W
    ("Bigcity", "Bigcity", 45.0, -96.0, "US", "MN", 2_000_000, "PPL"),  # ~79 km W: bigger, farther
    ("Midtown Heights", "Midtown Heights", 45.01, -95.01, "US", "MN", 5_000_000, "PPLX"),  # a section: no label
    # For trading size against distance, around (10, 20): no city within 25 km.
    ("Hamlet", "Hamlet", 10.36, 20.0, "TD", "", 600, "PPL"),  # ~40 km N: too small
    ("Market Town", "Market Town", 9.2, 20.0, "TD", "", 200_000, "PPL"),  # ~89 km S
    ("Distant City", "Distant City", 11.26, 20.0, "TD", "", 600_000, "PPL"),  # ~140 km N: bigger, farther
    ("Metropolis", "Metropolis", 5.4, 20.0, "TD", "", 5_000_000, "PPL"),  # ~511 km S: too far
    # No city near NC_TWIN (25, 45): labels name the region.
    ("Oasis", "Oasis", 25.5, 45.0, "SA", "", 800, "PPL"),  # ~56 km N, no admin1
    ("Faraway", "Faraway", 26.5, 45.0, "SA", "10", 90_000, "PPL"),  # ~167 km N, just too small
    # Which regions labels show.
    ("Krasnoyarsk", "Krasnoyarsk", 56.0, 92.9, "RU", "91", 1_090_000, "PPLA"),  # region repeats the city
    ("Ubari", "Ubari", 26.6, 12.8, "LY", "34", 120_000, "PPL"),  # Libya: regions not shown
    # Remote Tibet, for the region vote: the nearest village is miscoded as Hebei.
    ("Wrongcode", "Wrongcode", 33.02, 87.0, "CN", "10", 50, "PPL"),
    *((f"Tibet {k}", f"Tibet {k}", 33.05 + 0.05 * k, 87.0, "CN", "14", 50, "PPL") for k in range(6)),
    # Across a sea from a big foreign city, around (15, 30): the villages there say ER.
    *((f"Coast {k}", f"Coast {k}", 15.0 + 0.05 * k, 30.0, "ER", "", 700, "PPL") for k in range(5)),
    ("Portcity", "Portcity", 15.0, 31.0, "SA", "10", 3_000_000, "PPL"),  # ~107 km E: labels it, foreign
    # A sparsely mapped country (SS, north of lat 4 in BOUNDARY_POLYGONS) beside a densely mapped
    # one (UG, south of it): the towns nearest (4.3, 32) are mostly across the border.
    ("Northtown", "Northtown", 5.5, 32.0, "SS", "01", 400_000, "PPL"),
    *((f"Border {k}", f"Border {k}", 3.9 - 0.05 * k, 32.0, "UG", "N", 800, "PPL") for k in range(8)),
    # A region split after Natural Earth drew it: most of the old Almaty Region's towns are now in Zhetysu.
    *((f"Almaty {k}", f"Almaty {k}", 44.1 + 0.1 * k, 78.0, "KZ", "10", 900, "PPL") for k in range(2)),
    *((f"Zhetysu {k}", f"Zhetysu {k}", 44.6 + 0.1 * k, 78.0, "KZ", "17", 900, "PPL") for k in range(3)),
    # Search by name and region or country.
    ("Portland", "Portland", 45.52, -122.68, "US", "OR", 652_503, "PPLA2"),
    ("Portland", "Portland", 43.66, -70.25, "US", "ME", 68_408, "PPLA2"),
    ("Portland", "Portland", 50.57, -2.44, "GB", "ENG", 12_710, "PPL"),
    ("Paris", "Paris", 48.85, 2.35, "FR", "11", 2_138_551, "PPLC"),
    ("Paris", "Paris", 33.66, -95.56, "US", "TX", 24_782, "PPLA2"),
    ("Parisot", "Parisot", 44.26, 1.86, "FR", "76", 500, "PPL"),
    ("New York City", "New York City", 40.71, -74.01, "US", "NY", 8_804_190, "PPL"),
    ("Bridgetown", "Bridgetown", 13.1, -59.62, "BB", "08", 98_511, "PPLC"),
    ("Cambridge", "Cambridge", 42.37, -71.11, "US", "MA", 118_403, "PPL"),
    ("Cambridge", "Cambridge", 38.56, -76.08, "US", "MD", 500_000, "PPL"),  # not really this big
    # Regions people abbreviate by ISO 3166-2 code, where GeoNames' codes are numbers.
    ("Vancouver", "Vancouver", 49.25, -123.12, "CA", "02", 662_248, "PPL"),
    ("Vancouver", "Vancouver", 45.64, -122.66, "US", "WA", 196_442, "PPL"),
    ("Sydney", "Sydney", -33.87, 151.21, "AU", "02", 5_638_830, "PPLA"),
    ("Sydney", "Sydney", 46.14, -60.19, "CA", "07", 105_968, "PPL"),
    # A region's code (WA, GT-SA) against the start of another region's name (Waikato, Santiago Metropolitan).
    ("Hamilton", "Hamilton", -37.78, 175.28, "NZ", "E8", 192_100, "PPL"),
    ("Hamilton Hill", "Hamilton Hill", -32.08, 115.78, "AU", "08", 11_327, "PPL"),
    ("Santiago", "Santiago", -33.45, -70.67, "CL", "12", 4_837_295, "PPLC"),
    ("Santiago Sacatepéquez", "Santiago Sacatepequez", 14.63, -90.68, "GT", "17", 40_000, "PPL"),
    # Ranking: population, with whole own names boosted.
    ("Bordeaux", "Bordeaux", 44.84, -0.58, "FR", "75", 265_328, "PPLA"),  # also "Porto" in GeoNames
    ("Par", "Par", 50.35, -4.70, "GB", "ENG", 9_462, "PPL"),
    ("Pa", "Pa", 11.55, -3.26, "BF", "", 15_170, "PPL"),
    ("Chongqing", "Chongqing", 29.56, 106.55, "CN", "32", 22_000_000, "PPLA"),  # junk alternate name "Pa"
    ("Mexico City", "Mexico City", 19.43, -99.13, "MX", "09", 12_294_193, "PPLC"),
    ("Mexico", "Mexico", 15.07, 120.72, "PH", "03", 43_694, "PPL"),
]
# Alternate names beyond the name and ASCII name, by name (search only uses them for cities
# that can label). Portland's "Bridgetown" is a nickname.
GEONAMES_ALIASES = {
    "New York City": "NYC,Big Apple,NY", "Portland": "PDX,Bridgetown", "Bordeaux": "Porto",
    "Chongqing": "Pa,Chungking", "Mexico City": "Mexico,CDMX",
}
GEONAMES_ADMIN1 = {
    "BR.27": "São Paulo", "BR.13": "Maranhão", "BR.23": "Rio Grande do Sul", "PT.17": "Porto",
    "CH.ZH": "Zurich", "US.MN": "Minnesota", "SA.10": "Riyadh Region",
    "RU.91": "Krasnoyarsk Krai", "LY.34": "Sabha District", "CN.10": "Hebei", "CN.14": "Tibet",
    "SS.01": "Central Equatoria", "UG.N": "Northern Region",
    "NO.21": "Trøndelag", "KZ.10": "Almaty Region", "KZ.17": "Zhetysu Region",
    "US.OR": "Oregon", "US.ME": "Maine", "US.TX": "Texas", "US.NY": "New York", "US.MA": "Massachusetts", "US.MD": "Maryland", "GB.ENG": "England",
    "FR.11": "Île-de-France", "FR.76": "Occitanie", "BB.08": "Saint Michael",
    "CA.02": "British Columbia", "CA.07": "Nova Scotia", "AU.01": "Australian Capital Territory",
    "AU.02": "New South Wales", "US.WA": "Washington", "FR.75": "Nouvelle-Aquitaine", "CN.32": "Chongqing",
    "MX.09": "Mexico City", "PH.03": "Central Luzon", "NZ.E8": "Waikato", "AU.08": "Western Australia",
    "CL.12": "Santiago Metropolitan", "GT.17": "Sacatepéquez",
}
GEONAMES_COUNTRIES = {
    "BR": "Brazil", "PT": "Portugal", "CH": "Switzerland", "US": "United States", "SA": "Saudi Arabia", "TD": "Chad",
    "RU": "Russia", "LY": "Libya", "CN": "China", "ER": "Eritrea", "SS": "South Sudan", "UG": "Uganda",
    "NO": "Norway", "KZ": "Kazakhstan", "GB": "United Kingdom", "FR": "France", "BB": "Barbados",
    "CA": "Canada", "AU": "Australia", "BF": "Burkina Faso", "MX": "Mexico", "PH": "Philippines",
    "NZ": "New Zealand", "CL": "Chile", "GT": "Guatemala",
}
# ISO 3166 alpha-3 codes and internet domains, where search tests need them (else "XXX" and none).
GEONAMES_COUNTRY_CODES = {"US": ("USA", ".us"), "GB": ("GBR", ".uk"), "FR": ("FRA", ".fr")}


def write_geonames(directory: Path, rows=GEONAMES_ROWS) -> Path:
    """Write tiny cities500.txt / admin1CodesASCII.txt / countryInfo.txt files."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for gid, (name, ascii_, lat, lon, cc, a1, pop, code) in enumerate(rows, 1):
        alt = ",".join(filter(None, (name, ascii_, GEONAMES_ALIASES.get(name))))
        f = [str(gid), name, ascii_, alt, str(lat), str(lon), "P", code, cc, "", a1,
             "", "", "", str(pop), "", "100", "Etc/UTC", "2024-01-01"]
        lines.append("\t".join(f))
    (directory / "cities500.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "admin1CodesASCII.txt").write_text(
        "".join(f"{k}\t{v}\t{v}\t1\n" for k, v in GEONAMES_ADMIN1.items()), encoding="utf-8"
    )
    (directory / "countryInfo.txt").write_text(
        "# GeoNames country info\n#ISO\tISO3\tISO-Numeric\tfips\tCountry\tCapital\tArea\tPopulation\tContinent\ttld\n"
        + "".join(
            f"{cc}\t{iso3}\t000\t{cc}\t{name}\tCapital\t1\t1\tXX\t{tld}\n"
            for cc, name in GEONAMES_COUNTRIES.items()
            for iso3, tld in [GEONAMES_COUNTRY_CODES.get(cc, ("XXX", ""))]
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture(scope="session")
def geonames_dir(tmp_path_factory) -> Path:
    return write_geonames(tmp_path_factory.mktemp("geonames"))


def _box(lon0: float, lat0: float, lon1: float, lat1: float) -> list[tuple[float, float]]:
    """A clockwise exterior ring of (lon, lat)."""
    return [(lon0, lat0), (lon0, lat1), (lon1, lat1), (lon1, lat0), (lon0, lat0)]


# Natural Earth admin-1 polygons: (iso_a2, gn_a1_code, name, name_en, exterior ring[, iso_3166_2]).
BOUNDARY_POLYGONS = [
    ("US", "US.MN", "Minnesota", "Minnesota", _box(-97, 43, -93, 47)),
    ("LY", "LY.34", "Sabha", "Sabha", _box(12, 25, 15, 28)),
    ("SA", "SA.10", "Ar Riyad", "Riyadh", _box(43, 23, 47, 27)),  # not GeoNames' name: code unchecked
    ("CN", "CN.14", "Xizang", "Tibet", _box(86, 32, 88, 34)),
    ("ER", "-1", "Gash Barka", "Gash Barka", _box(29.5, 14.5, 30.5, 15.5)),
    # SS north of lat 4, UG south: the towns nearest (4.3, 32) are mostly across the border.
    ("SS", "SS.01", "Central Equatoria", "Central Equatoria", _box(31, 4, 33, 6)),
    ("UG", "-1", "Gulu", "Gulu", _box(31, 2, 33, 4)),
    ("NO", "NO.21", "Svalbard", "Svalbard", _box(10, 78, 20, 80)),  # a stale code: NO.21 is now Trøndelag
    ("KZ", "KZ.10", "Almaty", "Almaty", _box(77, 44, 79, 45)),
    # ISO 3166-2 codes, which search takes as region abbreviations where they're consistent.
    ("CA", "CA.02", "British Columbia", "British Columbia", _box(-125, 48, -120, 51), "CA-BC"),
    ("CA", "CA.07", "Nova Scotia", "Nova Scotia", _box(-62, 45, -59, 47), "CA-NS"),
    ("AU", "AU.02", "New South Wales", "New South Wales", _box(150, -35, 152, -32), "AU-NSW"),
    ("AU", "AU.02", "Lord Howe Island", "Lord Howe Island", _box(159, -32, 159.2, -31.4), "AU-NSW"),  # agrees
    ("AU", "AU.", "Jervis Bay Territory", "", _box(140, -36, 141, -35), "AU-X02~"),  # placeholders
    ("AU", "AU.08", "Western Australia", "Western Australia", _box(113, -35, 129, -14), "AU-WA"),
    # Consistent, but Guatemala is in REGIONLESS_COUNTRIES: no abbreviation.
    ("GT", "GT.17", "Sacatepéquez", "Sacatepéquez", _box(-90.9, 14.4, -90.5, 14.8), "GT-SA"),
]


def write_boundaries(directory: Path, polygons=BOUNDARY_POLYGONS) -> Path:
    """Write a tiny ne_10m_admin_1_states_provinces shapefile."""
    directory.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(directory / BOUNDARIES_FILE, shapeType=shapefile.POLYGON) as w:
        w.field("iso_a2", "C", size=4)
        w.field("gn_a1_code", "C", size=10)
        w.field("name", "C", size=40)
        w.field("name_en", "C", size=40)
        w.field("iso_3166_2", "C", size=10)
        for iso_a2, gn_a1_code, name, name_en, ring, *iso in polygons:
            w.poly([ring])
            w.record(iso_a2, gn_a1_code, name, name_en, iso[0] if iso else "-99")
    return directory


@pytest.fixture(scope="session")
def boundaries_dir(tmp_path_factory) -> Path:
    return write_boundaries(tmp_path_factory.mktemp("naturalearth"))
