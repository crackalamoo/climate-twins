"""Place names from GeoNames: labelling cells and searching by name.

Reads three files from https://download.geonames.org/export/dump/ (see README):
cities500.txt, admin1CodesASCII.txt and countryInfo.txt. Parsed once into a compact
places.npz cache next to them, which is rebuilt whenever a source file is newer.
"""

import bisect
import difflib
import logging
import os
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geo import EARTH_RADIUS_KM, great_circle_km, unit_vectors

log = logging.getLogger(__name__)

CITIES_FILE = "cities500.txt"
ADMIN1_FILE = "admin1CodesASCII.txt"
COUNTRY_FILE = "countryInfo.txt"
CACHE_FILE = "places.npz"
CACHE_VERSION = 4

# Labels only name places people have heard of, and population is the best fame signal
# in GeoNames: a city of CITY_MIN_POPULATION+ within CITY_KM, trading size against
# distance (max population / (km + offset)^power); else just the region it's in.
CITY_MIN_POPULATION = 100_000
CITY_KM = 500.0
CITY_DISTANCE_OFFSET_KM = 15.0
CITY_DISTANCE_POWER = 3
REGION_KM = 500.0  # the region is that of the nearest place of any size within this
NEAR_KM = 25.0  # closer than this to the city, the label is "near <city>"
SPELLING_SIMILARITY = 0.75  # an alternate name this close to the name is another spelling of it
REGION_VOTERS = 10  # a region-only label takes the most common region of this many nearest places

# Labels show a place's admin1 region only where it helps a typical reader: not in
# countries whose regions are obscure outside them (the country alone says as much),
# not when the region just repeats the city's name ("Kabul, Kabul"), and not when it
# is generic ("Central Province"). Search results always show the region.
_AFRICA_EXCEPT_ZA = (
    "AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ GW KE KM LR LS LY MA MG ML "
    "MR MU MW MZ NA NE NG RE RW SC SD SS SH SL SN SO ST SZ TD TG TN TZ UG YT ZM ZW"
)
_REGIONLESS_ELSEWHERE = (
    "SA YE OM IQ SY IR AF JO KW QA BH LB "  # Middle East
    "KZ UZ TM KG TJ MN AZ GE AM "  # Central Asia, Caucasus
    "GL IS NO SE FI BY RO "  # Nordic, Eastern Europe
    "VN TH MM NP BD LA KH LK BT KP TW HK MO "  # South and Southeast Asia
    "VU FJ SB NC PF "  # Pacific islands
    "GT HN SV NI PA CU DO HT "  # Central America, Caribbean
    "HU RS "  # counties and "Central Serbia"
    "CO VE BO PY SR GY"  # South America
)
REGIONLESS_COUNTRIES = frozenset((_AFRICA_EXCEPT_ZA + " " + _REGIONLESS_ELSEWHERE).split())
# Words that only say what kind of division a region is: ignored when comparing with the city.
_ADMIN_WORDS = frozenset(
    "province provincia region regione district oblast oblisi oblys krai kray governorate prefecture "
    "state city county department departamento municipality wilaya muhafazat voivodeship division "
    "capital metropolitan autonomous federal territory of the dc fd grad do si shi".split()
)
# Regions named only by a compass point or "central" say nothing.
_GENERIC_REGION_WORDS = frozenset(
    "north northern south southern east eastern west western central centre center far upper lower "
    "nord sud est ouest centrale capital region province district department".split()
)
EXACT_KM = 5.0  # closer than this, the label is just the place's name
# Historical, abandoned and destroyed places: dropped entirely.
SKIP_FEATURE_CODES = {"PPLH", "PPLQ", "PPLW", "PPLCH"}
# Sections of a populated place (suburbs, localities): searchable, but labels name a town.
NOT_LABEL_FEATURE_CODES = {"PPLX"}
COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_STR_FIELDS = ("name", "asciiname", "admin1", "country")


def normalize(text: str) -> str:
    """Case- and accent-insensitive search key: NFKD, drop combining marks, casefold."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _name_words(text: str) -> list[str]:
    """normalize()d words of a name, punctuation dropped: "Arkhangel'sk" -> ["arkhangelsk"]."""
    kept = "".join(c if c.isalnum() or c.isspace() else (" " if c in "-/,()" else "") for c in normalize(text))
    return ["saint" if w == "st" else w for w in kept.split()]


def region_repeats_city(region: str, name: str, *spellings: str) -> bool:
    """True when the region is just the city's name with an admin word: "Sabha District",
    "Ho Chi Minh City (HCMC)", "Krasnoyarsk Krai", "St.-Petersburg", "Tucuman" for San Miguel
    de Tucumán, "Arkhangelskaya" for Arkhangel'sk; or exactly another spelling of it
    ("Cuzco" for Cusco)."""
    def key(text: str) -> str:
        return " ".join(w for w in _name_words(text) if w not in _ADMIN_WORDS)

    r, c = key(region.split("(")[0]), key(name)
    if not r or not c:
        return False
    if r == c or c.startswith(r + " ") or c.endswith(" " + r):
        return True
    if len(c) >= 5 and r.startswith(c) and len(r) - len(c) <= 3:  # an inflected form: -skaya, -ia
        return True
    return any(key(s) == r for s in spellings)


def region_is_generic(region: str) -> bool:
    """True for regions named only by direction: "Centre", "Northern Region", "North-West"."""
    words = _name_words(region)
    return bool(words) and all(w in _GENERIC_REGION_WORDS for w in words)


def _chord(km: float) -> float:
    """Straight-line distance between unit-sphere points km apart along the surface."""
    return 2.0 * np.sin(min(km / EARTH_RADIUS_KM, np.pi) / 2.0)


def _parse(directory: Path) -> dict[str, np.ndarray]:
    """Parse the GeoNames text files (data only) into column arrays."""
    admin1: dict[str, str] = {}
    for line in (directory / ADMIN1_FILE).read_text(encoding="utf-8").split("\n"):
        f = line.split("\t")
        if len(f) >= 2 and f[0]:
            admin1[f[0]] = f[1]
    countries: dict[str, str] = {}
    for line in (directory / COUNTRY_FILE).read_text(encoding="utf-8").split("\n"):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 5 and f[0]:
            countries[f[0]] = f[4]

    cols: dict[str, list] = {
        k: [] for k in (*_STR_FIELDS, "alt_names", "country_code", "lat", "lon", "population", "labels")
    }
    with open(directory / CITIES_FILE, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 15 or f[7] in SKIP_FEATURE_CODES:
                continue
            try:
                lat, lon, pop = float(f[4]), float(f[5]), int(f[14] or 0)
            except ValueError:
                continue
            cc = f[8]
            cols["name"].append(f[1])
            cols["asciiname"].append(f[2] or f[1])
            cols["admin1"].append(admin1.get(f"{cc}.{f[10]}", ""))
            cols["alt_names"].append("|".join(_spellings(f[1], f[3])) if pop >= CITY_MIN_POPULATION else "")
            cols["country"].append(countries.get(cc, ""))
            cols["country_code"].append(cc)
            cols["lat"].append(lat)
            cols["lon"].append(lon)
            cols["population"].append(pop)
            cols["labels"].append(f[7] not in NOT_LABEL_FEATURE_CODES)
    out = {k: _pack(cols[k]) for k in (*_STR_FIELDS, "alt_names")}
    out["country_code"] = np.array(cols["country_code"], dtype="U2")
    out["lat"] = np.array(cols["lat"], dtype=np.float64)
    out["lon"] = np.array(cols["lon"], dtype=np.float64)
    out["population"] = np.array(cols["population"], dtype=np.int64)
    out["labels"] = np.array(cols["labels"], dtype=bool)
    out["version"] = np.array(CACHE_VERSION)
    return out


def _spellings(name: str, alternate_names: str) -> list[str]:
    """Alternate names that are other spellings of name ("Cuzco" for Cusco, "Banghazi" for
    Benghazi), not other names: GeoNames also lists old names, e.g. "Pernambuco" for Recife."""
    key = normalize(name)
    return [a for a in dict.fromkeys(alternate_names.split(","))
            if a and difflib.SequenceMatcher(None, normalize(a), key).ratio() >= SPELLING_SIMILARITY]


def compass(lat: float, lon: float, from_lat: float, from_lon: float) -> str:
    """8-point compass direction of (lat, lon) as seen from (from_lat, from_lon)."""
    p1, p2 = np.radians(from_lat), np.radians(lat)
    dl = np.radians(lon - from_lon)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    bearing = np.degrees(np.arctan2(y, x)) % 360
    return COMPASS[int((bearing + 22.5) // 45) % 8]


def _pack(strings: list[str]) -> np.ndarray:
    """Strings -> one UTF-8 byte array, newline-separated (none contain newlines)."""
    return np.frombuffer("\n".join(s.replace("\n", " ") for s in strings).encode("utf-8"), dtype=np.uint8)


def _unpack(blob: np.ndarray) -> list[str]:
    return blob.tobytes().decode("utf-8").split("\n")


def _load_columns(directory: Path) -> dict[str, np.ndarray]:
    sources = [directory / f for f in (CITIES_FILE, ADMIN1_FILE, COUNTRY_FILE)]
    cache = directory / CACHE_FILE
    newest = max(p.stat().st_mtime for p in sources)
    if cache.exists() and cache.stat().st_mtime >= newest:
        try:
            with np.load(cache, allow_pickle=False) as z:
                cols = {k: z[k] for k in z.files}
            if int(cols.get("version", -1)) == CACHE_VERSION:
                return cols
        except (OSError, ValueError, KeyError) as e:
            log.warning("ignoring unreadable %s: %s", cache, e)
    cols = _parse(directory)
    try:
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".npz.tmp")
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **cols)
        os.replace(tmp, cache)
    except OSError as e:
        log.warning("could not write %s: %s", cache, e)
    return cols


class Places:
    """GeoNames populated places with a spatial index and a name-prefix index."""

    def __init__(self, cols: dict[str, np.ndarray]):
        self.name, self.asciiname, self.admin1, self.country = (_unpack(cols[k]) for k in _STR_FIELDS)
        self.country_code = cols["country_code"].tolist()
        self.lat, self.lon = cols["lat"], cols["lon"]
        self.population = cols["population"]
        self.labels = cols["labels"]  # usable as a label (not a section of another place)
        self.alt_names = _unpack(cols["alt_names"])  # "|"-separated; only for cities that can label
        n = len(self.lat)
        assert all(
            len(a) == n for a in (self.name, self.asciiname, self.admin1, self.country, self.alt_names, self.country_code)
        )
        self.tree = cKDTree(unit_vectors(self.lat, self.lon))

        # One key per distinct normalized name or ASCII name of each place.
        keys: list[tuple[str, int]] = []
        for i in range(n):
            a, b = normalize(self.name[i]), normalize(self.asciiname[i])
            keys.append((a, i))
            if b != a:
                keys.append((b, i))
        keys.sort()
        self._keys = [k for k, _ in keys]
        self._key_place = np.fromiter((i for _, i in keys), dtype=np.int64, count=len(keys))
        # Population rank of each key's place (stable: ties by geonames file order).
        order = np.argsort(-self.population, kind="stable")
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        self._key_rank = rank[self._key_place]

    @classmethod
    def load(cls, directory: Path) -> "Places | None":
        """Places from a GeoNames directory, or None (with a warning) if files are missing."""
        missing = [f for f in (CITIES_FILE, ADMIN1_FILE, COUNTRY_FILE) if not (directory / f).is_file()]
        if missing:
            log.warning("GeoNames files missing from %s (%s): place names disabled", directory, ", ".join(missing))
            return None
        return cls(_load_columns(directory))

    def __len__(self) -> int:
        return len(self.lat)

    # -- labels --------------------------------------------------------------

    def _within(self, lat: float, lon: float, km: float) -> tuple[np.ndarray, np.ndarray]:
        """Label-eligible places within km of (lat, lon), and their distances."""
        idx = np.asarray(self.tree.query_ball_point(unit_vectors(lat, lon), _chord(km)), dtype=np.int64)
        idx = idx[self.labels[idx]]
        return idx, great_circle_km(lat, lon, self.lat[idx], self.lon[idx])

    def city_for(self, lat: float, lon: float) -> int | None:
        """The city that labels (lat, lon), or None: see CITY_MIN_POPULATION."""
        idx, km = self._within(lat, lon, CITY_KM)
        big = self.population[idx] >= CITY_MIN_POPULATION
        idx, km = idx[big], km[big]
        if idx.size == 0:
            return None
        score = self.population[idx] / (km + CITY_DISTANCE_OFFSET_KM) ** CITY_DISTANCE_POWER
        return int(idx[np.argmax(score)])

    def label_for(self, lat: float, lon: float) -> dict | None:
        """The Place label (see docs/ui-api.md) for a point, or None if nothing is near."""
        i = self.city_for(lat, lon)
        if i is None:
            return self._region_label(lat, lon)
        km = float(great_circle_km(lat, lon, self.lat[i], self.lon[i]))
        place = self.describe(i, for_label=True)
        base, direction = place["base_label"], None
        if km < EXACT_KM:
            tier, label = "exact", base
        elif km < NEAR_KM:
            tier, label = "near", f"near {base}"
        else:
            direction = compass(lat, lon, self.lat[i], self.lon[i])
            tier, label = "offset", f"{km:.0f} km {direction} of {base}"
        # tier/direction/km/base_label let clients rebuild `label` in other units.
        return {**place, "km": round(km, 1), "tier": tier, "direction": direction, "label": label}

    def _region_label(self, lat: float, lon: float) -> dict | None:
        """No city near: name only the region and country, by majority of the nearest places
        (GeoNames has a few miscoded villages, e.g. Tibetan hamlets filed under Hebei)."""
        idx, km = self._within(lat, lon, REGION_KM)
        if idx.size == 0:
            return None
        nearest = idx[np.argsort(km)[:REGION_VOTERS]]
        votes: dict[tuple[str, str], int] = {}
        for i in nearest:  # ties go to the nearer place, since dicts keep insertion order
            key = (self.country_code[i], self.admin1[i])
            votes[key] = votes.get(key, 0) + 1
        cc, admin1 = max(votes, key=votes.get)
        i = next(int(j) for j in nearest if (self.country_code[j], self.admin1[j]) == (cc, admin1))
        country = self.country[i]
        shown = admin1 if self._shows_region(i) else ""
        base = ", ".join(p for p in dict.fromkeys((shown, country)) if p)
        if not base:
            return None
        return {
            "name": shown or country,
            "admin1": admin1 or None,
            "country": country or None,
            "country_code": cc,
            "population": None,
            "lat": None,
            "lon": None,
            "km": None,
            "tier": "region",
            "direction": None,
            "base_label": base,
            "label": f"in {base}",
        }

    def _shows_region(self, i: int, city: str | None = None) -> bool:
        """Whether labels show place i's admin1 region (see REGIONLESS_COUNTRIES)."""
        admin1 = self.admin1[i]
        if not admin1 or self.country_code[i] in REGIONLESS_COUNTRIES or region_is_generic(admin1):
            return False
        if city is None:
            return True
        spellings = (self.asciiname[i], *self.alt_names[i].split("|"))
        return not region_repeats_city(admin1, city, *spellings)

    def describe(self, i: int, for_label: bool = False) -> dict:
        """Place i's fields. Search results always show the region in base_label; labels
        only where it helps (for_label, see _shows_region)."""
        name, admin1, country = self.name[i], self.admin1[i], self.country[i]
        shown = admin1 if not for_label or self._shows_region(i, name) else ""
        parts: list[str] = []
        for p in (name, shown, country):
            if p and p not in parts:
                parts.append(p)
        return {
            "name": name,
            "admin1": admin1 or None,
            "country": country or None,
            "country_code": self.country_code[i],
            "population": int(self.population[i]),
            "lat": round(float(self.lat[i]), 4),
            "lon": round(float(self.lon[i]), 4),
            "base_label": ", ".join(parts),  # "Name, Admin1, Country"
            "label": ", ".join(parts),
        }

    # -- search --------------------------------------------------------------

    def search(self, q: str, limit: int = 8) -> list[dict]:
        """Places whose name or ASCII name starts with q (accent/case-insensitive), most populous first."""
        key = normalize(q.strip())
        if len(key) < 2 or limit <= 0:
            return []
        lo = bisect.bisect_left(self._keys, key)
        hi = bisect.bisect_left(self._keys, key + "\U0010ffff", lo)
        if lo == hi:
            return []
        ranks = self._key_rank[lo:hi]
        # A place has at most two keys, so the best 2*limit keys hold the best `limit` places.
        k = min(2 * limit, len(ranks))
        best = np.argpartition(ranks, k - 1)[:k] if k < len(ranks) else np.arange(len(ranks))
        best = best[np.argsort(ranks[best])]
        out: list[int] = []
        for j in self._key_place[lo + best]:
            if j not in out:
                out.append(int(j))
                if len(out) == limit:
                    break
        return [self.describe(i) for i in out]
