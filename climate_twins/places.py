"""Place names from GeoNames: labelling cells and searching by name.

Reads three files from https://download.geonames.org/export/dump/ (see README):
cities500.txt, admin1CodesASCII.txt and countryInfo.txt. Parsed once into a compact
places.npz cache next to them, which is rebuilt whenever a source file is newer.
Where a point is (its country and region) comes from boundary polygons (boundaries.py).
Search matches a place's name, optionally followed by its region and/or country
("Portland, Oregon", "Paris France", "Portland OR, USA"), as GeoNames names and codes them,
plus regions' ISO 3166-2 codes from the boundary data where GeoNames' are numbers ("Vancouver, BC").
"""

import bisect
import difflib
import logging
import os
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import glyphs
from .boundaries import Boundaries
from .geo import EARTH_RADIUS_KM, great_circle_km, unit_vectors

log = logging.getLogger(__name__)

CITIES_FILE = "cities500.txt"
ADMIN1_FILE = "admin1CodesASCII.txt"
COUNTRY_FILE = "countryInfo.txt"
CACHE_FILE = "places.npz"
CACHE_VERSION = 8

# Labels only name places people have heard of, and population is the best fame signal
# in GeoNames: a city of CITY_MIN_POPULATION+ within CITY_KM, trading size against
# distance (max population / (km + offset)^power); else just the region it's in.
CITY_MIN_POPULATION = 100_000
CITY_KM = 500.0
CITY_DISTANCE_OFFSET_KM = 15.0
CITY_DISTANCE_POWER = 3
NEAR_KM = 25.0  # closer than this to the city, the label is "near <city>"
# Search also finds cities that can label (CITY_MIN_POPULATION+) by all their GeoNames alternate
# names: other languages, former names, abbreviations ("NYC"), airport codes ("PDX").
SPELLING_SIMILARITY = 0.75  # an alternate name this close to the name is another spelling of it
# Search ranks by population, but a place whose own name is the whole name typed ranks as if this many
# times as populous: a subjective choice. Big enough that "Porto" puts Porto (250k) above Porto Alegre
# (1.4M); small enough that while typing "Par", Paris (2.1M) is above Par, England (9k).
WHOLE_NAME_BOOST = 10
# A match by alternate name ranks as if this many times less populous: also subjective. GeoNames
# gives cities other places' names (Bordeaux is also "Porto", Perth "Partha", which would beat Paris
# at "Par"). Small enough that Mexico City (12M) by "Mexico" is above every place called Mexico.
ALIAS_DISCOUNT = 10
# Alternate names match only once the name typed is this long: two letters are too early to guess one,
# and GeoNames has junk short ones (Chongqing as "Pa").
ALIAS_MIN_CHARS = 3
# A place whose region or country the qualifier names whole (all of its name, or a code) ranks as if
# this many times as populous as one it names only in part: also subjective. Big enough that
# "Cambridge, MA" puts Cambridge, Massachusetts (110k) above Cambridge, Maryland (13k); small enough
# that a code doesn't beat a region name still being typed: "Santiago, Sa" is Santiago, Chile
# (Santiago Metropolitan), not Santiago Sacatepéquez, Guatemala (code SA). Hence a boost rather than
# putting whole qualifiers first outright.
WHOLE_QUALIFIER_BOOST = 10
# A qualifier typed as capitals of this many letters is a code, not the start of a name (unless it
# names nothing whole): "Valencia, CA" is California or Canada, not Carabobo, which would win on
# population. Lowercase "ca" could be either; one letter is too early to tell.
CODE_LETTERS = (2, 3)
# Weight of a match by alternate name, by a place's own name in part, and by its whole own name.
_MATCH_WEIGHTS = np.array([1, ALIAS_DISCOUNT, ALIAS_DISCOUNT * WHOLE_NAME_BOOST], dtype=np.int64)
# Naming boundary polygons (Places._polygon_areas): the towns inside one name its country, or
# a region it has no usable code for, only if there are at least this many and every one agrees.
POLYGON_MIN_TOWNS = 3

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
_STR_FIELDS = ("name", "asciiname", "country")


# Letters NFKD doesn't decompose into a base letter and a mark ("Bakı", "Łódź").
_UNDECOMPOSED = str.maketrans({"ı": "i", "ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ħ": "h"})


def normalize(text: str) -> str:
    """Case- and accent-insensitive search key: casefold, NFKD, drop combining marks. Idempotent."""
    decomposed = unicodedata.normalize("NFKD", text.casefold().translate(_UNDECOMPOSED))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _plain_words(text: str) -> list[str]:
    """normalize()d words of a name, punctuation dropped: "Arkhangel'sk" -> ["arkhangelsk"]."""
    return "".join(c if c.isalnum() or c.isspace() else (" " if c in "-/,()" else "") for c in normalize(text)).split()


def _name_words(text: str) -> list[str]:
    """_plain_words, with "St" as "Saint"."""
    return ["saint" if w == "st" else w for w in _plain_words(text)]


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


def _unanimous(values: list[str]) -> str | None:
    """The value all of values are, if there are POLYGON_MIN_TOWNS+ of them."""
    return values[0] if len(values) >= POLYGON_MIN_TOWNS and len(set(values)) == 1 else None


def _region_key(name: str) -> str:
    """A region's name for comparison: "Astrakhan'" and "Astrakhan Oblast", "Sakha (Yakutia)"
    and "Sakha", "Bakı" and "Baki" are the same."""
    return " ".join(w for w in _name_words(name.split("(")[0]) if w not in _ADMIN_WORDS)


def _words(text: str) -> str:
    """A name as space-separated normalize()d words, for comparing search qualifiers. Not "St" as
    "Saint", since a qualifier's last word may be unfinished ("United St")."""
    return " ".join(_plain_words(text))


def _prefixed(keys: list[str], prefix: str) -> slice:
    """The range of sorted keys that start with prefix."""
    lo = bisect.bisect_left(keys, prefix)
    return slice(lo, bisect.bisect_left(keys, prefix + "\U0010ffff", lo))


def _readings(text: str) -> Iterator[tuple[str, list[str]]]:
    """Ways to read a normalize()d search as a name and then qualifier words, split at a space or
    comma: "portland oregon" is the name "portland oregon", or "portland" then ["oregon"]."""
    yield text.rstrip(" ,"), []
    for i in range(1, len(text)):
        if text[i - 1] in " ," and text[i] not in " ,":
            name, words = text[:i].rstrip(" ,"), _plain_words(text[i:])
            if len(name) >= 2 and words:
                yield name, words


class _Qualifiers:
    """What can follow a place's name in a search: its region (admin1), its country, or both.
    A region by its GeoNames name or admin1 code ("Oregon", "OR"; "England", "ENG"), or by an
    abbreviation (Places._region_abbreviations: "BC" for GeoNames' CA.02, British Columbia). A
    country by name, ISO 3166 alpha-2 or alpha-3 code, or its internet domain where no other
    country shares it ("United States", "US", "USA"; "United Kingdom", "GB", "GBR", "UK").
    A code can name several regions (GeoNames' "OR" is Oregon and regions elsewhere): it names
    each of them, and the place's name and population decide."""

    def __init__(self, admin1_codes: list[str], admin1_names: list[str], country_codes: list[str],
                 country_names: list[str], iso3: list[str], tlds: list[str],
                 abbreviations: dict[str, str] | None = None):
        country = {cc: j for j, cc in enumerate(country_codes)}
        self.region_country = [country.get(code.split(".")[0], -1) for code in admin1_codes]
        self.region_keys, self.region_ids = self._sorted(admin1_names)
        self.region_codes: dict[str, list[int]] = {}
        for r, code in enumerate(admin1_codes):
            self.region_codes.setdefault(code.split(".", 1)[-1].casefold(), []).append(r)
        # An abbreviation adds a code for its region, unless GeoNames already gives that code to a region
        # of the same country: then it's that region's own code (US-OR is Oregon's "OR" anyway), or
        # another region's, which it must keep.
        region_index = {code: r for r, code in enumerate(admin1_codes)}
        own = {(self.region_country[r], code) for code, rs in self.region_codes.items() for r in rs}
        for code, abbreviation in (abbreviations or {}).items():
            r, key = region_index.get(code), abbreviation.casefold()
            if r is not None and (self.region_country[r], key) not in own:
                self.region_codes.setdefault(key, []).append(r)
        self.country_keys, self.country_ids = self._sorted(country_names)
        self.country_codes: dict[str, set[int]] = {}
        shared = Counter(tlds)
        for j, codes in enumerate(zip(country_codes, iso3, tlds)):
            for k, code in enumerate(codes):
                if code and (k < 2 or shared[code] == 1):
                    self.country_codes.setdefault(code.casefold(), set()).add(j)

    @staticmethod
    def _sorted(names: Iterable[str]) -> tuple[list[str], list[int]]:
        keyed = sorted((_words(n), j) for j, n in enumerate(names))
        return [k for k, _ in keyed], [j for _, j in keyed]

    @staticmethod
    def _named(keys: list[str], ids: list[int], codes: dict, text: str) -> dict[int, bool]:
        """id -> whether text is all of its name or a code (True), or the start of its name (False)."""
        span = _prefixed(keys, text)
        whole = bisect.bisect_right(keys, text, span.start, span.stop)
        named = dict.fromkeys(ids[whole:span.stop], False)
        named.update(dict.fromkeys([*ids[span.start:whole], *codes.get(text, ())], True))
        return named

    def match(self, words: list[str]) -> tuple[dict[int, bool], dict[int, bool]]:
        """The regions (admin1 indices) and countries (country indices) that qualifier words name, a
        region, a country, or a region then its country, each -> whether the words name it whole.
        The last name may be unfinished ("Ore", typed so far): names match by prefix, codes only whole."""
        text = " ".join(words)
        regions = self._named(self.region_keys, self.region_ids, self.region_codes, text)
        countries = self._named(self.country_keys, self.country_ids, self.country_codes, text)
        for s in range(1, len(words)):
            if not (named := self._named(self.country_keys, self.country_ids, self.country_codes, " ".join(words[s:]))):
                continue
            region = self._named(self.region_keys, self.region_ids, self.region_codes, " ".join(words[:s]))
            for r, whole in region.items():
                if whole and self.region_country[r] in named:
                    regions[r] = regions.get(r, False) or named[self.region_country[r]]
        return regions, countries


def _chord(km: float) -> float:
    """Straight-line distance between unit-sphere points km apart along the surface."""
    return 2.0 * np.sin(min(km / EARTH_RADIUS_KM, np.pi) / 2.0)


def _parse(directory: Path) -> dict[str, np.ndarray]:
    """Parse the GeoNames text files (data only) into column arrays."""
    admin1: dict[str, str] = {}
    admin1_ascii: list[str] = []
    for line in (directory / ADMIN1_FILE).read_text(encoding="utf-8").split("\n"):
        f = line.split("\t")
        if len(f) >= 2 and f[0]:
            admin1[f[0]] = f[1]
            admin1_ascii.append(f[2] if len(f) > 2 and f[2] else f[1])
    admin1_index = {code: j for j, code in enumerate(admin1)}
    countries: dict[str, str] = {}
    iso3: list[str] = []
    tlds: list[str] = []
    for line in (directory / COUNTRY_FILE).read_text(encoding="utf-8").split("\n"):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 5 and f[0]:
            countries[f[0]] = f[4]
            iso3.append(f[1])
            tlds.append(f[9].lstrip(".") if len(f) > 9 else "")

    cols: dict[str, list] = {
        k: [] for k in (*_STR_FIELDS, "alt_names", "aliases", "admin1_index", "country_code", "lat", "lon",
                        "population", "labels")
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
            cols["admin1_index"].append(admin1_index.get(f"{cc}.{f[10]}", -1))
            big = pop >= CITY_MIN_POPULATION
            cols["alt_names"].append("|".join(_spellings(f[1], f[3])) if big else "")
            cols["aliases"].append("|".join(a for a in dict.fromkeys(f[3].split(",")) if a) if big else "")
            cols["country"].append(countries.get(cc, ""))
            cols["country_code"].append(cc)
            cols["lat"].append(lat)
            cols["lon"].append(lon)
            cols["population"].append(pop)
            cols["labels"].append(f[7] not in NOT_LABEL_FEATURE_CODES)
    out = {k: _pack(cols[k]) for k in (*_STR_FIELDS, "alt_names", "aliases")}
    # Name tables by code, for the codes boundary polygons give ("US.CA" -> "California", "US").
    out["admin1_codes"], out["admin1_names"] = _pack(list(admin1)), _pack(list(admin1.values()))
    out["admin1_ascii_names"] = _pack(admin1_ascii)
    out["country_codes"], out["country_names"] = _pack(list(countries)), _pack(list(countries.values()))
    out["country_iso3"], out["country_tlds"] = _pack(iso3), _pack(tlds)  # "USA", "us"; for search
    out["admin1_index"] = np.array(cols["admin1_index"], dtype=np.int32)  # into admin1_codes, -1 if none
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

    def __init__(self, cols: dict[str, np.ndarray], boundaries: Boundaries | None = None):
        self.name, self.asciiname, self.country = (_unpack(cols[k]) for k in _STR_FIELDS)
        self.admin1_index = cols["admin1_index"]  # into admin1_codes, -1 if none
        # A name with a letter the site's type lacks is shown by its ASCII name (glyphs.py): search
        # still finds it by either, as both normalize alike.
        self.name = [n if glyphs.covers(n) else a for n, a in zip(self.name, self.asciiname)]
        admin1_names = [n if glyphs.covers(n) else a
                        for n, a in zip(_unpack(cols["admin1_names"]), _unpack(cols["admin1_ascii_names"]))]
        self.admin1 = [admin1_names[j] if j >= 0 else "" for j in self.admin1_index]
        self.admin1_by_code = dict(zip(_unpack(cols["admin1_codes"]), admin1_names))
        self.boundaries = boundaries
        self.country_by_code = dict(zip(_unpack(cols["country_codes"]), _unpack(cols["country_names"])))
        self.country_code = cols["country_code"].tolist()
        self.lat, self.lon = cols["lat"], cols["lon"]
        self.population = cols["population"]
        self.labels = cols["labels"]  # usable as a label (not a section of another place)
        self.alt_names = _unpack(cols["alt_names"])  # "|"-separated; only for cities that can label
        country_codes = _unpack(cols["country_codes"])
        country_index = {cc: j for j, cc in enumerate(country_codes)}
        self.country_index = np.array([country_index.get(cc, -1) for cc in self.country_code], dtype=np.int16)
        self.qualifiers = _Qualifiers(
            _unpack(cols["admin1_codes"]), _unpack(cols["admin1_names"]), country_codes,
            _unpack(cols["country_names"]), _unpack(cols["country_iso3"]), _unpack(cols["country_tlds"]),
            self._region_abbreviations() if boundaries is not None else None,
        )
        n = len(self.lat)
        assert all(
            len(a) == n for a in (self.name, self.asciiname, self.admin1, self.country, self.alt_names, self.country_code)
        )
        self.tree = cKDTree(unit_vectors(self.lat, self.lon))
        self.polygon_areas = self._polygon_areas() if boundaries is not None else []

        # One key per distinct normalized name or ASCII name of each place, and per other alternate
        # name of a city that can label (`aliases`, flagged as such).
        keys: list[tuple[str, bool, int]] = []
        for i, aliases in enumerate(_unpack(cols["aliases"])):
            own = {normalize(self.name[i]), normalize(self.asciiname[i])}
            keys += [(k, False, i) for k in own]
            if aliases:
                keys += [(k, True, i) for k in {normalize(a) for a in aliases.split("|")} - own]
        keys.sort()
        self._keys = [k for k, _, _ in keys]
        self._key_alias = np.fromiter((a for _, a, _ in keys), dtype=bool, count=len(keys))
        self._key_place = np.fromiter((i for _, _, i in keys), dtype=np.int32, count=len(keys))
        # Population rank of each place (stable: ties by geonames file order).
        self._rank = np.empty(n, dtype=np.int32)
        self._rank[np.argsort(-self.population, kind="stable")] = np.arange(n)

    @classmethod
    def load(cls, directory: Path, boundaries: Boundaries | None = None) -> "Places | None":
        """Places from a GeoNames directory, or None (with a warning) if files are missing.
        Labels only say which country and region a point is in (`area`) with boundaries."""
        missing = [f for f in (CITIES_FILE, ADMIN1_FILE, COUNTRY_FILE) if not (directory / f).is_file()]
        if missing:
            log.warning("GeoNames files missing from %s (%s): place names disabled", directory, ", ".join(missing))
            return None
        return cls(_load_columns(directory), boundaries)

    def __len__(self) -> int:
        return len(self.lat)

    # -- labels --------------------------------------------------------------

    def _within(self, lat: float, lon: float, km: float) -> tuple[np.ndarray, np.ndarray]:
        """Label-eligible places within km of (lat, lon), and their distances."""
        idx = np.asarray(self.tree.query_ball_point(unit_vectors(lat, lon), _chord(km)), dtype=np.int64)
        idx = idx[self.labels[idx]]
        return idx, great_circle_km(lat, lon, self.lat[idx], self.lon[idx])

    def city_for(self, lat: float, lon: float) -> int | None:
        """The city that labels (lat, lon), or None: see CITY_MIN_POPULATION. It can be across
        a border; `area` in the label then says where the point itself is."""
        idx, km = self._within(lat, lon, CITY_KM)
        big = self.population[idx] >= CITY_MIN_POPULATION
        idx, km = idx[big], km[big]
        if idx.size == 0:
            return None
        score = self.population[idx] / (km + CITY_DISTANCE_OFFSET_KM) ** CITY_DISTANCE_POWER
        return int(idx[np.argmax(score)])

    def big_cities(self, lat: float, lon: float, min_km: float, max_km: float, min_population: int,
                   metro_km: float) -> tuple[np.ndarray, np.ndarray]:
        """Label-eligible cities of min_population+ between min_km and max_km of (lat, lon), and their
        distances; but not one within metro_km of a more populous such city, as part of its metro
        area (GeoNames lists New York's boroughs as cities: the Bronx folds into New York)."""
        idx, km = self._within(lat, lon, max_km)
        keep = (self.population[idx] >= min_population) & (km >= min_km)
        idx, km = idx[keep], km[keep]
        part = np.zeros(idx.size, dtype=bool)
        for j, i in enumerate(idx):  # a handful of cities: one neighbourhood lookup each
            near, _ = self._within(self.lat[i], self.lon[i], metro_km)
            part[j] = bool(np.any(self.population[near] > self.population[i]))
        return idx[~part], km[~part]

    def _polygon_areas(self) -> list[tuple[str, str] | None]:
        """Each boundary polygon's (country code, GeoNames admin1 region or ""), or None if its
        country is unknown. Only names two sources agree on, never a majority of towns:
        - Country: the one all of its GeoNames towns are filed under (POLYGON_MIN_TOWNS+ of
          them), so places are named as GeoNames names them (Western Sahara, French Guiana);
          else Natural Earth's.
        - Region: its GeoNames admin1 code, if GeoNames gives that code the polygon's own name
          (a stale code names some other region: Svalbard's NO.21 is now Trøndelag). Else the
          one all of its towns in that country are in (POLYGON_MIN_TOWNS+; towns GeoNames gives
          no region don't count); else "" (labels show the country).
        Natural Earth predates some reforms, so a few areas get their former region (Papua for
        what's now Central Papua), never one they weren't in."""
        b = self.boundaries
        polygon = b.polygons_containing(self.lat, self.lon)
        towns: dict[int, list[int]] = {}
        for i in np.flatnonzero((polygon >= 0) & self.labels):
            towns.setdefault(int(polygon[i]), []).append(int(i))
        areas: list[tuple[str, str] | None] = []
        for j in range(len(b)):
            inside = towns.get(j, [])
            cc = _unanimous([self.country_code[i] for i in inside]) or b.country_codes[j]
            if cc is None:
                areas.append(None)
                continue
            coded = self._admin1_named(b.admin1_codes[j], cc, b.names[j])
            regions = [self.admin1[i] for i in inside if self.country_code[i] == cc and self.admin1[i]]
            region = coded if coded is not None else _unanimous(regions) or ""
            areas.append((cc, region))
        return areas

    def _region_abbreviations(self) -> dict[str, str]:
        """GeoNames admin1 code -> the abbreviation people write for that region, from the ISO 3166-2
        codes of its boundary polygons: "CA.02" -> "BC" (CA-BC). GeoNames' own codes for many
        countries are only numbers, so without these "Vancouver, BC" finds nothing. Not Natural
        Earth's `postal` field: it has NSW as "NS" and Queensland as "QL".
        A code is used only where it's surely that region's:
        - the ISO code is in the GeoNames code's country, and some polygon with that GeoNames code is
          named as GeoNames names the region (_admin1_named);
        - every polygon with that GeoNames code has the same ISO code (UK and Italian polygons are
          councils and provinces, many to one GeoNames region, each with its own ISO code);
        - it has 2+ characters, one a letter: one letter would match while a name is still being
          typed ("Springfield, I"), and numeric ones ("JP-13") clash with GeoNames' numeric codes.
        None for REGIONLESS_COUNTRIES: nobody types their codes (Guatemala's "SA"), which would only
        match a region name still being typed ("Santiago, Sa"). Their region names and GeoNames'
        own codes still match."""
        b = self.boundaries
        found: dict[str, set[str]] = {}
        trusted: set[str] = set()
        for code, iso, names in zip(b.admin1_codes, b.iso_codes, b.names):
            if code is None or iso is None:
                continue
            cc, abbreviation = iso.split("-")
            if code.startswith(cc + ".") and cc not in REGIONLESS_COUNTRIES:
                found.setdefault(code, set()).add(abbreviation)
                if self._admin1_named(code, cc, names) is not None:
                    trusted.add(code)
        abbreviations: dict[str, str] = {}
        for code in trusted:
            if len(found[code]) == 1:
                (a,) = found[code]
                if len(a) >= 2 and not a.isdigit():
                    abbreviations[code] = a
        return abbreviations

    def _admin1_named(self, code: str | None, cc: str, names: tuple[str, ...]) -> str | None:
        """GeoNames' region for admin1 code, if it's in country cc and is one of names
        (ignoring words like "Province"): a stale code names some other region."""
        region = self.admin1_by_code.get(code or "")
        if region is None or not code.startswith(cc + "."):
            return None
        return region if _region_key(region) in {_region_key(n) for n in names} - {""} else None

    def locate(self, lat: float, lon: float) -> tuple[str, str] | None:
        """(country code, admin1 region name or "") of where (lat, lon) is: those of the boundary
        polygon it's in (_polygon_areas). None without one (open sea, no boundary data)."""
        j = self.boundaries.polygon_at(lat, lon) if self.boundaries is not None else None
        return self.polygon_areas[j] if j is not None else None

    def _area(self, here: tuple[str, str] | None) -> dict | None:
        """The region and country from locate, the region only where labels show it."""
        if here is None:
            return None
        cc, admin1 = here
        shown = admin1 if self._shows_region(cc, admin1) else ""
        return {"admin1": shown or None, "country": self.country_by_code.get(cc) or None}

    def label_for(self, lat: float, lon: float) -> dict | None:
        """The Place label (see docs/api.md) for a point, or None if nothing is near."""
        here = self.locate(lat, lon)
        i = self.city_for(lat, lon)
        if i is None:
            return self._region_label(here)
        km = float(great_circle_km(lat, lon, self.lat[i], self.lon[i]))
        place = self.describe(i, for_label=True)
        area = self._area(here)
        # Across a border the city is only a landmark: its own region and country would say the
        # point is somewhere it isn't, so the label names the city alone and then where the point is.
        in_area = self._in_area(i, here, area)
        base, direction = (self.name[i] if in_area else place["base_label"]), None
        where = f", in {in_area}" if in_area else ""
        if km < EXACT_KM:
            # "Mannheim" alone would put Ludwigshafen, across the Rhine, in the wrong region.
            tier, label = "exact", f"near {base}{where}" if in_area else base
        elif km < NEAR_KM:
            tier, label = "near", f"near {base}{where}"
        else:
            direction = compass(lat, lon, self.lat[i], self.lon[i])
            tier, label = "offset", f"{km:.0f} km {direction} of {base}{where}"
        # tier/direction/km/base_label/in_area let clients rebuild `label` in other units.
        return {**place, "km": round(km, 1), "tier": tier, "direction": direction, "base_label": base,
                "in_area": in_area, "label": label, "area": area}

    def _in_area(self, i: int, here: tuple[str, str] | None, area: dict | None) -> str | None:
        """Where the point is, as label text ("Castille and León, Spain"; "France" where its region
        is unknown or not shown), when city i's label would otherwise name a region or country the
        point isn't in; else None. Only what the label claims can conflict: a region it hides can't,
        so Santiago's hidden region conflicts only with a point region labels would show (O'Higgins).
        Unknown is never a conflict (no area, or either region unknown in the same country): stale
        over wrong."""
        if here is None or area is None:
            return None
        cc, admin1 = here
        if cc == self.country_code[i]:
            if not admin1 or not self.admin1[i] or admin1 == self.admin1[i]:
                return None
            if not area["admin1"] and not self._label_region(i):
                return None  # neither the label nor the point's area names a region (Sweden)
        return ", ".join(p for p in (area["admin1"], area["country"]) if p) or None

    def _region_label(self, here: tuple[str, str] | None) -> dict | None:
        """No city near: name only the region and country the point is in (from locate)."""
        if here is None:
            return None
        cc, admin1 = here
        country = self.country_by_code.get(cc, "")
        shown = admin1 if self._shows_region(cc, admin1) else ""
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
            "in_area": None,  # the label is already where the point is
            "label": f"in {base}",
            "area": self._area(here),
        }

    def _shows_region(self, cc: str, admin1: str, city: str | None = None, spellings: tuple[str, ...] = ()) -> bool:
        """Whether labels show an admin1 region of country cc (see REGIONLESS_COUNTRIES): with a
        city, also not when the region repeats the city's name (or one of its spellings)."""
        if not admin1 or cc in REGIONLESS_COUNTRIES or region_is_generic(admin1):
            return False
        return city is None or not region_repeats_city(admin1, city, *spellings)

    def _label_region(self, i: int) -> str:
        """Place i's region as its label shows it, or "" where labels hide it (_shows_region)."""
        spellings = (self.asciiname[i], *self.alt_names[i].split("|"))
        return self.admin1[i] if self._shows_region(self.country_code[i], self.admin1[i], self.name[i], spellings) else ""

    def describe(self, i: int, for_label: bool = False) -> dict:
        """Place i's fields. Search results always show the region in base_label; labels
        only where it helps (for_label, see _shows_region)."""
        name, admin1, country = self.name[i], self.admin1[i], self.country[i]
        shown = self._label_region(i) if for_label else admin1
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
        """Places whose name (or ASCII or alternate name) starts with q, or with q's first words when
        the rest name the place's region and/or country (_Qualifiers): "Portland, Oregon", "Paris
        France". Accent- and case-insensitive. Ranked by population, scaled by WHOLE_NAME_BOOST,
        ALIAS_DISCOUNT and WHOLE_QUALIFIER_BOOST."""
        text = normalize(q).strip()
        if len(text.rstrip(" ,")) < 2 or limit <= 0:
            return []
        lo, hi = CODE_LETTERS
        codes = {normalize(w) for w in q.replace(",", " ").split() if lo <= len(w) <= hi and w.isalpha() and w.isupper()}
        places, boosts, levels = [], [], []
        for name, qualifier in _readings(text):
            span = _prefixed(self._keys, name)
            keys = np.arange(span.start, span.stop)
            if len(name) < ALIAS_MIN_CHARS:
                keys = keys[~self._key_alias[keys]]
            whole = keys < bisect.bisect_right(self._keys, name, span.start, span.stop)
            # Once the name typed is some place's whole name, an alternate name that goes on from it
            # with more words is another name altogether: Jakarta's nickname "New York Van Java" isn't
            # a New York. One it ends partway through a word of is still being typed: "Colo" (a town
            # in Iowa) finds Köln by "Cologne".
            if whole.any():
                more_words = np.array([self._keys[k][len(name)] == " " for k in keys[~whole]], dtype=bool)
                keep = whole | ~self._key_alias[keys]
                keep[~whole] |= ~more_words
                keys, whole = keys[keep], whole[keep]
            level = np.where(self._key_alias[keys], 0, np.where(whole, 2, 1))  # into _MATCH_WEIGHTS
            place = self._key_place[keys]
            boost = np.ones(keys.size, dtype=np.int64)  # WHOLE_QUALIFIER_BOOST where the qualifier names it whole
            if qualifier and keys.size:
                regions, countries = self.qualifiers.match(qualifier)
                region, country = self.admin1_index[place], self.country_index[place]

                def named(only_whole: bool) -> np.ndarray:
                    pick = lambda d: [k for k, w in d.items() if w or not only_whole]  # noqa: E731
                    return np.isin(region, pick(regions)) | np.isin(country, pick(countries))

                keep = named(False)
                # Typed as codes ("CA"): only what they name whole, if anything (else "SAN JOSE, CAL",
                # all capitals, would find nothing).
                if all(w in codes for w in qualifier) and (exact := named(True)).any():
                    keep = exact
                boost = np.where(named(True)[keep], WHOLE_QUALIFIER_BOOST, 1)
                level, place = level[keep], place[keep]
            places.append(place)
            boosts.append(boost)
            levels.append(level)
        place, level, boost = np.concatenate(places), np.concatenate(levels), np.concatenate(boosts)
        # By weighted population, then (between equals, as with no population) the better kind of name
        # match, then a whole qualifier, then population rank (stable: GeoNames file order).
        weight = self.population[place] * _MATCH_WEIGHTS[level] * boost
        order = np.lexsort((self._rank[place], -boost, -level, -weight))
        place = place[order]
        _, first = np.unique(place, return_index=True)  # each place at its best score
        return [self.describe(int(i)) for i in place[np.sort(first)[:limit]]]
