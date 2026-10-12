import os
from pathlib import Path

import pytest

from climate_twins.boundaries import Boundaries, _country_code
from climate_twins.places import (
    CACHE_FILE, Places, _region_key, _spellings, region_is_generic, region_repeats_city,
)

from .conftest import GEONAMES_ROWS, _box, write_boundaries, write_geonames


@pytest.fixture(scope="module")
def places(geonames_dir: Path, boundaries_dir: Path) -> Places:
    p = Places.load(geonames_dir, Boundaries.load(boundaries_dir))
    assert p is not None and p.boundaries is not None
    return p


def places_with(geonames_dir: Path, directory: Path, polygons) -> Places:
    return Places.load(geonames_dir, Boundaries.load(write_boundaries(directory, polygons)))


def test_skips_abandoned_places(places: Places):
    assert len(places) == len(GEONAMES_ROWS) - 1
    assert "Saõ Ghost" not in places.name


class TestLabel:
    def test_prefers_a_big_city_over_a_closer_small_one(self, places: Places):
        # Not the closer but tiny Smallville, the bigger but farther Bigcity, nor the section Midtown Heights.
        p = places.label_for(45.0, -95.0)
        assert (p["name"], p["tier"]) == ("Midtown", "near")
        assert places.label_for(45.0, -95.7)["name"] == "Bigcity"

    def test_city_balances_size_and_distance(self, places: Places):
        """Market Town (200k, 89 km) beats Distant City (600k, 140 km); the nearer Hamlet is
        too small to name and Metropolis is beyond 500 km."""
        p = places.label_for(10.0, 20.0)
        assert (p["name"], p["tier"], p["direction"]) == ("Market Town", "offset", "N")

    def test_exact_tier_has_no_prefix(self, places: Places):
        p = places.label_for(41.16, -8.6)
        assert p["tier"] == "exact" and p["label"] == p["base_label"]

    def test_region_when_no_city(self, places: Places):
        p = places.label_for(25.0, 45.0)
        assert p["tier"] == "region" and p["name"] is not None and p["km"] is None
        assert places.label_for(0.0, -140.0) is None  # nothing within 500 km

    def test_region_label_from_the_polygon(self, places: Places):
        p = places.label_for(33.0, 87.0)  # nearest village is miscoded as Hebei; the polygon says Tibet
        assert p["tier"] == "region" and p["label"] == "in Tibet, China"

    @pytest.mark.parametrize("lat, lon, query", [
        (56.0, 92.9, "krasno"),  # Krasnoyarsk Krai repeats the city
        (26.6, 12.8, "ubari"),  # Libya's regions are not shown
        (26.6, 14.0, "ubari"),  # nor when the label is an offset from the city
    ])
    def test_hidden_region_is_left_out_of_labels_but_not_search(self, places: Places, lat, lon, query):
        p = places.label_for(lat, lon)
        assert p["admin1"] not in p["label"]
        assert p["admin1"] in places.search(query)[0]["label"]

    def test_area_is_where_the_point_is_even_when_the_city_is_foreign(self, places: Places):
        p = places.label_for(15.0, 30.0)
        assert (p["name"], p["country"]) == ("Portcity", "Saudi Arabia")
        assert p["area"] == {"admin1": None, "country": "Eritrea"}

    @pytest.mark.parametrize("lat, lon, area", [
        (45.0, -95.0, {"admin1": "Minnesota", "country": "United States"}),
        (26.6, 14.0, {"admin1": None, "country": "Libya"}),  # Libya's regions are not shown
        (25.0, 45.0, {"admin1": None, "country": "Saudi Arabia"}),  # region tier, no admin1
    ])
    def test_area_has_the_shown_region_and_the_country(self, places: Places, lat, lon, area):
        assert places.label_for(lat, lon)["area"] == area

    def test_sections_are_searchable_but_never_labels(self, places: Places):
        assert [p["name"] for p in places.search("midtown")] == ["Midtown Heights", "Midtown"]
        assert places.label_for(45.01, -95.01)["name"] == "Midtown"


class TestArea:
    """Where a point is, from the boundary polygon it's in (BOUNDARY_POLYGONS)."""

    def test_the_polygon_says_which_country_not_the_nearest_towns(self, places: Places):
        # In SS, but 8 of its 9 nearest towns are across the border in UG.
        p = places.label_for(4.3, 32.0)
        assert p["name"] == "Northtown" and p["area"] == {"admin1": None, "country": "South Sudan"}
        assert places.label_for(3.5, 32.0)["area"]["country"] == "Uganda"

    def test_region_from_a_code_that_matches_the_polygon_name(self, places: Places):
        assert places.locate(4.3, 32.0) == ("SS", "Central Equatoria")
        assert places.locate(45.5, -94.0) == ("US", "Minnesota")

    def test_towns_dont_override_a_code(self, places: Places):
        assert places.locate(32.5, 87.5) == ("CN", "Tibet")  # one of its 7 towns is miscoded as Hebei
        # Split since Natural Earth drew it: 3 of its 5 towns are now in Zhetysu. The former
        # region, never one it wasn't in.
        assert places.locate(44.5, 78.0) == ("KZ", "Almaty Region")

    def test_a_stale_code_is_ignored(self, places: Places):
        assert places.locate(79.0, 15.0) == ("NO", "")  # NO.21 is Trøndelag in GeoNames, not Svalbard

    def test_region_from_towns_only_when_every_one_agrees(self, places: Places, geonames_dir: Path, tmp_path: Path):
        assert places.locate(3.1, 31.1) == ("UG", "Northern Region")  # no code; all 8 towns agree
        # No code, and one of 7 towns says Hebei: no region rather than a majority.
        p = places_with(geonames_dir, tmp_path, [("CN", "-1", "Xizang", "Tibet", _box(86, 32, 88, 34))])
        assert p.locate(32.5, 87.5) == ("CN", "")

    def test_country_as_geonames_files_the_towns(self, geonames_dir: Path, tmp_path: Path):
        # Natural Earth says SA, but all 5 GeoNames towns inside are filed under ER
        # (as GeoNames files Western Sahara as EH, French Guiana as GF).
        p = places_with(geonames_dir, tmp_path, [("SA", "-1", "Somewhere", "", _box(29.5, 14.5, 30.5, 15.5))])
        assert p.locate(15.0, 30.0) == ("ER", "")

    def test_no_country_without_a_code_or_agreeing_towns(self, geonames_dir: Path, tmp_path: Path):
        p = places_with(geonames_dir, tmp_path, [("-1", "-1", "Somaliland", "", _box(40, 8, 42, 10))])
        assert p.locate(9.0, 41.0) is None

    def test_a_point_just_offshore_takes_the_nearest_polygon(self, places: Places):
        assert places.locate(6.1, 32.0)[0] == "SS"  # 0.1 degrees outside SS's polygon

    def test_nothing_beyond_every_polygon(self, places: Places):
        assert places.locate(4.3, 34.0) is None  # 1 degree east of SS and UG
        assert places.label_for(4.3, 34.0)["area"] is None  # still labelled from a city

    def test_without_boundaries_labels_name_only_cities(self, geonames_dir: Path):
        p = Places.load(geonames_dir)
        assert p.locate(4.3, 32.0) is None and p.label_for(4.3, 32.0)["area"] is None
        assert p.label_for(25.0, 45.0) is None  # no city near and no region to name

    def test_wraps_longitude(self, places: Places):
        assert places.locate(4.3, 32.0 + 360.0)[0] == "SS"


class TestBoundaries:
    def test_no_boundary_files(self, tmp_path: Path, caplog):
        assert Boundaries.load(tmp_path) is None
        assert "missing" in caplog.text

    @pytest.mark.parametrize("iso_a2, gn_a1_code, expected", [
        ("US", "US.CA", "US"),
        ("RU", "UA.11", "UA"),  # disputed: the country GeoNames files its towns under
        ("GB", "GB.S6", "GB"),
        ("FR", "-1", "FR"),
        ("-1", "-1", None),  # Somaliland, sovereign base areas
    ])
    def test_country_code(self, iso_a2, gn_a1_code, expected):
        assert _country_code(iso_a2, gn_a1_code) == expected

    def test_a_point_in_two_polygons_takes_the_first(self, tmp_path: Path):
        box = _box(0, 0, 2, 2)
        b = Boundaries.load(write_boundaries(tmp_path, [("AA", "", "A", "", box), ("BB", "", "B", "", box)]))
        assert b.polygon_at(1.0, 1.0) == 0
        assert b.polygons_containing([1.0, 5.0], [1.0, 5.0]).tolist() == [0, -1]


@pytest.mark.parametrize("a, b", [
    ("Astrakhan'", "Astrakhan Oblast"),
    ("Sakha (Yakutia)", "Sakha"),
    ("Bakı", "Baki"),
    ("Sabha", "Sabha District"),
])
def test_region_key_matches_spellings(a, b):
    assert _region_key(a) == _region_key(b)


class TestSearch:
    def test_accent_and_case_insensitive_prefix(self, places: Places):
        for q in ["sao", "SÃO", "São", "  sao  "]:
            assert [p["name"] for p in places.search(q)] == ["São Paulo", "São Luís"]
        assert [p["name"] for p in places.search("zur")] == ["Zürich"]

    def test_sorted_by_population_and_deduplicated(self, places: Places):
        assert [p["name"] for p in places.search("porto")] == ["Porto Alegre", "Porto"]
        assert places.search("s") == []  # too short
        assert [p["name"] for p in places.search("sa", limit=2)] == ["São Paulo", "São Luís"]


def test_cache_is_used_and_rebuilt_when_stale(tmp_path: Path):
    d = write_geonames(tmp_path / "g")
    assert Places.load(d).search("porto")[0]["name"] == "Porto Alegre"
    cache = d / CACHE_FILE
    assert cache.exists()
    built = cache.stat().st_mtime

    # Rewrite the source with a different population; the newer txt must win over the cache.
    rows = [r if r[0] != "Porto" else (*r[:6], 9_000_000, r[7]) for r in GEONAMES_ROWS]
    write_geonames(d, rows)
    os.utime(d / "cities500.txt", (built + 10, built + 10))
    assert Places.load(d).search("porto")[0]["name"] == "Porto"


@pytest.mark.parametrize("region, city, repeats", [
    ("Mykolaiv", "Mykolayiv|Mykolaiv|Nikolaev", True),  # a spelling from GeoNames' alternate names
    ("Ho Chi Minh City (HCMC)", "Ho Chi Minh City", True),
    ("Santiago Metropolitan", "Santiago", True),
    ("Buenos Aires F.D.", "Buenos Aires", True),
    ("St.-Petersburg", "Saint Petersburg", True),
    ("Tucuman", "San Miguel de Tucumán", True),
    ("Queensland", "Cairns", False),
    ("Osaka", "Higashiosaka|Higashi Osaka", False),
    ("Gyeongsangbuk-do", "Gyeongsan", False),
    ("Gilgit-Baltistan", "Gilgit", False),  # the region says more than the city
    ("Jeju-do", "Jeju City", True),
    ("Delhi", "Deoli", False),  # similar spelling, different place
    ("New Jersey", "Jersey City", False),
    ("Taiwan", "Tainan", False),
])
def test_region_repeats_city(region, city, repeats):
    assert region_repeats_city(region, *city.split("|")) is repeats


@pytest.mark.parametrize("region, generic", [
    ("Centre", True), ("Northern Region", True), ("Centre-Nord", True),
    ("Western Australia", False), ("Northern Territory", False), ("East Java", False),
])
def test_region_is_generic(region, generic):
    assert region_is_generic(region) is generic


def test_spellings_keep_variants_not_other_names():
    assert _spellings("Cusco", "Cuzco,Qosqo,Cuzco,Cusco City") == ["Cuzco"]
    assert _spellings("Recife", "Pernambuco,Recife,Ресифи") == ["Recife"]
