import os
from pathlib import Path

import pytest

from climate_twins.places import (
    CACHE_FILE, Places, _spellings, region_is_generic, region_repeats_city,
)

from .conftest import GEONAMES_ROWS, write_geonames


@pytest.fixture(scope="module")
def places(geonames_dir: Path) -> Places:
    p = Places.load(geonames_dir)
    assert p is not None
    return p


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

    def test_region_label_is_majority_of_nearest_places(self, places: Places):
        p = places.label_for(33.0, 87.0)  # nearest village says Hebei, the next six say Tibet
        assert "Tibet" in p["label"]

    @pytest.mark.parametrize("lat, lon, query", [
        (56.0, 92.9, "krasno"),  # Krasnoyarsk Krai repeats the city
        (26.6, 12.8, "ubari"),  # Libya's regions are not shown
        (26.6, 14.0, "ubari"),  # nor when the label is an offset from the city
    ])
    def test_hidden_region_is_left_out_of_labels_but_not_search(self, places: Places, lat, lon, query):
        p = places.label_for(lat, lon)
        assert p["admin1"] not in p["label"]
        assert p["admin1"] in places.search(query)[0]["label"]

    def test_sections_are_searchable_but_never_labels(self, places: Places):
        assert [p["name"] for p in places.search("midtown")] == ["Midtown Heights", "Midtown"]
        assert places.label_for(45.01, -95.01)["name"] == "Midtown"


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
