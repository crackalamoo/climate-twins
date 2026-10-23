import os
from pathlib import Path

import pytest

from climate_twins.boundaries import Boundaries, _country_code
from climate_twins.places import (
    CACHE_FILE, REGIONLESS_COUNTRIES, Places, _region_key, _spellings, region_is_generic, region_repeats_city,
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
        assert p["in_area"] is None  # the label is already where the point is
        assert places.label_for(0.0, -140.0) is None  # nothing within 500 km

    def test_region_label_from_the_polygon(self, places: Places):
        p = places.label_for(33.0, 87.0)  # nearest village is miscoded as Hebei; the polygon says Tibet
        assert p["tier"] == "region" and p["label"] == "in Tibet, China"

    def test_region_repeating_the_city_is_left_out_of_labels_and_search(self, places: Places):
        assert places.label_for(56.0, 92.9)["label"] == "Krasnoyarsk, Russia"
        assert places.search("krasno")[0]["label"] == "Krasnoyarsk, Russia"

    @pytest.mark.parametrize("lat, lon, query", [
        (26.6, 12.8, "ubari"),  # Libya's regions are not shown
        (26.6, 14.0, "ubari"),  # nor when the label is an offset from the city
    ])
    def test_hidden_region_is_left_out_of_labels_but_not_search(self, places: Places, lat, lon, query):
        p = places.label_for(lat, lon)
        assert p["admin1"] not in p["label"]
        assert p["admin1"] in places.search(query)[0]["label"]

    @pytest.mark.parametrize("lat, lon, area", [
        (45.0, -95.0, {"admin1": "Minnesota", "country": "United States"}),
        (26.6, 14.0, {"admin1": None, "country": "Libya"}),  # Libya's regions are not shown
        (25.0, 45.0, {"admin1": None, "country": "Saudi Arabia"}),  # region tier, no admin1
    ])
    def test_area_has_the_shown_region_and_the_country(self, places: Places, lat, lon, area):
        assert places.label_for(lat, lon)["area"] == area

    def test_sections_are_searchable_but_never_labels(self, places: Places):
        assert {p["name"] for p in places.search("midtown")} == {"Midtown", "Midtown Heights"}
        assert places.label_for(45.01, -95.01)["name"] == "Midtown"


class TestArea:
    """Where a point is, from the boundary polygon it's in (BOUNDARY_POLYGONS)."""

    def test_the_polygon_says_which_country_not_the_nearest_towns(self, places: Places):
        # In SS, but 8 of its 9 nearest towns are across the border in UG.
        p = places.label_for(4.3, 32.0)
        assert p["name"] == "Northtown" and p["area"] == {"admin1": None, "country": "South Sudan"}

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

    def test_without_boundaries_labels_name_only_cities(self, geonames_dir: Path):
        p = Places.load(geonames_dir)
        assert p.locate(4.3, 32.0) is None and p.label_for(4.3, 32.0)["area"] is None
        assert p.label_for(25.0, 45.0) is None  # no city near and no region to name

    def test_wraps_longitude(self, places: Places):
        assert places.locate(4.3, 32.0 + 360.0)[0] == "SS"


@pytest.fixture(scope="module")
def brazil(geonames_dir: Path, tmp_path_factory) -> Places:
    """Porto Alegre (Rio Grande do Sul) and São Paulo (São Paulo, hidden as it repeats the city),
    each in its own region with a strip of another region to its west; south of Porto Alegre, a
    polygon of unknown region."""
    return places_with(geonames_dir, tmp_path_factory.mktemp("ne"), [
        ("BR", "BR.13", "Maranhão", "Maranhão", _box(-52, -31, -51.4, -29)),
        ("BR", "BR.23", "Rio Grande do Sul", "Rio Grande do Sul", _box(-51.4, -31, -50, -29)),
        ("BR", "BR.23", "Rio Grande do Sul", "Rio Grande do Sul", _box(-47.5, -24, -46.8, -23)),
        ("BR", "BR.27", "São Paulo", "São Paulo", _box(-46.8, -24.5, -46, -23)),
        ("BR", "-1", "Nowhere", "", _box(-52, -33, -50, -31)),  # no code, no towns: region unknown
    ])


class TestAcrossABorder:
    """A label never names a region or country the point isn't in: across a border from the
    city it names the city alone, then where the point is (`in_area`)."""

    def test_same_region_is_unchanged(self, places: Places, brazil: Places):
        p = places.label_for(45.0, -96.5)
        assert p["label"] == "39 km W of Bigcity, Minnesota, United States"
        assert p["in_area"] is None and p["base_label"] == "Bigcity, Minnesota, United States"
        assert brazil.label_for(-30.03, -51.0)["label"] == "near Porto Alegre, Rio Grande do Sul, Brazil"
        assert brazil.label_for(-23.55, -46.7)["label"] == "near São Paulo, Brazil"  # hidden, but the same

    def test_another_region_of_the_same_country(self, brazil: Places):
        p = brazil.label_for(-30.03, -51.6)
        assert p["label"] == "36 km W of Porto Alegre, in Maranhão, Brazil"
        assert (p["base_label"], p["in_area"]) == ("Porto Alegre", "Maranhão, Brazil")
        assert p["area"] == {"admin1": "Maranhão", "country": "Brazil"}
        assert (p["name"], p["admin1"]) == ("Porto Alegre", "Rio Grande do Sul")  # still the city's

    def test_a_hidden_city_region_conflicts_only_with_a_shown_point_region(self, brazil: Places):
        # São Paulo's region repeats the city, so the label claims none; the point's would be shown.
        assert brazil.label_for(-23.55, -46.9)["label"] == "28 km W of São Paulo, in Rio Grande do Sul, Brazil"

    def test_an_unknown_region_in_the_same_country_is_unchanged(self, brazil: Places):
        p = brazil.label_for(-31.2, -51.23)
        assert p["area"] == {"admin1": None, "country": "Brazil"}
        assert p["in_area"] is None and p["label"].endswith(" of Porto Alegre, Rio Grande do Sul, Brazil")

    def test_another_country_with_a_known_region(self, geonames_dir: Path, tmp_path: Path):
        p = places_with(geonames_dir, tmp_path, [("CN", "CN.14", "Xizang", "Tibet", _box(-97, 44, -96.02, 46))])
        assert p.label_for(45.0, -96.5)["label"] == "39 km W of Bigcity, in Tibet, China"
        # Within 5 km the label is normally just the city, but the point isn't in it.
        exact = p.label_for(45.0, -96.03)
        assert (exact["tier"], exact["label"]) == ("exact", "near Bigcity, in Tibet, China")
        assert (exact["base_label"], exact["in_area"]) == ("Bigcity", "Tibet, China")

    @pytest.mark.parametrize("lat, lon, label", [
        (15.0, 30.0, "107 km W of Portcity, in Eritrea"),  # region unknown
        (3.5, 32.0, "222 km S of Northtown, in Uganda"),  # region hidden ("Northern Region" is generic)
    ])
    def test_another_country_with_no_region_to_show(self, places: Places, lat, lon, label):
        p = places.label_for(lat, lon)
        assert p["label"] == label and p["in_area"] == p["area"]["country"]

    def test_a_country_whose_regions_labels_hide_is_unchanged(self, tmp_path: Path):
        # Kazakhstan's regions aren't shown, so the label claims none and can't be wrong.
        rows = [*GEONAMES_ROWS, ("Bigalmaty", "Bigalmaty", 43.25, 76.9, "KZ", "10", 2_000_000, "PPL")]
        p = places_with(write_geonames(tmp_path / "g", rows), tmp_path / "ne",
                        [("KZ", "KZ.17", "Zhetysu", "Zhetysu Region", _box(77.5, 43, 78.5, 43.5))])
        assert p.locate(43.25, 78.0) == ("KZ", "Zhetysu Region")
        label = p.label_for(43.25, 78.0)
        assert label["label"] == "89 km E of Bigalmaty, Kazakhstan" and label["in_area"] is None

    def test_unknown_area_is_unchanged(self, places: Places, geonames_dir: Path):
        assert places.locate(4.3, 34.0) is None  # 1 degree east of every polygon
        p = places.label_for(4.3, 34.0)  # still labelled from a city
        assert p["area"] is None and p["in_area"] is None and p["label"].endswith(" of Northtown, South Sudan")
        assert Places.load(geonames_dir).label_for(45.0, -96.5)["in_area"] is None  # no boundary data


class TestBoundaries:
    def test_no_boundary_files(self, tmp_path: Path, caplog):
        assert Boundaries.load(tmp_path) is None
        assert "missing" in caplog.text

    @pytest.mark.parametrize("iso_a2, gn_a1_code, expected", [
        ("US", "US.CA", "US"),
        ("RU", "UA.11", "UA"),  # disputed: the country GeoNames files its towns under
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

    def test_by_population_with_whole_names_boosted(self, places: Places):
        # Porto (250k) counts as 2.5M: above Porto Alegre (1.5M), which only starts with "porto".
        # Bordeaux (265k) is "Porto" too in GeoNames, but an alternate name counts as 26k.
        assert [p["name"] for p in places.search("porto")] == ["Porto", "Porto Alegre", "Bordeaux"]
        # While typing: Paris (2.1M) is above Par (9k, as 95k) and Pa (15k, as 150k); Par is above
        # Paris, Texas (25k).
        assert [p["name"] for p in places.search("par")] == ["Paris", "Par", "Paris", "Parisot"]
        assert [p["name"] for p in places.search("pa")][0] == "Paris"
        assert [p["name"] for p in places.search("mexico")] == ["Mexico City", "Mexico"]  # an alternate name
        # All names in part, by population: Bordeaux's "Porto" is an alternate name, so as 26k.
        assert [p["name"] for p in places.search("port")] == [
            "Portcity", "Porto Alegre", "Portland", "Porto", "Portland", "Bordeaux", "Portland"]
        assert places.search("s") == []  # too short
        assert [p["name"] for p in places.search("sa", limit=2)] == ["São Paulo", "Santiago"]

    @pytest.mark.parametrize("q", [
        "Portland, Oregon", "portland oregon", "Portland OR", "Portland, OR", "PORTLAND,OREGON",
        "Portland, Oregon, United States", "Portland Oregon USA", "Portland, OR, US",
        "Portland, Ore",  # still typing
        "Portland Oregon United St",
    ])
    def test_name_then_region_and_or_country(self, places: Places, q):
        assert [(p["name"], p["admin1"]) for p in places.search(q)] == [("Portland", "Oregon")]

    @pytest.mark.parametrize("q, expected", [
        ("Portland, United States", ["Oregon", "Maine"]),
        ("Portland, UK", ["England"]),  # a country by its internet domain
        ("Portland GBR", ["England"]),  # ... or ISO alpha-3 code
        ("Portland, England", ["England"]),
        ("Portland, England, United Kingdom", ["England"]),
        ("Portland, M", ["Maine"]),  # a region or country beginning with M
        ("Portland, Oregon, United Kingdom", []),  # Oregon isn't in the UK
        ("Portland, United States, Oregon", []),  # region before country
        ("Portland O", ["Oregon"]),
        ("Portland, ", ["Oregon", "Maine", "England"]),
    ])
    def test_qualifier_filters(self, places: Places, q, expected):
        assert [p["admin1"] for p in places.search(q)] == expected

    def test_whole_qualifiers_boosted(self, places: Places):
        # "MA" is Massachusetts' code and the start of Maryland: Cambridge, Massachusetts (118k) counts
        # WHOLE_QUALIFIER_BOOST times over Cambridge, Maryland (500k in the fixture).
        assert [p["admin1"] for p in places.search("Cambridge, Ma")] == ["Massachusetts", "Maryland"]
        assert [p["admin1"] for p in places.search("Cambridge, Mar")] == ["Maryland"]

    def test_capitals_are_a_code(self, places: Places):
        # Typed in capitals, "MA" and "WA" are codes, not the starts of Maryland and Waikato; "M" is too
        # short to tell; and capitals that name nothing whole still match names they start.
        assert [p["admin1"] for p in places.search("Cambridge, MA")] == ["Massachusetts"]
        assert [p["admin1"] for p in places.search("Cambridge, M")] == ["Maryland", "Massachusetts"]  # by population
        assert [p["admin1"] for p in places.search("Hamilton, WA")] == ["Western Australia"]
        assert [p["admin1"] for p in places.search("CAMBRIDGE, MAR")] == ["Maryland"]

    def test_a_code_doesnt_beat_a_region_name_being_typed(self, places: Places):
        # "Wa" is Western Australia's code, and the start of Waikato: population decides.
        assert [(p["name"], p["admin1"]) for p in places.search("Hamilton, Wa")] == [
            ("Hamilton", "Waikato"), ("Hamilton Hill", "Western Australia")]
        assert [p["name"] for p in places.search("Hamilton Hill, WA")] == ["Hamilton Hill"]
        assert [(p["name"], p["admin1"]) for p in places.search("Santiago, Sa")] == [
            ("Santiago", "Santiago Metropolitan"), ("Santiago Sacatepéquez", "Sacatepéquez")]

    def test_qualifier_codes_match_only_whole(self, places: Places):
        assert [p["admin1"] for p in places.search("Portland, U")] == ["Oregon", "Maine", "England"]  # United ...
        assert [p["admin1"] for p in places.search("Portland, Or")] == ["Oregon"]  # a code and a prefix
        assert places.search("Portland, ORE, US") == []  # codes are whole: no region "ORE"

    def test_country_alone(self, places: Places):
        assert [p["admin1"] for p in places.search("Paris, France")] == ["Île-de-France", "Occitanie"]  # Parisot
        assert places.search("paris fr")[0]["admin1"] == "Île-de-France"
        assert [p["admin1"] for p in places.search("Paris Texas")] == ["Texas"]
        assert [p["name"] for p in places.search("Pari, France")] == ["Paris", "Parisot"]  # the name too can be partial

    def test_the_suggested_label_finds_its_place(self, places: Places):
        for i in range(len(places)):
            p = places.describe(i)
            assert p in places.search(p["label"], limit=50), p["label"]

    def test_alternate_names_of_big_cities(self, places: Places):
        assert [p["name"] for p in places.search("NYC")] == ["New York City"]
        assert [p["name"] for p in places.search("big app")] == ["New York City"]
        assert [p["name"] for p in places.search("New York, NY")] == ["New York City", "New York Mills"]
        assert [p["admin1"] for p in places.search("pdx")] == ["Oregon"]
        # A place's own name before another's alternate name, though that city is bigger. GeoNames
        # lists "Bridgetown" for every Portland, but only those of 100,000+ match it.
        assert [(p["name"], p["admin1"]) for p in places.search("bridgetown")] == [
            ("Bridgetown", "Saint Michael"), ("Portland", "Oregon")]

    def test_alternate_names_going_on_past_a_whole_name(self, places: Places):
        # "New York" is a whole name (New York City's), and Jakarta's nickname "New York Van Java"
        # goes on from it with more words: another name altogether.
        assert "Jakarta" not in [p["name"] for p in places.search("new york", limit=50)]
        # Partway through a word of it, it's still being typed.
        assert [p["name"] for p in places.search("new york v")] == ["Jakarta"]

    def test_alternate_names_need_three_characters(self, places: Places):
        # Chongqing (22M) is "Pa" in GeoNames: two letters are too early to match an alternate name.
        assert "Chongqing" not in [p["name"] for p in places.search("pa", limit=50)]
        assert [p["name"] for p in places.search("chu")] == ["Chongqing"]  # "Chungking"
        assert places.search("ny") == []  # New York City's "NY"
        assert [p["name"] for p in places.search("cdm")] == ["Mexico City"]


class TestRegionAbbreviations:
    """ISO 3166-2 codes of boundary polygons as region qualifiers, where GeoNames' codes are numbers."""

    @pytest.mark.parametrize("q, expected", [
        ("Vancouver, BC", [("Vancouver", "British Columbia")]),
        ("vancouver bc canada", [("Vancouver", "British Columbia")]),
        ("Sydney NSW", [("Sydney", "New South Wales")]),  # from two polygons that agree
        ("Sydney, NS", [("Sydney", "Nova Scotia")]),
        ("Vancouver, WA", [("Vancouver", "Washington")]),  # GeoNames' own codes still work
        ("Sydney, N", [("Sydney", "New South Wales"), ("Sydney", "Nova Scotia")]),  # names by prefix
    ])
    def test_abbreviation_names_the_region(self, places: Places, q, expected):
        assert [(p["name"], p["admin1"]) for p in places.search(q)] == expected

    def test_only_consistent_codes(self, places: Places):
        assert places._region_abbreviations() == {"CA.02": "BC", "CA.07": "NS", "AU.02": "NSW", "AU.08": "WA"}

    def test_none_for_countries_whose_regions_labels_hide(self, places: Places):
        # Guatemala's GT-SA polygon is consistent, but labels hide its regions: "SA" is only the
        # start of Sacatepéquez, not its code.
        assert "GT" in REGIONLESS_COUNTRIES and "GT.17" not in places._region_abbreviations()
        regions, _ = places.qualifiers.match(["sa"])
        assert regions[places.qualifiers.region_ids[places.qualifiers.region_keys.index("sacatepequez")]] is False
        # Its region's name and GeoNames' own code still match.
        for q in ["Santiago Sacatepéquez, Sacatepéquez", "Santiago Sacatepequez 17", "Santiago Sacatepéquez, Sacatepéquez, Guatemala"]:
            assert [p["name"] for p in places.search(q)] == ["Santiago Sacatepéquez"], q

    def test_without_boundaries_there_are_none(self, geonames_dir: Path):
        p = Places.load(geonames_dir)
        assert p.search("Vancouver, BC") == []
        assert [p["admin1"] for p in p.search("Vancouver, British Columbia")] == ["British Columbia"]

    @pytest.mark.parametrize("polygons", [
        # One GeoNames code, polygons with different ISO codes: many to one, like UK councils.
        [("CA", "CA.02", "British Columbia", "", _box(-125, 48, -120, 51), "CA-BC"),
         ("CA", "CA.02", "Vancouver Island", "", _box(-128, 48, -125, 51), "CA-VI")],
        # A stale GeoNames code: CA.07 is Nova Scotia, not this polygon.
        [("CA", "CA.07", "British Columbia", "", _box(-125, 48, -120, 51), "CA-BC")],
        # The ISO code is another country's.
        [("CA", "CA.02", "British Columbia", "", _box(-125, 48, -120, 51), "US-BC")],
        # Numbers and single letters.
        [("CA", "CA.02", "British Columbia", "", _box(-125, 48, -120, 51), "CA-59")],
        [("CA", "CA.02", "British Columbia", "", _box(-125, 48, -120, 51), "CA-B")],
        # No GeoNames code or a placeholder ISO code.
        [("CA", "-1", "British Columbia", "", _box(-125, 48, -120, 51), "CA-BC")],
        [("CA", "CA.02", "British Columbia", "", _box(-125, 48, -120, 51), "CA-X1~")],
    ])
    def test_unusable_codes_are_skipped(self, geonames_dir: Path, tmp_path: Path, polygons):
        p = places_with(geonames_dir, tmp_path, polygons)
        assert p._region_abbreviations() == {}
        assert p.search("Vancouver, BC") == []

    def test_one_abbreviation_for_two_regions_names_both(self, geonames_dir: Path, tmp_path: Path):
        p = places_with(geonames_dir, tmp_path, [
            ("AU", "AU.02", "New South Wales", "", _box(150, -35, 152, -32), "AU-NSW"),
            ("AU", "AU.01", "Australian Capital Territory", "", _box(148, -36, 149.5, -35), "AU-NSW"),
        ])
        regions, _ = p.qualifiers.match(["nsw"])
        assert {p.qualifiers.region_keys[p.qualifiers.region_ids.index(r)] for r in regions} == {
            "new south wales", "australian capital territory"}

    def test_never_takes_a_code_geonames_gives_another_region(self, geonames_dir: Path, tmp_path: Path):
        # US-MN on Oregon's polygon: "MN" stays Minnesota's alone.
        p = places_with(geonames_dir, tmp_path, [("US", "US.OR", "Oregon", "", _box(-124, 42, -117, 46), "US-MN")])
        assert p._region_abbreviations() == {"US.OR": "MN"}
        assert p.search("Portland, MN") == []
        assert [r["admin1"] for r in p.search("Portland, OR")] == ["Oregon"]


def test_cache_is_used_and_rebuilt_when_stale(tmp_path: Path):
    d = write_geonames(tmp_path / "g")
    assert Places.load(d).search("porto")[0]["population"] == 249_633
    cache = d / CACHE_FILE
    assert cache.exists()
    built = cache.stat().st_mtime

    # Rewrite the source with a different population; the newer txt must win over the cache.
    rows = [r if r[0] != "Porto" else (*r[:6], 9_000_000, r[7]) for r in GEONAMES_ROWS]
    write_geonames(d, rows)
    os.utime(d / "cities500.txt", (built + 10, built + 10))
    assert Places.load(d).search("porto")[0]["population"] == 9_000_000


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


def test_names_with_letters_the_type_lacks_show_their_ascii_names(tmp_path: Path):
    rows = [("Ḩalwān", "Halwan", 29.84, 31.33, "EG", "", 230_000, "PPLA2"),
            ("Kraków", "Krakow", 50.06, 19.94, "PL", "", 770_000, "PPLA")]
    places = Places.load(write_geonames(tmp_path, rows))
    assert [p["name"] for q in ("halwan", "Ḩalwān") for p in places.search(q)] == ["Halwan", "Halwan"]
    assert places.search("krakow")[0]["name"] == "Kraków"
