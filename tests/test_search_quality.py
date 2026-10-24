"""Search against the real GeoNames data: realistic queries and the place each should suggest first.

A fixed list to weigh ranking changes by, not a list of every case that works. Skipped without
data/geonames (and data/naturalearth, for regions' ISO codes such as "BC")."""

from pathlib import Path

import pytest

from climate_twins.boundaries import Boundaries
from climate_twins.places import CITIES_FILE, Places

DATA = Path(__file__).resolve().parent.parent / "data"

# (query, the start of the top suggestion's label)
QUERIES = [
    # Major cities, whole and as typed so far.
    ("paris", "Paris, Île-de-France"),
    ("london", "London, England"),
    ("tokyo", "Tokyo, Japan"),
    ("new york", "New York City"),
    ("new y", "New York City"),
    ("los ang", "Los Angeles, California"),
    ("san fran", "San Francisco, California"),
    ("chic", "Chicago, Illinois"),
    ("sao p", "São Paulo, Brazil"),
    ("São Pau", "São Paulo, Brazil"),
    ("delh", "Delhi"),
    ("berl", "Berlin, Germany"),
    ("barc", "Barcelona, Catalonia"),
    ("amst", "Amsterdam, North Holland"),
    ("mel", "Melbourne, Victoria"),
    ("buenos", "Buenos Aires"),
    ("ist", "Istanbul"),
    ("sing", "Singapore"),
    ("hong k", "Hong Kong"),
    ("mexico", "Mexico City"),
    ("cairo", "Cairo, Egypt"),
    ("athens", "Athens, Attica, Greece"),
    ("rome", "Rome, Lazio, Italy"),
    ("porto", "Porto, Portugal"),
    # Ambiguous names: the best-known first, a qualifier for the rest.
    ("portland", "Portland, Oregon"),
    ("springfield", "Springfield, Missouri"),
    ("paris tx", "Paris, Texas"),
    ("Paris, Texas", "Paris, Texas"),
    ("london on", "London, Ontario"),
    ("London, Ontario", "London, Ontario"),
    ("cambridge", "Cambridge, England"),
    ("cambridge ma", "Cambridge, Massachusetts"),
    ("san jose", "San Jose, California"),
    ("san jose costa rica", "San José, Costa Rica"),
    ("birmingham", "Birmingham, England"),
    ("birmingham al", "Birmingham, Alabama"),
    ("perth", "Perth, Western Australia"),
    ("perth scotland", "Perth, Scotland"),
    # Qualifiers.
    ("Portland, Oregon", "Portland, Oregon"),
    ("portland or", "Portland, Oregon"),
    ("portland me", "Portland, Maine"),
    ("Vancouver BC", "Vancouver, British Columbia"),
    ("vancouver wa", "Vancouver, Washington"),
    ("Sydney, NS", "Sydney, Nova Scotia"),
    ("Kansas City, MO", "Kansas City, Missouri"),
    ("richmond va", "Richmond, Virginia"),
    ("Columbus, Ohio", "Columbus, Ohio"),
    ("Valencia, Spain", "Valencia, Spain"),
    # Accents, abbreviations, other names, other scripts.
    ("munchen", "Munich, Bavaria"),
    ("sao paulo", "São Paulo, Brazil"),
    ("urumqi", "Ürümqi"),
    ("zurich", "Zürich, Switzerland"),
    ("montreal", "Montréal, Quebec"),
    ("bogota", "Bogotá"),
    ("malmo", "Malmö"),
    ("koln", "Köln"),
    ("cologne", "Köln"),
    ("krakow", "Kraków"),
    ("saint louis", "St. Louis, Missouri"),
    ("st louis", "St. Louis, Missouri"),
    ("st. petersburg", "Saint Petersburg, Russia"),
    ("st paul", "Saint Paul, Minnesota"),
    ("nyc", "New York City"),
    ("roma", "Rome, Lazio"),
    ("wien", "Vienna"),
    ("praha", "Prague"),
    ("lisboa", "Lisbon"),
    ("moskva", "Moscow"),
    ("kiev", "Kyiv"),
    ("bombay", "Mumbai"),
    ("Москва", "Moscow"),
    ("서울", "Seoul"),
    # Small towns.
    ("bozeman", "Bozeman, Montana"),
    ("hilo", "Hilo, Hawaii"),
    ("ushuaia", "Ushuaia"),
    ("tromso", "Tromsø"),
    ("moab", "Moab, Utah"),
    ("alice springs", "Alice Springs"),
    ("hallstatt", "Hallstatt"),
]


@pytest.fixture(scope="module")
def places() -> Places:
    if not (DATA / "geonames" / CITIES_FILE).is_file():
        pytest.skip("no GeoNames data in data/geonames")
    p = Places.load(DATA / "geonames", Boundaries.load(DATA / "naturalearth"))
    assert p is not None
    return p


@pytest.mark.parametrize("q, top", QUERIES)
def test_top_suggestion(places: Places, q: str, top: str):
    labels = [p["label"] for p in places.search(q)]
    assert labels and labels[0].startswith(top), labels
