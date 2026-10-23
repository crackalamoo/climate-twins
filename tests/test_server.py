import io
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from climate_twins.boundaries import Boundaries
from climate_twins.features import VARIANCE_TARGET
from climate_twins.index import MONTHLY_VARS
from climate_twins.places import Places
import climate_twins.server as server
from climate_twins import page, preview
from climate_twins.index import QueryCache
from climate_twins.geo import great_circle_km
from climate_twins.server import app

from .conftest import BOUNDARY_POLYGONS, NC_QUERY, NC_TWIN, write_geonames


def make_client(data: Path, geonames: Path, boundaries: Path | None = None) -> Iterator[TestClient]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLIMATE_TWINS_DATA", str(data))
        mp.setenv("CLIMATE_TWINS_GEONAMES", str(geonames))
        # Without boundaries, a directory with no shapefile: never the real data/naturalearth.
        mp.setenv("CLIMATE_TWINS_BOUNDARIES", str(boundaries or geonames))
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def client(built: tuple[Path, dict], geonames_dir: Path, boundaries_dir: Path) -> Iterator[TestClient]:
    yield from make_client(built[0], geonames_dir, boundaries_dir)


@pytest.fixture(scope="module")
def client_without_places(built: tuple[Path, dict], tmp_path_factory) -> Iterator[TestClient]:
    yield from make_client(built[0], tmp_path_factory.mktemp("no-geonames"))


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["cells"] > 0 and body["pca_components"] >= 1
    assert body["pca_variance_kept"] >= VARIANCE_TARGET
    assert body["source"].startswith("TerraClimate")


def test_similar(client: TestClient):
    lat, lon = NC_QUERY
    r = client.get("/similar", params={"lat": lat, "lon": lon, "n": 3, "detail": "true"})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["matches"]) <= 3
    top = body["matches"][0]
    assert (top["cell"]["lat"], top["cell"]["lon"]) == NC_TWIN
    assert set(top["monthly"]) == set(MONTHLY_VARS)
    assert set(body["params"]) == {"n", "exclude_km", "suppress_km", "align"}
    assert body["params"]["n"] == 3
    assert isinstance(body["elapsed_ms"], int)
    assert (body["query"]["place"]["name"], body["query"]["place"]["tier"]) == ("Midtown", "near")
    assert top["place"]["tier"] == "region"


def test_climate(client: TestClient):
    r = client.get("/climate", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1]})
    assert r.status_code == 200
    assert r.json()["cell"]["lat"] == NC_QUERY[0]
    assert r.json()["place"]["name"] == "Midtown"


@pytest.mark.parametrize("path", ["/similar", "/climate", "/tiles/0/0/0.png"])
def test_ocean_is_404(client: TestClient, path: str):
    r = client.get(path, params={"lat": 0, "lon": -140})
    assert r.status_code == 404
    assert "on land" in r.json()["detail"]


def test_places_search(client: TestClient):
    r = client.get("/places/search", params={"q": "SAO", "limit": 1})
    assert r.status_code == 200
    (p,) = r.json()
    assert p["name"] == "São Paulo" and p["label"]
    assert client.get("/places/search").json() == []
    (p,) = client.get("/places/search", params={"q": "Portland, OR, USA"}).json()
    assert p["label"] == "Portland, Oregon, United States"
    assert client.get("/places/search", params={"q": p["label"]}).json() == [p]


def head(html: str) -> tuple[str, str]:
    title = html.split("<title>")[1].split("</title>")[0]
    return title, html.split('<meta name="description" content="')[1].split('"')[0]


def test_page_describes_the_query_in_its_url(client: TestClient):
    params = {"lat": NC_QUERY[0], "lon": NC_QUERY[1], "exclude": 1000, "suppress": "x"}
    place = client.get("/climate", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1]}).json()["place"]
    title, description = head(client.get("/", params=params).text)
    assert title == f"{page.short_headline(place)} · Climate Twins"
    assert description.startswith("The places on Earth whose climate is most like ")


@pytest.mark.parametrize("params", [{}, {"lat": "x", "lon": "y"}, {"lat": 0, "lon": -30}, {"lat": 95, "lon": 0}])
def test_page_without_a_query_on_land_describes_the_site(client: TestClient, params: dict):
    assert head(client.get("/", params=params).text) == ("Climate Twins", page.SITE_DESCRIPTION)


def test_page_names_places_like_the_ui():
    city = {"tier": "exact", "name": "Brussels", "country": "Belgium", "label": "Brussels, Belgium", "in_area": None}
    offset = {"tier": "offset", "name": "Cairns", "km": 1234.4, "direction": "NW", "label": "1234 km NW of Cairns, Queensland, Australia",
              "area": {"admin1": "Queensland", "country": "Australia"}}
    region = {"tier": "region", "name": "Iran", "label": "in Iran", "area": {"admin1": None, "country": "Iran"}}
    assert [page.short_headline(p) for p in (city, offset, region)] == ["Brussels' climate twins", "1,234 km NW of Cairns", "Iran"]
    assert [page.headline(p) for p in (city, offset, region)] == [
        "Brussels' climate twins", "Climate twins of a spot 1,234 km NW of Cairns", "Climate twins of a spot in Iran"]
    assert [page.card_lines(p) for p in (city, {**city, "in_area": "Flanders, Belgium"}, offset, region)] == [
        ("Climate twins of", "Brussels"), ("Climate twins of a spot", "near Brussels"),
        ("Climate twins of a spot", "1,234 km NW of Cairns"), ("Climate twins of a spot", "in Iran")]
    assert page.description(offset, [city, region, None, None]) == (
        "The places on Earth whose climate is most like the spot 1234 km NW of Cairns, Queensland, Australia: "
        "Brussels, Belgium; Iran, and 2 more."
    )


def card(r) -> bytes:
    assert r.status_code == 200 and r.headers["cache-control"] == server.TILE_CACHE_CONTROL
    assert Image.open(io.BytesIO(r.content)).size == (1200, 630)
    return r.content


def test_preview_card(client: TestClient):
    site = card(client.get("/preview.png"))
    for params in ({"lat": "x", "lon": "y"}, {"lat": 0, "lon": -30}, {"lat": 95, "lon": 0}):
        assert card(client.get("/preview.png", params=params)) == site
    assert card(client.get("/preview.png", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1], "exclude": "x"})) != site


def test_concurrent_first_cards_build_each_plate_once(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(preview, "_plates", QueryCache(2))
    build, builds = preview.Plate.build, []

    def slow_build(*args):
        builds.append(args)
        time.sleep(0.2)
        return build(*args)

    monkeypatch.setattr(preview.Plate, "build", slow_build)
    index = client.app.state.index
    with ThreadPoolExecutor(4) as pool:
        plates = list(pool.map(lambda _: preview.plate(index, None, 1), range(4)))
    assert len(builds) == 1 and all(p is plates[0] for p in plates)


def test_heavy_requests_run_two_at_a_time(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    index = client.app.state.index
    similar, lock, running, most = index.similar, threading.Lock(), [0], [0]

    def slow_similar(*args, **kwargs):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.1)
        with lock:
            running[0] -= 1
        return similar(*args, **kwargs)

    monkeypatch.setattr(index, "similar", slow_similar)
    params = {"lat": NC_QUERY[0], "lon": NC_QUERY[1], "n": 2}
    with ThreadPoolExecutor(6) as pool:
        statuses = list(pool.map(lambda _: client.get("/similar", params=params).status_code, range(6)))
    assert statuses == [200] * 6 and most[0] == server.HEAVY_REQUESTS


def test_page_previews_with_its_card(client: TestClient):
    query = f"lat={NC_QUERY[0]}&lon={NC_QUERY[1]}&align=calendar"
    html = client.get(f"/?{query}").text
    image = unescape(html.split('<meta property="og:image" content="')[1].split('"')[0])
    assert image == f"http://testserver/preview.png?{query}"
    assert '<meta name="twitter:card" content="summary_large_image">' in html


def test_health_counts_boundaries(client: TestClient, built: tuple[Path, dict], geonames_dir: Path):
    assert client.get("/health").json()["boundaries"] == len(BOUNDARY_POLYGONS)
    for c in make_client(built[0], geonames_dir):
        assert c.get("/health").json()["boundaries"] == 0


def test_without_geonames(client_without_places: TestClient):
    c = client_without_places
    lat, lon = NC_QUERY
    body = c.get("/similar", params={"lat": lat, "lon": lon, "n": 2}).json()
    assert body["query"]["place"] is None and all(m["place"] is None for m in body["matches"])
    assert c.get("/climate", params={"lat": lat, "lon": lon}).json()["place"] is None
    assert c.get("/places/search", params={"q": "sao"}).json() == []
    assert c.get("/health").json()["places"] == 0


def test_tile(client: TestClient):
    lat, lon = NC_QUERY
    for align in ("auto", "calendar"):
        r = client.get("/tiles/0/0/0.png", params={"lat": lat, "lon": lon, "align": align})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert "max-age" in r.headers["cache-control"]
        assert Image.open(io.BytesIO(r.content)).size == (256, 256)


def test_tile_out_of_range_is_400(client: TestClient):
    r = client.get("/tiles/1/2/0.png", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1]})
    assert r.status_code == 400


@pytest.mark.parametrize("path", ["/", "/style.css"])
def test_static_files_revalidate(client: TestClient, path: str):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert client.get(path, headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_similar_has_a_ruler(client: TestClient):
    body = client.get("/similar", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1], "n": 1}).json()
    assert set(body["ruler"]) == {"min_km", "max_km", "min_population", "cities"}
    assert "ruler_comparison" in body["matches"][0]


# The ruler on the 2-degree synthetic world (base_index), whose cell centres sit on odd degrees:
# cities around RULER_QUERY (in "North America", and in Minnesota's boundary polygon), most on cell
# centres so they snap. (name, lat, lon, country, population). The big ones 50-500 km away are
# candidates; too near, too far, too small, part of a bigger city's metro and unsnappable ones aren't.
RULER_QUERY = (45.0, -95.0)
RULER_ROWS = [
    ("Suburbia", 45.0, -95.3, "US", 1_000_000),  # ~24 km: the query's own metro area
    ("Bigcity", 45.0, -97.0, "US", 2_000_000),  # ~157 km
    ("Borough", 45.0, -96.8, "US", 1_200_000),  # ~16 km from Bigcity, which is bigger: part of it
    ("Eastville", 45.0, -91.0, "US", 800_000),  # ~315 km
    ("Grenzstadt", 47.0, -99.0, "CH", 600_000),  # ~370 km, across a border
    ("Lakeside", 49.0, -95.0, "US", 550_000),  # ~445 km
    ("Westport", 45.0, -101.0, "US", 500_000),  # ~472 km
    ("Offcentre", 44.0, -98.0, "US", 700_000),  # ~250 km, but on a cell corner: >25 km from any centre
    ("Hamlet", 43.0, -95.0, "US", 400_000),  # too small
    ("Faraway", 45.0, -103.0, "US", 900_000),  # ~629 km: too far
]


@pytest.fixture(scope="module")
def ruler_request(base_index, tmp_path_factory, boundaries_dir: Path) -> SimpleNamespace:
    """A stand-in for the Request ruler() reads: the synthetic world and the cities above."""
    rows = [(n, n, lat, lon, cc, "MN" if cc == "US" else "", pop, "PPL") for n, lat, lon, cc, pop in RULER_ROWS]
    places = Places.load(write_geonames(tmp_path_factory.mktemp("ruler-geonames"), rows), Boundaries.load(boundaries_dir))
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(index=base_index, places=places)))


def candidate_distances(ruler_request: SimpleNamespace, align: str = "auto") -> dict[str, float]:
    """The candidates' distances from RULER_QUERY, straight from the index (the expected values)."""
    index = ruler_request.app.state.index
    q = index.query(index.locate(*RULER_QUERY)[0], align)
    names = ("Bigcity", "Eastville", "Grenzstadt", "Lakeside", "Westport")
    rows = {n: next(r for r in RULER_ROWS if r[0] == n) for n in names}
    return {n: float(index.distance_at(q, [index.locate(lat, lon)[0]])[0]) for n, (_, lat, lon, _, _) in rows.items()}


def test_ruler(ruler_request: SimpleNamespace):
    lat, lon = RULER_QUERY
    expected = candidate_distances(ruler_request)
    # A match exactly as close as each candidate ties with it, so every candidate is listed.
    ruler, comparisons = server.ruler(ruler_request, lat, lon, "auto", list(expected.values()))
    cities = {c["name"]: c for c in ruler["cities"]}
    assert set(cities) == set(expected)
    assert [c["distance"] for c in ruler["cities"]] == sorted(c["distance"] for c in ruler["cities"])
    assert cities["Bigcity"]["label"] == "Bigcity"
    assert cities["Grenzstadt"]["label"] == "Grenzstadt, Switzerland"  # the query is in the US
    assert cities["Westport"]["km_from_query"] == round(great_circle_km(lat, lon, 45.0, -101.0))
    assert cities["Westport"]["population"] == 500_000
    for name, comparison in zip(expected, comparisons):
        assert comparison == {"kind": "about", "city": list(cities).index(name)}
    # The same measure as matches' `distance`, from the query to the city's land cell, in either alignment.
    for align in ("auto", "calendar"):
        want = candidate_distances(ruler_request, align)
        got, _ = server.ruler(ruler_request, lat, lon, align, list(want.values()))
        for c in got["cities"]:
            assert c["distance"] == pytest.approx(want[c["name"]], abs=1e-4)


def test_ruler_candidates(ruler_request: SimpleNamespace):
    places = ruler_request.app.state.places
    idx, _ = places.big_cities(*RULER_QUERY, server.RULER_MIN_KM, server.CITY_KM, server.RULER_MIN_POPULATION, server.RULER_MIN_KM)
    # Not Suburbia (the query's metro), Borough (Bigcity's), Hamlet (too small) or Faraway (too far).
    assert {places.name[i] for i in idx} == {"Bigcity", "Eastville", "Grenzstadt", "Lakeside", "Westport", "Offcentre"}


def test_ruler_comparisons(ruler_request: SimpleNamespace):
    """A tie if any (the nearest in distance), else "closer than" the most similar city it clearly
    beats; "all" when it clearly beats every one; None when it beats none. Only the cities named,
    and the most similar one, are listed."""
    dist = candidate_distances(ruler_request)
    band = server.RULER_BAND
    best, worst = min(dist.values()), max(dist.values())
    tie = lambda d: [n for n, c in dist.items() if abs(d - c) <= band * c]
    # A distance that ties with no candidate but clearly beats some: "closer than" the most similar of those.
    gap = next(d for d in np.linspace(best, worst, 4000) if not tie(d) and any(d < c * (1 - band) for c in dist.values()))
    beaten = min((c, n) for n, c in dist.items() if gap < c * (1 - band))[1]
    ruler, comparisons = server.ruler(ruler_request, *RULER_QUERY, "auto", [best * 0.5, gap, worst * 10])
    names = [c["name"] for c in ruler["cities"]]
    assert comparisons[0] == {"kind": "all"}
    assert comparisons[1] == {"kind": "closer", "city": names.index(beaten)}
    assert comparisons[2] is None
    assert set(names) == {beaten, min(dist, key=dist.get)}


def test_ruler_without_places(client_without_places: TestClient):
    body = client_without_places.get("/similar", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1], "n": 1}).json()
    assert body["ruler"]["cities"] == []

