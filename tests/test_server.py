import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from climate_twins.features import VARIANCE_TARGET
from climate_twins.index import MONTHLY_VARS
from climate_twins.server import app

from .conftest import NC_QUERY, NC_TWIN


def make_client(data: Path, geonames: Path) -> Iterator[TestClient]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLIMATE_TWINS_DATA", str(data))
        mp.setenv("CLIMATE_TWINS_GEONAMES", str(geonames))
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def client(built: tuple[Path, dict], geonames_dir: Path) -> Iterator[TestClient]:
    yield from make_client(built[0], geonames_dir)


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
    assert all("place" in m for m in body["matches"])
    for place in (body["query"]["place"], top["place"]):
        assert {"tier", "direction", "base_label", "label"} <= set(place)
        assert isinstance(place["label"], str) and place["label"]
    assert body["query"]["place"]["tier"] == "near"
    assert body["query"]["place"]["name"] == "Midtown"
    assert top["place"]["tier"] == "region"


def test_climate(client: TestClient):
    r = client.get("/climate", params={"lat": NC_QUERY[0], "lon": NC_QUERY[1]})
    assert r.status_code == 200
    assert r.json()["cell"]["lat"] == NC_QUERY[0]
    assert r.json()["place"]["name"] == "Midtown"
    base_label = r.json()["place"]["base_label"]
    assert isinstance(base_label, str) and base_label


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
