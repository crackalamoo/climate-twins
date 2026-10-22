"""HTTP API over the climate index.

    uv run climate-twins-serve            # 0.0.0.0:8000, index from data/processed

    GET /health
    GET /climate?lat=..&lon=..
    GET /similar?lat=..&lon=..[&n=20&exclude_km=1500&suppress_km=400&align=auto&detail=false]
    GET /places/search?q=..[&limit=8]
    GET /tiles/{z}/{x}/{y}.png?lat=..&lon=..[&align=auto&exclude_km=1500]
    GET /heatmap/legend
    GET /?lat=..&lon=..   the web UI, its <head> describing the query
    GET /preview.png?lat=..&lon=..   its link-preview card

See docs/api.md.
"""

import argparse
import functools
import hashlib
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import page, preview, tiles
from .index import DEFAULT_EXCLUDE_KM, MAX_SNAP_KM, Align, ClimateIndex, NoLandError
from .boundaries import Boundaries
from .places import CITY_KM, Places

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "processed"
DEFAULT_GEONAMES = ROOT / "data" / "geonames"
DEFAULT_BOUNDARIES = ROOT / "data" / "naturalearth"
TILE_CACHE_CONTROL = "public, max-age=86400"
# The UI's bundle files have fixed names and import each other, so a browser mixing a fresh app.js
# with a stale cached maplibre-gl-shared.js fails to start. Revalidate every load; ETags keep
# unchanged files a cheap 304.
STATIC_CACHE_CONTROL = "no-cache"
DEFAULT_PORT = 8000
STATIC_DIR = Path(__file__).resolve().parent / "static"
# The page's description counts matches as the UI lists them (N_RESULTS in web/app.ts).
PAGE_MATCHES = 20
# The ruler: well-known cities near the query, scored like matches, so the UI can say a match is
# "closer to Portland's climate than Seattle is". Cities are label-eligible ones of
# RULER_MIN_POPULATION+, out to the labels' CITY_KM. Nearer than RULER_MIN_KM is the query's own metro
# area (Vancouver, WA is 15 km from Portland), where "closer than Vancouver" says nothing.
RULER_MIN_KM = 50.0
# Only major cities: with every 100k+ city, nearly every match is "about as close as" some suburb
# (Renton, Coquitlam). A city within RULER_MIN_KM of a more populous one folds into its metro area.
RULER_MIN_POPULATION = 500_000
# "About as close" as a ruler city: a match's distance within this fraction of the city's, either
# way. A judgment call: distances within ~10% are hard to tell apart in the charts.
RULER_BAND = 0.1

Lat = Annotated[float, Query(ge=-90, le=90)]
Lon = Annotated[float, Query(ge=-180, le=180)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    app.state.index = ClimateIndex.load(Path(os.environ.get("CLIMATE_TWINS_DATA", DEFAULT_DATA)))
    boundaries = Boundaries.load(Path(os.environ.get("CLIMATE_TWINS_BOUNDARIES", DEFAULT_BOUNDARIES)))
    app.state.places = Places.load(Path(os.environ.get("CLIMATE_TWINS_GEONAMES", DEFAULT_GEONAMES)), boundaries)
    app.state.load_seconds = round(time.time() - t0, 1)
    yield


def place_label(request: Request, lat: float, lon: float) -> dict | None:
    places: Places | None = request.app.state.places
    return places.label_for(lat, lon) if places is not None else None


def ruler(request: Request, lat: float, lon: float, align: Align, distances: list[float]) -> tuple[dict, list]:
    """Compare each match `distance` with big cities RULER_MIN_KM..CITY_KM from (lat, lon), scored
    on the same scale. Returns the ruler (its cities: those a comparison names, plus the most
    similar one, most similar first) and, per match, its comparison or None:
    - {"kind": "all"}: clearly closer (by more than RULER_BAND) than every city;
    - {"kind": "about", "city": i}: within RULER_BAND of a city; of several, the nearest in distance;
    - {"kind": "closer", "city": i}: else, clearly closer than a city; of several, the most similar
      one it beats (the strongest claim it can make);
    - None: no cities, or clearly less similar than all of them.
    `city` indexes the ruler's `cities`. Every candidate is scored: a few hundred cells at most.
    """
    out = {"min_km": RULER_MIN_KM, "max_km": CITY_KM, "min_population": RULER_MIN_POPULATION, "cities": []}
    none: list = [None] * len(distances)
    places: Places | None = request.app.state.places
    if places is None:
        return out, none
    idx, km = places.big_cities(lat, lon, RULER_MIN_KM, CITY_KM, RULER_MIN_POPULATION, RULER_MIN_KM)
    if idx.size == 0:
        return out, none
    dist = request.app.state.index.distances_to(lat, lon, align, places.lat[idx], places.lon[idx])
    ok = ~np.isnan(dist)  # a city with no land cell within reach (a tiny island) is skipped
    idx, km, dist = idx[ok], km[ok], dist[ok]
    if idx.size == 0:
        return out, none
    picks: list[tuple[str, int] | None] = []
    for d in distances:
        if np.all(d < dist * (1 - RULER_BAND)):
            picks.append(("all", -1))
        elif (tie := np.flatnonzero(np.abs(d - dist) <= RULER_BAND * dist)).size:
            picks.append(("about", int(tie[np.argmin(np.abs(d - dist[tie]))])))
        elif (beaten := np.flatnonzero(d < dist * (1 - RULER_BAND))).size:
            picks.append(("closer", int(beaten[np.argmin(dist[beaten])])))
        else:
            picks.append(None)
    keep = sorted({int(np.argmin(dist))} | {p[1] for p in picks if p and p[1] >= 0}, key=lambda j: (dist[j], j))
    position = {j: k for k, j in enumerate(keep)}
    # Near the query, the city's name alone is clear, unless it's across a border: from Portland,
    # "Vancouver" reads as Vancouver, WA (too close to be a candidate), so it's "Vancouver, Canada".
    here = places.locate(lat, lon)
    for j in keep:
        i = int(idx[j])
        foreign = here is not None and places.country_code[i] != here[0] and places.country[i]
        out["cities"].append(
            {
                "name": places.name[i],
                "label": f"{places.name[i]}, {places.country[i]}" if foreign else places.name[i],
                "population": int(places.population[i]),
                "lat": round(float(places.lat[i]), 4),
                "lon": round(float(places.lon[i]), 4),
                "km_from_query": round(float(km[j]), 0),
                "distance": round(float(dist[j]), 4),
            }
        )
    comparisons = [None if p is None else {"kind": p[0]} if p[1] < 0 else {"kind": p[0], "city": position[p[1]]} for p in picks]
    return out, comparisons


app = FastAPI(title="Climate Twins", lifespan=lifespan)


def no_land(e: NoLandError) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"{e}. Points must be on land (Antarctica is not indexed), within {MAX_SNAP_KM:g} km of the coast.",
    )


@app.get("/health")
def health(request: Request) -> dict:
    index: ClimateIndex = request.app.state.index
    return {
        "status": "ok",
        "cells": len(index),
        "pca_components": index.model.k,
        "pca_variance_kept": round(float(index.model.explained_variance_ratio.sum()), 4),
        "built": index.meta.get("built"),
        "source": index.meta.get("source"),
        "places": len(request.app.state.places) if request.app.state.places is not None else 0,
        "boundaries": len(places.boundaries) if (places := request.app.state.places) and places.boundaries else 0,
        "query_cache_entries": len(index.query_cache),
        "scan_cache_entries": len(index.scan_cache),
        "load_seconds": request.app.state.load_seconds,
    }


@app.get("/climate")
def climate(request: Request, lat: Lat, lon: Lon) -> dict:
    """Monthly normals and summary for the land cell nearest (lat, lon)."""
    try:
        result = request.app.state.index.describe(lat, lon)
    except NoLandError as e:
        raise no_land(e) from e
    result["place"] = place_label(request, lat, lon)
    return result


@app.get("/similar")
def similar(
    request: Request,
    lat: Lat,
    lon: Lon,
    n: Annotated[int, Query(ge=1, le=200)] = 20,
    exclude_km: Annotated[float, Query(ge=0, le=20000, description="ignore cells this close to the query")] = DEFAULT_EXCLUDE_KM,
    suppress_km: Annotated[float, Query(ge=0, le=20000, description="minimum spacing between results")] = 400,
    align: Annotated[Align, Query(description="auto: also match seasons shifted 6 months")] = "auto",
    detail: bool = False,
) -> dict:
    t0 = time.perf_counter()
    try:
        result = request.app.state.index.similar(lat, lon, n, exclude_km, suppress_km, align, detail)
    except NoLandError as e:
        raise no_land(e) from e
    result["query"]["place"] = place_label(request, lat, lon)
    for m in result["matches"]:
        m["place"] = place_label(request, m["cell"]["lat"], m["cell"]["lon"])
    result["ruler"], comparisons = ruler(request, lat, lon, align, [m["distance"] for m in result["matches"]])
    for m, comparison in zip(result["matches"], comparisons):
        m["ruler_comparison"] = comparison
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


@app.get("/places/search")
def places_search(
    request: Request,
    q: Annotated[str, Query(max_length=100)] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 8,
) -> list[dict]:
    """Autocomplete: places whose name starts with q, optionally followed by their region and/or
    country ("Portland, Oregon"). See Places.search for ranking."""
    places: Places | None = request.app.state.places
    return places.search(q, limit) if places is not None else []


@app.get("/tiles/{z}/{x}/{y}.png", response_class=Response)
def tile(
    request: Request,
    z: Annotated[int, PathParam()],
    x: Annotated[int, PathParam()],
    y: Annotated[int, PathParam()],
    lat: Lat,
    lon: Lon,
    align: Annotated[Align, Query()] = "auto",
    exclude_km: Annotated[float, Query(ge=0, le=20000)] = DEFAULT_EXCLUDE_KM,
) -> Response:
    """The similarity heatmap for the query (lat, lon, align) as a 256x256 PNG tile.

    Percentiles are relative to land beyond exclude_km, matching /similar; similar land
    inside it is grey."""
    try:
        tiles.check_tile(z, x, y)
    except tiles.TileRangeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    index: ClimateIndex = request.app.state.index
    try:
        qi, _ = index.locate(lat, lon)
    except NoLandError as e:
        raise no_land(e) from e
    png = tiles.render_tile(index, index.query(qi, align), z, x, y, exclude_km, center=(lat, lon))
    return Response(content=png, media_type="image/png", headers={"Cache-Control": TILE_CACHE_CONTROL})


@app.get("/heatmap/legend")
def heatmap_legend() -> dict:
    return tiles.legend()


@dataclass(frozen=True)
class PageQuery:
    """The query in a page's URL, rounded and clamped; None where /similar's default applies."""

    lat: float
    lon: float
    align: Align
    exclude_km: float | None
    suppress_km: float | None


def page_query(request: Request) -> PageQuery | None:
    """The query in the page's URL, read the way the UI reads it (unusable values fall back to
    defaults), or None without a usable point."""
    q = request.query_params

    def num(key: str) -> float | None:
        try:
            v = float(q[key])
        except (KeyError, ValueError):
            return None
        return v if math.isfinite(v) else None

    lat, lon = num("lat"), num("lon")
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    exclude, suppress = num("exclude"), num("suppress")
    return PageQuery(
        round(lat, 4), round(lon, 4), "calendar" if q.get("align") == "calendar" else "auto",
        None if exclude is None else min(max(exclude, 0.0), 20000.0),
        None if suppress is None else min(max(suppress, 0.0), 20000.0),
    )


# Each view of a shared link asks for the same page and card, so their matches are kept (a few
# hundred bytes each).
@functools.lru_cache(maxsize=1024)
def cached_page_matches(index: ClimateIndex, q: PageQuery) -> tuple[tuple[float, float, float], ...] | None:
    """The query's matches as (lat, lon, top % of land), best first, or None off land."""
    options = {k: v for k, v in (("exclude_km", q.exclude_km), ("suppress_km", q.suppress_km)) if v is not None}
    try:
        result = index.similar(q.lat, q.lon, PAGE_MATCHES, align=q.align, **options)
    except NoLandError:
        return None
    return tuple((m["cell"]["lat"], m["cell"]["lon"], 100 - m["more_similar_than_pct_of_land"]) for m in result["matches"])


# Their labels too (a few KB each).
@functools.lru_cache(maxsize=1024)
def cached_page_labels(index: ClimateIndex, places: Places, q: PageQuery) -> tuple[dict, list] | None:
    """The Place labels of the query and its matches (those the description names; None for the
    rest), or None off land or where nothing is near enough to name."""
    matches = cached_page_matches(index, q)
    if matches is None or (place := places.label_for(q.lat, q.lon)) is None:
        return None
    named = [places.label_for(lat, lon) for lat, lon, _ in matches[: page.NAMED_MATCHES]]
    return place, named + [None] * (len(matches) - len(named))


def page_labels(request: Request, q: PageQuery | None) -> tuple[dict, list] | None:
    places: Places | None = request.app.state.places
    return cached_page_labels(request.app.state.index, places, q) if q and places else None


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request) -> Response:
    """The web UI, its <head> describing the query in the URL (page.py). Revalidated like the
    static files."""
    place, matches = page_labels(request, page_query(request)) or (None, None)
    image = request.url_for("preview_image").replace(query=request.url.query)
    body = page.render((STATIC_DIR / "index.html").read_text(), str(request.url), str(image), place, matches)
    etag = f'"{hashlib.sha256(body.encode()).hexdigest()[:32]}"'
    headers = {"Cache-Control": STATIC_CACHE_CONTROL, "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return HTMLResponse(body, headers=headers)


# Cards are ~100-150 KB: this keeps at most ~20 MB of them.
@functools.lru_cache(maxsize=128)
def cached_preview(index: ClimateIndex, places: Places | None, q: PageQuery | None) -> bytes:
    """The card for q, which has labels (page_labels), or with None the site's."""
    boundaries = places.boundaries if places else None
    if q is None:
        return preview.render(index, boundaries, page.SITE_DESCRIPTION.rstrip("."), page.SITE_NAME)
    center = (q.lat, q.lon)
    return preview.render(
        index, boundaries, *page.card_lines(cached_page_labels(index, places, q)[0]),
        index.query(index.locate(*center)[0], q.align), DEFAULT_EXCLUDE_KM if q.exclude_km is None else q.exclude_km,
        center, cached_page_matches(index, q),
    )


@app.get("/preview.png", response_class=Response)
def preview_image(request: Request) -> Response:
    """The page's link-preview card (preview.py) for the query in the URL, read as the page reads it:
    the site's where the page describes the site."""
    q = page_query(request)
    png = cached_preview(request.app.state.index, request.app.state.places, q if page_labels(request, q) else None)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": TILE_CACHE_CONTROL})


class RevalidatedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = STATIC_CACHE_CONTROL
        return response


# The web UI. Mounted at "/" and so must stay after every API route.
app.mount("/", RevalidatedStaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the Climate Twins API.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="index directory from climate-twins-build")
    args = p.parse_args()
    os.environ["CLIMATE_TWINS_DATA"] = str(args.data)
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
