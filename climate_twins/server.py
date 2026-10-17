"""HTTP API over the climate index.

    uv run climate-twins-serve            # 0.0.0.0:8000, index from data/processed

    GET /health
    GET /climate?lat=..&lon=..
    GET /similar?lat=..&lon=..[&n=20&exclude_km=1500&suppress_km=400&align=auto&detail=false]
    GET /places/search?q=..[&limit=8]
    GET /tiles/{z}/{x}/{y}.png?lat=..&lon=..[&align=auto&exclude_km=1500]
    GET /heatmap/legend

See docs/ui-api.md.
"""

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request, Response
from fastapi.staticfiles import StaticFiles

from . import tiles
from .index import DEFAULT_EXCLUDE_KM, MAX_SNAP_KM, Align, ClimateIndex, NoLandError
from .boundaries import Boundaries
from .places import CITY_KM, Places

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "processed"
DEFAULT_GEONAMES = ROOT / "data" / "geonames"
DEFAULT_BOUNDARIES = ROOT / "data" / "naturalearth"
TILE_CACHE_CONTROL = "public, max-age=86400"
# The UI's ES modules import each other, so a browser mixing a fresh app.js with a stale
# cached map.js fails to start. Revalidate every load; ETags keep unchanged files a cheap 304.
STATIC_CACHE_CONTROL = "no-cache"
DEFAULT_PORT = 8000
STATIC_DIR = Path(__file__).resolve().parent / "static"
# The ruler: well-known cities near the query, scored like matches, so the UI can say a match is
# "closer to Portland's climate than Seattle is". Cities are label-eligible ones of
# RULER_MIN_POPULATION+, out to the labels' CITY_KM. Nearer than RULER_MIN_KM is the query's own metro
# area (Vancouver, WA is 15 km from Portland), where "closer than Vancouver" says nothing.
RULER_MIN_KM = 50.0
# Only major cities: with every 100k+ city, nearly every match is "about as close as" some suburb
# (Renton, Coquitlam, East Los Angeles). A city within RULER_MIN_KM of a more populous one is part of
# its metro area too, and folds into it (the Bronx into New York).
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
    country ("Portland, Oregon"); whole names first, then the most populous."""
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
