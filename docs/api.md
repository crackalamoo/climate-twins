# API

Served by the same FastAPI app as the web UI (`climate_twins/static/`, at `/`). All endpoints are
GET and return JSON unless noted; all units are metric. Errors: 404 when a point isn't on land
(or within 25 km of it; Antarctica isn't indexed), 400/422 for bad parameters, with
`{"detail": "..."}`.

A query is `(lat, lon, align)`; the server snaps `lat/lon` to the nearest land cell (~4.6 km grid).

## `GET /similar?lat&lon&n&exclude_km&suppress_km&align&detail`

| Param | Default | |
|---|---|---|
| `n` | 20 | Number of matches (1–200). |
| `exclude_km` | 1500 | Ignore cells this close to the query. |
| `suppress_km` | 400 | Minimum spacing between matches. |
| `align` | `auto` | `auto` also compares against the query's seasons shifted 6 months; `calendar` compares the same months only. |
| `detail` | false | Include the 12-month arrays for the query and each match. |

Returns `query` (the requested point, its `cell` and `snapped_km`, `summary`, `place`, and with
`detail` its months), `params`, `matches`, `ruler` and `elapsed_ms`. Each match has:
- `rank`, `cell` (lat/lon of the cell centre), `km_from_query`.
- `distance`: lower is more similar, in weighted, standardized feature units. Matches are ranked by it.
- `more_similar_than_pct_of_land`: area-weighted share of land beyond `exclude_km` that is less similar,
  on the heatmap's scale (the PCA projection), so it needn't fall in rank order.
- `seasons_shifted_6_months`.
- `difference_share`: each feature's share of the squared difference.
- `difference_d2`: each feature's part of the squared difference, in `distance`'s units squared
  (they sum to `distance`²).
- `summary`: annual and extreme values.
- `place`: a [Place label](#place-label) for the cell centre, or `null`.
- `ruler_comparison`: how it compares with `ruler.cities` (below), or `null`.

`ruler`: big cities near the query, scored on the same scale, so the UI can say "closer to
Portland's climate than Seattle is" (`ruler()` in `server.py`).

```json
"ruler": {
  "min_km": 50, "max_km": 500, "min_population": 500000,
  "cities": [{"name": "Vancouver", "label": "Vancouver, Canada", "population": 662248,
              "lat": 49.2497, "lon": -123.1193, "km_from_query": 416, "distance": 0.6995}]
}
```

`cities` are those some comparison names plus the most similar one, most similar first; `[]`
when none qualify or there are no place names. `label` adds the country when it differs from the
query point's. A `ruler_comparison` is one of:
- `{"kind": "all"}`: clearly closer than every ruler city;
- `{"kind": "about", "city": i}`: within ±10% of city `i`'s distance;
- `{"kind": "closer", "city": i}`: clearly closer than city `i`, the most similar one it beats.

## `GET /climate?lat&lon`

Monthly normals and summary for the nearest land cell, plus `place` for the requested point.

## `GET /places/search?q&limit=8`

Autocomplete. `q` (2+ characters) is the start of a place's name or ASCII name, or (for cities of
100k+) the whole of a GeoNames alternate name of 3+ characters (`"NYC"`, `"Roma"`), optionally
followed by its region and/or country, as a name or code: `"Portland, Oregon"`, `"portland or"`,
`"Vancouver BC"`, `"Paris France"`. Case- and accent-insensitive; "St" is "Saint". Ranked by
population, weighted by how well the name and qualifier match (`Places.search` in `places.py` has the
rules). `limit` is 1–50.

Returns `[{name, admin1, country, country_code, population, lat, lon, base_label, label, matched}]`,
where `label` (equal to `base_label`) is `"Porto Alegre, Rio Grande do Sul, Brazil"` style, and
`matched` is the alternate name the place was found by (`"Roma"` for Rome), or null.

## `GET /tiles/{z}/{x}/{y}.png?lat&lon&align=auto&exclude_km=1500`

The similarity heatmap as 256×256 Web Mercator PNG tiles, z 0–10 (z≈5 is native resolution).
Coloured by percentile against land beyond `exclude_km` (as `more_similar_than_pct_of_land`),
only roughly the top 5% visible; similar land within `exclude_km` is grey. Colours come from
`/heatmap/legend`. `Cache-Control: public, max-age=86400`.

## `GET /heatmap/legend`

```json
{"stops": [{"top_pct": 5.0, "rgba": [255, 224, 120, 90]}, ...,
           {"top_pct": 0.01, "rgba": [84, 13, 110, 255]}],
 "excluded": {"rgba": [105, 110, 125, 178], "label": "Similar, but within the exclusion radius"}}
```

Alpha is 0–255. Colours interpolate in log(`top_pct`); rarer than the last stop takes its colour.

## `GET /?lat&lon&exclude&suppress&align`

The web UI. Its `<head>` (title, description, Open Graph tags) describes the query in the URL, in
the UI's parameters, for link previews; without a usable query on land, the site (`page.py`).
`og:image` is the same URL's `/preview.png`.

## `GET /preview.png?lat&lon&exclude&suppress&align`

The page's link-preview card, read from the same parameters the same way: a 1200×630 PNG of the
query's heatmap on an Equal Earth world map, its star and top 5 matches, under the place as the
panel names it; without a usable query on land, the site's card (`preview.py`).
`Cache-Control: public, max-age=86400`.

## `GET /health`

Status, index size and build info, `places` (0 without place names), cache sizes, load time.

## Place label

```json
{
  "name": "Talca", "admin1": "Maule", "country": "Chile", "country_code": "CL",
  "population": 197479, "lat": -35.43, "lon": -71.67, "km": 12.3,
  "tier": "near", "direction": null, "base_label": "Talca, Maule, Chile", "in_area": null,
  "label": "near Talca, Maule, Chile", "area": {"admin1": "Maule", "country": "Chile"}
}
```

Names a well-known city (100k+) near the point, else the region it's in (`Places.label_for` in
`places.py` has the rules). `label` is ready to show, in km, by `tier`:
- `exact`: `"Talca, Maule, Chile"` (under 5 km);
- `near`: `"near Talca, Maule, Chile"` (under 25 km);
- `offset`: `"97 km NW of Cairns, Queensland, Australia"`, with `direction`;
- `region`: `"in Western Australia, Australia"`; then `population`, `lat`, `lon` and `km` are `null`.

`area` is where the point itself is (`admin1` `null` where labels hide regions). When the city's
region or country differs from it, the label ends with the point's own instead, held in `in_area`,
and `base_label` is just the city: `"93 km SW of Oviedo, in Castille and León, Spain"`. Clients
rebuild the label from `tier`, `km`, `direction`, `base_label` and `in_area` to show miles.

Without GeoNames data every `place` is `null` and search returns `[]`.
