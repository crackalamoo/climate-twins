# UI ⇄ API contract

The web UI (`climate_twins/static/`) is served at `/` by the same FastAPI app as
the API. All endpoints are GET and return JSON unless noted.

## Query identity

A "query" means `(lat, lon, align)`. The server snaps `lat/lon` to the nearest land
cell. Per query (keyed by `(cell index, align)`) it caches what every request needs,
~2 MB: the query's projected vectors and its percentile reference, a sample of land
(LRU, 16 queries). `/similar` also needs the distance to every land cell, ~36 MB; only
the latest is kept, so moving the sliders on one query doesn't recompute it. A tile
computes distances for just the cells under its pixels (a few ms), so tiles never
depend on that scan and visitors browsing at once don't evict each other's.

## Existing endpoints (extended)

`GET /similar?lat&lon&n&exclude_km&suppress_km&align&detail`:

| Param | Default | |
|---|---|---|
| `n` | 20 | Number of results (1–200). |
| `exclude_km` | 1500 | Ignore cells this close to the query. |
| `suppress_km` | 400 | Minimum spacing between results. |
| `align` | `auto` | `auto` also compares against the query's seasons shifted 6 months (hemisphere flip, or a tropical wet season across the equator); `calendar` compares the same months only. |
| `detail` | false | Include the 12-month arrays for the query and each match. |

Each match has:
- `cell` (lat/lon of the cell centre) and `km_from_query`.
- `distance`: lower is more similar, in units of the weighted, standardized features.
- `more_similar_than_pct_of_land`: the area-weighted share of land beyond
  `exclude_km` that is less similar.
- `seasons_shifted_6_months`.
- `difference_share`: which features make up the difference: each one's share of the
  squared difference over all features (`distance` is the same, up to the PCA truncation).
- `summary`: annual and extreme values.
- `place`: a **Place label** (below), or `null`. The query's `place` is for the
  requested `lat/lon`; a match's is for its cell centre.

`GET /climate?lat&lon`: as before, plus a `place` field (for the requested `lat/lon`).

`GET /health`: as before, plus `places` (number of GeoNames places loaded; 0 when
place names are unavailable), `query_cache_entries` and `scan_cache_entries`.

Errors: 404 (not on land or within 25 km of it), 422 (bad params), with
`{"detail": "..."}`.

Place names load from `data/geonames/` (or `$CLIMATE_TWINS_GEONAMES`) if present;
the first start caches them in `places.npz` there (~1 s).

### Place label

```json
{
  "name": "Talca",
  "admin1": "Maule",
  "country": "Chile",
  "country_code": "CL",
  "population": 197479,
  "lat": -35.43, "lon": -71.67,
  "km": 12.3,
  "tier": "near",
  "direction": null,
  "base_label": "Talca, Maule, Chile",
  "label": "near Talca, Maule, Chile",
  "area": {"admin1": "Maule", "country": "Chile"}
}
```

Labels only name places people are likely to have heard of, using population as the
fame signal:
1. A **city of 100,000+** people within 500 km of the cell, chosen by size against
   distance: the highest population / (km + 15)³. A big city a bit farther beats a
   smaller one nearby, but a metropolis doesn't win from hundreds of km away.
2. Otherwise just the **region** the cell is in (its `area`, below), e.g.
   `"in Western Australia, Australia"`, or the country alone where the region is unknown.
3. Otherwise (no `area`) `null`.

The region (admin1) only appears in labels where it helps:
- Not in countries whose regions are obscure outside them (`REGIONLESS_COUNTRIES`
  in `places.py`: Africa except South Africa, the Middle East, Central Asia, the
  Nordics, mainland Southeast Asia, Central America, and a few more). There
  labels read `"Ubari, Libya"` / `"in Libya"`.
- Not when it repeats the city's name: `"Kabul, Afghanistan"`, not
  `"Kabul, Kabul, Afghanistan"`; the same for `"Krasnoyarsk Krai"`,
  `"St.-Petersburg"` and other spellings of the name GeoNames lists (`"Cuzco"`).
- Not when it is only a direction (`"Centre"`, `"Northern Region"`).
`admin1` always holds the raw region, and search results always show it.

Sections of a populated place (GeoNames feature code `PPLX`: suburbs and localities)
are never used as labels, though search still finds them.

`label` is a ready-to-display metric string, by `tier`:
- `"exact"`: `"Talca, Maule, Chile"` when `km` < 5.
- `"near"`: `"near Talca, Maule, Chile"` when `km` < 25.
- `"offset"`: `"97 km NW of Cairns, Queensland, Australia"` otherwise. `direction`
  is the 8-point compass direction of the cell as seen from the place (`null` for
  the other tiers).
- `"region"`: `"in Western Australia, Australia"` when no city qualifies. Then
  `base_label` is `"Western Australia, Australia"`, `name` is the region (or the
  country when admin1 is unknown), and `population`, `lat`, `lon` and `km` are `null`.

`area` is where the cell itself is: `{admin1, country}`, with `admin1` `null` where
labels don't show regions (the rules above) or the region is unknown. It can differ
from the city's: the labelling city may be across a border or a sea, as in
`"210 km SW of Al Ḩudaydah, Yemen"` with `area` `{"admin1": null, "country": "Eritrea"}`.
It's the name of the Natural Earth admin-1 polygon the cell is in (or, for a coastal
cell just offshore, the nearest one within 0.25°), or `null` if there's none (open
sea, the Caspian, a few tiny islands, or no boundary data). Each polygon is named in
GeoNames' terms, using only names two sources agree on, never a majority of towns:
- **Country**: the one all its GeoNames towns are filed under (3+ towns), so places
  are named as GeoNames names them (Western Sahara, French Guiana); else Natural
  Earth's (GeoNames admin1 code's country first, so a disputed region gets the
  country GeoNames files its towns under). Neither (Somaliland, Siachen): `area` is `null`.
- **Region**: the polygon's GeoNames admin1 code, if GeoNames gives that code the
  polygon's own name (some codes are stale: Svalbard's NO.21 is now Trøndelag). Else
  the region all its towns in that country are in (3+). Else unknown: labels show the
  country alone.
A region is the polygon's own or none, never a majority of nearby towns. Natural
Earth predates some reforms, so a few areas get their former region (Papua for
what's now Central Papua, Almaty Region for Zhetysu), never one they weren't in. The
nearest towns aren't used at all: GeoNames maps some countries far more densely than
their neighbours, so the towns nearest a cell in South Sudan are mostly in Uganda.
The UI's short name for a place not in or near a city is `area.admin1` when the label
names that region, else `area.country`.

`base_label` is the place's own name part (`"Cairns, Queensland, Australia"`). The
UI rebuilds the label from `tier`, `km`, `direction` and `base_label` so it can show
miles; it falls back to `label` if `tier` is missing.

Duplicate parts are dropped: when `admin1` equals `name`, or `admin1` is missing.
`admin1` and `country` are `null` when unknown.

Place names are optional: when the server has no GeoNames data, every `place` is
`null` and `/places/search` returns `[]`.

## New endpoints

### `GET /places/search?q=<text>&limit=8`

Autocomplete for the search box.
- Matching is a case- and accent-insensitive **prefix** match on the place's name
  and its ASCII name. A query of 2+ characters is required, otherwise the result
  is `[]`.
- Results are sorted by population, descending.
- Returns `[{name, admin1, country, country_code, population, lat, lon, base_label, label}]`.
  Here `label` (equal to `base_label`) is `"Porto Alegre, Rio Grande do Sul, Brazil"` style, with duplicates
  dropped as above (so Porto, whose admin1 is also "Porto", is `"Porto, Portugal"`).
  `limit` is 1–50.

### `GET /tiles/{z}/{x}/{y}.png?lat&lon&align=auto&exclude_km=1500`

The similarity heatmap as standard XYZ Web Mercator tiles.
- 256×256 RGBA PNG.
- Each pixel takes the land cell under its center (nearest-neighbour). Pixels over
  ocean, Antarctica, or areas below the visibility cutoff are fully transparent.
- Color is by **percentile**: the share of land area (area-weighted) **beyond
  `exclude_km` from the query** that is *less* similar. This is the same reference
  as `/similar`'s `more_similar_than_pct_of_land`. Only roughly the top 5% is
  visible, with stronger color for rarer similarity. The colors come from
  `/heatmap/legend`.
- Land within `exclude_km` of the query that would be visible is drawn in the
  legend's `excluded` grey. It is similar, but skipped by the search, and it
  doesn't use up the top of the scale. With `exclude_km=0`, percentiles are over
  all land and there is no grey.
- `z` runs 0–6; the data is about 4.6 km per cell, so z≈5 is native resolution. The
  client sets source `maxzoom: 6` and overzooms beyond that. (The server also
  accepts z up to 10.)
- Errors: 404 if the query isn't near land; 400 for out-of-range `z/x/y`.
- Response header `Cache-Control: public, max-age=86400`. Tiles are deterministic
  for a given build and query.

### `GET /heatmap/legend`

```json
{"stops": [
  {"top_pct": 5.0,  "rgba": [255, 224, 120, 90]},
  {"top_pct": 1.0,  "rgba": [253, 141, 60, 160]},
  {"top_pct": 0.1,  "rgba": [215, 25, 50, 220]},
  {"top_pct": 0.01, "rgba": [84, 13, 110, 255]}
],
 "excluded": {"rgba": [105, 110, 125, 178], "label": "Similar, but within the exclusion radius"}}
```

(Alpha is 0–255, as in the PNG.) Anything rarer than the last stop uses its colour.

`top_pct` is "top X% most similar land". Color is interpolated between stops in
log(top_pct), and anything below the 5% stop is transparent. The UI renders its
legend from this endpoint.

## UI behaviour summary

- MapLibre GL JS (CDN) with the OpenFreeMap `positron` basemap
  (`https://tiles.openfreemap.org/styles/positron`). The heatmap raster layer goes
  **before the `water` layer**, so water, borders, roads and labels draw on top.
  Raster opacity is about 0.8, with linear resampling.
- Inputs:
  - the search box (`/places/search`);
  - a "my location" button (browser geolocation);
  - clicking the map, which runs a new query there;
  - sliders for `exclude_km` and `suppress_km`;
  - a season-shift checkbox (`align=auto|calendar`).
- Outputs:
  - the heatmap;
  - numbered pins for the matches and a star for the query;
  - a results list (place label, km away, `distance` shown as "climate difference",
    percentile, ⇄ when seasons are shifted);
  - clicking a result flies the map to it and shows a comparison panel. The panel
    has 12-month charts (temperature band tmin–tmax plus precipitation bars, query
    vs match, with seasons aligned when shifted) built with Observable Plot (CDN),
    and a bar for `difference_share`.
- State lives only in the URL (`?lat&lon&exclude&suppress&align`). No
  localStorage.
- Units: a sidebar toggle switches the display between metric (°C, mm, km) and US
  (°F, in, mi). The default is US iff the browser locale's region is the US
  (`new Intl.Locale(navigator.language).maximize().region`). The choice is kept in
  memory only (not in the URL), and switching re-renders the last response
  without refetching. The API, and the `exclude`/`suppress` URL parameters, stay
  in km; the slider readouts show miles rounded to 10.
