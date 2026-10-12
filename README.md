# Climate Twins

Finds places whose climate resembles a given location, beyond the obvious nearby
ones. It searches every ~4 km land cell on Earth (9.0 M cells, Antarctica excluded)
using TerraClimate's 1991–2020 monthly normals.

## Setup

```sh
uv sync

# 1. TerraClimate v1.1 1991–2020 climatology (~530 MB) into data/raw/
for v in tmax tmin ppt vap srad ws; do
  curl -fo data/raw/TerraClimate_19912020_$v.nc \
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/TERRACLIMATE_ALL/climatology/TerraClimate_19912020_$v.nc"
done

# 2. Optional, for place names: GeoNames (~55 MB) into data/geonames/
mkdir -p data/geonames
for f in cities500.zip admin1CodesASCII.txt countryInfo.txt; do
  curl -fo data/geonames/$f "https://download.geonames.org/export/dump/$f"
done
unzip -o data/geonames/cities500.zip -d data/geonames/
# and Natural Earth admin-1 boundaries (~15 MB): without them labels only name cities,
# never which country or region a point itself is in
mkdir -p data/naturalearth
curl -fo data/naturalearth/admin1.zip \
  https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip
unzip -o data/naturalearth/admin1.zip -d data/naturalearth/

# 3. Build the index (~30 s, ~3.4 GB peak RAM) -> data/processed/ (~1.8 GB)
uv run climate-twins-build

# 4. Serve the UI and API on 0.0.0.0:8000 (~1.0 GB RAM)
uv run climate-twins-serve
```

Tests: `uv run pytest -q` (synthetic data; doesn't need the real index).

## API

`/similar`, `/climate`, `/places/search`, `/tiles/{z}/{x}/{y}.png`,
`/heatmap/legend` and `/health`; all metric. See `docs/ui-api.md` for the full contract.

```sh
curl 'localhost:8000/similar?lat=37.77&lon=-122.42&n=5'
```

## Method

1. **Features**: 12 monthly values each of daily high and low temperature, dewpoint
   (from vapor pressure), `log(mm + 15)` precipitation, sky clearness (surface solar
   radiation ÷ top-of-atmosphere radiation, so sunny vs grey without the latitude
   effect) and wind speed: 72 numbers per cell.
2. **Standardize** each feature over all land (area-weighted), then **weight** them
   so that, among close matches, highs and lows count most, then humidity and rain,
   then sun and wind. Weights are `WEIGHTS` in `climate_twins/features.py`;
   rebuild after changing them.
3. **PCA** (fit on the data plus a 6-month season-shifted copy, no whitening) keeps
   enough components for 99% of the variance.
4. **Search**: brute force over all cells, taking the better of the normal and the
   6-month-shifted seasons, then keep the best cells at least `suppress_km` apart and
   beyond `exclude_km` of the query.

## Limitations

- Dewpoint is coarse (TerraClimate stores vapor pressure in 0.1 kPa steps).
- Cells are ~4 km averages with no elevation correction; a city in a steep valley can
  differ from its cell.
- Normals only: nothing about variability, extremes or storms.
- Place labels name a nearby large city, not the cell itself (rules in
  `docs/ui-api.md`). Names are from [GeoNames](https://www.geonames.org/)
  (CC BY 4.0); which country a point is in, from [Natural Earth](https://www.naturalearthdata.com/)
  (public domain).
