"""Loading the built index and answering "where else has this climate?" queries."""

import json
import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .features import FEATURES, RAW_VARS, FeatureModel, RawCodec, perceptual, physical, shift_seasons
from .geo import Grid, cos_of_km, great_circle_km, unit_vectors

Align = Literal["auto", "calendar"]

MAX_SNAP_KM = 25.0
# Far enough that a regional climate (the Russian plain, the US Midwest) yields
# distant analogs rather than its own neighbours.
DEFAULT_EXCLUDE_KM = 1500.0
INITIAL_CANDIDATES = 4096
PICK_CHUNK = 1024
MONTHLY_VARS = ("tmax", "tmin", "dewpoint", "ppt", "srad", "clearness", "ws")
QUERY_CACHE_SIZE = 8  # each entry holds a float32 per land cell (~36 MB for the real index)
MAX_REFERENCES = 32  # per cached query: one per exclusion radius used


class NoLandError(LookupError):
    """The query point is not within MAX_SNAP_KM of any indexed land cell."""


def climate_summary(p: Mapping[str, np.ndarray]) -> dict:
    """Headline numbers for one cell's (12,) monthly physical arrays."""
    tmean = (p["tmax"] + p["tmin"]) / 2
    return {
        "annual_mean_temp_c": round(float(tmean.mean()), 1),
        "warmest_month_high_c": round(float(p["tmax"].max()), 1),
        "coldest_month_low_c": round(float(p["tmin"].min()), 1),
        "annual_precip_mm": round(float(p["ppt"].sum())),
        "wettest_month_precip_mm": round(float(p["ppt"].max())),
        "driest_month_precip_mm": round(float(p["ppt"].min())),
        "muggiest_month_dewpoint_c": round(float(p["dewpoint"].max()), 1),
        "mean_solar_radiation_w_m2": round(float(p["srad"].mean())),
        "mean_clearness": round(float(p["clearness"].mean()), 2),
        "mean_wind_m_s": round(float(p["ws"].mean()), 1),
    }


@dataclass(frozen=True)
class QueryDistances:
    """Squared climate distance from one query cell to every land cell (read-only arrays)."""

    qi: int
    align: Align
    variants: np.ndarray  # (v, d) full query vectors: as-is, then seasons shifted
    qp: np.ndarray  # (v, k) their PCA projections
    d2: np.ndarray  # (cells,) min over variants
    sample_d2: np.ndarray  # d2 of each cell in the area-weighted sample, for percentiles
    sample_km: np.ndarray  # km from the query cell to each sample cell
    _refs: dict = field(default_factory=dict, repr=False, compare=False)

    def reference(self, exclude_km: float = 0.0) -> np.ndarray:
        """Sorted sample d2 over land at least exclude_km from the query cell.

        Percentiles are relative to this land only, so a large climate region around
        the query (which the search excludes) doesn't use up the top of the scale.
        Falls back to all land if the exclusion would leave none.
        """
        key = max(float(exclude_km), 0.0)
        ref = self._refs.get(key)
        if ref is None:
            keep = self.sample_d2[self.sample_km >= key] if key > 0 else self.sample_d2
            ref = np.sort(keep if keep.size else self.sample_d2)
            ref.flags.writeable = False
            if len(self._refs) >= MAX_REFERENCES:
                self._refs.clear()
            self._refs[key] = ref  # a racing duplicate computes the same array: harmless
        return ref

    def top_pct(self, d2: np.ndarray | float, exclude_km: float = 0.0) -> np.ndarray:
        """Share (%) of land beyond exclude_km at least as similar as d2: 0 = most similar."""
        ref = self.reference(exclude_km)
        return 100.0 * np.searchsorted(ref, d2, side="right") / len(ref)


class QueryCache:
    """Thread-safe LRU of QueryDistances; concurrent misses on one key compute it once."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict[tuple, QueryDistances] = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict[tuple, threading.Lock] = {}

    def get(self, key: tuple, compute) -> QueryDistances:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                if key in self._items:  # another thread finished it while we waited
                    self._items.move_to_end(key)
                    return self._items[key]
            try:
                value = compute()
                with self._lock:
                    self._items[key] = value
                    while len(self._items) > self.size:
                        self._items.popitem(last=False)
            finally:
                with self._lock:
                    self._key_locks.pop(key, None)
            return value

    def __len__(self) -> int:
        return len(self._items)


def climate_monthly(p: Mapping[str, np.ndarray]) -> dict:
    return {v: [round(float(x), 1) for x in p[v]] for v in MONTHLY_VARS}


class ClimateIndex:
    def __init__(
        self,
        grid: Grid,
        codecs: Mapping[str, RawCodec],
        model: FeatureModel,
        cells: np.ndarray,
        raw: Mapping[str, np.ndarray],
        pcs: np.ndarray,
        sample: np.ndarray,
        meta: Mapping | None = None,
    ):
        self.grid = grid
        self.codecs = dict(codecs)
        self.model = model
        self.cells = cells
        self.raw = dict(raw)
        self.pcs = pcs
        self.sample = sample
        self.meta = dict(meta or {})
        self.norms = np.einsum("ij,ij->i", pcs, pcs)
        self.lat = grid.lat_of(cells)
        self.lon = grid.lon_of(cells)
        self.query_cache = QueryCache(QUERY_CACHE_SIZE)

    @classmethod
    def load(cls, path: Path) -> "ClimateIndex":
        meta = json.loads((path / "model.json").read_text())
        return cls(
            grid=Grid(**meta["grid"]),
            codecs={v: RawCodec(**c) for v, c in meta["codecs"].items()},
            model=FeatureModel.from_dict(meta["model"]),
            cells=np.load(path / "cells.npy"),
            # Raw values are only read for a handful of cells per query: leave them on disk.
            raw={v: np.load(path / f"raw_{v}.npy", mmap_mode="r") for v in RAW_VARS},
            pcs=np.load(path / "pcs.npy"),
            sample=np.load(path / "sample.npy"),
            meta={k: v for k, v in meta.items() if k != "model"},
        )

    def __len__(self) -> int:
        return len(self.cells)

    # -- per-cell data -------------------------------------------------------

    def physical(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.atleast_1d(idx)
        return physical({v: self.raw[v][idx] for v in RAW_VARS}, self.codecs, self.lat[idx])

    def vectors(self, idx: np.ndarray) -> np.ndarray:
        return self.model.vectors(perceptual(self.physical(idx)))

    def locate(self, lat: float, lon: float, max_km: float = MAX_SNAP_KM) -> tuple[int, float]:
        """Nearest land cell to (lat, lon) and its distance in km."""
        row, col = self.grid.row_col(lat, lon)
        row_radius = math.ceil(max_km / self.grid.cell_km) + 1
        col_radius = min(
            math.ceil(row_radius / max(math.cos(math.radians(lat)), 1e-3)), self.grid.nlon // 2
        )
        flat = self.grid.window(row, col, row_radius, col_radius)
        pos = np.minimum(np.searchsorted(self.cells, flat), len(self.cells) - 1)
        cand = pos[self.cells[pos] == flat]
        if cand.size:
            d = great_circle_km(lat, lon, self.lat[cand], self.lon[cand])
            j = int(np.argmin(d))
            if d[j] <= max_km:
                return int(cand[j]), float(d[j])
        raise NoLandError(f"no indexed land within {max_km:g} km of ({lat}, {lon})")

    def positions_at(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Position of the land cell containing each point (no snapping), or -1."""
        flat = self.grid.flat_of(lat, lon).astype(self.cells.dtype)  # same dtype: no 9M-cell cast
        pos = np.minimum(np.searchsorted(self.cells, flat), len(self.cells) - 1)
        return np.where((flat >= 0) & (self.cells[pos] == flat), pos, -1)

    def describe(self, lat: float, lon: float) -> dict:
        i, snap_km = self.locate(lat, lon)
        p = {k: v[0] for k, v in self.physical(i).items()}
        return {
            "cell": self._cell_json(i),
            "snapped_km": round(snap_km, 1),
            "summary": climate_summary(p),
            "monthly": climate_monthly(p),
        }

    # -- search --------------------------------------------------------------

    def similar(
        self,
        lat: float,
        lon: float,
        n: int = 20,
        exclude_km: float = DEFAULT_EXCLUDE_KM,
        suppress_km: float = 400.0,
        align: Align = "auto",
        detail: bool = False,
    ) -> dict:
        """The n most similar cells at least exclude_km from the query and suppress_km apart.

        align="auto" also compares against the query's seasons rolled by six months and
        keeps whichever alignment is closer per cell, so southern-hemisphere analogs of a
        northern location (and vice versa) are found without any equator rule.
        """
        qi, snap_km = self.locate(lat, lon)
        qd = self.distances(qi, align)
        qp, d2 = qd.qp, qd.d2
        picked = self._pick(d2, lat, lon, n, exclude_km, suppress_km)

        q_phys = {k: v[0] for k, v in self.physical(qi).items()}
        matches = []
        for rank, i in enumerate(picked, 1):
            # Exact distances: d2 uses the fast |x|^2 - 2x.q + |q|^2 expansion in float32,
            # which is off by ~1e-5 and so reports ~0.003 for an identical climate.
            per_variant = ((self.pcs[i].astype(np.float64) - qp) ** 2).sum(axis=1)
            shifted = bool(len(qp) > 1 and per_variant[1] < per_variant[0])
            matches.append(
                {
                    "rank": rank,
                    "cell": self._cell_json(i),
                    "km_from_query": round(float(great_circle_km(lat, lon, self.lat[i], self.lon[i])), 0),
                    "distance": round(float(np.sqrt(per_variant.min())), 4),
                    "more_similar_than_pct_of_land": round(100.0 - float(qd.top_pct(d2[i], exclude_km)), 3),
                    "seasons_shifted_6_months": shifted,
                    "difference_share": self._difference_share(qd.variants[int(shifted)], i),
                    **self._climate_json(i, detail),
                }
            )
        return {
            "query": {
                "lat": lat,
                "lon": lon,
                "cell": self._cell_json(qi),
                "snapped_km": round(snap_km, 1),
                "summary": climate_summary(q_phys),
                **({"monthly": climate_monthly(q_phys)} if detail else {}),
            },
            "params": {"n": n, "exclude_km": exclude_km, "suppress_km": suppress_km, "align": align},
            "matches": matches,
        }

    def distances(self, qi: int, align: Align = "auto") -> QueryDistances:
        """Distances from land cell qi to every land cell, cached per (qi, align)."""
        return self.query_cache.get((int(qi), align), lambda: self._compute_distances(int(qi), align))

    def _compute_distances(self, qi: int, align: Align) -> QueryDistances:
        qvec = self.vectors(qi)[0]
        variants = np.stack([qvec, shift_seasons(qvec)] if align == "auto" else [qvec])
        qp = self.model.project(variants)
        d2 = self.norms - 2 * (self.pcs @ qp[0]) + float(qp[0] @ qp[0])
        for q in qp[1:]:
            np.minimum(d2, self.norms - 2 * (self.pcs @ q) + float(q @ q), out=d2)
        np.maximum(d2, 0.0, out=d2)
        sample_d2 = d2[self.sample]
        sample_km = great_circle_km(self.lat[qi], self.lon[qi], self.lat[self.sample], self.lon[self.sample])
        for a in (variants, qp, d2, sample_d2, sample_km):
            a.flags.writeable = False
        return QueryDistances(qi, align, variants, qp, d2, sample_d2, sample_km)

    def _pick(
        self, d2: np.ndarray, lat: float, lon: float, n: int, exclude_km: float, suppress_km: float
    ) -> list[int]:
        """Greedy spatial non-max suppression over cells in order of climate distance.

        Cells are visited best-first (ties by index) and a cell is picked unless it is
        within exclude_km of the query or suppress_km of an earlier pick -- the same as
        greedy suppression over every cell, but visiting only as many cells as needed:
        bands of d2 holding ~INITIAL_CANDIDATES, then 8x more, and so on.
        """
        if n <= 0:
            return []
        total = len(d2)
        query = unit_vectors(lat, lon)
        cos_exclude, cos_suppress = cos_of_km(exclude_km), cos_of_km(suppress_km)
        picks_xyz = np.empty((n, 3))
        picked: list[int] = []
        lo, k = -np.inf, max(INITIAL_CANDIDATES, 1)
        while True:
            hi = float(np.partition(d2, k)[k]) if k < total else np.inf
            band = np.flatnonzero((d2 >= lo) & (d2 < hi))
            band = band[np.argsort(d2[band], kind="stable")]
            xyz = unit_vectors(self.lat[band], self.lon[band])
            if exclude_km > 0:
                far = xyz @ query <= cos_exclude
                band, xyz = band[far], xyz[far]
            for start in range(0, len(band), PICK_CHUNK):
                chunk = xyz[start : start + PICK_CHUNK]
                first_new = len(picked)
                if suppress_km > 0 and picked:
                    clear = (chunk @ picks_xyz[:first_new].T).max(axis=1) <= cos_suppress
                    survivors = np.flatnonzero(clear)
                else:
                    survivors = range(len(chunk))
                # Survivors are clear of earlier chunks' picks; check those made in this chunk.
                for j in survivors:
                    if suppress_km > 0 and len(picked) > first_new:
                        if (picks_xyz[first_new : len(picked)] @ chunk[j]).max() > cos_suppress:
                            continue
                    picks_xyz[len(picked)] = chunk[j]
                    picked.append(int(band[start + j]))
                    if len(picked) == n:
                        return picked
            if hi == np.inf:
                return picked
            lo, k = hi, min(k * 8, total)

    def _difference_share(self, qvec: np.ndarray, i: int) -> dict:
        """Share of the (full, unprojected) squared distance contributed by each feature."""
        diff = (self.vectors(i)[0] - qvec).reshape(len(FEATURES), -1)
        per = (diff.astype(np.float64) ** 2).sum(axis=1)
        total = per.sum() or 1.0
        return {f: round(float(x / total), 3) for f, x in zip(FEATURES, per)}

    def _cell_json(self, i: int) -> dict:
        return {"index": int(i), "lat": round(float(self.lat[i]), 4), "lon": round(float(self.lon[i]), 4)}

    def _climate_json(self, i: int, detail: bool) -> dict:
        p = {k: v[0] for k, v in self.physical(i).items()}
        out = {"summary": climate_summary(p)}
        if detail:
            out["monthly"] = climate_monthly(p)
        return out
