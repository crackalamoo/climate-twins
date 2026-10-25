"""Loading the built index and answering "where else has this climate?" queries."""

import json
import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
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
# Matches are picked by the scan's projected distance, this many times as many as asked for, and
# the closest by full distance are kept: the ranking, `distance` and the verdicts then agree. More
# gains nothing measurable (the spacing has already suppressed the rest), and past ~1.5x the
# picker can need a far larger band of cells when most near the query are excluded (~4x the time).
RESORT_FACTOR = 1.5
MONTHLY_VARS = ("tmax", "tmin", "dewpoint", "ppt", "srad", "clearness", "ws")
# Per query, tiles need only its projected vectors and its percentile references (distances to a
# sample of land), ~3 MB: enough are kept for everyone browsing at once. /similar needs the distance
# to every land cell (a float32 each, ~36 MB): only the latest is kept, for slider tweaks.
QUERY_CACHE_SIZE = 16
SCAN_CACHE_SIZE = 1
MAX_REFERENCES = 8  # per cached query: one per exclusion radius used (and projected or full)


def _d2(qp: np.ndarray, qq: np.ndarray, pcs: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Squared distance from each row of pcs to the nearest of the projected query vectors qp:
    |x|^2 - 2x.q + |q|^2, the min over them, clipped at 0 (float32 rounding)."""
    # float(): a NumPy float64 scalar would promote the float32 result to float64 (twice the memory).
    d2 = np.asarray(norms - 2 * (pcs @ qp[0]) + float(qq[0]))  # an array even for a single cell
    for q, q2 in zip(qp[1:], qq[1:]):
        np.minimum(d2, norms - 2 * (pcs @ q) + float(q2), out=d2)
    return np.maximum(d2, 0.0, out=d2)


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
class Query:
    """One query cell, ready to measure climate distances from (read-only arrays). Distances to
    cells come from ClimateIndex.d2_at (a few: a tile's) or ClimateIndex.scan (every land cell)."""

    qi: int
    align: Align
    variants: np.ndarray  # (v, d) full query vectors: as-is, then seasons shifted
    qp: np.ndarray  # (v, k) their PCA projections
    qq: np.ndarray  # (v,) their squared norms
    sample_d2: np.ndarray  # d2 of each cell in the area-weighted sample, for percentiles
    sample_full_d2: np.ndarray  # the same, unprojected: for matches, ranked by full distance
    sample_km: np.ndarray  # km from the query cell to each sample cell
    _refs: dict = field(default_factory=dict, repr=False, compare=False)

    def reference(self, exclude_km: float = 0.0, full: bool = False) -> np.ndarray:
        """Sorted sample d2 (projected, or full) over land at least exclude_km from the query cell.

        Percentiles are relative to this land only, so a large climate region around
        the query (which the search excludes) doesn't use up the top of the scale.
        Falls back to all land if the exclusion would leave none.
        """
        km = max(float(exclude_km), 0.0)
        ref = self._refs.get((km, full))
        if ref is None:
            d2 = self.sample_full_d2 if full else self.sample_d2
            keep = d2[self.sample_km >= km] if km > 0 else d2
            ref = np.sort(keep if keep.size else d2)
            ref.flags.writeable = False
            if len(self._refs) >= MAX_REFERENCES:
                self._refs.clear()
            self._refs[(km, full)] = ref  # a racing duplicate computes the same array: harmless
        return ref

    def top_pct(self, d2: np.ndarray | float, exclude_km: float = 0.0, full: bool = False) -> np.ndarray:
        """Share (%) of land beyond exclude_km at least as similar as d2 (projected, or full):
        0 = most similar."""
        ref = self.reference(exclude_km, full)
        return 100.0 * np.searchsorted(ref, d2, side="right") / len(ref)


class QueryCache:
    """Thread-safe LRU; concurrent misses on one key compute it once."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict[tuple, threading.Lock] = {}

    def get(self, key: tuple, compute):
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
        self.query_cache = QueryCache(QUERY_CACHE_SIZE)
        self.scan_cache = QueryCache(SCAN_CACHE_SIZE)

    @classmethod
    def load(cls, path: Path) -> "ClimateIndex":
        meta = json.loads((path / "model.json").read_text())
        return cls(
            grid=Grid(**meta["grid"]),
            codecs={v: RawCodec(**c) for v, c in meta["codecs"].items()},
            model=FeatureModel.from_dict(meta["model"]),
            # Memory-mapped: the OS keeps what's read in its page cache, which it can drop under memory
            # pressure where a loaded copy couldn't be. Raw values are only read for a handful of cells
            # per query; cells and pcs are read throughout, and are as fast once cached.
            cells=np.load(path / "cells.npy", mmap_mode="r"),
            raw={v: np.load(path / f"raw_{v}.npy", mmap_mode="r") for v in RAW_VARS},
            pcs=np.load(path / "pcs.npy", mmap_mode="r"),
            sample=np.load(path / "sample.npy"),
            meta={k: v for k, v in meta.items() if k != "model"},
        )

    def __len__(self) -> int:
        return len(self.cells)

    # Cell positions, computed from the cell ids when asked: storing them for every cell would
    # take 138 MB, and every use needs only a few.
    def lat_of(self, idx) -> np.ndarray:
        return self.grid.lat_of(self.cells[idx])

    def lon_of(self, idx) -> np.ndarray:
        return self.grid.lon_of(self.cells[idx])

    def physical(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.atleast_1d(idx)
        return physical({v: self.raw[v][idx] for v in RAW_VARS}, self.codecs, self.lat_of(idx))

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
            d = great_circle_km(lat, lon, self.lat_of(cand), self.lon_of(cand))
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
        q = self.query(qi, align)
        d2 = self.scan(q)
        picked = np.array(self._pick(d2, lat, lon, math.ceil(RESORT_FACTOR * n), exclude_km, suppress_km), dtype=np.int64)
        full = self._full_d2(q, picked)
        keep = np.lexsort((picked, full.min(axis=1)))[:n]  # any subset of the picks is still spaced

        q_phys = {k: v[0] for k, v in self.physical(qi).items()}
        matches = []
        for rank, (i, per_variant) in enumerate(zip(picked[keep].tolist(), full[keep]), 1):
            shifted = bool(len(per_variant) > 1 and per_variant[1] < per_variant[0])
            matches.append(
                {
                    "rank": rank,
                    "cell": self._cell_json(i),
                    "km_from_query": round(float(great_circle_km(lat, lon, self.lat_of(i), self.lon_of(i))), 0),
                    "distance": round(float(np.sqrt(per_variant.min())), 4),  # = distance_at
                    # By full distance, like the ranking, so a better rank never reads as less similar.
                    "more_similar_than_pct_of_land": round(
                        100.0 - float(q.top_pct(per_variant.min(), exclude_km, full=True)), 3
                    ),
                    "seasons_shifted_6_months": shifted,
                    **self._difference(q.variants[int(shifted)], i),
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

    def query(self, qi: int, align: Align = "auto") -> Query:
        """Query cell qi, cached per (qi, align)."""
        return self.query_cache.get((int(qi), align), lambda: self._make_query(int(qi), align))

    def _make_query(self, qi: int, align: Align) -> Query:
        qvec = self.vectors(qi)[0]
        variants = np.stack([qvec, shift_seasons(qvec)] if align == "auto" else [qvec])
        qp = self.model.project(variants)
        qq = np.array([float(v @ v) for v in qp])
        sample_d2 = _d2(qp, qq, self.pcs[self.sample], self.norms[self.sample])
        sample_full_d2 = _d2(variants, np.einsum("ij,ij->i", variants, variants), *self._sample_vectors)
        sample_lat, sample_lon = self.lat_of(self.sample), self.lon_of(self.sample)
        sample_km = great_circle_km(self.lat_of(qi), self.lon_of(qi), sample_lat, sample_lon).astype(np.float32)
        for a in (variants, qp, qq, sample_d2, sample_full_d2, sample_km):
            a.flags.writeable = False
        return Query(qi, align, variants, qp, qq, sample_d2, sample_full_d2, sample_km)

    @cached_property
    def _sample_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """The sample's full vectors and their squared norms (~60 MB), for full-distance percentiles."""
        v = self.vectors(self.sample)
        return v, np.einsum("ij,ij->i", v, v)

    def _full_d2(self, q: Query, pos) -> np.ndarray:
        """(len(pos), v) full (unprojected) squared distances from the cells at pos to each of q's
        alignments, in float64. The scan's projected d2 leaves out the PCA's truncated part."""
        pos = np.asarray(pos, dtype=np.int64)
        if not len(pos):
            return np.empty((0, len(q.variants)))
        return ((self.vectors(pos).astype(np.float64)[:, None, :] - q.variants) ** 2).sum(axis=2)

    def distance_at(self, q: Query, pos) -> np.ndarray:
        """The `distance` /similar reports for each cell at pos: full, the best of q's alignments."""
        return np.sqrt(self._full_d2(q, pos).min(axis=1))

    def distances_to(self, lat: float, lon: float, align: Align, lats, lons) -> np.ndarray:
        """Climate distance from the query (lat, lon, align) to each point (lats, lons), on the same
        scale as matches' `distance`. Points snap to land as queries do; NaN where one can't."""
        q = self.query(self.locate(lat, lon)[0], align)
        lats, lons = np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)
        # The cell a point is in has the nearest centre, which locate() would snap it to (if within
        # MAX_SNAP_KM: always, on a ~4 km grid). Only the rest (a coastal city whose cell is sea) need
        # its slower search (~2 ms each).
        pos = self.positions_at(lats, lons)
        found = pos >= 0
        far = great_circle_km(lats[found], lons[found], self.lat_of(pos[found]), self.lon_of(pos[found])) > MAX_SNAP_KM
        pos[np.flatnonzero(found)[far]] = -1
        for j in np.flatnonzero(pos < 0):
            try:
                pos[j] = self.locate(float(lats[j]), float(lons[j]))[0]
            except NoLandError:
                pass
        out = np.full(len(pos), np.nan)
        ok = pos >= 0
        if ok.any():
            out[ok] = self.distance_at(q, pos[ok])
        return out

    def d2_at(self, q: Query, pos: np.ndarray) -> np.ndarray:
        """Squared climate distance from q to the cells at pos (the min over q's alignments)."""
        return _d2(q.qp, q.qq, self.pcs[pos], self.norms[pos])

    def scan(self, q: Query) -> np.ndarray:
        """d2_at every land cell (read-only), cached per (qi, align)."""

        def compute():
            d2 = _d2(q.qp, q.qq, self.pcs, self.norms)
            d2.flags.writeable = False
            return d2

        return self.scan_cache.get((q.qi, q.align), compute)

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
            xyz = unit_vectors(self.lat_of(band), self.lon_of(band))
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

    def _difference(self, qvec: np.ndarray, i: int) -> dict:
        """Each feature's part of the (full, unprojected) squared distance: as a share, and as is.

        As is (difference_d2) is in the weighted, standardized units the search measures in, so the
        UI can size a gap ("slightly sunnier") on the search's own scale; they sum to distance**2.
        """
        diff = (self.vectors(i)[0] - qvec).reshape(len(FEATURES), -1)
        per = (diff.astype(np.float64) ** 2).sum(axis=1)
        total = per.sum() or 1.0
        return {
            "difference_share": {f: round(float(x / total), 3) for f, x in zip(FEATURES, per)},
            "difference_d2": {f: round(float(x), 4) for f, x in zip(FEATURES, per)},
        }

    def _cell_json(self, i: int) -> dict:
        return {"index": int(i), "lat": round(float(self.lat_of(i)), 4), "lon": round(float(self.lon_of(i)), 4)}

    def _climate_json(self, i: int, detail: bool) -> dict:
        p = {k: v[0] for k, v in self.physical(i).items()}
        out = {"summary": climate_summary(p)}
        if detail:
            out["monthly"] = climate_monthly(p)
        return out
