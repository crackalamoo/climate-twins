"""Turning monthly climate normals into the vectors that climate distance is measured on.

Pipeline for one cell (12 months of each raw variable):
  1. physical units   -- decode the stored integers (scale/offset from the netCDF files)
  2. perceptual units -- vapor pressure -> dewpoint, precipitation -> log(mm + PPT_OFFSET_MM),
                         solar radiation -> sky clearness (fraction of top-of-atmosphere sunlight)
  3. standardize      -- per feature, pooled over months, area-weighted over all land
  4. weight           -- how much each feature should matter (WEIGHTS)
  5. PCA              -- rotate and keep the components holding VARIANCE_TARGET of the
                         variance. No whitening: whitening would cancel the weights.

Seasons are aligned at query time by also comparing against the query rolled by
six months, so the PCA is fit on the data plus its six-month-rolled copy and
represents both alignments equally well.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

RAW_VARS = ("tmax", "tmin", "ppt", "vap", "srad", "ws")
FEATURES = ("tmax", "tmin", "dewpoint", "ppt", "clearness", "ws")
MONTHS = 12
SEASON_SHIFT = 6

# Relative importance of each feature after standardization. These are calibrated, not
# chosen directly: what matters is each feature's share of the distance among close
# matches, which also depends on how much it varies between otherwise similar places.
# The weights were tuned (iteratively, on the top 20 matches of 100 random land
# locations at default settings) so those shares are about:
#   daily highs 24%, nightly lows 18%, humidity 15%, rain 20%, sunniness 15%, wind 8%
# i.e. temperature first (highs above lows), and wind least: it is the least
# perceptible and least reliable variable. Re-calibrate if features change.
WEIGHTS = {
    "tmax": 1.0,
    "tmin": 0.837,
    "dewpoint": 0.417,
    "ppt": 0.186,
    "clearness": 0.138,
    "ws": 0.092,
}
VARIANCE_TARGET = 0.99

# Precipitation is compared on a log scale so 100 vs 500 mm counts as much as 20 vs
# 100 mm; the offset keeps dry months alike. With log1p, 2 vs 10 mm (both dry) counts
# nearly as much as 100 vs 500 mm, ~40% of the precipitation difference among top matches.
PPT_OFFSET_MM = 15.0

# Sunniness is the clearness index, surface / top-of-atmosphere solar radiation (~0.75
# clear, ~0.3 overcast); raw radiation mostly tracks latitude, which temperature already
# captures. Near polar night the ratio is meaningless, so it blends toward the cell's
# annual clearness once top-of-atmosphere radiation falls to about this level.
CLEARNESS_BLEND_W_M2 = 20.0
SOLAR_CONSTANT_W_M2 = 1367.0
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Identifies the feature definitions above; a saved model must match it to load.
TRANSFORM = (
    f"tmax,tmin,dewpoint(vap),log(ppt+{PPT_OFFSET_MM:g}),"
    f"clearness(srad/toa,blend {CLEARNESS_BLEND_W_M2:g}),ws"
)

# TerraClimate stores vapor pressure in 0.1 kPa steps, so 0 means "below 0.05 kPa".
# Use the middle of that bucket (dewpoint about -37 C) rather than log(0).
VAP_FLOOR_HPA = 0.25


@dataclass(frozen=True)
class RawCodec:
    """Stored integer -> physical value, as given by the netCDF scale_factor/add_offset."""

    scale: float
    offset: float

    def decode(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype=np.float64) * self.scale + self.offset


def dewpoint_c(vap_kpa: np.ndarray) -> np.ndarray:
    """Dewpoint (C) from vapor pressure (kPa), inverting the Magnus formula."""
    e_hpa = np.maximum(np.asarray(vap_kpa, dtype=np.float64) * 10.0, VAP_FLOOR_HPA)
    g = np.log(e_hpa / 6.112)
    return 243.04 * g / (17.625 - g)


def toa_radiation(lat_deg: np.ndarray) -> np.ndarray:
    """(n, 12) monthly mean top-of-atmosphere solar radiation (W/m^2, 24-hour mean).

    FAO-56 extraterrestrial radiation (eqs. 21-25), averaged over every day of each month.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    uniq, inverse = np.unique(lat, return_inverse=True)  # grid cells share few latitudes
    phi = np.radians(uniq)[:, None]
    day = np.arange(1, 366)
    dr = 1 + 0.033 * np.cos(2 * np.pi * day / 365)
    decl = 0.409 * np.sin(2 * np.pi * day / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
    daily = (SOLAR_CONSTANT_W_M2 / np.pi) * dr * (
        ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws)
    )
    starts = np.concatenate([[0], np.cumsum(DAYS_IN_MONTH)[:-1]])
    monthly = np.add.reduceat(daily, starts, axis=1) / DAYS_IN_MONTH
    return monthly[inverse.reshape(lat.shape)]


def clearness(srad: np.ndarray, toa: np.ndarray) -> np.ndarray:
    """Clearness index srad / toa in [0, 1], blended toward the annual value in dark months."""
    annual = srad.sum(axis=-1, keepdims=True) / toa.sum(axis=-1, keepdims=True)
    k = (srad + CLEARNESS_BLEND_W_M2 * annual) / (toa + CLEARNESS_BLEND_W_M2)
    return np.clip(k, 0.0, 1.0)


def physical(
    raw: Mapping[str, np.ndarray], codecs: Mapping[str, RawCodec], lat: np.ndarray
) -> dict[str, np.ndarray]:
    """Decode raw (n, 12) integer arrays to physical units, adding derived dewpoint,
    top-of-atmosphere radiation and clearness for cells at latitudes `lat` (n,)."""
    phys = {v: codecs[v].decode(raw[v]) for v in RAW_VARS}
    phys["dewpoint"] = dewpoint_c(phys["vap"])
    phys["toa"] = toa_radiation(lat)
    phys["clearness"] = clearness(phys["srad"], phys["toa"])
    return phys


def perceptual(phys: Mapping[str, np.ndarray]) -> np.ndarray:
    """(n, len(FEATURES), 12) array of the features before standardization."""
    cols = {
        "tmax": phys["tmax"],
        "tmin": phys["tmin"],
        "dewpoint": phys["dewpoint"],
        "ppt": np.log(np.maximum(phys["ppt"], 0.0) + PPT_OFFSET_MM),
        "clearness": phys["clearness"],
        "ws": phys["ws"],
    }
    return np.stack([cols[f] for f in FEATURES], axis=1)


def shift_seasons(x: np.ndarray) -> np.ndarray:
    """Roll the month axis by six months. Accepts (..., F, 12) or flattened (..., F*12)."""
    if x.shape[-1] == MONTHS:
        return np.roll(x, SEASON_SHIFT, axis=-1)
    shaped = x.reshape(*x.shape[:-1], len(FEATURES), MONTHS)
    return np.roll(shaped, SEASON_SHIFT, axis=-1).reshape(x.shape)


@dataclass(frozen=True)
class FeatureModel:
    feature_mean: np.ndarray  # (F,)
    feature_std: np.ndarray  # (F,)
    weights: np.ndarray  # (F,)
    pca_mean: np.ndarray  # (F*12,)
    components: np.ndarray  # (F*12, k), orthonormal columns
    explained_variance_ratio: np.ndarray  # (k,)

    @property
    def k(self) -> int:
        return self.components.shape[1]

    def vectors(self, perceptual_features: np.ndarray) -> np.ndarray:
        """(n, F, 12) perceptual features -> (n, F*12) standardized, weighted vectors."""
        z = (perceptual_features - self.feature_mean[:, None]) / self.feature_std[:, None]
        z *= self.weights[:, None]
        return z.reshape(len(z), -1).astype(np.float32)

    def project(self, vectors: np.ndarray) -> np.ndarray:
        return ((vectors - self.pca_mean) @ self.components).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "features": list(FEATURES),
            "transform": TRANSFORM,
            **{k: v.tolist() for k, v in self.__dict__.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "FeatureModel":
        if list(d["features"]) != list(FEATURES):
            raise ValueError(f"model was built for features {d['features']}, code expects {FEATURES}")
        if d.get("transform") != TRANSFORM:
            raise ValueError(
                f"model was built with transform {d.get('transform')!r}, code expects {TRANSFORM!r}; "
                "rebuild with `uv run climate-twins-build`"
            )
        names = cls.__dataclass_fields__
        return cls(**{k: np.asarray(d[k], dtype=np.float64) for k in names})


def fit_feature_model(
    raw: Mapping[str, np.ndarray],
    codecs: Mapping[str, RawCodec],
    lat: np.ndarray,
    weights: Mapping[str, float] = WEIGHTS,
    variance_target: float = VARIANCE_TARGET,
    chunk: int = 500_000,
) -> FeatureModel:
    """Fit standardization and PCA over all cells, weighting each by its area (cos lat).

    `raw` maps each RAW_VAR to an (N, 12) array (memmaps are fine; it is read in chunks);
    `lat` is each cell's latitude.
    """
    lat = np.asarray(lat, dtype=np.float64)
    area_weight = np.cos(np.radians(lat))
    n = len(area_weight)
    nf = len(FEATURES)
    w_total = float(np.sum(area_weight, dtype=np.float64))

    def chunks():
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            feats = perceptual(physical({v: raw[v][sl] for v in RAW_VARS}, codecs, lat[sl]))
            yield feats, np.asarray(area_weight[sl], dtype=np.float64)

    # Pass 1: area-weighted mean and std of each feature, pooled over months.
    s1 = np.zeros(nf)
    s2 = np.zeros(nf)
    for feats, w in chunks():
        s1 += np.einsum("i,ifm->f", w, feats)
        s2 += np.einsum("i,ifm->f", w, feats**2)
    mean = s1 / (w_total * MONTHS)
    std = np.sqrt(np.maximum(s2 / (w_total * MONTHS) - mean**2, 0.0))
    std[std == 0] = 1.0

    partial = FeatureModel(
        feature_mean=mean,
        feature_std=std,
        weights=np.array([weights[f] for f in FEATURES], dtype=np.float64),
        pca_mean=np.zeros(nf * MONTHS),
        components=np.eye(nf * MONTHS),
        explained_variance_ratio=np.ones(nf * MONTHS) / (nf * MONTHS),
    )

    # Pass 2: area-weighted first and second moments of the weighted vectors.
    m1 = np.zeros(nf * MONTHS)
    m2 = np.zeros((nf * MONTHS, nf * MONTHS))
    for feats, w in chunks():
        x = partial.vectors(feats).astype(np.float64)
        m1 += w @ x
        m2 += x.T @ (x * w[:, None])
    m1 /= w_total
    m2 /= w_total

    # Average with the six-month-rolled copy so both season alignments are represented.
    perm = shift_seasons(np.arange(nf * MONTHS))
    mu = (m1 + m1[perm]) / 2
    second = (m2 + m2[np.ix_(perm, perm)]) / 2
    cov = second - np.outer(mu, mu)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    ratio = evals / evals.sum()
    k = int(np.searchsorted(np.cumsum(ratio), variance_target) + 1)
    comps = evecs[:, :k]
    # Deterministic signs. The covariance is symmetric under the season roll, so each
    # component's largest loading ties with its rolled twin; orient against a fixed
    # generic vector instead, which no component is orthogonal to in practice.
    reference = np.random.default_rng(0).standard_normal(nf * MONTHS)
    comps *= np.where(reference @ comps < 0, -1.0, 1.0)

    return FeatureModel(
        feature_mean=mean,
        feature_std=std,
        weights=partial.weights,
        pca_mean=mu,
        components=comps,
        explained_variance_ratio=ratio[:k],
    )
