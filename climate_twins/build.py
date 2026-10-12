"""Build the search index from the TerraClimate 1991-2020 climatology netCDF files.

    uv run climate-twins-build [--raw-dir data/raw] [--out data/processed]

Outputs (in --out):
  model.json       grid geometry, raw decoding, standardization, weights, PCA
  cells.npy        int32 (N,)    flat grid index of each land cell, ascending
  raw_<var>.npy    int16 (N, 12) stored monthly values, decoded via model.json codecs
  pcs.npy          float32 (N, k) PCA coordinates searched at query time
  sample.npy       int64 (S,)    area-weighted random cells, for similarity percentiles
"""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np

from .features import RAW_VARS, RawCodec, fit_feature_model, perceptual, physical
from .geo import Grid

ROOT = Path(__file__).resolve().parent.parent
FILE_PATTERN = "TerraClimate_19912020_{var}.nc"
PERCENTILE_SAMPLE = 200_000
CHUNK = 500_000


def read_var(path: Path, var: str) -> tuple[np.ndarray, np.ndarray, RawCodec, Grid]:
    """Stored integers (12, nlat*nlon), validity mask, codec, and grid of one variable."""
    with netCDF4.Dataset(path) as ds:
        v = ds[var]
        v.set_auto_maskandscale(False)
        data = np.asarray(v[:])
        valid = data != v._FillValue
        codec = RawCodec(scale=float(getattr(v, "scale_factor", 1.0)), offset=float(getattr(v, "add_offset", 0.0)))
        grid = Grid.from_coords(np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:]))
    if data.shape[0] != 12:
        raise ValueError(f"{path}: expected 12 months, got shape {data.shape}")
    return data.reshape(12, -1), valid.reshape(12, -1).all(axis=0), codec, grid


def build(raw_dir: Path, out: Path, min_lat: float) -> None:
    t0 = time.time()
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    grid = cells = None
    codecs: dict[str, RawCodec] = {}
    for var in RAW_VARS:
        data, valid, codec, var_grid = read_var(raw_dir / FILE_PATTERN.format(var=var), var)
        if grid is None:
            grid = var_grid
            lat_all = grid.lat_of(np.arange(grid.nlat * grid.nlon))
            cells = np.flatnonzero(valid & (lat_all >= min_lat)).astype(np.int32)
            del lat_all
            print(f"{len(cells):,} land cells north of {min_lat} deg")
        elif var_grid != grid:
            raise ValueError(f"{var}: grid {var_grid} differs from {grid}")
        if not valid[cells].all():
            raise ValueError(f"{var}: {np.count_nonzero(~valid[cells])} land cells are missing data")
        values = data[:, cells].T
        if values.min() < np.iinfo(np.int16).min or values.max() > np.iinfo(np.int16).max:
            raise ValueError(f"{var}: stored values do not fit int16")
        np.save(tmp / f"raw_{var}.npy", np.ascontiguousarray(values, dtype=np.int16))
        codecs[var] = codec
        del data, valid, values
        print(f"  {var}: scale={codec.scale} offset={codec.offset} ({time.time() - t0:.0f}s)")

    np.save(tmp / "cells.npy", cells)
    raw = {v: np.load(tmp / f"raw_{v}.npy", mmap_mode="r") for v in RAW_VARS}
    lat = grid.lat_of(cells)
    area = np.cos(np.radians(lat))

    model = fit_feature_model(raw, codecs, lat)
    cum = np.cumsum(model.explained_variance_ratio)
    print(f"PCA: {model.k} components keep {cum[-1]:.2%} of variance ({time.time() - t0:.0f}s)")

    pcs = np.lib.format.open_memmap(tmp / "pcs.npy", mode="w+", dtype=np.float32, shape=(len(cells), model.k))
    for start in range(0, len(cells), CHUNK):
        sl = slice(start, min(start + CHUNK, len(cells)))
        phys = physical({v: raw[v][sl] for v in RAW_VARS}, codecs, lat[sl])
        pcs[sl] = model.project(model.vectors(perceptual(phys)))
    pcs.flush()
    del pcs, raw

    rng = np.random.default_rng(0)
    size = min(PERCENTILE_SAMPLE, len(cells))
    np.save(tmp / "sample.npy", np.sort(rng.choice(len(cells), size=size, replace=False, p=area / area.sum())))

    meta = {
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "TerraClimate v1.1 monthly climatology 1991-2020",
        "files": [FILE_PATTERN.format(var=v) for v in RAW_VARS],
        "min_lat": min_lat,
        "n_cells": int(len(cells)),
        "grid": grid.to_dict(),
        "codecs": {v: c.__dict__ for v, c in codecs.items()},
        "model": model.to_dict(),
    }
    (tmp / "model.json").write_text(json.dumps(meta, indent=1))

    if out.exists():
        old = out.with_name(out.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        out.rename(old)
        tmp.rename(out)
        shutil.rmtree(old)
    else:
        tmp.rename(out)
    print(f"index written to {out} ({time.time() - t0:.0f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--min-lat", type=float, default=-60.0, help="drop cells south of this (default: Antarctica)")
    args = p.parse_args()
    build(args.raw_dir, args.out, args.min_lat)


if __name__ == "__main__":
    main()
