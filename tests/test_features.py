import numpy as np
import pytest

from climate_twins.features import (
    FEATURES,
    MONTHS,
    VARIANCE_TARGET,
    FeatureModel,
    clearness,
    dewpoint_c,
    fit_feature_model,
    perceptual,
    toa_radiation,
    physical,
    shift_seasons,
)

from .conftest import CODECS, World, make_world

F = len(FEATURES)


@pytest.fixture(scope="module")
def fitted() -> tuple[World, FeatureModel, np.ndarray]:
    """(world, model fitted on it, cell latitudes)."""
    w = make_world()
    lat = w.grid.lat_of(w.cells)
    return w, fit_feature_model(w.raw, w.codecs, lat), lat


def world_vectors(w: World, model: FeatureModel) -> np.ndarray:
    return model.vectors(perceptual(physical(w.raw, w.codecs, w.grid.lat_of(w.cells))))


def test_zero_vapor_dewpoint_is_a_finite_floor():
    d = dewpoint_c(np.array([0.0, -0.1]))
    assert np.all(np.isfinite(d))
    assert d[0] == pytest.approx(-37.3, abs=0.5)
    assert d[1] == d[0]


def test_shift_seasons_flat_matches_monthly():
    """Flattened (..., F*12) vectors roll within each feature, never across them."""
    x = np.random.default_rng(0).normal(size=(3, F, MONTHS))
    flat = shift_seasons(x.reshape(3, -1))
    assert flat.shape == (3, F * MONTHS)
    np.testing.assert_array_equal(flat, shift_seasons(x).reshape(3, -1))
    np.testing.assert_array_equal(shift_seasons(x[0].ravel()), shift_seasons(x[0]).ravel())


class TestFeatureModel:
    def test_standardization_is_area_weighted_unit(self, fitted):
        w, model, lat = fitted
        area = np.cos(np.radians(lat))
        z = world_vectors(w, model).reshape(-1, F, MONTHS) / model.weights[:, None]
        wt = area / area.sum()
        mean = np.einsum("i,ifm->f", wt, z) / MONTHS
        var = np.einsum("i,ifm->f", wt, z.astype(np.float64) ** 2) / MONTHS - mean**2
        np.testing.assert_allclose(mean, 0, atol=1e-4)
        np.testing.assert_allclose(var, 1, atol=1e-4)

    def test_explained_variance_sorted_and_target_met_minimally(self, fitted):
        r = fitted[1].explained_variance_ratio
        assert np.all(np.diff(r) <= 0)
        assert r.sum() >= VARIANCE_TARGET
        assert r[:-1].sum() < VARIANCE_TARGET  # no more components than needed
        assert 1 < len(r) < F * MONTHS

    def test_pca_is_season_symmetric(self, fitted):
        """Fit on data plus its 6-month roll: the mean is invariant under the roll, and so is each
        component up to sign (which is why signs can't be pinned by the largest loading)."""
        m = fitted[1].pca_mean
        np.testing.assert_allclose(shift_seasons(m), m, atol=1e-12)
        for c in fitted[1].components.T:
            s = shift_seasons(c)
            assert np.allclose(s, c, atol=1e-8) or np.allclose(s, -c, atol=1e-8)

    def test_projected_distance_approximates_full_distance(self, fitted):
        w, model, _ = fitted
        v = world_vectors(w, model).astype(np.float64)
        pcs = model.project(v).astype(np.float64)
        rng = np.random.default_rng(2)
        a, b = rng.integers(0, len(v), size=(2, 2000))
        full = ((v[a] - v[b]) ** 2).sum(axis=1)
        proj = ((pcs[a] - pcs[b]) ** 2).sum(axis=1)
        # Projection onto orthonormal components can only shrink distances ...
        assert np.all(proj <= full * (1 + 1e-5) + 1e-6)
        # ... and with 99% of variance kept it shrinks them only a little.
        assert proj.sum() / full.sum() > 0.97
        assert np.corrcoef(np.sqrt(proj), np.sqrt(full))[0, 1] > 0.995

    def test_chunked_fit_matches_single_pass(self, fitted):
        w, model, lat = fitted
        chunked = fit_feature_model(w.raw, w.codecs, lat, chunk=97)
        np.testing.assert_allclose(chunked.feature_mean, model.feature_mean, rtol=1e-10)
        np.testing.assert_allclose(chunked.feature_std, model.feature_std, rtol=1e-10)
        # Signs are pinned against a fixed reference vector, so they must agree too.
        np.testing.assert_allclose(chunked.components, model.components, atol=1e-6)


def test_model_with_other_transform_is_rejected(fitted):
    d = fitted[1].to_dict()
    d["transform"] = "tmax,tmin,dewpoint(vap),log1p(ppt),srad,ws"
    with pytest.raises(ValueError, match="rebuild"):
        FeatureModel.from_dict(d)


def test_dry_months_count_less_than_wet_ones():
    """2 vs 10 mm (both dry) must differ far less than 100 vs 500 mm."""
    def ppt(mm: float) -> float:
        raw = {v: np.zeros((1, MONTHS), dtype=np.int16) for v in CODECS}
        raw["ppt"][:] = round(mm * 10)
        return float(perceptual(physical(raw, CODECS, np.zeros(1)))[0, FEATURES.index("ppt"), 0])

    assert abs(ppt(2) - ppt(10)) < 0.3 * abs(ppt(100) - ppt(500))


class TestClearness:
    def test_toa_radiation_reference_values(self):
        toa = toa_radiation(np.array([0.0, 80.0, -80.0]))
        assert toa.shape == (3, MONTHS)
        assert toa[0].mean() == pytest.approx(417, abs=3)  # equator annual mean, W/m^2
        assert toa[1, [0, 1, 10, 11]].max() < 1 and toa[1, 5] > 480  # polar night / midnight sun
        np.testing.assert_allclose(toa[2], np.roll(toa[1], 6), atol=40)  # mirrored hemispheres

    def test_clearness_blends_in_polar_night_and_is_bounded(self):
        toa = toa_radiation(np.array([80.0]))
        srad = 0.5 * toa
        srad[0, 5] = 0.3 * toa[0, 5]  # a cloudier June
        k = clearness(srad, toa)
        annual = srad.sum() / toa.sum()
        assert k[0, 0] == pytest.approx(annual)  # no sun in January: annual value
        assert k[0, 5] == pytest.approx(0.3, abs=0.02)
        assert np.all((0 <= clearness(toa * 2, toa)) & (clearness(toa * 2, toa) <= 1))

