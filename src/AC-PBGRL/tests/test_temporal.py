import numpy as np

from ac_pbgrl.events import GraphEvent
from ac_pbgrl.models.temporal import AdaptivePotentialKF, VarianceCalibrator


def test_scalar_kf_matches_closed_form_first_update():
    filt = AdaptivePotentialKF(p0=1.0, q_stable=0.1, r_min=0.2, r_max=0.2)
    mean0, variance0 = filt.update(7, 2.0, 0.2, step=0)
    assert mean0 == 2.0
    assert variance0 == 1.0
    mean1, variance1 = filt.update(7, 4.0, 0.2, step=1)
    prior = 1.1
    gain = prior / (prior + 0.2)
    assert np.isclose(mean1, 2.0 + gain * 2.0)
    assert np.isclose(variance1, (1.0 - gain) * prior)


def test_hard_event_resets_and_ttl_expires():
    filt = AdaptivePotentialKF(ttl_steps=2)
    filt.update(1, 1.0, 0.2, step=0)
    mean, variance = filt.update(1, 9.0, 0.2, GraphEvent.VISITED, step=1)
    assert mean == 9.0 and variance == filt.p0
    mean, variance = filt.update(1, 3.0, 0.2, step=5)
    assert mean == 3.0 and variance == filt.p0
    assert filt.event_count == 1
    filt.retire(1)
    assert 1 not in filt.records


def test_variance_temperature_calibration():
    calibrator = VarianceCalibrator()
    temperature = calibrator.fit(np.ones(4), np.asarray([2.0, -2.0, 2.0, -2.0]))
    assert np.isclose(temperature, 4.0)
    assert np.isclose(calibrator(0.5), 2.0)
