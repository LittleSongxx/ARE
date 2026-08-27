import numpy as np

from ac_pbgrl.evaluation.metrics import ranking_metrics, uncertainty_metrics
from ac_pbgrl.evaluation.statistics import holm_adjust, paired_comparison


def test_ranking_and_uncertainty_metrics():
    ranking = ranking_metrics([1, 2, 3], [1, 2, 4], [True, True, True])
    assert ranking["spearman"] == 1.0
    assert ranking["top1_regret"] == 0.0
    uncertainty = uncertainty_metrics([0, 1], [1, 1], [0, 2], [True, True])
    assert np.isfinite(uncertainty["nll"])
    assert 0 <= uncertainty["coverage_95"] <= 1


def test_paired_statistics_and_holm_are_well_formed():
    result = paired_comparison([1, 2, 3, 4], [2, 3, 4, 5], bootstrap_samples=500)
    assert result["difference"] == -1.0
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    with_missing = holm_adjust([0.01, np.nan, 0.2])
    assert np.isnan(with_missing[1])
    assert 0 <= with_missing[0] <= with_missing[2] <= 1
