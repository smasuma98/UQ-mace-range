from orchestr_range_al.metrics import mae, rmse


def test_metrics():
    err = [-1.0, 1.0, 3.0]
    assert mae(err) == 5.0 / 3.0
    assert round(rmse(err), 12) == round((11.0 / 3.0) ** 0.5, 12)
