import numpy as np
import pytest


def test_topology_parsing():
    from gwopt.topology import parse_topology, signature, canonical_key

    topo = "AEDDBCDBA-LSLSDLDHSLSL"
    center, border = parse_topology(topo, 3)
    assert center["11"] == "A"
    assert center["22"] == "B"

    sig = signature(topo, 3)
    assert isinstance(sig, tuple)
    assert len(sig) > 0

    assert canonical_key(["center11", "reflectivity"]) == ("center11", "reflectivity")
    assert canonical_key([["center11", "reflectivity"]]) == ("center11", "reflectivity")


def test_tracker():
    from gwopt.optimizer import Tracker

    tracker = Tracker()
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    losses = np.array([0.5, 0.3])
    feasible = np.array([True, True])

    tracker.update(points, losses, feasible)
    assert tracker.best_loss == 0.3
    assert tracker.best_feasible_loss == 0.3
    assert np.allclose(tracker.best_x, [3.0, 4.0])
    assert tracker.iter == 1


def test_tracker_infeasible():
    from gwopt.optimizer import Tracker

    tracker = Tracker()
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    losses = np.array([0.5, 0.3])
    feasible = np.array([True, False])

    tracker.update(points, losses, feasible)
    assert tracker.best_feasible_loss == 0.5
    assert np.allclose(tracker.best_feasible_x, [1.0, 2.0])


def test_clean_nans():
    import jax.numpy as jnp
    from gwopt.optimizer import clean_nans

    vals = jnp.array([1.0, float('nan'), 3.0])
    safe = clean_nans(vals)
    assert float(safe[1]) == 1e10

    grads = jnp.array([[1.0, 2.0], [float('nan'), 1.0], [3.0, 4.0]])
    safe_v, safe_g = clean_nans(vals, grads)
    assert float(safe_g[1, 0]) == 0.0


def test_linear_anneal():
    from gwopt.optimizer import linear_anneal

    assert linear_anneal(0.0, 1.0, 0.0, 0.5) == 1.0
    assert linear_anneal(0.5, 1.0, 0.0, 0.5) == 0.0
    assert linear_anneal(0.25, 1.0, 0.0, 0.5) == 0.5


def test_prior_loading():
    from gwopt.warmstart import Prior, find_prior

    path = find_prior()
    prior = Prior(path)
    assert len(prior.keys_arr) > 0
    assert len(prior.donor_loss) > 0


def test_box_helpers():
    from gwopt.box import get_bounds, unit_to_physical, physical_to_unit

    class FakeObj:
        bounds = np.array([[0.0, 0.0], [1.0, 2.0]])

    lo, hi, span = get_bounds(FakeObj())
    assert np.allclose(span, [1.0, 2.0])

    u = np.array([0.5, 0.5])
    x = unit_to_physical(u, lo, span)
    assert np.allclose(x, [0.5, 1.0])

    u_back = physical_to_unit(x, lo, span)
    assert np.allclose(u_back, [0.5, 0.5])
