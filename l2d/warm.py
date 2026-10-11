import os

import numpy as np

from name_gen import pairs_for_topology, canonical_key, signature

_EPS = 1e-12  # tight so bounded->unbounded->bounded round-trip preserves loss to ~1e-6
_BCODE = {"L": 0, "S": 1, "D": 2, "H": 3}


class Prior:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.keys_arr = z["keys_arr"]
        self.key_to_id = {k: i for i, k in enumerate(self.keys_arr)}
        self.key_prop = z["key_prop"]
        self.prop_arr = z["prop_arr"]
        self.prop_bounds = z["prop_bounds"]
        self.prop_to_idx = {p: i for i, p in enumerate(self.prop_arr)}
        self.per_key_top = z["per_key_top"]
        self.per_key_quant = z["per_key_quantiles"]
        self.per_key_count = z["per_key_count"]
        self.donor_offsets = z["donor_offsets"]
        self.donor_key_id = z["donor_key_id"]
        self.donor_value = z["donor_value"]
        self.donor_size = z["donor_size"]
        self.donor_loss = z["donor_loss"]
        self.donor_len = z["donor_len"]
        self.donor_topo = np.array([t.decode() for t in z["donor_topo_bytes"]])
        self.donor_sig_int = z["donor_sig_int"]
        self.donor_sig_bnd = z["donor_sig_bnd"]

    def donor_map(self, d):
        s, e = int(self.donor_offsets[d]), int(self.donor_offsets[d + 1])
        return dict(zip(self.donor_key_id[s:e].tolist(), self.donor_value[s:e].tolist()))

    def sample_top(self, kid, n, rng):
        v = self.per_key_top[kid]
        if v.size == 0 or self.per_key_count[kid] == 0:
            return np.zeros(n, dtype=np.float64)
        return rng.choice(v, size=n, replace=True).astype(np.float64)


def _canon_pairs(obj):
    out = [canonical_key(p) for p in obj.optimization_pairs]
    assert len(out) == int(obj.n_params)
    return out


def _hamming_size3(sig, si, sb):
    ints = np.array(sig[:9], dtype=np.int8)
    bnds = np.array([_BCODE[c] for c in sig[9:21]], dtype=np.int8)
    return (si != ints[None, :]).sum(1) + (sb != bnds[None, :]).sum(1)


def _nn_donor_indices(prior, sig, size, m):
    if size == 3:
        good = np.where(prior.donor_size == 3)[0]
        if good.size == 0: return np.zeros(0, dtype=int)
        h = _hamming_size3(sig, prior.donor_sig_int[good], prior.donor_sig_bnd[good])
        return good[np.lexsort((prior.donor_loss[good], h))[:m]]
    good = np.where(prior.donor_size == size)[0]
    if good.size == 0: return np.zeros(0, dtype=int)
    return good[np.argsort(prior.donor_loss[good])[:m]]


def _prop_bounds_vector(obj):
    b = np.asarray(obj.bounds, dtype=np.float64)
    return b[0].copy(), b[1].copy()


def _bounded_to_unbounded(v, lo, hi):
    span = hi - lo
    u = np.where(span > 0, (v - lo) / np.where(span > 0, span, 1.0), 0.5)
    u = np.clip(u, _EPS, 1.0 - _EPS)
    return np.log(u / (1.0 - u))


def _assemble_from_donor(prior, d, name_keys, lo, hi, rng):
    dm = prior.donor_map(d)
    out = np.empty(len(name_keys), dtype=np.float64)
    for i, (c, p) in enumerate(name_keys):
        kid = prior.key_to_id.get(f"{c}/{p}")
        if kid is not None and kid in dm: out[i] = dm[kid]
        elif kid is not None:              out[i] = float(prior.sample_top(kid, 1, rng)[0])
        else:                               out[i] = float(rng.uniform(lo[i], hi[i]))
    return np.clip(out, lo, hi)


def _assemble_prior_draw(prior, name_keys, lo, hi, rng):
    out = np.empty(len(name_keys), dtype=np.float64)
    for i, (c, p) in enumerate(name_keys):
        kid = prior.key_to_id.get(f"{c}/{p}")
        if kid is not None: out[i] = float(prior.sample_top(kid, 1, rng)[0])
        else:               out[i] = float(rng.uniform(lo[i], hi[i]))
    return np.clip(out, lo, hi)


def build_starts(objective, k, prior, rng_seed=None, n_donors=8, n_prior=3):
    rng = np.random.default_rng(rng_seed or 0)
    nk = _canon_pairs(objective)
    lo, hi = _prop_bounds_vector(objective)

    topo, size = None, None
    try:
        params = objective.problem_spec.get("params", objective.problem_spec)
        if "topology" in params and "size" in params:
            topo, size = params["topology"], int(params["size"])
    except Exception:
        pass

    starts = []
    if topo is not None:
        for d in _nn_donor_indices(prior, signature(topo, size), size, n_donors):
            starts.append(_bounded_to_unbounded(_assemble_from_donor(prior, int(d), nk, lo, hi, rng), lo, hi))
    for _ in range(n_prior):
        starts.append(_bounded_to_unbounded(_assemble_prior_draw(prior, nk, lo, hi, rng), lo, hi))
    while len(starts) < k:
        u = rng.uniform(_EPS, 1.0 - _EPS, size=len(nk))
        starts.append(_bounded_to_unbounded(lo + (hi - lo) * u, lo, hi))
    return np.stack(starts[:k]).astype(np.float64)


def find_prior_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here, "prior.npz"),
              os.path.join(here, "..", "l2d", "prior.npz"),
              os.path.join(here, "..", "prior.npz")]:
        if os.path.isfile(c): return os.path.abspath(c)
    raise FileNotFoundError("prior.npz not found")


def donor_starts(obj, prior, rng, count, ordinal):
    nk = _canon_pairs(obj)
    lo, hi = _prop_bounds_vector(obj)
    spec = obj.problem_spec.get("params", obj.problem_spec)
    topo, size = spec.get("topology"), spec.get("size")
    if topo is None or size is None:
        x = _assemble_prior_draw(prior, nk, lo, hi, rng)
    else:
        ids = _nn_donor_indices(prior, signature(topo, int(size)), int(size), count)
        x = (_assemble_from_donor(prior, int(ids[ordinal % len(ids)]), nk, lo, hi, rng)
             if len(ids) else _assemble_prior_draw(prior, nk, lo, hi, rng))
    return _bounded_to_unbounded(x, lo, hi)
