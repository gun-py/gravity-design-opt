import os
import numpy as np
from topology import pairs_for_topology, canonical_key, signature

EPS = 1e-12
BOUNDARY_CODES = {"L": 0, "S": 1, "D": 2, "H": 3}


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

    def get_donor_params(self, idx):
        s, e = int(self.donor_offsets[idx]), int(self.donor_offsets[idx + 1])
        return dict(zip(self.donor_key_id[s:e].tolist(),
                        self.donor_value[s:e].tolist()))

    def sample_top_values(self, key_id, n, rng):
        vals = self.per_key_top[key_id]
        if vals.size == 0 or self.per_key_count[key_id] == 0:
            return np.zeros(n, dtype=np.float64)
        return rng.choice(vals, size=n, replace=True).astype(np.float64)


def _get_param_keys(obj):
    out = [canonical_key(p) for p in obj.optimization_pairs]
    assert len(out) == int(obj.n_params)
    return out


def _hamming_dist(sig, ref_ints, ref_bounds):
    ints = np.array(sig[:9], dtype=np.int8)
    bounds = np.array([BOUNDARY_CODES[c] for c in sig[9:21]], dtype=np.int8)
    return (ref_ints != ints[None, :]).sum(1) + (ref_bounds != bounds[None, :]).sum(1)


def _nearest_donors(prior, sig, size, count):
    if size == 3:
        candidates = np.where(prior.donor_size == 3)[0]
        if candidates.size == 0:
            return np.zeros(0, dtype=int)
        dist = _hamming_dist(sig, prior.donor_sig_int[candidates],
                             prior.donor_sig_bnd[candidates])
        order = np.lexsort((prior.donor_loss[candidates], dist))
        return candidates[order[:count]]
    candidates = np.where(prior.donor_size == size)[0]
    if candidates.size == 0:
        return np.zeros(0, dtype=int)
    return candidates[np.argsort(prior.donor_loss[candidates])[:count]]


def _get_bounds(obj):
    b = np.asarray(obj.bounds, dtype=np.float64)
    return b[0].copy(), b[1].copy()


def _to_logit(values, lo, hi):
    span = hi - lo
    u = np.where(span > 0, (values - lo) / np.where(span > 0, span, 1.0), 0.5)
    u = np.clip(u, EPS, 1.0 - EPS)
    return np.log(u / (1.0 - u))


def _fill_from_donor(prior, donor_idx, param_keys, lo, hi, rng):
    donor_params = prior.get_donor_params(donor_idx)
    out = np.empty(len(param_keys), dtype=np.float64)
    for i, (comp, prop) in enumerate(param_keys):
        kid = prior.key_to_id.get(f"{comp}/{prop}")
        if kid is not None and kid in donor_params:
            out[i] = donor_params[kid]
        elif kid is not None:
            out[i] = float(prior.sample_top_values(kid, 1, rng)[0])
        else:
            out[i] = float(rng.uniform(lo[i], hi[i]))
    return np.clip(out, lo, hi)


def _fill_from_prior(prior, param_keys, lo, hi, rng):
    out = np.empty(len(param_keys), dtype=np.float64)
    for i, (comp, prop) in enumerate(param_keys):
        kid = prior.key_to_id.get(f"{comp}/{prop}")
        if kid is not None:
            out[i] = float(prior.sample_top_values(kid, 1, rng)[0])
        else:
            out[i] = float(rng.uniform(lo[i], hi[i]))
    return np.clip(out, lo, hi)


def build_starts(obj, k, prior, rng_seed=None, n_donors=8, n_prior=3):
    rng = np.random.default_rng(rng_seed or 0)
    param_keys = _get_param_keys(obj)
    lo, hi = _get_bounds(obj)

    topo, size = None, None
    try:
        params = obj.problem_spec.get("params", obj.problem_spec)
        if "topology" in params and "size" in params:
            topo, size = params["topology"], int(params["size"])
    except Exception:
        pass

    starts = []
    if topo is not None:
        sig = signature(topo, size)
        for d in _nearest_donors(prior, sig, size, n_donors):
            x = _fill_from_donor(prior, int(d), param_keys, lo, hi, rng)
            starts.append(_to_logit(x, lo, hi))
    for _ in range(n_prior):
        x = _fill_from_prior(prior, param_keys, lo, hi, rng)
        starts.append(_to_logit(x, lo, hi))
    while len(starts) < k:
        u = rng.uniform(EPS, 1.0 - EPS, size=len(param_keys))
        starts.append(_to_logit(lo + (hi - lo) * u, lo, hi))
    return np.stack(starts[:k]).astype(np.float64)


def find_prior():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(here, "prior.npz"),
                 os.path.join(here, "..", "gwopt", "prior.npz"),
                 os.path.join(here, "..", "prior.npz")]:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError("prior.npz not found")
