# offline: train the M4 kernel-mixture prior from Differometor-30k top-loss designs
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "l2d"))

from name_gen import pairs_for_topology, canonical_key, signature

DATA = os.path.join(ROOT, "dataset", "dataset.h5")
PRIOR = os.path.join(ROOT, "l2d", "prior.npz")
OUT = os.path.join(ROOT, "entries", "m4_priorgen", "weights.npz")

TRAIN_LOSS_MAX = 0.3
MAX = 3000
BT = ["L", "S", "D", "H"]


def feat(t, s):
    if s != 3: return np.zeros(9 + 12 * 4, np.float32)
    sig = signature(t, s)
    ints = np.array(sig[:9], np.float32)
    bnd = np.zeros((12, 4), np.float32)
    for i, c in enumerate(sig[9:]): bnd[i, BT.index(c)] = 1.0
    return np.concatenate([ints, bnd.reshape(-1)])


def v2q(v, qt):
    if not np.isfinite(v): return np.nan
    if qt[-1] <= qt[0]: return 0.5
    return float(np.interp(v, qt, np.array([0., .05, .25, .5, .75, .95, 1.])))


def main():
    prior = np.load(PRIOR, allow_pickle=False)
    keys_arr = prior["keys_arr"]
    per_key_q = prior["per_key_quantiles"]
    key_prop_id = np.array([list(prior["prop_arr"]).index(p) for p in prior["key_prop"]], np.int8)
    key_to_id = {k: i for i, k in enumerate(keys_arr)}
    K = len(keys_arr)

    h5 = h5py.File(DATA, "r"); e = h5["entries"][:]
    losses = e["loss"].astype(np.float32); sizes = e["size"].astype(np.int8)
    topos = np.array([t.decode() for t in e["topology_string"]])
    po, pl = e["param_offset"].astype(np.int64), e["param_length"].astype(np.int32)

    mask = (losses < TRAIN_LOSS_MAX) & (sizes == 3)
    idx = np.where(mask)[0][np.argsort(losses[mask])][:MAX]
    D = len(idx); print(f"donors: {D}")

    bp = h5["bounded_params"][:]
    F = 9 + 12 * 4
    dq = np.full((D, K), np.nan, np.float32)
    df = np.zeros((D, F), np.float32)
    dl = np.zeros(D, np.float32); ds = np.zeros(D, np.int8)

    for d, i in enumerate(idx):
        df[d] = feat(topos[i], int(sizes[i])); dl[d] = losses[i]; ds[d] = int(sizes[i])
        try:
            pairs = pairs_for_topology(topos[i], int(sizes[i]))
        except Exception:
            continue
        vals = bp[int(po[i]):int(po[i]) + int(pl[i])]
        for j, p in enumerate(pairs):
            k = canonical_key(p); kid = key_to_id.get(f"{k[0]}/{k[1]}")
            if kid is None: continue
            dq[d, kid] = v2q(float(vals[j]), per_key_q[kid])

    w = np.exp(-dl / 0.1); w /= w.sum()
    mu = np.zeros(K, np.float32); sd = np.zeros(K, np.float32)
    for kk in range(K):
        col = dq[:, kk]; m = np.isfinite(col)
        if m.sum() < 3:
            mu[kk], sd[kk] = 0.5, 0.3; continue
        wt = w[m]; wt = wt / wt.sum()
        m_ = float((wt * col[m]).sum())
        var = float((wt * (col[m] - m_) ** 2).sum())
        mu[kk], sd[kk] = m_, float(np.sqrt(var) + 1e-3)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, donor_q=dq, donor_feat=df, donor_loss=dl, donor_size=ds,
                        key_prop_id=key_prop_id, global_mean_q=mu, global_std_q=sd)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
