# offline: label bounded_params by (component, property), pack a lean prior.npz for warm.py
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "l2d"))

from name_gen import pairs_for_topology, canonical_key, signature

DATA = os.path.join(ROOT, "dataset", "dataset.h5")
OUT = os.path.join(ROOT, "l2d", "prior.npz")

TOP_DECILE_FRAC = 0.10
SAMPLES_PER_KEY = 256
DONOR_LOSS_MAX = 0.4
MAX_DONORS = 4000

PROP_BOUNDS = {
    "reflectivity": (1e-12, 1.0 - 1e-12),
    "tuning":       (-360.0, 360.0),
    "db":           (0.0, 10.0),
    "angle":        (-360.0, 360.0),
    "power":        (0.0, 200.0),
    "mass":         (0.01, 200.0),
    "length":       (0.1, 4000.0),
}


def main():
    os.makedirs(os.path.join(ROOT, "l2d"), exist_ok=True)
    print(f"reading {DATA}")
    t0 = time.time()
    with h5py.File(DATA, "r") as h5:
        e = h5["entries"][:]; n = len(e)
        losses = e["loss"].astype(np.float32)
        sizes = e["size"].astype(np.int16)
        po, pl = e["param_offset"].astype(np.int64), e["param_length"].astype(np.int32)
        topos = np.array([t.decode() for t in e["topology_string"]])
        print(f"{n} entries {time.time()-t0:.1f}s")

        key_set = {}
        entry_keys = [None] * n
        for i in range(n):
            keys = [canonical_key(p) for p in pairs_for_topology(topos[i], int(sizes[i]))]
            ids = []
            for k in keys:
                if k not in key_set: key_set[k] = len(key_set)
                ids.append(key_set[k])
            entry_keys[i] = ids

        keys_sorted = sorted(key_set.keys())
        remap = np.empty(len(key_set), np.int32)
        for k, old in key_set.items(): remap[old] = keys_sorted.index(k)
        entry_keys = [remap[np.asarray(ids)].tolist() for ids in entry_keys]

        K = len(keys_sorted)
        keys_arr = np.array([f"{c}/{p}" for c, p in keys_sorted])
        key_prop = np.array([p for _, p in keys_sorted])
        prop_arr = np.array(sorted(set(key_prop.tolist())))
        prop_bounds = np.array([[PROP_BOUNDS[p][0], PROP_BOUNDS[p][1]] for p in prop_arr], np.float64)
        print(f"K={K}")

        by_v = [[] for _ in range(K)]
        by_l = [[] for _ in range(K)]
        bp = h5["bounded_params"][:]
        for i in np.argsort(po):
            off, L = int(po[i]), int(pl[i]); vals = bp[off:off + L]
            ids = entry_keys[i]; l = float(losses[i])
            for kid, v in zip(ids, vals):
                by_v[kid].append(float(v)); by_l[kid].append(l)

        per_key_top = np.zeros((K, SAMPLES_PER_KEY), np.float32)
        per_key_quant = np.zeros((K, 7), np.float32)
        per_key_count = np.zeros(K, np.int32)
        rng = np.random.default_rng(0)
        for kid in range(K):
            v = np.asarray(by_v[kid], np.float64); l = np.asarray(by_l[kid], np.float64)
            per_key_count[kid] = v.size
            if v.size == 0: continue
            per_key_quant[kid] = np.quantile(v, [0, .05, .25, .5, .75, .95, 1.0]).astype(np.float32)
            n_top = max(1, int(np.ceil(v.size * TOP_DECILE_FRAC)))
            top_v = v[np.argpartition(l, min(n_top, v.size - 1))[:n_top]]
            per_key_top[kid] = rng.choice(top_v, size=SAMPLES_PER_KEY, replace=True).astype(np.float32)

        donor_all = np.where(losses < DONOR_LOSS_MAX)[0]
        donor_idx = donor_all[np.argsort(losses[donor_all])][:MAX_DONORS]
        D = len(donor_idx); print(f"donors: {D}")

        d_off = np.zeros(D + 1, np.int64)
        d_kid_l, d_val_l = [], []
        d_size = np.zeros(D, np.int8); d_loss = np.zeros(D, np.float32); d_len = np.zeros(D, np.int32)
        d_topo = []
        d_sint = np.full((D, 9), -1, np.int8); d_sbnd = np.full((D, 12), -1, np.int8)
        bcode = {"L": 0, "S": 1, "D": 2, "H": 3}

        for d, i in enumerate(donor_idx):
            off, L = int(po[i]), int(pl[i])
            d_off[d + 1] = d_off[d] + L
            # float64 preserves near-bound reflectivity precision (round-trip stays at ~1e-6 loss delta)
            d_kid_l.append(np.asarray(entry_keys[i], np.int32))
            d_val_l.append(bp[off:off + L].astype(np.float64))
            d_size[d] = int(sizes[i]); d_loss[d] = float(losses[i]); d_len[d] = L
            d_topo.append(topos[i])
            if int(sizes[i]) == 3:
                sig = signature(topos[i], 3)
                d_sint[d, :9] = np.array(sig[:9], np.int8)
                d_sbnd[d, :12] = np.array([bcode[c] for c in sig[9:9 + 12]], np.int8)

        np.savez_compressed(
            OUT,
            keys_arr=keys_arr, key_prop=key_prop, prop_arr=prop_arr, prop_bounds=prop_bounds,
            per_key_top=per_key_top, per_key_quantiles=per_key_quant, per_key_count=per_key_count,
            donor_offsets=d_off,
            donor_key_id=np.concatenate(d_kid_l).astype(np.int32),
            donor_value=np.concatenate(d_val_l).astype(np.float64),
            donor_len=d_len, donor_size=d_size, donor_loss=d_loss,
            donor_topo_bytes=np.array(d_topo, dtype="S64"),
            donor_sig_int=d_sint, donor_sig_bnd=d_sbnd,
        )
        print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
