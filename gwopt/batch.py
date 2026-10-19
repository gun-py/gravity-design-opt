import os


def _try_warmup(warmup_fn, batch_size):
    try:
        warmup_fn(batch_size=batch_size)
        return True
    except Exception:
        try:
            import jax
            for dev in jax.local_devices():
                try:
                    dev.memory_stats()
                except Exception:
                    pass
        except Exception:
            pass
        return False


def pick_batch_size(obj, kind="grad_aux", cap=19, floor=1, verbose=True):
    env = os.environ.get("L2D_BATCH")
    warmup_fns = {
        "value": obj.warmup_vmap_value,
        "grad": obj.warmup_vmap_value_and_grad,
        "grad_aux": obj.warmup_vmap_value_and_grad_aux,
    }
    fn = warmup_fns[kind]

    if env:
        b = max(floor, min(cap, int(env)))
        _try_warmup(fn, b)
        if verbose:
            print(f"[gwopt] batch={b} ({kind}, from $L2D_BATCH)", flush=True)
        return b

    for b in [x for x in (19, 16, 12, 10, 8, 6, 4, 3, 2, 1) if floor <= x <= cap]:
        if _try_warmup(fn, b):
            if verbose:
                print(f"[gwopt] batch={b} ({kind})", flush=True)
            return b

    if verbose:
        print(f"[gwopt] batch={floor} ({kind}, all probes failed)", flush=True)
    return floor
