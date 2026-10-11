# auto-size the vmap batch. the organisers' batch-19 figure is for an 80GB H100; the
# simulator runs in float64 (differometor sets jax_enable_x64 at import) so buffers are
# 2x, and a 16-40GB GPU OOMs long before 19. warmup is pre-clock and free, so probe there.
import os


def _try(fn, b):
    try:
        fn(batch_size=b)
        return True
    except Exception:
        try:
            import jax
            for d in jax.local_devices():
                try: d.memory_stats()
                except Exception: pass
        except Exception:
            pass
        return False


def pick_batch(obj, kind="grad_aux", cap=19, floor=1, verbose=True):
    """Largest batch that warms up without OOM. Honours $L2D_BATCH if set."""
    env = os.environ.get("L2D_BATCH")
    if env:
        b = max(floor, min(cap, int(env)))
        fn = {"value": obj.warmup_vmap_value,
              "grad": obj.warmup_vmap_value_and_grad,
              "grad_aux": obj.warmup_vmap_value_and_grad_aux}[kind]
        _try(fn, b)
        if verbose: print(f"[l2d] batch={b} ({kind}, from $L2D_BATCH)", flush=True)
        return b

    fn = {"value": obj.warmup_vmap_value,
          "grad": obj.warmup_vmap_value_and_grad,
          "grad_aux": obj.warmup_vmap_value_and_grad_aux}[kind]
    for b in [x for x in (19, 16, 12, 10, 8, 6, 4, 3, 2, 1) if floor <= x <= cap]:
        if _try(fn, b):
            if verbose: print(f"[l2d] batch={b} ({kind})", flush=True)
            return b
    if verbose: print(f"[l2d] batch={floor} ({kind}, all probes failed)", flush=True)
    return floor
