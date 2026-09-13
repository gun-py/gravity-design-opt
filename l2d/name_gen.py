_BS = set("ABCD")


def _ip(n): return [f"{r}{c}" for r in range(1, n + 1) for c in range(1, n + 1)]


def _bp(n):
    g = n + 2
    ps = [f"0{c}" for c in range(1, g - 1)]
    for r in range(1, g - 1):
        ps += [f"{r}0", f"{r}{g - 1}"]
    ps += [f"{g - 1}{c}" for c in range(1, g - 1)]
    return ps


def parse_topology(t, n):
    a, b = t.split("-")
    return dict(zip(_ip(n), a)), dict(zip(_bp(n), b))


def pairs_for_topology(t, n):
    C, B = parse_topology(t, n)
    out = []
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            xy = f"{r}{c}"
            if C[xy] in _BS:
                out += [[f"center{xy}", "reflectivity"], [f"center{xy}", "tuning"], [f"center{xy}sus", "mass"]]
            for m in ("ml", "mr", "mt", "mb"):
                out += [[f"{m}{xy}", "reflectivity"], [f"{m}{xy}", "tuning"]]
            for m in ("ml", "mr", "mt", "mb"):
                out.append([f"{m}{xy}sus", "mass"])

    def _bc(p, k):
        e = [[f"m{p}", "reflectivity"], [f"m{p}", "tuning"], [f"m{p}sus", "mass"]]
        if k == "L": e.append([f"boundary{p}", "power"])
        elif k == "S": e += [[f"boundary{p}", "db"], [f"boundary{p}", "angle"]]
        elif k == "H": e += [[f"boundary{p}bhbs", "reflectivity"], [f"boundary{p}bhbs", "tuning"]]
        return e

    for x in range(1, n + 1):
        out += _bc(f"{x}0", B[f"{x}0"]) + [[f"m{x}0_ml{x}1", "length"]]
        out += _bc(f"{x}{n + 1}", B[f"{x}{n + 1}"]) + [[f"mr{x}{n}_m{x}{n + 1}", "length"]]
    for y in range(1, n + 1):
        out += _bc(f"0{y}", B[f"0{y}"]) + [[f"m0{y}_mt1{y}", "length"]]
        out += _bc(f"{n + 1}{y}", B[f"{n + 1}{y}"]) + [[f"mb{n}{y}_m{n + 1}{y}", "length"]]
    for k in range(1, n):
        out.append([[f"mr{x}{k}_ml{x}{k + 1}", "length"] for x in range(1, n + 1)])
    for k in range(1, n):
        out.append([[f"mt{k + 1}{y}_mb{k}{y}", "length"] for y in range(1, n + 1)])
    return out


def signature(t, n):
    C, B = parse_topology(t, n)
    return tuple(1 if C[p] in _BS else 0 for p in _ip(n)) + tuple(B[p] for p in _bp(n))


def canonical_key(p):
    return canonical_key(p[0]) if isinstance(p[0], (list, tuple)) else (p[0], p[1])
