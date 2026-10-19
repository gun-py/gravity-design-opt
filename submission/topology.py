SPLITTERS = set("ABCD")


def _inner_pos(n):
    return [f"{r}{c}" for r in range(1, n + 1) for c in range(1, n + 1)]


def _border_pos(n):
    g = n + 2
    ps = [f"0{c}" for c in range(1, g - 1)]
    for r in range(1, g - 1):
        ps += [f"{r}0", f"{r}{g - 1}"]
    ps += [f"{g - 1}{c}" for c in range(1, g - 1)]
    return ps


def parse_topology(topo, size):
    a, b = topo.split("-")
    return dict(zip(_inner_pos(size), a)), dict(zip(_border_pos(size), b))


def pairs_for_topology(topo, size):
    center_map, border_map = parse_topology(topo, size)
    out = []
    for r in range(1, size + 1):
        for c in range(1, size + 1):
            xy = f"{r}{c}"
            if center_map[xy] in SPLITTERS:
                out += [[f"center{xy}", "reflectivity"],
                        [f"center{xy}", "tuning"],
                        [f"center{xy}sus", "mass"]]
            for m in ("ml", "mr", "mt", "mb"):
                out += [[f"{m}{xy}", "reflectivity"], [f"{m}{xy}", "tuning"]]
            for m in ("ml", "mr", "mt", "mb"):
                out.append([f"{m}{xy}sus", "mass"])

    def _boundary(pos, kind):
        e = [[f"m{pos}", "reflectivity"], [f"m{pos}", "tuning"],
             [f"m{pos}sus", "mass"]]
        if kind == "L":
            e.append([f"boundary{pos}", "power"])
        elif kind == "S":
            e += [[f"boundary{pos}", "db"], [f"boundary{pos}", "angle"]]
        elif kind == "H":
            e += [[f"boundary{pos}bhbs", "reflectivity"],
                  [f"boundary{pos}bhbs", "tuning"]]
        return e

    n = size
    for x in range(1, n + 1):
        out += _boundary(f"{x}0", border_map[f"{x}0"])
        out += [[f"m{x}0_ml{x}1", "length"]]
        out += _boundary(f"{x}{n+1}", border_map[f"{x}{n+1}"])
        out += [[f"mr{x}{n}_m{x}{n+1}", "length"]]
    for y in range(1, n + 1):
        out += _boundary(f"0{y}", border_map[f"0{y}"])
        out += [[f"m0{y}_mt1{y}", "length"]]
        out += _boundary(f"{n+1}{y}", border_map[f"{n+1}{y}"])
        out += [[f"mb{n}{y}_m{n+1}{y}", "length"]]
    for k in range(1, n):
        out.append([[f"mr{x}{k}_ml{x}{k+1}", "length"] for x in range(1, n+1)])
    for k in range(1, n):
        out.append([[f"mt{k+1}{y}_mb{k}{y}", "length"] for y in range(1, n+1)])
    return out


def signature(topo, size):
    center_map, border_map = parse_topology(topo, size)
    return (tuple(1 if center_map[p] in SPLITTERS else 0 for p in _inner_pos(size))
            + tuple(border_map[p] for p in _border_pos(size)))


def canonical_key(pair):
    if isinstance(pair[0], (list, tuple)):
        return canonical_key(pair[0])
    return (pair[0], pair[1])
