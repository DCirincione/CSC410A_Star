
#!/usr/bin/env python3
"""
A* pathfinding for a 2D-embedded, undirected graph defined in a text file.

Input format:
- First line: integer N = number of cities
- Next N lines: "Label,x,y" with Label alphabetic, x and y numbers (ints or floats)
  * The first label among these N is the start
  * The last label among these N is the goal
- Remaining lines: "U,V" edges (undirected). Either U,V or V,U indicates the same edge.
  * Lines starting with '#' or blank lines are ignored anywhere.

Edge cost: Euclidean distance between city coordinates.
Heuristic: Euclidean distance from current city to goal.

Usage:
    python a_star.py /path/to/graph.data
Options:
    --show-debug   Print g, h, and f-values as nodes are processed
"""
from __future__ import annotations
import math
import sys
import heapq
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Optional

@dataclass(order=True)
class PQItem:
    f: float
    tie: int
    node: str = field(compare=False)

def _parse_line_csv(line: str, expected: int) -> List[str]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {len(parts)} in line: {line!r}")
    return parts

def parse_graph(path: str):
    """Parse the graph file and return (start, goal, coords, adj)."""
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Strip blanks and comments
    lines = []
    for ln in raw_lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)

    if not lines:
        raise ValueError("Empty (or comment-only) file.")

    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError("First non-empty line must be an integer N (number of cities).")

    if n <= 0:
        raise ValueError("N must be positive.")
    if len(lines) < 1 + n:
        raise ValueError(f"Expected at least {1+n} lines, got {len(lines)}.")

    cities_order: List[str] = []
    coords: Dict[str, Tuple[float, float]] = {}

    # Read N city lines
    for i in range(1, 1+n):
        label, xs, ys = _parse_line_csv(lines[i], 3)
        if not label.isalpha():
            raise ValueError(f"City label must be alphabetic: got {label!r}")
        try:
            x = float(xs)
            y = float(ys)
        except ValueError:
            raise ValueError(f"Invalid coordinates for {label!r}: {xs!r}, {ys!r}")
        if label in coords:
            raise ValueError(f"Duplicate city label: {label}")
        coords[label] = (x, y)
        cities_order.append(label)

    start = cities_order[0]
    goal = cities_order[-1]

    # Build adjacency
    adj: Dict[str, Dict[str, float]] = {c: {} for c in cities_order}

    for i in range(1+n, len(lines)):
        u, v = _parse_line_csv(lines[i], 2)
        if u not in coords or v not in coords:
            raise ValueError(f"Edge references unknown city: {u},{v}")
        ux, uy = coords[u]
        vx, vy = coords[v]
        w = math.hypot(ux - vx, uy - vy)
        adj[u][v] = w
        adj[v][u] = w

    return start, goal, coords, adj

def heuristic(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def reconstruct_path(parent: Dict[str, Optional[str]], goal: str) -> List[str]:
    path: List[str] = []
    cur: Optional[str] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

def a_star(start: str, goal: str, coords: Dict[str, Tuple[float, float]], adj: Dict[str, Dict[str, float]], show_debug: bool=False):
    """Run A* and return (path, explored_order, total_cost)."""
    if start not in coords or goal not in coords:
        raise ValueError("Start or goal not found in coords.")
    h = {v: heuristic(coords[v], coords[goal]) for v in coords}
    g = {v: float("inf") for v in coords}
    g[start] = 0.0
    parent: Dict[str, Optional[str]] = {start: None}
    explored_order: List[str] = []

    counter = 0
    pq: List[PQItem] = [PQItem(g[start] + h[start], counter, start)]
    heapq.heapify(pq)
    in_open: Set[str] = {start}
    closed: Set[str] = set()

    while pq:
        item = heapq.heappop(pq)
        u = item.node
        if u in closed:
            continue
        in_open.discard(u)
        explored_order.append(u)
        closed.add(u)
        if show_debug:
            print(f"[POP] {u}: g={g[u]:.6f}, h={h[u]:.6f}, f={g[u]+h[u]:.6f}")
        if u == goal:
            path = reconstruct_path(parent, goal)
            return path, explored_order, g[goal]

        for v, w in adj[u].items():
            if v in closed:
                continue
            tentative = g[u] + w
            if tentative < g[v]:
                g[v] = tentative
                parent[v] = u
                counter += 1
                heapq.heappush(pq, PQItem(g[v] + h[v], counter, v))
                in_open.add(v)
                if show_debug:
                    print(f"  [RELAX] {u} -> {v}: w={w:.6f}, g[v]={g[v]:.6f}, h[v]={h[v]:.6f}, f={g[v]+h[v]:.6f}")

    # No path
    return [], explored_order, float("inf")

def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="A* on a 2D embedded, undirected graph.")
    parser.add_argument("input_file", help="Path to input .data file")
    parser.add_argument("--show-debug", action="store_true", help="Print g/h/f updates during search")
    args = parser.parse_args(argv)

    start, goal, coords, adj = parse_graph(args.input_file)
    path, explored, cost = a_star(start, goal, coords, adj, show_debug=args.show_debug)

    print(f"Start: {start}")
    print(f"Goal:  {goal}")
    print("Explored order:")
    print("  " + ", ".join(explored))
    if path:
        print("Shortest path:")
        print("  " + " -> ".join(path))
        print(f"Total cost: {cost:.6f}")
        return 0
    else:
        print("No path found.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
