#Daniel Cirincione
#4 September 2025
#UTampa CSC410 A-Star Assignment
'''
This program implements the A* search algorithm on a 2D embedded, undirected graph.
The input is a .data file specifying the number of cities, their labels and coordinates,
and edges between cities. The program finds the shortest path from the first city listed
to the last city listed using A*, printing the explored order, the shortest path, and total cost.
'''

#imports
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
    #split a line by commas, strip whitespace, and verify expected number of parts
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {len(parts)} in line: {line!r}")
    return parts

def parse_graph(path: str):
    """Parse the graph file and return (start, goal, coords, adj)."""
    #read all lines from the input file
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    #strip blanks and comments (lines starting with '#')
    lines = []
    for ln in raw_lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)

    if not lines:
        raise ValueError("Empty (or comment-only) file.")

    #parse number of cities N from first line
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

    #parse city labels and coordinates
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

    #define start and goal cities
    start = cities_order[0]
    goal = cities_order[-1]

    #initialize adjacency dictionary for graph edges
    adj: Dict[str, Dict[str, float]] = {c: {} for c in cities_order}

    #parse edges and compute Euclidean distances as weights
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
    #euclidean distance heuristic between points a and b
    return math.hypot(a[0]-b[0], a[1]-b[1])

def reconstruct_path(parent: Dict[str, Optional[str]], goal: str) -> List[str]:
    #reconstruct path from start to goal using parent pointers
    path: List[str] = []
    cur: Optional[str] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path

def a_star(start: str, goal: str, coords: Dict[str, Tuple[float, float]], adj: Dict[str, Dict[str, float]], show_debug: bool=False):
    """Run A* and return (path, explored_order, total_cost)."""
    #validate start and goal presence
    if start not in coords or goal not in coords:
        raise ValueError("Start or goal not found in coords.")
    #precompute heuristic values for all nodes
    h = {v: heuristic(coords[v], coords[goal]) for v in coords}
    #initialize g-score (cost from start) for all nodes to infinity
    g = {v: float("inf") for v in coords}
    g[start] = 0.0
    #parent dictionary to reconstruct path
    parent: Dict[str, Optional[str]] = {start: None}
    #list to record order nodes are explored
    explored_order: List[str] = []

    counter = 0  #tie breaker counter for priority queue
    #initialize priority queue with start node
    pq: List[PQItem] = [PQItem(g[start] + h[start], counter, start)]
    heapq.heapify(pq)
    in_open: Set[str] = {start}  #track nodes in open set
    closed: Set[str] = set()     #track nodes already processed

    while pq:
        #pop node with lowest f = g + h
        item = heapq.heappop(pq)
        u = item.node
        if u in closed:
            continue
        in_open.discard(u)
        explored_order.append(u)
        closed.add(u)
        if show_debug:
            print(f"[POP] {u}: g={g[u]:.6f}, h={h[u]:.6f}, f={g[u]+h[u]:.6f}")
            print()
        #if goal reached, reconstruct and return path
        if u == goal:
            path = reconstruct_path(parent, goal)
            return path, explored_order, g[goal]

        #relax edges from u to neighbors v
        for v, w in adj[u].items():
            if v in closed:
                continue
            tentative = g[u] + w
            if tentative < g[v]:
                #found better path to v
                g[v] = tentative
                parent[v] = u
                counter += 1
                heapq.heappush(pq, PQItem(g[v] + h[v], counter, v))
                in_open.add(v)
                if show_debug:
                    print(f"    [RELAX] {u} -> {v}: w={w:.6f}, g[v]={g[v]:.6f}, h[v]={h[v]:.6f}, f={g[v]+h[v]:.6f}")

    #no path found
    return [], explored_order, float("inf")

def main(argv: list[str]) -> int:
    import argparse
    #setup command line argument parser
    parser = argparse.ArgumentParser(description="A* on a 2D embedded, undirected graph.")
    parser.add_argument("input_file", help="Path to input .data file")
    parser.add_argument("--show-debug", action="store_true", help="Print g/h/f updates during search")
    args = parser.parse_args(argv)

    #parse graph from input file
    start, goal, coords, adj = parse_graph(args.input_file)
    #run A* search
    path, explored, cost = a_star(start, goal, coords, adj, show_debug=args.show_debug)

    #print results
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
