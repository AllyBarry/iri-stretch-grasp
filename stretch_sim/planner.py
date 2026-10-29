import heapq
import math
from typing import List, Optional, Tuple

from stretch_sim.geometry import Point, point_segment_distance
from stretch_sim.world import WorldMap

Cell = Tuple[int, int]


class GridPlanner:
    '''
    8-connected A* on an occupancy grid built from the walls and furniture, inflated by
    the robot's radius, followed by line-of-sight shortcutting so paths are a few
    straight segments rather than a staircase of cells.
    '''
    def __init__(self, world: WorldMap, robot_radius: float = 0.22, resolution: float = 0.05,
                 wall_thickness: float = 0.05):
        self.res = resolution
        self.nx = int(math.ceil(world.width / resolution))
        self.ny = int(math.ceil(world.height / resolution))
        inflate = robot_radius
        self.blocked = [[False] * self.ny for _ in range(self.nx)]

        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.cell_centre((i, j))
                if x < inflate or y < inflate or x > world.width - inflate or y > world.height - inflate:
                    self.blocked[i][j] = True
                elif any(rect.contains(x, y, inflate) for rect in world.blocking_rects()):
                    self.blocked[i][j] = True
                elif any(point_segment_distance((x, y), a, b) < inflate + wall_thickness / 2.0
                         for a, b in world.walls):
                    self.blocked[i][j] = True

    def cell(self, p: Point) -> Cell:
        i = min(max(int(p[0] / self.res), 0), self.nx - 1)
        j = min(max(int(p[1] / self.res), 0), self.ny - 1)
        return (i, j)

    def cell_centre(self, c: Cell) -> Point:
        return ((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res)

    def free(self, c: Cell) -> bool:
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and not self.blocked[c[0]][c[1]]

    def _nearest_free(self, c: Cell, max_radius: int = 10) -> Optional[Cell]:
        '''The robot can end up slightly inside the inflated region, e.g. after a grasp shifts the base.'''
        if self.free(c):
            return c
        for r in range(1, max_radius + 1):
            ring = [(c[0] + di, c[1] + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                    if max(abs(di), abs(dj)) == r]
            ring = [n for n in ring if self.free(n)]
            if ring:
                return min(ring, key=lambda n: math.hypot(n[0] - c[0], n[1] - c[1]))
        return None

    def plan(self, start: Point, goal: Point) -> Optional[List[Point]]:
        '''Waypoints from start to goal (both included), or None if the goal is unreachable.'''
        s = self._nearest_free(self.cell(start))
        g = self._nearest_free(self.cell(goal), max_radius=3)
        if s is None or g is None:
            return None

        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        frontier = [(h(s), 0.0, s)]
        came_from = {s: None}
        cost = {s: 0.0}
        while frontier:
            _, c_cost, current = heapq.heappop(frontier)
            if current == g:
                break
            if c_cost > cost[current]:
                continue
            for di, dj, step in moves:
                n = (current[0] + di, current[1] + dj)
                # No corner cutting past blocked cells.
                if not self.free(n) or not self.free((current[0] + di, current[1])) or not self.free((current[0], current[1] + dj)):
                    continue
                new_cost = c_cost + step
                if new_cost < cost.get(n, math.inf):
                    cost[n] = new_cost
                    came_from[n] = current
                    heapq.heappush(frontier, (new_cost + h(n), new_cost, n))
        else:
            return None

        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = came_from[c]
        cells.reverse()

        points = [start] + [self.cell_centre(c) for c in self._shortcut(cells)[1:-1]] + [goal]
        return points

    def _line_free(self, a: Cell, b: Cell) -> bool:
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for k in range(1, n + 1):
            t = k / n
            c = (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t))
            if not self.free(c):
                return False
        return True

    def _shortcut(self, cells: List[Cell]) -> List[Cell]:
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_free(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out
