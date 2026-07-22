"""Grid path planning constrained to user-defined working regions."""

from __future__ import annotations

import heapq
import math
from typing import Any


Point = tuple[float, float]
GridPoint = tuple[int, int]


def point_in_work(point: Point, regions: list[dict[str, Any]]) -> bool:
    x, y = point
    return any(
        region.get("type") == "work"
        and float(region["x"]) <= x <= float(region["x"]) + float(region["width"])
        and float(region["y"]) <= y <= float(region["y"]) + float(region["height"])
        for region in regions
    )


def point_has_clearance(point: Point, regions: list[dict[str, Any]], clearance: float) -> bool:
    if not point_in_work(point, regions):
        return False
    if clearance <= 0:
        return True
    x, y = point
    offsets = (
        (-clearance, 0), (clearance, 0), (0, -clearance), (0, clearance),
        (-clearance * .71, -clearance * .71), (clearance * .71, -clearance * .71),
        (clearance * .71, clearance * .71), (-clearance * .71, clearance * .71),
    )
    return all(point_in_work((x + dx, y + dy), regions) for dx, dy in offsets)


def segment_is_safe(start: Point, end: Point, regions: list[dict[str, Any]], clearance: float) -> bool:
    distance = math.dist(start, end)
    samples = max(2, int(math.ceil(distance / .65)))
    return all(
        point_has_clearance(
            (start[0] + (end[0] - start[0]) * index / samples,
             start[1] + (end[1] - start[1]) * index / samples),
            regions,
            clearance,
        )
        for index in range(samples + 1)
    )


def _nearest_safe_node(point: Point, safe_nodes: set[GridPoint], step: float) -> GridPoint | None:
    if not safe_nodes:
        return None
    return min(safe_nodes, key=lambda node: math.dist(point, (node[0] * step, node[1] * step)))


def _simplify(points: list[Point], regions: list[dict[str, Any]], clearance: float) -> list[Point]:
    if len(points) <= 2:
        return points
    result = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1 and not segment_is_safe(points[anchor], points[candidate], regions, clearance):
            candidate -= 1
        result.append(points[candidate])
        anchor = candidate
    return result


def plan_work_route(
    start: Point,
    target: Point,
    regions: list[dict[str, Any]],
    *,
    clearance: float = 1.4,
    step: float = 2.0,
) -> list[Point]:
    """Return normalized waypoints after start; raise ValueError if no safe route exists."""
    work_regions = [region for region in regions if region.get("type") == "work"]
    if not work_regions:
        raise ValueError("Сначала нарисуйте рабочую область")
    if not point_in_work(start, work_regions):
        raise ValueError("Агент находится вне рабочей поверхности")
    if not point_in_work(target, work_regions):
        raise ValueError("Точка назначения находится вне рабочей поверхности")
    if not point_has_clearance(target, work_regions, clearance):
        raise ValueError("Точка слишком близко к границе рабочей поверхности")
    if segment_is_safe(start, target, work_regions, clearance):
        return [target]

    limit = int(round(100.0 / step))
    safe_nodes: set[GridPoint] = {
        (gx, gy)
        for gx in range(limit + 1)
        for gy in range(limit + 1)
        if point_has_clearance((gx * step, gy * step), work_regions, clearance)
    }
    start_node = _nearest_safe_node(start, safe_nodes, step)
    target_node = _nearest_safe_node(target, safe_nodes, step)
    if start_node is None or target_node is None:
        raise ValueError("Рабочая поверхность слишком узкая для безопасного маршрута")

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    frontier: list[tuple[float, GridPoint]] = [(0.0, start_node)]
    came_from: dict[GridPoint, GridPoint | None] = {start_node: None}
    cost: dict[GridPoint, float] = {start_node: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == target_node:
            break
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in safe_nodes:
                continue
            if dx and dy and ((current[0] + dx, current[1]) not in safe_nodes or (current[0], current[1] + dy) not in safe_nodes):
                continue
            new_cost = cost[current] + math.hypot(dx, dy)
            if new_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = new_cost
            priority = new_cost + math.dist(neighbor, target_node)
            heapq.heappush(frontier, (priority, neighbor))
            came_from[neighbor] = current

    if target_node not in came_from:
        raise ValueError("Рабочие области не соединены — безопасного маршрута нет")
    nodes: list[GridPoint] = []
    cursor: GridPoint | None = target_node
    while cursor is not None:
        nodes.append(cursor)
        cursor = came_from[cursor]
    nodes.reverse()
    points = [start] + [(node[0] * step, node[1] * step) for node in nodes[1:]] + [target]
    simplified = _simplify(points, work_regions, clearance)
    return simplified[1:]
