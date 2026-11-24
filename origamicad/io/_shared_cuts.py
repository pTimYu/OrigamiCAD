"""Remove duplicate laser cuts on coincident straight lines."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product
import math


_Point = tuple[float, float]
_Segment = tuple[_Point, _Point]
_LineKey = tuple[int, int, int]
_CUT_TOLERANCE_MM = 1e-7
_DIRECTION_TOLERANCE = 1e-7
_NEIGHBOR_BUCKETS = tuple(product((-1, 0, 1), repeat=3))


@dataclass
class _CutLine:
    anchor: _Point
    direction: _Point
    cuts: list[tuple[int, _Segment]] = field(default_factory=list)


def _offset(direction: _Point, origin: _Point, point: _Point) -> float:
    return direction[0] * (point[1] - origin[1]) - direction[1] * (point[0] - origin[0])


def _line_key(direction: _Point, offset: float, offset_bucket: float) -> _LineKey:
    return (
        math.floor(direction[0] / _DIRECTION_TOLERANCE),
        math.floor(direction[1] / _DIRECTION_TOLERANCE),
        math.floor(offset / offset_bucket),
    )


def _find_line(
    buckets: dict[_LineKey, list[_CutLine]],
    key: _LineKey,
    direction: _Point,
    start: _Point,
    end: _Point,
) -> _CutLine | None:
    for di, dj, dk in _NEIGHBOR_BUCKETS:
        for line in buckets.get((key[0] + di, key[1] + dj, key[2] + dk), ()):
            ux, uy = line.direction
            if abs(ux * direction[1] - uy * direction[0]) > _DIRECTION_TOLERANCE:
                continue
            if (
                abs(_offset(line.direction, line.anchor, start)) <= _CUT_TOLERANCE_MM
                and abs(_offset(line.direction, line.anchor, end)) <= _CUT_TOLERANCE_MM
            ):
                return line
    return None


def _project(point: _Point, line: _CutLine) -> float:
    return (
        (point[0] - line.anchor[0]) * line.direction[0]
        + (point[1] - line.anchor[1]) * line.direction[1]
    )


def _unique_intervals(line: _CutLine) -> Iterable[tuple[int, _Segment]]:
    if len(line.cuts) == 1:
        yield line.cuts[0]
        return

    intervals = []
    for index, (start, end) in line.cuts:
        left, right = _project(start, line), _project(end, line)
        reverse = left > right
        if reverse:
            left, right = right, left
            start, end = end, start
        intervals.append((left, -right, index, start, end, reverse))

    covered_until = -math.inf
    covered_end = None
    for left, negative_right, index, start, end, reverse in sorted(intervals):
        right = -negative_right
        if covered_end is not None and left < covered_until:
            if right <= covered_until + _CUT_TOLERANCE_MM:
                continue
            start = covered_end
        covered_until, covered_end = right, end
        yield index, (end, start) if reverse else (start, end)


def merge_shared_cuts(segments: Iterable[_Segment]) -> list[_Segment]:
    """Deduplicate common cuts in millimetres, keeping adjacent cuts separate."""
    segments = list(segments)
    if not segments:
        return []

    origin = segments[0][0]
    extent = max(
        max(abs(point[0] - origin[0]), abs(point[1] - origin[1]))
        for segment in segments
        for point in segment
    )
    # A small change in direction can move the offset of a distant line.
    # Widen the search bucket to account for it, then check each endpoint.
    offset_bucket = _CUT_TOLERANCE_MM + 2.0 * extent * _DIRECTION_TOLERANCE
    buckets: dict[_LineKey, list[_CutLine]] = {}
    lines = []

    lengths = [math.dist(start, end) for start, end in segments]
    # Long segments give the most stable direction for a group of cuts.
    for index in sorted(range(len(segments)), key=lengths.__getitem__, reverse=True):
        start, end = segments[index]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = lengths[index]
        if length == 0.0:
            continue
        direction = dx / length, dy / length
        offset = _offset(direction, origin, start)
        key = _line_key(direction, offset, offset_bucket)
        line = _find_line(buckets, key, direction, start, end)
        if line is None:
            line = _CutLine(start, direction)
            lines.append(line)
            reverse_key = _line_key((-direction[0], -direction[1]), -offset, offset_bucket)
            buckets.setdefault(key, []).append(line)
            buckets.setdefault(reverse_key, []).append(line)
        line.cuts.append((index, (start, end)))

    cuts: list[_Segment | None] = [None] * len(segments)
    for line in lines:
        for index, segment in _unique_intervals(line):
            cuts[index] = segment
    return [segment for segment in cuts if segment is not None]
