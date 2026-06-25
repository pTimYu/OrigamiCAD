from __future__ import annotations

from pathlib import Path
from typing import Literal


CreaseStyle = Literal["solid", "dashed"]


def save_dxf(
    metadata: dict,
    filename: str | Path,
    include_creases: bool = True,
    crease_style: CreaseStyle = "dashed",
    include_construction: bool = False,
    include_rigid: bool = True,
    include_side: bool = True,
) -> Path:
    """Save 2D metadata as an ASCII DXF file."""
    path = Path(filename)
    path.write_text(
        dxf_string_from_metadata(
            metadata,
            include_creases=include_creases,
            crease_style=crease_style,
            include_construction=include_construction,
            include_rigid=include_rigid,
            include_side=include_side,
        ),
        encoding="ascii",
        newline="\n",
    )
    return path


def dxf_string_from_metadata(
    metadata: dict,
    include_creases: bool = True,
    crease_style: CreaseStyle = "dashed",
    include_construction: bool = False,
    include_rigid: bool = True,
    include_side: bool = True,
) -> str:
    """Convert 2D metadata to a simple DXF string made of LINE entities."""
    if crease_style not in {"solid", "dashed"}:
        raise ValueError("crease_style must be 'solid' or 'dashed'.")

    points = metadata.get("points", {})
    lines = metadata.get("lines", {})
    unit = metadata.get("metadata", {}).get("unit", "mm")

    dxf = _DxfWriter(
        unit=unit,
        crease_style=crease_style,
        include_creases=include_creases,
        include_construction=include_construction,
        include_rigid=include_rigid,
        include_side=include_side,
    )
    dxf.write_header()
    dxf.write_tables()
    dxf.begin_entities()

    for line_id, line in lines.items():
        kind = line.get("kind", "side")

        if kind in {"valley", "mountain"} and not include_creases:
            continue
        if kind == "construction" and not include_construction:
            continue
        if kind == "rigid" and not include_rigid:
            continue
        if kind == "side" and not include_side:
            continue

        start = line["start"]
        end = line["end"]
        if start not in points or end not in points:
            raise ValueError(
                f"Line '{line_id}' references missing point(s): {start}, {end}."
            )

        dxf.add_line(
            points[start],
            points[end],
            kind=kind,
        )

    dxf.end_entities()
    return dxf.to_string()


class _DxfWriter:
    def __init__(
        self,
        unit: str = "mm",
        crease_style: CreaseStyle = "dashed",
        include_creases: bool = True,
        include_construction: bool = False,
        include_rigid: bool = True,
        include_side: bool = True,
    ):
        self.unit = unit
        self.crease_style = crease_style
        self.include_creases = include_creases
        self.include_construction = include_construction
        self.include_rigid = include_rigid
        self.include_side = include_side
        self.rows: list[str] = []

    def pair(self, code: int, value) -> None:
        self.rows.extend([str(code), str(value)])

    def write_header(self) -> None:
        self.pair(0, "SECTION")
        self.pair(2, "HEADER")
        self.pair(9, "$ACADVER")
        self.pair(1, "AC1015")
        self.pair(9, "$INSUNITS")
        self.pair(70, _dxf_unit_code(self.unit))
        self.pair(0, "ENDSEC")

    def write_tables(self) -> None:
        self.pair(0, "SECTION")
        self.pair(2, "TABLES")
        self._write_linetype_table()
        self._write_layer_table()
        self.pair(0, "ENDSEC")

    def _write_linetype_table(self) -> None:
        self.pair(0, "TABLE")
        self.pair(2, "LTYPE")
        self.pair(70, 2)

        self.pair(0, "LTYPE")
        self.pair(2, "CONTINUOUS")
        self.pair(70, 0)
        self.pair(3, "Solid line")
        self.pair(72, 65)
        self.pair(73, 0)
        self.pair(40, 0.0)

        self.pair(0, "LTYPE")
        self.pair(2, "DASHED")
        self.pair(70, 0)
        self.pair(3, "Dashed __ __ __")
        self.pair(72, 65)
        self.pair(73, 2)
        self.pair(40, 6.0)
        self.pair(49, 3.0)
        self.pair(74, 0)
        self.pair(49, -3.0)
        self.pair(74, 0)

        self.pair(0, "ENDTAB")

    def _write_layer_table(self) -> None:
        self.pair(0, "TABLE")
        self.pair(2, "LAYER")
        layer_defs = self._layer_defs()
        self.pair(70, len(layer_defs))

        for layer, color, linetype in layer_defs:
            self.pair(0, "LAYER")
            self.pair(2, layer)
            self.pair(70, 0)
            self.pair(62, color)
            self.pair(6, linetype)

        self.pair(0, "ENDTAB")

    def begin_entities(self) -> None:
        self.pair(0, "SECTION")
        self.pair(2, "ENTITIES")

    def end_entities(self) -> None:
        self.pair(0, "ENDSEC")
        self.pair(0, "EOF")

    def add_line(self, start, end, kind: str) -> None:
        layer, color, linetype = self._line_properties(kind)
        x0, y0 = _xy(start)
        x1, y1 = _xy(end)

        self.pair(0, "LINE")
        self.pair(8, layer)
        self.pair(62, color)
        self.pair(6, linetype)
        self.pair(10, _number(x0))
        self.pair(20, _number(y0))
        self.pair(30, "0.0")
        self.pair(11, _number(x1))
        self.pair(21, _number(y1))
        self.pair(31, "0.0")

    def _line_properties(self, kind: str) -> tuple[str, int, str]:
        if kind == "valley":
            return "CREASE_VALLEY", 5, self._crease_linetype()
        if kind == "mountain":
            return "CREASE_MOUNTAIN", 1, self._crease_linetype()
        if kind == "rigid":
            return "RIGID", 8, "CONTINUOUS"
        if kind == "construction":
            return "CONSTRUCTION", 9, "DASHED"
        return "CUT_SIDE", 7, "CONTINUOUS"

    def _crease_linetype(self) -> str:
        return "DASHED" if self.crease_style == "dashed" else "CONTINUOUS"

    def _layer_defs(self) -> list[tuple[str, int, str]]:
        layers = []
        if self.include_side:
            layers.append(("CUT_SIDE", 7, "CONTINUOUS"))
        if self.include_rigid:
            layers.append(("RIGID", 8, "CONTINUOUS"))
        if self.include_creases:
            layers.extend(
                [
                    ("CREASE_VALLEY", 5, self._crease_linetype()),
                    ("CREASE_MOUNTAIN", 1, self._crease_linetype()),
                ]
            )
        if self.include_construction:
            layers.append(("CONSTRUCTION", 9, "DASHED"))
        return layers

    def to_string(self) -> str:
        return "\n".join(self.rows) + "\n"


def _xy(coords) -> tuple[float, float]:
    if len(coords) < 2:
        raise ValueError(f"Point coordinate must contain at least x and y: {coords}")
    return float(coords[0]), float(coords[1])


def _number(value: float) -> str:
    return f"{float(value):.12g}"


def _dxf_unit_code(unit: str) -> int:
    return {
        "in": 1,
        "inch": 1,
        "inches": 1,
        "ft": 2,
        "feet": 2,
        "mi": 3,
        "mile": 3,
        "miles": 3,
        "mm": 4,
        "millimeter": 4,
        "millimeters": 4,
        "cm": 5,
        "centimeter": 5,
        "centimeters": 5,
        "m": 6,
        "meter": 6,
        "meters": 6,
    }.get(str(unit).lower(), 0)
