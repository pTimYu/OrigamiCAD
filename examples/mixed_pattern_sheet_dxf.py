"""Fill one sheet with plain and punched hexagonal crease patterns."""

from pathlib import Path

from origamicad import TwoDDrawer
from origamicad.io import SheetPattern, save_sheet_dxf
from origamicad.patterns.hexagon import draw_hex_loops


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DXF_PATH = OUTPUT_DIR / "mixed_pattern_sheet.dxf"


def main() -> None:
    plain = TwoDDrawer(unit="mm")
    punched = TwoDDrawer(unit="mm")
    
    draw_hex_loops(plain, n=2, l=15)
    draw_hex_loops(punched, n=2, l=15, enable_hole_punch_outer=2.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layout = save_sheet_dxf(
        [SheetPattern("plain", plain)], # SheetPattern("punched", punched, quantity=2)],
        DXF_PATH,
        width_mm=558.8,
        height_mm=711.2,
        margin_mm=2.0,
        spacing_mm=0.0,
    )
    print(f"Placed {layout.counts['plain']} plain patterns.")
    print(f"Footprint coverage is {layout.utilization:.1%} of the sheet.")
    print(f"Saved sheet DXF to {DXF_PATH}")


if __name__ == "__main__":
    main()
