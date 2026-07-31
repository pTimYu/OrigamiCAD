# OrigamiCAD

OrigamiCAD is a small Python toolkit for drawing 2D origami crease patterns,
solving folded 3D kinematic configurations, projecting them back to 2D, and
exporting JSON, DXF, STL, or STEP files.

Typical imports:

```python
from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import build_packaging, solve_kinematics

pattern = TwoDDrawer()
build_packaging(pattern)

model = Cadder.from_drawer(pattern)
solve_kinematics(model, final_dihedral=135.0)
```

The reusable core is kept separate from pattern-specific code:

```text
origamicad/
  core/                 # generic 2D/3D models and constraint solver
  patterns/
    hexagon/
      layout.py         # hexagon geometry and metadata generation
      kinematics.py     # hexagon-specific constraint setup and solving
      metadata.py       # shared hexagon metadata types
  io/                   # JSON, DXF, STL, and STEP export
```

Run the 2D and 3D examples from the project root with:

```bash
python -m examples.hexagon_example_2d
python -m examples.hexagon_example_3d
```
