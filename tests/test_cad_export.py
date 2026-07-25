from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from origamicad import Cadder
from origamicad.io.cad_export import save_step


def _four_layer_model() -> Cadder:
    model = Cadder(unit="mm")
    for layer_index in range(4):
        prefix = f"layer_{layer_index}::"
        z = float(layer_index)
        for point_id, x, y in (
            ("p0", 0.0, 0.0),
            ("p1", 1.0, 0.0),
            ("p2", 0.0, 1.0),
        ):
            model.add_point(f"{prefix}{point_id}", x, y, z)
        model.surfaces[f"{prefix}panel"] = {
            "vertices": [
                f"{prefix}p0",
                f"{prefix}p1",
                f"{prefix}p2",
            ],
        }
    return model


class StepLayerPartTests(unittest.TestCase):
    def test_exports_four_named_step_products(self) -> None:
        model = _four_layer_model()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layers.step"
            save_step(model, path, separate_layer_parts=True)
            step_text = path.read_text(encoding="ascii")

        self.assertEqual(step_text.count("PRODUCT('Layer_"), 4)
        self.assertEqual(
            step_text.count("SHAPE_DEFINITION_REPRESENTATION("),
            4,
        )
        self.assertEqual(
            step_text.count("MANIFOLD_SURFACE_SHAPE_REPRESENTATION("),
            4,
        )
        for layer_index in range(1, 5):
            self.assertIn(f"PRODUCT('Layer_{layer_index}'", step_text)

    def test_rejects_unstacked_surface_ids(self) -> None:
        model = Cadder(unit="mm")
        model.add_point("p0", 0.0, 0.0, 0.0)
        model.add_point("p1", 1.0, 0.0, 0.0)
        model.add_point("p2", 0.0, 1.0, 0.0)
        model.surfaces["panel"] = {
            "vertices": ["p0", "p1", "p2"],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layers.step"
            with self.assertRaisesRegex(
                ValueError,
                "layer_<index>::<surface_id>",
            ):
                save_step(model, path, separate_layer_parts=True)


if __name__ == "__main__":
    unittest.main()
