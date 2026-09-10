import sys
import unittest
from pathlib import Path

from PIL import Image
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_id.attributes.colour import estimate_colour  # noqa: E402


class ColourBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = yaml.safe_load((REPO_ROOT / "configs" / "stage2_baseline.yaml").read_text(encoding="utf-8"))
        cls.config = config["colour_baseline"]

    def test_red_image(self) -> None:
        result = estimate_colour(Image.new("RGB", (100, 100), (220, 25, 25)), self.config)
        self.assertEqual(result.label, "red")
        self.assertEqual(result.status, "baseline_estimate")
        self.assertEqual(result.confidence, 1.0)

    def test_blue_image(self) -> None:
        result = estimate_colour(Image.new("RGB", (100, 100), (25, 80, 220)), self.config)
        self.assertEqual(result.label, "blue")
        self.assertEqual(result.status, "baseline_estimate")


if __name__ == "__main__":
    unittest.main()
