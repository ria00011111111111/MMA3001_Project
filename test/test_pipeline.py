
"""Structural integration checks for the occupancy pipeline."""

import ast
import unittest
import run as pipeline


class TestOccupancyPipeline(unittest.TestCase):

    def test_seven_scripts_are_configured(self):
        self.assertEqual(len(pipeline.SCRIPTS), 7)

    def test_script_names_are_unique(self):
        self.assertEqual(len(pipeline.SCRIPTS), len(set(pipeline.SCRIPTS)))

    def test_all_scripts_exist(self):
        for filename in pipeline.SCRIPTS:
            with self.subTest(script=filename):
                path = pipeline.SCRIPT_DIR / filename
                self.assertTrue(path.is_file(), f"Missing script: {path}")

    def test_all_scripts_have_valid_python_syntax(self):
        for filename in pipeline.SCRIPTS:
            with self.subTest(script=filename):
                path = pipeline.SCRIPT_DIR / filename
                source = path.read_text(encoding="utf-8-sig")
                ast.parse(source, filename=str(path))

    def test_script_names_end_in_python_extension(self):
        for filename in pipeline.SCRIPTS:
            with self.subTest(script=filename):
                self.assertTrue(filename.endswith(".py"))


if __name__ == "__main__":
    unittest.main()
