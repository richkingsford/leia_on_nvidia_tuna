import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_MAIN


class TestManualTrainingStreamFooter(unittest.TestCase):
    def test_cyan_shade_footer_lines_use_expected_hex_palette(self):
        lines = a_MAIN._cyan_shade_footer_lines()

        self.assertEqual(len(lines), 6)
        self.assertIn("#018F97", lines[0])
        self.assertIn("#005A5F", lines[1])
        self.assertIn("#017E87", lines[-1])

    def test_stream_footer_html_includes_cyan_shades_section(self):
        footer_html = a_MAIN.stream_footer_html()

        self.assertIn("Crown Brick Shades", footer_html)
        self.assertIn("Shared 1% Floor", footer_html)
        self.assertIn("#018F97", footer_html)
        self.assertIn("#005A5F", footer_html)
        self.assertIn("#007179", footer_html)
        self.assertIn("#049397", footer_html)
        self.assertIn("#00898F", footer_html)
        self.assertIn("#017E87", footer_html)
        self.assertIn("Forward (R): pwm=103, pwr=0.306, t=255ms", footer_html)
        self.assertIn("Left (Q): pwm=102, pwr=0.301, t=135ms", footer_html)


if __name__ == "__main__":
    unittest.main()
