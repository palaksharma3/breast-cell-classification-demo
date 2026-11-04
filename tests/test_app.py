from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from model import EXAMPLES, FEATURE_NAMES, example_values

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class AppTests(unittest.TestCase):
    def start(self):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        return app

    def submit(self, app):
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)

    def test_first_render_and_disclaimer(self):
        app = self.start()
        self.assertEqual(len(app.number_input), 30)
        self.assertIn("not for medical diagnosis", app.warning[0].value)
        self.assertEqual(len(app.table), 1)
        self.assertFalse(any("Model prediction:" in h.value for h in app.subheader))

    def test_both_examples(self):
        app = self.start()
        for source in EXAMPLES:
            app.selectbox[0].set_value(source).run()
            self.submit(app)
            self.assertTrue(any("Model prediction:" in h.value for h in app.subheader))
            self.assertEqual(len(app.error), 0)

    def test_edited_example(self):
        app = self.start()
        app.number_input[0].set_value(18.0)
        self.submit(app)
        self.assertTrue(any("Model prediction:" in h.value for h in app.subheader))

    def test_blank_custom_input_rejected(self):
        app = self.start()
        app.selectbox[0].set_value("Enter my own measurements").run()
        self.assertTrue(all(widget.value is None for widget in app.number_input))
        self.submit(app)
        self.assertIn("no blank fields", app.error[0].value)
        self.assertFalse(any("Model prediction:" in h.value for h in app.subheader))

    def test_complete_custom_input(self):
        app = self.start()
        app.selectbox[0].set_value("Enter my own measurements").run()
        for name, value in example_values(next(iter(EXAMPLES))).items():
            app.number_input(key=f"Enter my own measurements:{name}").set_value(value)
        self.submit(app)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any("Model prediction:" in h.value for h in app.subheader))

    def test_out_of_range_input_rejected(self):
        app = self.start()
        app.number_input[0].set_value(-1.0)
        self.submit(app)
        self.assertIn("observed dataset range", app.error[0].value)
        self.assertFalse(any("Model prediction:" in h.value for h in app.subheader))

    def test_changing_source_clears_previous_result(self):
        app = self.start()
        self.submit(app)
        app.selectbox[0].set_value("Enter my own measurements").run()
        self.assertFalse(any("Model prediction:" in h.value for h in app.subheader))


if __name__ == "__main__":
    unittest.main()
