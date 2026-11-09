"""
Streamlit AppTest suite to simulate navigating every single page of the dashboard
and verifying that no exceptions or unhandled errors are raised during UI rendering.
"""

import unittest
import sys
from pathlib import Path

# Add root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from streamlit.testing.v1 import AppTest
    HAS_APP_TEST = True
except ImportError:
    HAS_APP_TEST = False


class TestStreamlitAppNavigation(unittest.TestCase):
    def setUp(self):
        if not HAS_APP_TEST:
            self.skipTest("streamlit.testing.v1.AppTest not available")

    def test_app_loads_home_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"Exception on Home page: {at.exception}")

    def test_app_dataset_explorer_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        # Navigate to Dataset Explorer
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value("📁 Dataset Explorer").run()
            self.assertFalse(at.exception, f"Exception on Dataset Explorer page: {at.exception}")

    def test_app_eda_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value("📊 Exploratory Data Analysis (EDA)").run()
            self.assertFalse(at.exception, f"Exception on EDA page: {at.exception}")

    def test_app_forecasting_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value("🔮 Price Forecasting Studio").run()
            self.assertFalse(at.exception, f"Exception on Forecasting page: {at.exception}")

    def test_app_model_comparison_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value("⚖️ Model Evaluation & Comparison").run()
            self.assertFalse(at.exception, f"Exception on Model Comparison page: {at.exception}")

    def test_app_external_factors_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value("🌐 External Factor Analysis").run()
            self.assertFalse(at.exception, f"Exception on External Factors page: {at.exception}")

    def test_app_reports_page(self):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        if len(at.sidebar.radio) > 0:
            at.sidebar.radio[0].set_value("📄 Reports & Data Export").run()
            self.assertFalse(at.exception, f"Exception on Reports page: {at.exception}")


if __name__ == "__main__":
    unittest.main()
