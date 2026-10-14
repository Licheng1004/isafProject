"""Protect the public demo without downloading gated foundation-model weights."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

def test_public_app_starts_without_foundation_model():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()
    assert not app.exception
    assert not app.error
    assert app.sidebar.toggle[0].value is False
    assert any("Which model" in title.value for title in app.title)
