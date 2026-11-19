from pathlib import Path

import pytest

from geoevidence.store import build_index


def test_demo_answers_from_real_fixture(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "demo.sqlite"
    build_index(root / "data/demo.geojson", target)
    monkeypatch.setenv("GEOEVIDENCE_INDEX", str(target))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert result.status == "answered"
    assert result.evidence[0].document_id == "lag-drain"
