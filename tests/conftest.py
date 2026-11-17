import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_bytes():
    return (ROOT / "data" / "sample_sales.csv").read_bytes()


@pytest.fixture
def sample_ds(sample_bytes):
    from app.ingest import load_file
    return load_file("sample.csv", sample_bytes)[0]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def make_png(color=(200, 30, 30), size=(64, 48)):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()
