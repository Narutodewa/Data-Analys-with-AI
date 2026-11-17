import io

from tests.conftest import make_png


def up(client, name, data):
    r = client.post("/api/upload", files=[("files", (name, io.BytesIO(data)))])
    assert r.status_code == 200, r.text
    return r.json()


def test_full_flow(client, sample_bytes):
    j = up(client, "s.csv", sample_bytes)
    assert not j["errors"]
    ds = j["datasets"][0]
    i = ds["id"]

    rows = client.get(f"/api/datasets/{i}/rows", params={"limit": 5, "sort": "pendapatan", "desc": True}).json()
    assert rows["total"] == 304 and len(rows["rows"]) == 5

    iss = client.get(f"/api/datasets/{i}/issues").json()
    assert any(x["id"] == "duplicates" for x in iss["issues"])

    r = client.post(f"/api/datasets/{i}/clean", json={"issue_id": "missing:kepuasan", "action": "fill_mean"})
    assert r.status_code == 200
    r = client.post(f"/api/datasets/{i}/clean", json={"issue_id": "missing:kepuasan", "action": "fill_mean"})
    assert r.status_code == 404  # sudah bersih

    assert client.put(f"/api/datasets/{i}/cell", json={"row": 0, "col": "jumlah", "value": "7"}).status_code == 200
    assert client.put(f"/api/datasets/{i}/cell", json={"row": 0, "col": "jumlah", "value": "abc"}).status_code == 400

    assert client.post(f"/api/datasets/{i}/undo").status_code == 200
    assert client.get(f"/api/datasets/{i}/eda").json()["overview"]["rows"] == 304
    assert client.post(f"/api/datasets/{i}/chart", json={"type": "bar", "x": "wilayah", "y": "pendapatan", "agg": "sum"}).status_code == 200
    assert client.post(f"/api/datasets/{i}/chart", json={"type": "bar", "x": "zzz"}).status_code == 400
    assert client.post(f"/api/datasets/{i}/recommend", json={"goal": "tren pendapatan"}).json()["charts"]
    assert client.post(f"/api/datasets/{i}/pivot", json={"index": ["wilayah"], "values": "jumlah", "agg": "sum"}).json()["total"] >= 4
    assert client.post(f"/api/datasets/{i}/columns", json={"name": "tot", "expr": "jumlah * 2"}).status_code == 200
    a = client.post(f"/api/datasets/{i}/ask", json={"question": "apa temuan utama?"}).json()
    assert a["answer"]

    for fmt, magic in (("csv", b"tanggal"), ("json", b"["), ("xlsx", b"PK")):
        r = client.get(f"/api/datasets/{i}/export", params={"fmt": fmt})
        assert r.status_code == 200 and magic in r.content[:200]
    assert client.get(f"/api/datasets/{i}/export", params={"fmt": "exe"}).status_code == 422

    assert client.post(f"/api/datasets/{i}/reset").status_code == 200
    assert client.delete(f"/api/datasets/{i}").status_code == 200
    assert client.get(f"/api/datasets/{i}").status_code == 404


def test_upload_error_is_friendly(client):
    j = up(client, "bad.exe", b"zzz")
    assert j["datasets"] == [] and "belum didukung" in j["errors"][0]["error"]


def test_multiple_images_merge_and_thumb(client):
    r = client.post("/api/upload", files=[("files", ("a.png", io.BytesIO(make_png()))), ("files", ("b.png", io.BytesIO(make_png((0, 255, 0)))))])
    ds = r.json()["datasets"]
    assert len(ds) == 1 and ds[0]["kind"] == "image" and ds[0]["rows"] == 2
    t = client.get(f"/api/datasets/{ds[0]['id']}/thumb", params={"name": "a.png"})
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"


def test_sample_and_paste_and_ai_config(client):
    assert client.post("/api/sample").json()["datasets"][0]["rows"] == 304
    assert client.post("/api/paste", json={"text": "a,b\n1,2\n3,4"}).json()["datasets"][0]["rows"] == 2
    r = client.post("/api/ai/config", json={"provider": "openai", "api_key": "secret", "model": "m"}).json()
    assert r["enabled"] and "secret" not in str(r) and r["model"] == "m"
    client.post("/api/ai/config", json={"provider": "anthropic"})


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
