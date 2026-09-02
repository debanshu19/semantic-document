"""Route-level tests for the browser-native open (file upload) and save
(download) flows.

No native OS dialogs here at all -- opening uses a plain multipart file
upload (exactly what `<input type=file>` sends), and saving is just
serving a file for the browser's own download manager to handle. Both
are tested the same way any other FastAPI file upload/download endpoint
would be: TestClient with `files=`, no GUI involved.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main, sdoc


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LIBRARY_DIR", tmp_path / "library")
    return TestClient(main.app)


def test_index_shows_new_draft_form(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "New document" in resp.text


def test_save_creates_draft_and_redirects(client):
    resp = client.post("/save", data={"title": "Hello World", "content": "some text"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/?doc=hello-world"
    assert sdoc.draft_path(main.LIBRARY_DIR, "hello-world").exists()


def test_finalize_writes_into_library_and_redirects(client):
    client.post("/save", data={"title": "Foo Doc", "content": "the quick brown fox"})
    resp = client.post("/finalize", data={"doc": "foo-doc"}, follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/?doc=foo-doc"
    assert sdoc.final_path(main.LIBRARY_DIR, "foo-doc").exists()
    assert not sdoc.draft_path(main.LIBRARY_DIR, "foo-doc").exists()


def test_download_serves_the_finalized_file(client, tmp_path):
    client.post("/save", data={"title": "Downloadable", "content": "grab this file"})
    client.post("/finalize", data={"doc": "downloadable"})

    resp = client.get("/documents/downloadable/download")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].endswith('filename="downloadable.sdoc"')
    on_disk = sdoc.final_path(main.LIBRARY_DIR, "downloadable").read_bytes()
    assert resp.content == on_disk


def test_download_missing_document_404s(client):
    resp = client.get("/documents/does-not-exist/download")
    assert resp.status_code == 404


def test_open_upload_adds_valid_sdoc_to_library(client, tmp_path):
    # Build a real finalized .sdoc somewhere outside the library to upload.
    source_dir = tmp_path / "source"
    sdoc.create_draft(source_dir, "myrecipe", title="My Recipe", content="mix flour, sugar, and eggs")
    built = sdoc.finalize_draft(source_dir, "myrecipe")

    resp = client.post(
        "/open-upload",
        files={"file": ("myrecipe.sdoc", built.read_bytes(), "application/octet-stream")},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/?doc=myrecipe"
    assert sdoc.final_path(main.LIBRARY_DIR, "myrecipe").exists()


def test_open_upload_rejects_garbage_file(client):
    resp = client.post(
        "/open-upload",
        files={"file": ("not-real.sdoc", b"this is not a sqlite database", "application/octet-stream")},
    )
    assert resp.status_code == 200
    assert "not a valid .sdoc file" in resp.text
    # nothing should have been added to the library
    assert list(main.LIBRARY_DIR.glob("*.sdoc")) == [] if main.LIBRARY_DIR.exists() else True


def test_open_upload_dedupes_name_collisions(client, tmp_path):
    source_dir = tmp_path / "source"
    sdoc.create_draft(source_dir, "dup", title="Dup", content="first version")
    first = sdoc.finalize_draft(source_dir, "dup")

    client.post("/open-upload", files={"file": ("dup.sdoc", first.read_bytes(), "application/octet-stream")})
    resp = client.post("/open-upload", files={"file": ("dup.sdoc", first.read_bytes(), "application/octet-stream")}, follow_redirects=False)

    assert resp.headers["location"] == "/?doc=dup-2"
