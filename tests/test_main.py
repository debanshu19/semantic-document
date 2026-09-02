"""Route-level tests for the save/finalize/open-with-native-dialog flow.

The native OS dialogs (app.native_dialog) are monkeypatched here rather
than exercised for real -- there's no GUI in CI, and unit-testing "did
tkinter pop a window" isn't the point anyway. What matters, and what
these tests actually check, is that app.main wires the dialog's result
(or its absence, i.e. Cancel) into the right sdoc.py calls and redirects.
"""
from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app import main, native_dialog, sdoc


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


def test_finalize_saves_to_native_dialog_chosen_path(client, tmp_path, monkeypatch):
    client.post("/save", data={"title": "Foo Doc", "content": "the quick brown fox"})
    chosen = tmp_path / "wherever" / "custom-name.sdoc"
    monkeypatch.setattr(native_dialog, "ask_save_path", lambda default_name: chosen)

    resp = client.post("/finalize", data={"doc": "foo-doc"}, follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == f"/?path={quote(str(chosen))}"
    assert chosen.exists()
    assert not sdoc.draft_path(main.LIBRARY_DIR, "foo-doc").exists()


def test_finalize_cancel_leaves_draft_untouched(client, monkeypatch):
    client.post("/save", data={"title": "Cancel Doc", "content": "content"})
    monkeypatch.setattr(native_dialog, "ask_save_path", lambda default_name: None)
    monkeypatch.setattr(native_dialog, "available", lambda: True)

    resp = client.post("/finalize", data={"doc": "cancel-doc"}, follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/?doc=cancel-doc"
    assert sdoc.draft_path(main.LIBRARY_DIR, "cancel-doc").exists()
    assert not sdoc.final_path(main.LIBRARY_DIR, "cancel-doc").exists()


def test_finalize_falls_back_to_library_when_no_dialog_available(client, monkeypatch):
    """Headless server, no Tk -- shouldn't brick finalize entirely."""
    client.post("/save", data={"title": "Headless Doc", "content": "content"})
    monkeypatch.setattr(native_dialog, "ask_save_path", lambda default_name: None)
    monkeypatch.setattr(native_dialog, "available", lambda: False)

    resp = client.post("/finalize", data={"doc": "headless-doc"}, follow_redirects=False)

    assert resp.status_code == 303
    assert sdoc.final_path(main.LIBRARY_DIR, "headless-doc").exists()


def test_browse_open_redirects_to_chosen_path(client, monkeypatch, tmp_path):
    chosen = tmp_path / "somewhere.sdoc"
    client.post("/save", data={"title": "Browsable", "content": "browsable content"})
    monkeypatch.setattr(native_dialog, "ask_save_path", lambda default_name: chosen)
    client.post("/finalize", data={"doc": "browsable"})

    monkeypatch.setattr(native_dialog, "ask_open_path", lambda: chosen)
    resp = client.get("/browse/open", follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == f"/?path={quote(str(chosen))}"


def test_browse_open_cancel_redirects_home(client, monkeypatch):
    monkeypatch.setattr(native_dialog, "ask_open_path", lambda: None)
    resp = client.get("/browse/open", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
