import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from alayarenderer_flash.demo import preview
from alayarenderer_flash.demo.preview import (
    MAX_POST_BODY_BYTES,
    RequestBodyError,
    origin_allowed,
    parse_content_length,
    parse_json_object,
)


@pytest.mark.parametrize("raw", ["-1", "abc", str(MAX_POST_BODY_BYTES + 1)])
def test_content_length_rejected(raw):
    with pytest.raises(RequestBodyError):
        parse_content_length(raw)


def test_content_length_accepted():
    assert parse_content_length(None) == 0
    assert parse_content_length(str(MAX_POST_BODY_BYTES)) == MAX_POST_BODY_BYTES


@pytest.mark.parametrize("raw", [b"[1]", b'"x"', b"null", b"{bad", b"\xff"])
def test_json_body_must_be_object(raw):
    with pytest.raises(RequestBodyError):
        parse_json_object(raw)


def test_origin_policy():
    extra = frozenset({"https://demo.example.org"})
    assert origin_allowed(None, "127.0.0.1:8095", extra)
    assert origin_allowed("http://127.0.0.1:8095", "127.0.0.1:8095", extra)
    assert origin_allowed("https://demo.example.org/", "127.0.0.1:8095", extra)
    assert not origin_allowed("https://evil.example", "127.0.0.1:8095", extra)
    assert not origin_allowed("null", "127.0.0.1:8095", extra)


def _serve(tmp_path, monkeypatch, read_only=False, token=""):
    capture = tmp_path / "capture"
    capture.mkdir()
    input_state = capture / "input.json"
    input_state.write_text("{}")
    monkeypatch.setenv("STK_PROMPT_CONTROL_DIR", str(tmp_path / "control"))
    monkeypatch.setenv("ALAYA_PREVIEW_ADMIN_TOKEN", token)
    monkeypatch.delenv("ALAYA_PREVIEW_ADMIN_TOKEN_FILE", raising=False)
    monkeypatch.delenv("ALAYA_PREVIEW_ALLOWED_ORIGINS", raising=False)
    handler = preview.make_handler(capture, None, 720, 360, 72, 24.0, 6.0, input_state, read_only)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(server, path, body=b"{}", headers=None):
    conn = http.client.HTTPConnection(*server.server_address, timeout=5)
    conn.putrequest("POST", path)
    for key, value in {"Content-Length": str(len(body)), **(headers or {})}.items():
        conn.putheader(key, value)
    conn.endheaders()
    if body and int((headers or {}).get("Content-Length", len(body))) >= 0:
        conn.send(body)
    resp = conn.getresponse()
    status, payload = resp.status, json.loads(resp.read() or b"{}")
    conn.close()
    return status, payload


def test_post_bodies_are_bounded(tmp_path, monkeypatch):
    server = _serve(tmp_path, monkeypatch)
    try:
        for path in ("/api/track", "/api/prompt", "/input"):
            assert _post(server, path, b"", {"Content-Length": "-1"})[0] == 400
            assert _post(server, path, b"", {"Content-Length": str(MAX_POST_BODY_BYTES + 1)})[0] == 413
            assert _post(server, path, b"[1]")[0] == 400
        assert _post(server, "/api/prompt", b'{"prompt":"x"}', {"Origin": "https://evil.example"})[0] == 403
    finally:
        server.shutdown()


def test_recording_requires_admin_token(tmp_path, monkeypatch):
    server = _serve(tmp_path, monkeypatch)
    try:
        assert _post(server, "/record/start") == (403, {"ok": False, "error": "admin_disabled"})
        assert _post(server, "/record/stop")[0] == 403
    finally:
        server.shutdown()
    (tmp_path / "with-token").mkdir()
    server = _serve(tmp_path / "with-token", monkeypatch, token="s3cret")
    try:
        assert _post(server, "/record/start")[1]["error"] == "admin_token_required"
        assert _post(server, "/record/start", headers={"X-Alaya-Admin-Token": "wrong"})[0] == 403
    finally:
        server.shutdown()


def test_read_only_observer_rejects_prompt(tmp_path, monkeypatch):
    server = _serve(tmp_path, monkeypatch, read_only=True)
    try:
        status, _ = _post(server, "/api/prompt", b'{"prompt":"watercolor"}')
        assert status == 403
        assert not (tmp_path / "control").exists()
    finally:
        server.shutdown()
