import subprocess
import sys

import pytest

from medical_harness import cli
from medical_harness.paths import ROOT


@pytest.mark.parametrize("content,code", [(b'{"source":1,"source":2}', "INVALID_JSON"),
                                       (b"x" * 131073, "PAYLOAD_TOO_LARGE")])
def test_cli_rejects_ambiguous_or_oversized_json_before_http(tmp_path, monkeypatch, capsys, content, code):
    source = tmp_path / "case.json"
    source.write_bytes(content)
    token = tmp_path / "test.token"
    token.write_text("public-test-token")
    monkeypatch.setattr(sys, "argv", ["medical-harness", "--token-file", str(token), "case-create", str(source)])
    def forbidden(*args):
        pytest.fail("invalid JSON must not reach HTTP")
    monkeypatch.setattr(cli, "api_call", forbidden)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    assert capsys.readouterr().err.strip() == code


def test_cli_server_rejects_outside_data_dir():
    outside = ROOT.parent / "audit-escaped-service-must-not-exist"
    assert not outside.exists()
    result = subprocess.run([sys.executable, "-m", "medical_harness.cli", "serve", "--data-dir", str(outside)],
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "inside implementation" in result.stderr
    assert not outside.exists()


@pytest.mark.parametrize("token", ["", "short", "x" * 32 + " internal-space", "非ASCII" * 10])
def test_cli_server_refuses_invalid_persisted_credential(tmp_path, token):
    (tmp_path / "operator.token").write_text(token)
    result = subprocess.run([sys.executable, "-m", "medical_harness.cli", "serve", "--data-dir", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "invalid operator token file" in result.stderr
    assert not (tmp_path / "harness.sqlite3").exists()


def test_cli_server_rejects_lock_symlink_escape(tmp_path):
    outside = ROOT.parent / "audit-escaped-lock-must-not-exist"
    assert not outside.exists()
    (tmp_path / "service.lock").symlink_to(outside)
    result = subprocess.run([sys.executable, "-m", "medical_harness.cli", "serve", "--data-dir", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "inside implementation" in result.stderr
    assert not outside.exists()
