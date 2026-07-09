import subprocess
import sys


def test_validate_runs():
    r = subprocess.run([sys.executable, "main.py", "validate", "--quick"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "72." in r.stdout
