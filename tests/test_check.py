from lapscale.check import run_checks
from lapscale.cli import main


def test_check_reports_environment_without_backends():
    res = run_checks(log=lambda *_: None)
    names = [r[0] for r in res]
    assert names[:2] == ["python", "opencv"] and "torch" in names


def test_check_reports_missing_backend_instead_of_crashing(monkeypatch):
    import lapscale.depth.moge as moge_mod

    class Broken:                      # independent of whether MoGe is installed in this environment
        def __init__(self, *a, **k):
            raise ImportError("MoGe not available")

    monkeypatch.setattr(moge_mod, "MoGeBackend", Broken)
    res = run_checks(moge=["v2"], log=lambda *_: None)
    row = [r for r in res if r[0] == "depth moge v2"][0]
    assert row[1] is False and "Error" in row[2]
    assert main(["check", "--moge", "v2"]) == 1
