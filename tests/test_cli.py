import pytest

from lapscale.cli import main


def test_target_and_show(tmp_path, capsys):
    out = tmp_path / "board.png"
    assert main(["calib", "target", "--target", "charuco", "--cols", "5", "--rows", "4",
                 "--square-mm", "5", "--marker-mm", "3.7", "--out", str(out)]) == 0
    assert out.exists()


def test_missing_target_params():
    with pytest.raises(SystemExit):
        main(["calib", "target", "--target", "charuco", "--out", "x.png"])


def test_config_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("calib:\n  target:\n    target: checkerboard\n    cols: 6\n    rows: 4\n    square-mm: 5\n")
    out = tmp_path / "b.png"
    assert main(["--config", str(cfg), "calib", "target", "--out", str(out)]) == 0
    assert out.exists()


def test_unimplemented_stage():
    assert main(["scale"]) == 2
