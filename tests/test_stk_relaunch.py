import json
import re
import subprocess
from pathlib import Path

SERVICE = Path(__file__).parents[1] / "src/alayarenderer_flash/demo/scripts/alaya_stk_checkpoint_service.sh"


def _relaunch(tmp_path, live_control):
    """Run the STK supervisor's relaunch helpers against a live_control.json."""
    text = SERVICE.read_text()
    helpers = text[text.index("active_track() {") : text.index("launch_owned_child() {")]
    path = tmp_path / "live_control.json"
    if live_control is not None:
        path.write_text(json.dumps(live_control, separators=(",", ":")))
    script = (
        f"live_control_path={path}\n"
        "command=(env TRACK=black_forest SOURCE_SESSION_ID=black_forest-rt SCREENSIZE=960x540 bash launcher.sh)\n"
        f"{helpers}\nrelaunch_command\nprintf '%s\\n' \"${{relaunch[@]}}\"\n"
    )
    return subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout.split()


def test_relaunch_follows_viewer_track(tmp_path):
    args = _relaunch(tmp_path, {"prompt": "night", "track_id": "zengarden", "source_session_id": "s"})
    assert "TRACK=zengarden" in args and "SCREENSIZE=960x540" in args
    assert any(re.fullmatch(r"SOURCE_SESSION_ID=zengarden-\d+", a) for a in args)


def test_relaunch_keeps_default_without_control(tmp_path):
    args = _relaunch(tmp_path, None)
    assert "TRACK=black_forest" in args and "SOURCE_SESSION_ID=black_forest-rt" in args
