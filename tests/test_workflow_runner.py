"""Static check: a fork-headed pull_request can never resolve to the fleet runner."""
import glob
import re
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"
FLEET = "fleet-k3s"


def _resolve(expr, event, head_repo, repo="o/r", ci_runner=FLEET):
    """Evaluate the runs-on expression by translating it to Python."""
    m = re.fullmatch(r"\$\{\{(.*)\}\}", expr.strip())
    assert m, expr
    py = m.group(1)
    py = py.replace("github.event_name", "EV").replace(
        "github.event.pull_request.head.repo.full_name", "HEAD"
    ).replace("github.repository", "REPO").replace("vars.CI_RUNNER", "CI")
    py = py.replace("&&", " and ").replace("||", " or ").replace("!=", "!=")
    return eval(py, {}, {"EV": event, "HEAD": head_repo, "REPO": repo, "CI": ci_runner})


def _wf(name):
    return yaml.safe_load((WF / name).read_text())


def test_fork_pr_never_fleet_runner():
    wf = _wf("test_bot.yaml")
    assert "pull_request" in (wf.get(True) or wf.get("on"))
    for job in wf["jobs"].values():
        expr = job["runs-on"]
        assert _resolve(expr, "pull_request", "fork/r") == "ubuntu-latest"
        assert _resolve(expr, "pull_request", "o/r") == FLEET
        assert _resolve(expr, "push", "") == FLEET


def test_no_pull_request_target_and_no_secrets_in_test_bot():
    for f in glob.glob(str(WF / "*.y*ml")):
        assert "pull_request_target" not in Path(f).read_text(), f
    assert "secrets." not in (WF / "test_bot.yaml").read_text()
