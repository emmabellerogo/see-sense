"""Matched configs and the experiment runner, against mocked APIs only."""
from __future__ import annotations

import copy
import io
import json
import urllib.error

import pytest

from experiments import run_experiments as rx
from experiments.configs import (
    ABLATED_FIELD, MATCHED_FIELDS, VARIANTS, Protocol, build_plan, check_matched, fingerprint,
)

PROTO = Protocol(dataset_id=1, epochs=2, batch_size=32, lr=0.01, split_seed=3)


def _argv(*extra, seeds="0", group="g"):
    return ["--dataset-id", "1", "--group", group, "--seeds", seeds, "--epochs", "2",
            "--batch-size", "32", "--lr", "0.01", "--split-seed", "3", *extra]


class ApiViaTestClient:
    """Runner transport that calls the real FastAPI app (database mocked in conftest)."""

    def __init__(self, client):
        self.client, self.calls = client, []

    def request(self, method, path, payload=None):
        self.calls.append((method, path))
        r = self.client.request(method, path, json=payload)
        return r.status_code, r.json()


# ------------------------------------------------------------------ configs
def test_six_variants_with_unique_seeded_names():
    plan = build_plan(PROTO, "grp", [0, 1])
    assert len(plan) == 12
    names = [p["run_name"] for p in plan]
    assert len(set(names)) == 12
    assert {n.rsplit("-seed", 1)[0] for n in names} == set(VARIANTS)
    assert {p["run_name"] for p in plan if p["train_seed"] == 1} == {f"{v}-seed1" for v in VARIANTS}


def test_everything_is_matched_except_the_declared_technique():
    plan = {p["run_name"]: p for p in build_plan(PROTO, "grp", [4])}
    full = plan["full_improved-seed4"]
    for variant, field in ABLATED_FIELD.items():
        diff = {k for k in full if plan[f"{variant}-seed4"][k] != full[k]} - {"run_name"}
        assert diff == {field}, variant
    for p in plan.values():
        assert all(p[f] == full[f] for f in MATCHED_FIELDS)
    assert (full["model_variant"], full["use_batchnorm"], full["dropout"], full["augment"], full["scheduler"]) == (
        "improved", True, 0.3, True, "cosine")
    base = plan["baseline-seed4"]
    assert (base["model_variant"], base["use_batchnorm"], base["dropout"], base["augment"], base["scheduler"]) == (
        "baseline", False, 0.0, False, "none")
    # Ablations keep the other three techniques on.
    assert plan["no_dropout-seed4"]["use_batchnorm"] and plan["no_dropout-seed4"]["augment"]
    assert plan["no_scheduler-seed4"]["scheduler"] == "none" and plan["no_scheduler-seed4"]["dropout"] == 0.3


def test_tampering_with_a_matched_field_is_caught():
    plan = build_plan(PROTO, "grp", [0])
    bad = copy.deepcopy(plan)
    bad[2]["batch_size"] = 64
    with pytest.raises(ValueError, match="batch_size"):
        check_matched(bad)
    bad = copy.deepcopy(plan)
    bad[2]["lr"] = 0.5
    with pytest.raises(ValueError, match="lr"):
        check_matched(bad)
    bad = copy.deepcopy(plan)
    ablation = next(p for p in bad if p["run_name"].startswith("no_dropout"))
    ablation["augment"] = False  # a second, undeclared difference
    with pytest.raises(ValueError, match="no_dropout"):
        check_matched(bad)


@pytest.mark.parametrize("bad", [dict(dropout=0.0), dict(augment_max_shift=0), dict(scheduler="none")])
def test_protocol_that_would_make_an_ablation_a_noop_is_rejected(bad):
    with pytest.raises(ValueError):
        Protocol(dataset_id=1, epochs=2, batch_size=32, lr=0.01, split_seed=3, **bad)


def test_fingerprint_is_stable_and_sensitive():
    a, b = build_plan(PROTO, "g", [0]), build_plan(PROTO, "g", [0])
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(build_plan(Protocol(1, 3, 32, 0.01, 3), "g", [0]))
    assert fingerprint(a) != fingerprint(build_plan(PROTO, "g2", [0]))


# ------------------------------------------------- end to end (real API code, fake DB)
@pytest.fixture
def api(client):
    client.post("/datasets", json={"name": "d", "n_rows": 200, "noise": 0.1, "seed": 42})
    return ApiViaTestClient(client)


def test_runner_persists_all_variants_for_every_seed(api, client, tmp_path):
    out, lines = tmp_path / "runs.jsonl", []
    code = rx.main(_argv("--out", str(out), seeds="0,1"), api=api, log=lines.append)
    assert code == 0, lines
    rows = list(client._store["runs"].values())
    assert len(rows) == 12 and len(client._store["artifacts"]) == 12
    assert {r["run_name"] for r in rows} == {f"{v}-seed{s}" for v in VARIANTS for s in (0, 1)}
    for seed in (0, 1):
        group = [r for r in rows if r["train_seed"] == seed]
        assert len({(r["dataset_id"], r["split_sha256"], r["split_seed"], r["data_seed"]) for r in group}) == 1
        assert {(r["epochs"], r["batch_size"], r["lr"], r["optimizer"]) for r in group} == {(2, 32, 0.01, "adam")}
        assert all(r["experiment_group"] == "g" for r in group)
    base = next(r for r in rows if r["run_name"] == "baseline-seed0")
    assert base["use_batchnorm"] is False and base["dropout"] == 0.0 and base["param_count"] == 4388
    assert next(r for r in rows if r["run_name"] == "no_batchnorm-seed0")["use_batchnorm"] is False
    # Test metrics stay out of the console and the out-file by default.
    text = "\n".join(lines)
    assert "TEST accuracy" not in text and "best_val_loss" in text
    saved = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(saved) == 12 and all("accuracy" not in s and "saved_row" not in s for s in saved)
    assert all(s["status"] == "saved" for s in saved)


def test_show_test_flag_prints_test_metrics(api):
    lines = []
    assert rx.main(_argv("--variants", "baseline", "--show-test"), api=api, log=lines.append) == 0
    assert any("TEST accuracy" in l for l in lines)


def test_rerunning_the_same_names_skips_instead_of_duplicating(api, client):
    assert rx.main(_argv("--variants", "baseline,full_improved"), api=api, log=lambda m: None) == 0
    lines = []
    assert rx.main(_argv("--variants", "baseline,full_improved"), api=api, log=lines.append) == 0
    assert len(client._store["runs"]) == 2
    assert any("2 already existed" in l for l in lines)


# ------------------------------------------------------- scripted responses
class Scripted:
    """Returns canned responses; records requests."""

    def __init__(self, train=None, saved=None, health=(200, {"status": "ok"})):
        self.train, self.saved, self.health, self.calls = train, saved, health, []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if path == "/healthz":
            return self.health
        if method == "POST":
            return self.train(payload, len([c for c in self.calls if c[0] == "POST"]))
        return self.saved(path, self.calls)


def _good_row(payload, run_id=1, **over):
    n = payload["epochs"]
    base = "baseline" == payload["model_variant"]
    row = {
        "id": run_id, "dataset_id": payload["dataset_id"], "lr": payload["lr"],
        "batch_size": payload["batch_size"], "epochs": n, "accuracy": 0.9, "macro_f1": 0.9,
        "experiment_group": payload["experiment_group"], "run_name": payload["run_name"],
        "model_variant": payload["model_variant"], "use_batchnorm": False if base else payload["use_batchnorm"],
        "dropout": 0.0 if base else payload["dropout"], "augment": payload["augment"],
        "augment_max_shift": payload["augment_max_shift"], "scheduler": payload["scheduler"],
        "data_seed": 42, "split_seed": payload["split_seed"], "train_seed": payload["train_seed"],
        "split_sha256": "h" * 64, "n_train": 140, "n_val": 20, "n_test": 40,
        "config": {"n_rows": 200, "img_size": 28, "noise": 0.1, "test_size": 0.2, "val_size": 0.1},
        "confusion_matrix": [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]],
        "train_loss_history": [0.5] * n, "val_loss_history": [0.4] * n,
        "val_accuracy_history": [0.9] * n, "lr_history": [0.01] * n,
        "best_epoch": 0, "param_count": 4436, "train_seconds": 1.0,
    }
    row.update(over)
    return row


def _scripted(fail_at=None, per_run=None, **fixed):
    """Scripted API whose GET returns the row of the most recent POST.

    ``fixed`` overrides saved-row fields for every run; ``per_run(payload)`` returns
    overrides for one run.
    """
    store = {}

    def train(payload, n):
        if fail_at == n:
            return 500, {"detail": "boom"}
        store["last"] = _good_row(payload, run_id=n, **{**fixed, **(per_run(payload) if per_run else {})})
        return 200, {"run_id": n}

    return Scripted(train=train, saved=lambda path, calls: (200, store["last"]))


def test_first_failure_stops_the_run_unless_keep_going():
    plan = build_plan(PROTO, "g", [0])
    res = rx.run_plan(_scripted(fail_at=2), plan, log=lambda m: None)
    assert [r["status"] for r in res] == ["saved", "failed"]
    res = rx.run_plan(_scripted(fail_at=2), plan, keep_going=True, log=lambda m: None)
    assert [r["status"] for r in res] == ["saved", "failed", "saved", "saved", "saved", "saved"]
    assert "HTTP 500" in res[1]["error"]


def test_exit_code_is_nonzero_on_failure():
    assert rx.main(_argv(), api=_scripted(fail_at=1), log=lambda m: None) == 1


def test_saved_run_that_does_not_match_the_request_is_flagged():
    plan = build_plan(PROTO, "g", [0], ["no_dropout"])
    res = rx.run_plan(_scripted(dropout=0.3), plan, log=lambda m: None)  # API "ignored" dropout=0
    assert res[0]["status"] == "failed" and "dropout" in res[0]["error"]


def test_missing_history_or_confusion_matrix_is_flagged():
    plan = build_plan(PROTO, "g", [0], ["baseline"])
    res = rx.run_plan(_scripted(val_loss_history=[0.4]), plan, log=lambda m: None)
    assert "val_loss_history" in res[0]["error"]
    res = rx.run_plan(_scripted(confusion_matrix=None), plan, log=lambda m: None)
    assert "confusion matrix" in res[0]["error"]


def test_runs_that_did_not_share_a_split_are_reported():
    plan = build_plan(PROTO, "g", [0], ["baseline", "full_improved"])
    api = _scripted(per_run=lambda p: {"split_sha256": ("a" if p["model_variant"] == "baseline" else "b") * 64})
    res = rx.run_plan(api, plan, log=lambda m: None)
    assert [r["status"] for r in res] == ["saved", "saved"]
    assert len(rx.check_splits(res)) == 1
    # Different dataset settings with an identical hash are also not "the same split".
    api = _scripted(per_run=lambda p: {"config": {
        "n_rows": 200, "img_size": 28, "noise": 0.1 if p["model_variant"] == "baseline" else 0.4,
        "test_size": 0.2, "val_size": 0.1}})
    assert len(rx.check_splits(rx.run_plan(api, plan, log=lambda m: None))) == 1
    assert rx.check_splits(rx.run_plan(_scripted(), plan, log=lambda m: None)) == []


def test_dry_run_prints_the_plan_and_never_touches_the_network():
    class Boom:
        def request(self, *a, **k):
            raise AssertionError("network used in dry run")

    lines = []
    assert rx.main(_argv("--dry-run", seeds="0,1"), api=Boom(), log=lines.append) == 0
    assert "12 runs" in lines[0] and len([l for l in lines if l.startswith("{")]) == 12


def test_frozen_fingerprint_must_match():
    lines = []
    rx.main(_argv("--dry-run"), log=lines.append)
    fp = lines[0].rsplit(" ", 1)[1]
    assert rx.main(_argv("--dry-run", "--expect-fingerprint", fp), log=lambda m: None) == 0
    assert rx.main(_argv("--dry-run", "--expect-fingerprint", "0" * 64), log=lambda m: None) == 2
    changed = _argv("--dry-run", "--expect-fingerprint", fp)
    changed[changed.index("--epochs") + 1] = "9"
    assert rx.main(changed, log=lambda m: None) == 2


def test_unhealthy_or_unreachable_api_stops_before_training():
    api = Scripted(health=(200, {"status": "degraded"}))
    assert rx.main(_argv(), api=api, log=lambda m: None) == 1
    assert [c[1] for c in api.calls] == ["/healthz"]

    class Down:
        def request(self, *a, **k):
            raise rx.ApiError("nope")

    assert rx.main(_argv(), api=Down(), log=lambda m: None) == 1


def test_protocol_values_are_required_and_bad_ones_rejected():
    with pytest.raises(SystemExit):
        rx.build_parser().parse_args(["--dataset-id", "1", "--group", "g", "--seeds", "0"])
    assert rx.main(_argv("--dropout", "0"), log=lambda m: None) == 2
    assert rx.main(_argv(seeds="0,0"), log=lambda m: None) == 2


# --------------------------------------------------------------- HTTP client
def test_api_client_parses_json_errors_and_connection_failures(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 409, "Conflict", {}, io.BytesIO(b'{"detail": "dup"}'))

    monkeypatch.setattr(rx.urllib.request, "urlopen", fake_urlopen)
    assert rx.ApiClient("http://x/").request("POST", "/train", {"a": 1}) == (409, {"detail": "dup"})

    def refuse(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(rx.urllib.request, "urlopen", refuse)
    with pytest.raises(rx.ApiError):
        rx.ApiClient("http://x").request("GET", "/healthz")
