"""The committed Fashion-MNIST protocol and served-model configuration are consistent and were fixed in advance."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from experiments.configs import VARIANTS, Protocol, build_plan, check_matched
from experiments.run_experiments import build_parser

ROOT = Path(__file__).parent.parent
PROTO = json.loads((ROOT / "config" / "fashion_protocol.json").read_text())
SERVED = json.loads((ROOT / "config" / "served_model.json").read_text())


def _protocol(dataset_id: int = 1) -> Protocol:
    p = PROTO["protocol"]
    return Protocol(dataset_id=dataset_id, epochs=p["epochs"], batch_size=p["batch_size"], lr=p["lr"],
                    split_seed=p["split_seed"], val_size=p["val_size"], train_limit=p["train_limit"],
                    dropout=p["dropout"], augment_max_shift=p["augment_max_shift"], scheduler=p["scheduler"])


def test_protocol_builds_a_matched_plan_for_every_variant_and_seed():
    plan = build_plan(_protocol(), PROTO["experiment_group"], PROTO["seeds"])
    check_matched(plan)
    assert len(plan) == len(VARIANTS) * len(PROTO["seeds"])
    assert {p["run_name"].rsplit("-seed", 1)[0] for p in plan} == set(VARIANTS)


def test_protocol_uses_the_full_dataset_unless_a_subset_is_justified():
    assert PROTO["protocol"]["train_limit"] is None
    assert PROTO["protocol"]["val_size"] == 0.1


def test_protocol_is_marked_as_fixed_before_official_results():
    assert PROTO["status"] in {"proposed", "frozen"}
    assert PROTO["chosen_before_official_results"] is True
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", PROTO["chosen_on"])
    assert PROTO["dataset"] == "fashion-mnist"


def test_served_model_is_one_of_the_planned_runs_and_not_selected_by_test_score():
    names = {p["run_name"] for p in build_plan(_protocol(), PROTO["experiment_group"], PROTO["seeds"])}
    assert SERVED["run_name"] in names
    assert SERVED["experiment_group"] == PROTO["experiment_group"]
    assert PROTO["served_model"]["run_name"] == SERVED["run_name"]
    assert SERVED["chosen_before_official_results"] is True
    assert SERVED["served_run_id"] is None or isinstance(SERVED["served_run_id"], int)


def test_the_documented_command_is_accepted_by_the_runner_parser():
    cmd = PROTO["command_template"].replace("<DATASET_ID>", "1")
    args = build_parser().parse_args(cmd.split("python -m experiments.run_experiments ")[1].split())
    assert args.group == PROTO["experiment_group"] and args.epochs == PROTO["protocol"]["epochs"]
    assert [int(s) for s in args.seeds.split(",")] == PROTO["seeds"]
    assert args.lr == PROTO["protocol"]["lr"] and args.batch_size == PROTO["protocol"]["batch_size"]
    p = PROTO["protocol"]
    assert (args.dropout, args.max_shift, args.scheduler, args.split_seed, args.val_size) == (
        p["dropout"], p["augment_max_shift"], p["scheduler"], p["split_seed"], p["val_size"])


def test_protocol_records_the_pilot_that_justified_the_budget_without_test_data():
    pilot = PROTO["pilot"]
    assert pilot["used_test_data"] is False and pilot["measured_seconds_per_epoch_mean"] > 0
    assert (ROOT / pilot["summary"]).is_file()


def test_dataset_download_exists_only_in_local_tools():
    """The API (Render) and UI (Streamlit) must never download the dataset."""
    for folder in ("api", "ui"):
        for f in (ROOT / folder).rglob("*.py"):
            text = f.read_text()
            assert "fashion.download" not in text and "urlopen" not in text and "urllib.request" not in text, f
            assert "fetch_fashion_mnist" not in text or "tools/fetch_fashion_mnist" in text or "python tools/" in text, f
    users = [f.relative_to(ROOT) for f in ROOT.rglob("*.py")
             if ".venv" not in f.parts and "download(" in f.read_text() and "fashion" in f.read_text()
             and f.name not in {"fashion.py", "test_protocol_config.py"}]
    assert {u.parts[0] for u in users} <= {"tools", "tests"}, users


def test_a_frozen_fingerprint_if_recorded_matches_the_plan_it_names():
    fp = PROTO["frozen_fingerprint"]
    if fp is None:
        pytest.skip("protocol not frozen yet: record the dry-run fingerprint in config/fashion_protocol.json")
    from experiments.configs import fingerprint

    assert PROTO["status"] == "frozen"
    ds = PROTO.get("frozen_dataset_id")
    assert isinstance(ds, int)
    assert fingerprint(build_plan(_protocol(ds), PROTO["experiment_group"], PROTO["seeds"])) == fp
