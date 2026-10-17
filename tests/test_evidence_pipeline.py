"""The export / packaging / ingestion tools around the official Fashion-MNIST records (no network, no Supabase)."""
from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path

import pytest

from experiments import export_records, ingest_sql_export, package_sql
from ui import evidence

ROOT = Path(__file__).resolve().parent.parent
PROTO = json.loads((ROOT / "config" / "fashion_protocol.json").read_text())
RECORDS = ROOT / "experiments" / "results" / "fashion-official-1-records" / "runs.json"


@pytest.mark.skipif(not RECORDS.is_file(), reason="official records not exported")
class TestVerifyOfficialRecords:
    data = json.loads(RECORDS.read_text()) if RECORDS.is_file() else {}

    def test_the_saved_official_records_pass_every_matched_protocol_check(self):
        assert export_records.verify(self.data["runs"], self.data["datasets"], PROTO) == []
        assert sorted(r["id"] for r in self.data["runs"]) == list(range(53, 71))

    @pytest.mark.parametrize("field,value", [("epochs", 19), ("batch_size", 64), ("split_sha256", "x"), ("dropout", 0.5)])
    def test_a_mismatched_run_is_reported(self, field, value):
        runs = copy.deepcopy(self.data["runs"])
        next(r for r in runs if r["run_name"] == "no_scheduler-seed1")[field] = value
        assert export_records.verify(runs, self.data["datasets"], PROTO)

    def test_a_missing_run_is_reported(self):
        runs = [r for r in self.data["runs"] if r["run_name"] != "baseline-seed2"]
        assert any("missing" in p for p in export_records.verify(runs, self.data["datasets"], PROTO))

    def test_the_served_run_is_one_of_the_verified_records(self):
        served = json.loads((ROOT / "config" / "served_model.json").read_text())
        run = next(r for r in self.data["runs"] if r["id"] == served["served_run_id"])
        assert (run["run_name"], run["experiment_group"]) == (served["run_name"], served["experiment_group"])
        assert run["dataset_id"] == PROTO["frozen_dataset_id"]
        assert run["config"]["class_names"] == list(evidence.FASHION_CLASSES)
        assert run["config"]["preprocessing"]["mean"] == 0.286 and run["config"]["preprocessing"]["std"] == 0.353


class TestPackagedSql:
    sql = package_sql.package("fashion-official-1")

    def test_it_is_read_only_and_covers_every_query(self):
        body = re.sub(r"--[^\n]*", "", self.sql).lower()
        assert not re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant)\b", body)
        assert "replace-with-group" not in body
        for n in (1, 2, 3, 4, 5, 6, 7, 9, 10):
            assert f"'q{n}' as query_id" in body.replace("'Q", "'q")
        assert "'meta'" in body and body.count("union all") >= 9

    def test_the_committed_file_matches_the_generator(self):
        assert (ROOT / "docs" / "fashion_official_1_export.sql").read_text() == self.sql

    def test_the_header_hash_matches_the_queries_file(self):
        import hashlib
        digest = hashlib.sha256(package_sql.SRC.read_bytes()).hexdigest()
        assert digest in self.sql


class TestIngest:
    def _export(self, tmp_path, as_csv):
        rows = [{"query_id": "q4", "rows": [{"run_name": "baseline-seed0", "flag": True, "x": None, "n": 1.5}]},
                {"query_id": "meta", "rows": [{"executed_at": "2026-10-17T00:00:00Z", "server_version": "PostgreSQL 15",
                                               "database": "postgres", "experiment_group": "g",
                                               "queries_file": "docs/comparison_queries.sql", "queries_file_sha256": "ab"}]}]
        p = tmp_path / ("e.csv" if as_csv else "e.json")
        if as_csv:
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query_id", "rows"])
                w.writerows([r["query_id"], json.dumps(r["rows"])] for r in rows)
        else:
            p.write_text(json.dumps(rows))
        return p

    @pytest.mark.parametrize("as_csv", [True, False])
    def test_csv_and_json_exports_give_the_same_files_and_provenance(self, tmp_path, as_csv):
        out = tmp_path / "o"
        ingest_sql_export.main(str(self._export(tmp_path, as_csv)), str(out), "supabase-sql-editor")
        assert list(csv.DictReader(open(out / "q4.csv"))) == [{"run_name": "baseline-seed0", "flag": "true", "x": "", "n": "1.5"}]
        prov = (out / "PROVENANCE.md").read_text()
        assert "supabase-sql-editor" in prov and "PostgreSQL 15" in prov

    def test_an_export_without_the_database_meta_row_is_refused(self, tmp_path):
        p = tmp_path / "e.json"
        p.write_text(json.dumps([{"query_id": "q4", "rows": []}]))
        with pytest.raises(SystemExit):
            ingest_sql_export.main(str(p), str(tmp_path / "o"))


class TestProvenanceLabel:
    def test_only_a_supabase_source_counts_as_supabase_sql(self, tmp_path):
        for text, expect in [("- source: **supabase-sql-editor**", True), ("- source: **LOCAL REPLICA, NOT Supabase SQL.**", False)]:
            (tmp_path / "PROVENANCE.md").write_text(text)
            assert evidence.load_provenance(tmp_path)["is_supabase_sql"] is expect
        assert evidence.load_provenance(tmp_path / "nope")["is_supabase_sql"] is False
