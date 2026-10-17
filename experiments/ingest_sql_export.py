"""Turn the packaged SQL export (docs/fashion_official_1_export.sql, run once in Supabase) into q*.csv files.

    python -m experiments.ingest_sql_export <export.csv|export.json> <out_dir> [--source supabase-sql-editor]

Accepts the SQL editor's CSV download (columns query_id, rows) or "copy as JSON" / the JSON download.
Writes <out_dir>/q1.csv ... plus PROVENANCE.md (sha256 of the export, the executed_at / server_version / SQL
file hash reported by the database itself, row counts). Refuses to run without the ``meta`` row, because
without it nothing shows that the numbers came from a SQL execution.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)


def load(path: Path) -> dict[str, list[dict]]:
    raw = path.read_text()
    if raw.lstrip().startswith(("[", "{")):
        data = json.loads(raw)
        rows = data if isinstance(data, list) else data.get("rows", data)
    else:
        rows = list(csv.DictReader(raw.splitlines(keepends=True)))
    out = {}
    for r in rows:
        v = r["rows"]
        out[r["query_id"]] = json.loads(v) if isinstance(v, str) else v
    return out


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def main(export: str, out_dir: str, source: str = "unspecified") -> None:
    src = Path(export)
    res = load(src)
    if "meta" not in res:
        raise SystemExit("export has no 'meta' row: not produced by the packaged SQL (or truncated)")
    meta = res["meta"][0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for q, rows in sorted(res.items()):
        if q == "meta":
            continue
        cols = list(rows[0]) if rows else []
        with open(out / f"{q}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(cols)
            w.writerows([fmt(r.get(c)) for c in cols] for r in rows)
        counts[q] = len(rows)
    prov = [
        "# SQL provenance", "",
        f"- source: **{source}**",
        f"- export file: `{src.name}` sha256 `{hashlib.sha256(src.read_bytes()).hexdigest()}`",
        f"- executed_at (reported by the database): {meta.get('executed_at')}",
        f"- server_version: {meta.get('server_version')}", f"- database: {meta.get('database')}",
        f"- experiment_group: {meta.get('experiment_group')}",
        f"- queries file: {meta.get('queries_file')} sha256 `{meta.get('queries_file_sha256')}`",
        "", "| query | rows |", "|---|---|", *[f"| {q} | {n} |" for q, n in counts.items()], "",
    ]
    (out / "PROVENANCE.md").write_text("\n".join(prov))
    print(f"wrote {len(counts)} csv files to {out}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:]]
    src = a[a.index("--source") + 1] if "--source" in a else "unspecified"
    main(a[0], a[1], src)
