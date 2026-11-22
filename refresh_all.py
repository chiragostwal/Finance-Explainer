import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import yaml

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

DB_CFG = next(db for db in CONFIG.get("databases", []) if db.get("name") == "nse_universe")
DB_PATH = BASE_DIR / DB_CFG.get("path", "NSE-universe-raw.duckdb")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("refresh_all")


def compute_unstructured_diff_report(conn: duckdb.DuckDBPyConnection) -> dict:
    """Computes a git-diff-style content hash over unstructured fields in company_intelligence."""
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    if "company_intelligence" not in tables:
        return {"total_rows": 0, "missing_llm": 0, "changed_or_new": 0}

    rows = conn.execute("""
        SELECT ticker, symbol,
               COALESCE(latest_transcript_url, '') AS t_url,
               COALESCE(latest_presentation_url, '') AS p_url,
               COALESCE(latest_xbrl_url, '') AS x_url,
               COALESCE(cleaned_presentation_text, '') AS p_txt,
               COALESCE(cleaned_transcript_text, '') AS t_txt,
               COALESCE(latest_news_text, '') AS n_txt,
               extracted_facts_json,
               audit_flags_json
        FROM company_intelligence
    """).fetchall()

    missing_llm = 0
    flagged_audit = 0
    fingerprints = {}

    for ticker, symbol, t_url, p_url, x_url, p_txt, t_txt, n_txt, facts_j, audit_j in rows:
        if not facts_j:
            missing_llm += 1
        if audit_j and "WARN" in str(audit_j):
            flagged_audit += 1
        payload = f"{t_url}|{p_url}|{x_url}|{len(p_txt)}|{hashlib.sha256(p_txt.encode('utf-8', errors='ignore')).hexdigest()[:12]}|{len(t_txt)}|{hashlib.sha256(t_txt.encode('utf-8', errors='ignore')).hexdigest()[:12]}|{hashlib.sha256(n_txt.encode('utf-8', errors='ignore')).hexdigest()[:12]}"
        fingerprints[ticker] = payload

    # Compare against prior snapshot file if present
    snap_path = BASE_DIR / ".unstructured_diff_snapshot.txt"
    prev_map = {}
    if snap_path.exists():
        for line in snap_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                prev_map[k] = v

    changed_or_new = sum(1 for k, v in fingerprints.items() if prev_map.get(k) != v)
    snap_path.write_text(
        "\n".join(f"{k}\t{v}" for k, v in sorted(fingerprints.items())),
        encoding="utf-8",
    )
    return {
        "total_rows": len(rows),
        "missing_llm": missing_llm,
        "flagged_audit": flagged_audit,
        "changed_or_new_since_last_run": changed_or_new,
    }


def run_cmd(args: list[str], label: str):
    logger.info(f"[{label}] Running: {' '.join(args)}")
    proc = subprocess.run([sys.executable, *args], cwd=str(BASE_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"[{label}] exited with code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Diff-aware full refresh for NSE 750 pipeline")
    parser.add_argument("--top-n", type=int, default=0, help="Top N stocks for intelligence extraction (0 = all)")
    parser.add_argument("--top-k", type=int, default=20, help="Top K display rows for quant_stats_engine")
    parser.add_argument("--skip-price-download", action="store_true", help="Skip yfinance price download in quant engine")
    parser.add_argument("--force-reclean", action="store_true", help="Force re-extracting filings even if unchanged")
    parser.add_argument("--symbols", nargs="*", default=["BAJAJ-AUTO", "BAJFINANCE"], help="Symbols to analyze in analyst_engine")
    args = parser.parse_args()

    # 1. Inspect unstructured diff before running extraction
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        diff_stats = compute_unstructured_diff_report(conn)
        logger.info(f"Unstructured Diff Summary: {diff_stats}")
    finally:
        conn.close()

    # 2. Refresh universe rankings & deep fundamentals
    run_cmd(["nse_refresh_pipeline.py", "--top-n", str(args.top_n)], "NSE Master & Fundamentals")

    # 3. Refresh corporate intelligence (skips unchanged URLs unless --force-reclean)
    intel_args = ["competitive_inteligence.py", "--top-n", str(args.top_n)]
    if args.force_reclean:
        intel_args.append("--force-reclean")
    run_cmd(intel_args, "Corporate Intelligence Extraction")

    # 4. Refresh prices, point-in-time tables & 70-col quant/DCF signals
    quant_args = ["quant_stats_engine.py", "--top-k", str(args.top_k)]
    if args.skip_price_download:
        quant_args.append("--skip-price-download")
    run_cmd(quant_args, "Quant & DCF Feature Store")

    # 5. Refresh commodities/FX & generate analyst verdicts
    for sym in args.symbols:
        run_cmd(["analyst_engine.py", "--symbol", sym], f"Analyst Verdict ({sym})")

    logger.info(f"Full refresh completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()