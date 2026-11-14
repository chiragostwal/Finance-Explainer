import argparse
import io
import logging
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

# ==========================================================
# 0. LOAD CONFIGURATION & LOGGING
# ==========================================================
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

log_cfg = CONFIG.get("logging", {})
logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO")),
    format=log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)

# Suppress Pandas 4.0 and yfinance internal warnings
warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore")


def get_current_iso_week() -> str:
    """Returns ISO year and week string, e.g. '2026-W45'."""
    return datetime.now().strftime("%Y-W%V")


def resolve_sync_week(db_config: dict) -> str:
    """Uses sync_week from config.yaml if explicitly set, otherwise dynamic ISO week."""
    cfg_week = db_config.get("sync_week")
    if cfg_week and str(cfg_week).strip().lower() != "auto":
        return str(cfg_week).strip()
    return get_current_iso_week()


# ==========================================================
# 1. DATABASE INITIALIZATION
# ==========================================================
def init_db(conn: duckdb.DuckDBPyConnection, db_name: str):
    """Initializes DuckDB tables if they do not exist."""
    logger.info(f"Initializing database schema for: {db_name}")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nse_universe_ranks (
            ticker VARCHAR PRIMARY KEY,
            symbol VARCHAR,
            company_name VARCHAR,
            isin VARCHAR,
            face_value DOUBLE,
            market_cap DOUBLE,
            market_cap_cr DOUBLE,
            market_cap_rank INTEGER,
            sync_week VARCHAR,
            updated_at VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS equities_master (
            ticker VARCHAR PRIMARY KEY,
            company_name VARCHAR,
            sector VARCHAR,
            industry VARCHAR,
            market_cap_category VARCHAR,
            market_cap_rank INTEGER,
            market_cap DOUBLE,
            market_cap_cr DOUBLE,
            pe_ratio DOUBLE,
            forward_pe DOUBLE,
            peg_ratio DOUBLE,
            pb_ratio DOUBLE,
            ps_ratio DOUBLE,
            ev_to_ebitda DOUBLE,
            roe DOUBLE,
            roa DOUBLE,
            gross_margin DOUBLE,
            operating_margin DOUBLE,
            ebitda_margin DOUBLE,
            net_profit_margin DOUBLE,
            debt_to_equity DOUBLE,
            current_ratio DOUBLE,
            quick_ratio DOUBLE,
            beta DOUBLE,
            free_cash_flow DOUBLE,
            operating_cash_flow DOUBLE,
            held_by_insiders DOUBLE,
            held_by_institutions DOUBLE,
            revenue_growth DOUBLE,
            earnings_growth DOUBLE,
            friday_close DOUBLE,
            dist_from_200dma DOUBLE,
            dist_from_50dma DOUBLE,
            fifty_two_week_high DOUBLE,
            fifty_two_week_low DOUBLE,
            deep_intelligence_json VARCHAR,
            sync_week VARCHAR,
            updated_at VARCHAR
        )
    """)


# ==========================================================
# 2. DATA FETCHERS & RANKING ENGINE
# ==========================================================
def fetch_nse_master(db_config: dict) -> pd.DataFrame:
    """Fetches and cleans the NSE Master Equity List."""
    try:
        logger.info(f"Fetching NSE Master List from {db_config['nse_url']}...")
        req = urllib.request.Request(
            db_config["nse_url"],
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read().decode("utf-8")

        master_df = pd.read_csv(io.StringIO(data))
        master_df.columns = master_df.columns.str.strip()

        eq_df = master_df[master_df["SERIES"] == db_config["nse_series"]].copy()
        eq_df["TICKER"] = eq_df["SYMBOL"].astype(str).str.strip() + ".NS"
        return eq_df
    except Exception as e:
        logger.error(f"Failed to fetch NSE Master List for {db_config['name']}: {e}")
        raise


def fetch_yahoo_data(ticker: str, retries: int = 3, sleep: float = 0.35) -> dict:
    """Fetches Yahoo Finance .info dictionary with rate-limit throttling and retries."""
    for attempt in range(retries):
        try:
            time.sleep(sleep)  # Always throttle before hitting Yahoo Finance
            t = yf.Ticker(ticker)
            info = t.info
            if info and ("longName" in info or "shortName" in info):
                return info
            logger.warning(f"Empty/delisted payload for {ticker}, retry {attempt + 1}/{retries}...")
        except Exception as e:
            logger.warning(f"Error fetching {ticker} ({attempt + 1}/{retries}): {e}")
            time.sleep(sleep * 2)

    logger.error(f"Failed to fetch data for {ticker} after {retries} attempts.")
    return {}


def refresh_universe_rankings(conn: duckdb.DuckDBPyConnection, db_config: dict, force: bool = False):
    """Sweeps live market caps and updates nse_universe_ranks table."""
    current_week = resolve_sync_week(db_config)
    existing_count = conn.execute(
        "SELECT COUNT(*) FROM nse_universe_ranks WHERE sync_week = ?", [current_week]
    ).fetchone()[0]

    if existing_count > 2000 and not force:
        logger.info(
            f"Universe rankings already synced for {db_config['name']} ({current_week}, {existing_count} stocks). Skipping."
        )
        return

    eq_df = fetch_nse_master(db_config)
    tickers = eq_df["TICKER"].tolist()
    sleep_interval = min(float(db_config.get("sleep_interval", 0.35)), 0.15)

    logger.info(f"Sweeping live market caps for {len(tickers)} stocks...")
    market_caps = {}

    for ticker in tqdm(tickers, desc="Step 1: Fetching Market Caps"):
        try:
            t = yf.Ticker(ticker)
            mcap = t.fast_info.get("market_cap", 0)
            if not mcap or mcap == 0:
                mcap = t.info.get("marketCap", 0)
            market_caps[ticker] = float(mcap or 0.0)
            time.sleep(sleep_interval)
        except Exception:
            market_caps[ticker] = 0.0

    eq_df["MARKET_CAP"] = eq_df["TICKER"].map(market_caps).fillna(0.0)
    eq_df["MARKET_CAP_CR"] = (eq_df["MARKET_CAP"] / 1e7).round(2)

    ranked_universe = (
        eq_df[eq_df["MARKET_CAP"] > 0]
        .sort_values(by="MARKET_CAP", ascending=False)
        .reset_index(drop=True)
    )
    ranked_universe["MARKET_CAP_RANK"] = ranked_universe.index + 1
    ranked_universe["SYNC_WEEK"] = current_week
    ranked_universe["UPDATED_AT"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("DELETE FROM nse_universe_ranks")
    conn.execute("""
        INSERT INTO nse_universe_ranks
        SELECT 
            TICKER, SYMBOL, "NAME OF COMPANY", "ISIN NUMBER", "FACE VALUE",
            MARKET_CAP, MARKET_CAP_CR, MARKET_CAP_RANK, SYNC_WEEK, UPDATED_AT
        FROM ranked_universe
    """)
    logger.info(
        f"Ranked and stored {len(ranked_universe)} active NSE equities in {db_config['path']} (Week {current_week})."
    )


def classify_market_cap(rank: int, mcap_cr: float) -> str:
    """Classifies stocks using SEBI rank rules + Micro-Cap tiering."""
    if rank <= 100:
        return "Large_Cap"
    elif rank <= 250:
        return "Mid_Cap"
    elif rank <= 500:
        return "Small_Cap"
    elif mcap_cr >= 1000:
        return "Micro_Cap"
    return "Nano_Cap"


# ==========================================================
# 3. DEEP FUNDAMENTALS SYNC
# ==========================================================
def refresh_deep_fundamentals(conn: duckdb.DuckDBPyConnection, db_config: dict, top_n: int):
    """Fetches deep fundamentals for top ranked stocks with weekly checkpointing."""
    current_week = resolve_sync_week(db_config)
    limit_sql = f"LIMIT {top_n}" if top_n > 0 else ""

    query = f"""
        SELECT r.ticker, r.company_name, r.market_cap, r.market_cap_cr, r.market_cap_rank
        FROM (
            SELECT * FROM nse_universe_ranks 
            ORDER BY market_cap_rank ASC 
            {limit_sql}
        ) r
        LEFT JOIN equities_master e ON r.ticker = e.ticker AND e.sync_week = ?
        WHERE e.ticker IS NULL
        ORDER BY r.market_cap_rank ASC
    """

    pending_df = conn.execute(query, [current_week]).df()
    if pending_df.empty:
        logger.info(f"All target stocks are already synced for {db_config['name']} ({current_week})!")
        return

    logger.info(f"Fetching deep fundamentals for {len(pending_df)} remaining stocks (Week {current_week})...")
    failed = []
    retries = int(db_config.get("max_retries", 3))
    sleep_sec = float(db_config.get("sleep_interval", 0.35))

    for _, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="Step 2: Deep Fundamentals"):
        ticker = row["ticker"]
        rank = int(row["market_cap_rank"])
        try:
            info = fetch_yahoo_data(ticker, retries=retries, sleep=sleep_sec)
            if not info:
                failed.append(ticker)
                continue

            mcap = float(info.get("marketCap") or row["market_cap"] or 0.0)
            mcap_cr = round(mcap / 1e7, 2)
            cap_cat = classify_market_cap(rank, mcap_cr)

            close = float(info.get("previousClose") or info.get("currentPrice") or 0.0)
            sma200 = float(info.get("twoHundredDayAverage") or close)
            sma50 = float(info.get("fiftyDayAverage") or close)
            dist_200 = ((close - sma200) / sma200) * 100 if sma200 else 0.0
            dist_50 = ((close - sma50) / sma50) * 100 if sma50 else 0.0
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            conn.execute("""
                INSERT INTO equities_master VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    NULL, ?, ?
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap_category = EXCLUDED.market_cap_category,
                    market_cap_rank = EXCLUDED.market_cap_rank,
                    market_cap = EXCLUDED.market_cap,
                    market_cap_cr = EXCLUDED.market_cap_cr,
                    pe_ratio = EXCLUDED.pe_ratio,
                    forward_pe = EXCLUDED.forward_pe,
                    peg_ratio = EXCLUDED.peg_ratio,
                    pb_ratio = EXCLUDED.pb_ratio,
                    ps_ratio = EXCLUDED.ps_ratio,
                    ev_to_ebitda = EXCLUDED.ev_to_ebitda,
                    roe = EXCLUDED.roe,
                    roa = EXCLUDED.roa,
                    gross_margin = EXCLUDED.gross_margin,
                    operating_margin = EXCLUDED.operating_margin,
                    ebitda_margin = EXCLUDED.ebitda_margin,
                    net_profit_margin = EXCLUDED.net_profit_margin,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    current_ratio = EXCLUDED.current_ratio,
                    quick_ratio = EXCLUDED.quick_ratio,
                    beta = EXCLUDED.beta,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    held_by_insiders = EXCLUDED.held_by_insiders,
                    held_by_institutions = EXCLUDED.held_by_institutions,
                    revenue_growth = EXCLUDED.revenue_growth,
                    earnings_growth = EXCLUDED.earnings_growth,
                    friday_close = EXCLUDED.friday_close,
                    dist_from_200dma = EXCLUDED.dist_from_200dma,
                    dist_from_50dma = EXCLUDED.dist_from_50dma,
                    fifty_two_week_high = EXCLUDED.fifty_two_week_high,
                    fifty_two_week_low = EXCLUDED.fifty_two_week_low,
                    deep_intelligence_json = COALESCE(EXCLUDED.deep_intelligence_json, equities_master.deep_intelligence_json),
                    sync_week = EXCLUDED.sync_week,
                    updated_at = EXCLUDED.updated_at
            """, (
                ticker,
                info.get("longName") or info.get("shortName") or row["company_name"],
                info.get("sector", "Unknown"),
                info.get("industry", "Unknown"),
                cap_cat,
                rank,
                mcap,
                mcap_cr,
                round(info.get("trailingPE") or 0.0, 2),
                round(info.get("forwardPE") or 0.0, 2),
                round(info.get("pegRatio") or 0.0, 2),
                round(info.get("priceToBook") or 0.0, 2),
                round(info.get("priceToSalesTrailing12Months") or 0.0, 2),
                round(info.get("enterpriseToEbitda") or 0.0, 2),
                round((info.get("returnOnEquity") or 0.0) * 100, 2),
                round((info.get("returnOnAssets") or 0.0) * 100, 2),
                round((info.get("grossMargins") or 0.0) * 100, 2),
                round((info.get("operatingMargins") or 0.0) * 100, 2),
                round((info.get("ebitdaMargins") or 0.0) * 100, 2),
                round((info.get("profitMargins") or 0.0) * 100, 2),
                round((info.get("debtToEquity") or 0.0) / 100, 2),
                round(info.get("currentRatio") or 0.0, 2),
                round(info.get("quickRatio") or 0.0, 2),
                round(info.get("beta") or 0.0, 2),
                float(info.get("freeCashflow") or 0.0),
                float(info.get("operatingCashflow") or 0.0),
                round((info.get("heldPercentInsiders") or 0.0) * 100, 2),
                round((info.get("heldPercentInstitutions") or 0.0) * 100, 2),
                round((info.get("revenueGrowth") or 0.0) * 100, 2),
                round((info.get("earningsGrowth") or 0.0) * 100, 2),
                round(close, 2),
                round(dist_200, 2),
                round(dist_50, 2),
                round(info.get("fiftyTwoWeekHigh") or 0.0, 2),
                round(info.get("fiftyTwoWeekLow") or 0.0, 2),
                current_week,
                now_str
            ))
        except Exception as e:
            logger.error(f"Failed to insert data for {ticker}: {e}")
            failed.append(ticker)
            time.sleep(1.0)

    total_synced = conn.execute(
        "SELECT COUNT(*) FROM equities_master WHERE sync_week = ?", [current_week]
    ).fetchone()[0]
    logger.info(
        f"Weekly Sync Complete! {total_synced} stocks up-to-date in '{db_config['path']}'. Failed: {len(failed)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly NSE Universe & Deep Fundamentals Sync Pipeline")
    parser.add_argument("--database", type=str, default="nse_universe", help="Database name from config.yaml")
    parser.add_argument("--top-n", type=int, default=None, help="Override top_n from config (0 = all stocks)")
    parser.add_argument("--force-rank", action="store_true", help="Force re-running the market cap sweep")
    args = parser.parse_args()

    db_config = next((db for db in CONFIG["databases"] if db["name"] == args.database), None)
    if not db_config:
        logger.error(f"Database '{args.database}' not found in config.yaml.")
        sys.exit(1)

    target_top_n = args.top_n if args.top_n is not None else int(db_config.get("top_n", 750))
    db_file_path = Path(__file__).parent / db_config["path"]

    conn = duckdb.connect(str(db_file_path))
    try:
        init_db(conn, db_config["name"])
        refresh_universe_rankings(conn, db_config, force=args.force_rank)
        refresh_deep_fundamentals(conn, db_config, top_n=target_top_n)
    finally:
        conn.close()