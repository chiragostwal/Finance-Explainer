import argparse
import io
import os
import time
import urllib.request
import warnings
from datetime import datetime
import duckdb
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore")

# Configuration (can be moved to a config file or environment variables)
CONFIG = {
    "DB_PATH": "NSE-universe-raw.duckdb",
    "NSE_URL": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "YF_TIMEOUT": 10,
    "YF_SLEEP": 0.35,
    "MAX_RETRIES": 3,
}

def get_current_iso_week() -> str:
    """Returns ISO year and week string, e.g. '2026-W45'."""
    return datetime.now().strftime("%Y-W%V")

def init_db(conn: duckdb.DuckDBPyConnection):
    """Initializes DuckDB tables if they do not exist."""
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

def fetch_nse_master() -> pd.DataFrame:
    """Fetches and cleans the NSE Master List."""
    try:
        logger.info("Fetching NSE Master List...")
        req = urllib.request.Request(CONFIG["NSE_URL"], headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=CONFIG["YF_TIMEOUT"]) as response:
            data = response.read().decode('utf-8')
        master_df = pd.read_csv(io.StringIO(data))
        master_df.columns = master_df.columns.str.strip()
        eq_df = master_df[master_df['SERIES'] == 'EQ'].copy()
        eq_df['TICKER'] = eq_df['SYMBOL'].astype(str).str.strip() + ".NS"
        return eq_df
    except Exception as e:
        logger.error(f"Failed to fetch NSE Master List: {e}")
        raise

def fetch_yahoo_data(ticker: str, retries: int = CONFIG["MAX_RETRIES"]) -> dict:
    """Fetches Yahoo Finance data with retries."""
    for i in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info and ("longName" in info or "shortName" in info):
                return info
            else:
                logger.warning(f"Empty or invalid data for {ticker}, retrying...")
                time.sleep(CONFIG["YF_SLEEP"])
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}, retrying... ({i+1}/{retries})")
            time.sleep(CONFIG["YF_SLEEP"])
    logger.error(f"Failed to fetch data for {ticker} after {retries} attempts.")
    return {}

def refresh_universe_rankings(conn: duckdb.DuckDBPyConnection, force: bool = False):
    """Fetches and updates nse_universe_ranks table."""
    current_week = get_current_iso_week()
    existing_count = conn.execute(
        "SELECT COUNT(*) FROM nse_universe_ranks WHERE sync_week = ?", [current_week]
    ).fetchone()[0]
    if existing_count > 2000 and not force:
        logger.info(f"Universe rankings already synced for {current_week}. Skipping.")
        return

    logger.info("Fetching and cleaning NSE Master List...")
    eq_df = fetch_nse_master()

    logger.info(f"Sweeping live market caps for {len(eq_df)} stocks...")
    market_caps = {}
    for ticker in tqdm(eq_df['TICKER'].tolist(), desc="Fetching Market Caps"):
        try:
            t = yf.Ticker(ticker)
            mcap = t.fast_info.get('market_cap', 0)
            if not mcap or mcap == 0:
                mcap = t.info.get('marketCap', 0)
            market_caps[ticker] = float(mcap or 0.0)
            time.sleep(CONFIG["YF_SLEEP"])
        except Exception:
            market_caps[ticker] = 0.0

    eq_df['MARKET_CAP'] = eq_df['TICKER'].map(market_caps).fillna(0.0)
    eq_df['MARKET_CAP_CR'] = (eq_df['MARKET_CAP'] / 1e7).round(2)
    ranked_universe = (
        eq_df[eq_df['MARKET_CAP'] > 0]
        .sort_values(by='MARKET_CAP', ascending=False)
        .reset_index(drop=True)
    )
    ranked_universe['MARKET_CAP_RANK'] = ranked_universe.index + 1
    ranked_universe['SYNC_WEEK'] = current_week
    ranked_universe['UPDATED_AT'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("DELETE FROM nse_universe_ranks")
    conn.execute("""
        INSERT INTO nse_universe_ranks
        SELECT 
            TICKER, SYMBOL, "NAME OF COMPANY", "ISIN NUMBER", "FACE VALUE",
            MARKET_CAP, MARKET_CAP_CR, MARKET_CAP_RANK, SYNC_WEEK, UPDATED_AT
        FROM ranked_universe
    """)
    logger.info(f"Ranked and stored {len(ranked_universe)} active NSE equities for {current_week}.")

def classify_market_cap(rank: int, mcap_cr: float) -> str:
    """Classifies stocks by market cap using a custom rule set."""
    if rank <= 100:
        return "Large_Cap"
    elif rank <= 250:
        return "Mid_Cap"
    elif rank <= 500:
        return "Small_Cap"
    elif mcap_cr >= 1000:
        return "Micro_Cap"
    return "Nano_Cap"

def refresh_deep_fundamentals(conn: duckdb.DuckDBPyConnection, top_n: int = 750):
    """Fetches deep fundamentals for top ranked stocks."""
    current_week = get_current_iso_week()
    query = """
        SELECT r.ticker, r.company_name, r.market_cap, r.market_cap_cr, r.market_cap_rank
        FROM nse_universe_ranks r
        LEFT JOIN equities_master e ON r.ticker = e.ticker AND e.sync_week = ?
        WHERE e.ticker IS NULL
        ORDER BY r.market_cap_rank ASC
    """
    if top_n > 0:
        query = f"""
            SELECT r.ticker, r.company_name, r.market_cap, r.market_cap_cr, r.market_cap_rank
            FROM (SELECT * FROM nse_universe_ranks ORDER BY market_cap_rank ASC LIMIT {top_n}) r
            LEFT JOIN equities_master e ON r.ticker = e.ticker AND e.sync_week = ?
            WHERE e.ticker IS NULL
            ORDER BY r.market_cap_rank ASC
        """

    pending_df = conn.execute(query, [current_week]).df()
    if pending_df.empty:
        logger.info(f"All target stocks are already synced for {current_week}!")
        return

    logger.info(f"Fetching deep fundamentals for {len(pending_df)} remaining stocks...")
    failed = []
    for _, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="Fetching Deep Data"):
        ticker = row['ticker']
        rank = int(row['market_cap_rank'])
        try:
            info = fetch_yahoo_data(ticker)
            if not info:
                failed.append(ticker)
                continue

            mcap = float(info.get("marketCap") or row['market_cap'] or 0.0)
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
                    sync_week = EXCLUDED.sync_week,
                    updated_at = EXCLUDED.updated_at
            """, (
                ticker,
                info.get("longName") or info.get("shortName") or row['company_name'],
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
    logger.info(f"Weekly Sync Complete! {total_synced} stocks up-to-date for {current_week}. Failed: {len(failed)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly NSE DuckDB Refresh Pipeline")
    parser.add_argument("--top-n", type=int, default=750, help="Number of top ranked stocks for deep pull (0 = all)")
    parser.add_argument("--force-rank", action="store_true", help="Force re-running the market cap sweep")
    args = parser.parse_args()

    conn = duckdb.connect(CONFIG["DB_PATH"])
    init_db(conn)
    refresh_universe_rankings(conn, force=args.force_rank)
    refresh_deep_fundamentals(conn, top_n=args.top_n)
    conn.close()