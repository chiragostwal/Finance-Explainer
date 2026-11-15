import argparse
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from normalizer import (
    extract_guidance_catalog,
    extract_news_catalog,
    extract_typed_facts_from_Sources,
)

# ==========================================================
# 0. CONFIGURATION & LOGGING
# ==========================================================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

log_cfg = config.get("logging", {})
logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO")),
    format=log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)

DB_CONFIG = next(db for db in config["databases"] if db["name"] == "nse_universe")
DB_PATH = BASE_DIR / DB_CONFIG["path"]


# ==========================================================
# 1. FULL 750-STOCK SYNC OF FACTS / EVENTS / GUIDANCE
# ==========================================================
def sync_point_in_time_tables_from_cache(conn: duckdb.DuckDBPyConnection):
    """Populates `facts`, `events`, and `guidance` across all 750 stocks by combining:
       Tier 1: normalizer.py (XBRL + Slides + Concall)
       Tier 2: ALL grounded segment_facts from extracted_facts_json
       Tier 3: equities_master baseline TTM facts + LLM management_guidance summary."""
    logger.info("Synchronizing `facts`, `events`, and `guidance` tables across all 750 stocks...")
    rows = conn.execute("""
        SELECT ci.ticker, ci.symbol, ci.xbrl_numbers_json,
               ci.cleaned_presentation_text, ci.cleaned_transcript_text,
               ci.latest_news_text, ci.extracted_facts_json, ci.management_guidance,
               em.revenue_growth, em.earnings_growth, em.operating_margin, em.roe, em.roa, em.market_cap_cr
        FROM company_intelligence ci
        LEFT JOIN equities_master em ON ci.ticker = em.ticker
    """).fetchall()

    as_of_date = datetime.now().strftime("%Y-%m-%d")
    facts_batch = []
    events_batch = []
    guidance_batch = []

    for (
        ticker, symbol, xbrl_str, pres_txt, trans_txt,
        news_txt, ext_json_str, mgmt_guid_str,
        rev_g, eps_g, opm, roe, roa, mcap_cr
    ) in rows:
        try:
            xbrl_dict = json.loads(xbrl_str) if xbrl_str else {}
        except Exception:
            xbrl_dict = {}

        pres_slice = (pres_txt or "")[:6000]
        trans_slice = (trans_txt or "")[:9000]
        news_slice = (news_txt or "").strip()

        typed_facts = extract_typed_facts_from_Sources(xbrl_dict, pres_slice, trans_slice)
        news_items = extract_news_catalog(news_slice)
        primary_period = typed_facts[0]["period"] if typed_facts else xbrl_dict.get("Period_End", "Q2 FY27")
        guidance_items = extract_guidance_catalog(pres_slice, trans_slice, doc_period=primary_period)

        seen_fact_vals = {f["value"][:30].lower() for f in typed_facts}

        # Tier 2: Always merge additional segment_facts from extracted_facts_json if typed_facts < 5
        if len(typed_facts) < 5 and ext_json_str:
            try:
                ext_data = json.loads(ext_json_str)
                for seg in ext_data.get("segment_facts", []):
                    s_name = str(seg.get("segment_name") or "Consolidated").strip()
                    perf = str(seg.get("financial_performance") or "").strip()
                    # Clean conversational artifacts like 'AUM, as you can see,:'
                    perf = re.sub(r",\s*as you can see,?\s*:", ":", perf, flags=re.I)
                    perf = re.sub(r"^\[F\d{3}\]\s*", "", perf)
                    if perf and re.search(r"\d", perf) and "not explicitly" not in perf.lower():
                        val_key = perf[:30].lower()
                        if val_key not in seen_fact_vals:
                            seen_fact_vals.add(val_key)
                            fid = f"F{len(typed_facts) + 1:03d}"
                            typed_facts.append({
                                "id": fid,
                                "period": primary_period,
                                "scope": s_name,
                                "metric": "Reported Segment / Metric",
                                "value": perf,
                                "change": str(seg.get("reported_driver") or "Reported")[:110],
                                "src": "filing_json"
                            })
            except Exception:
                pass

        # Tier 3: Ensure every stock has at least 3 baseline TTM facts so the Analyst packet is always complete
        if len(typed_facts) < 3:
            def fmt_pct(val) -> Optional[str]:
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return None
                v = float(val)
                if abs(v) < 1.5 and v != 0:
                    v *= 100.0
                return f"{v:+.1f}%"

            rg_str = fmt_pct(rev_g)
            eg_str = fmt_pct(eps_g)
            opm_str = fmt_pct(opm)
            roe_str = fmt_pct(roe)
            roa_str = fmt_pct(roa)

            if rg_str and eg_str:
                typed_facts.append({
                    "id": f"F{len(typed_facts) + 1:03d}",
                    "period": "TTM",
                    "scope": "Consolidated",
                    "metric": "TTM Revenue & Earnings Growth",
                    "value": f"Revenue Growth: {rg_str} YoY | Earnings Growth: {eg_str} YoY",
                    "change": f"{rg_str} YoY",
                    "src": "equities_master"
                })
            if roe_str or opm_str:
                typed_facts.append({
                    "id": f"F{len(typed_facts) + 1:03d}",
                    "period": "TTM",
                    "scope": "Consolidated",
                    "metric": "TTM Return & Margin Profile",
                    "value": f"ROE: {roe_str or 'N/A'} | ROA: {roa_str or 'N/A'} | OPM: {opm_str or 'N/A'}",
                    "change": f"Market Cap: ₹{float(mcap_cr or 0):,.0f} Cr",
                    "src": "equities_master"
                })

        # Merge LLM-extracted management_guidance summary if it adds information beyond guidance_items
        if mgmt_guid_str and "not explicitly stated" not in mgmt_guid_str.lower():
            clean_mg = re.sub(r"^\[G\d{2}\]\s*", "", mgmt_guid_str.strip())
            existing_g_text = " ".join(g["text"].lower() for g in guidance_items)
            if len(clean_mg) >= 25 and clean_mg[:35].lower() not in existing_g_text:
                guidance_items.insert(0, {
                    "id": "G01",
                    "period": primary_period,
                    "modality": "GUIDANCE",
                    "text": clean_mg[:300],
                    "src": "filing_summary"
                })
                # Re-number G01..G06 cleanly
                for idx_g, g_item in enumerate(guidance_items[:6], start=1):
                    g_item["id"] = f"G{idx_g:02d}"
                guidance_items = guidance_items[:6]

        for f in typed_facts:
            facts_batch.append((
                ticker, symbol, f["period"], f["scope"], f["metric"],
                f["value"], f["change"], f["src"], f["id"], as_of_date
            ))
        for n in news_items:
            events_batch.append((
                ticker, symbol, n["date"], n["fiscal_period"],
                n["event_type"], n["headline"], n["id"], as_of_date
            ))
        for g in guidance_items:
            guidance_batch.append((
                ticker, symbol, g["period"], g["modality"],
                g["text"], g["src"], g["id"], as_of_date
            ))

    conn.execute("DELETE FROM facts")
    conn.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", facts_batch)

    conn.execute("DELETE FROM events")
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", events_batch)

    conn.execute("DELETE FROM guidance")
    conn.executemany("INSERT INTO guidance VALUES (?, ?, ?, ?, ?, ?, ?, ?)", guidance_batch)

    conn.execute("CHECKPOINT")
    f_cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM facts").fetchone()[0]
    e_cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM events").fetchone()[0]
    g_cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM guidance").fetchone()[0]
    logger.info(
        f"Synced across 750 stocks -> facts: {len(facts_batch):,} rows ({f_cnt} stocks) | "
        f"events: {len(events_batch):,} rows ({e_cnt} stocks) | guidance: {len(guidance_batch):,} rows ({g_cnt} stocks)"
    )


# ==========================================================
# 2. DAILY OHLCV PRICE & QUANT SCHEMA (WITH DCF & CASH RUNWAY)
# ==========================================================
def init_prices_and_quant_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("DROP TABLE IF EXISTS stock_quant_signals")
    conn.execute("""
        CREATE TABLE stock_quant_signals (
            ticker VARCHAR PRIMARY KEY,
            symbol VARCHAR,
            company_name VARCHAR,
            sector VARCHAR,
            industry VARCHAR,
            market_cap_rank INTEGER,
            market_cap_cr DOUBLE,
            market_cap_category VARCHAR,
            current_price DOUBLE,

            -- Cross-Sectional Multiples & Profitability
            pe_ratio DOUBLE,
            forward_pe DOUBLE,
            pb_ratio DOUBLE,
            ev_to_ebitda DOUBLE,
            roe_pct DOUBLE,
            roa_pct DOUBLE,
            operating_margin_pct DOUBLE,
            net_profit_margin_pct DOUBLE,
            debt_to_equity DOUBLE,
            fcf_yield_pct DOUBLE,
            ocf_to_pat_ratio DOUBLE,
            revenue_growth_pct DOUBLE,
            earnings_growth_pct DOUBLE,
            pe_rerating_ratio DOUBLE,

            -- DCF, Reverse DCF, Cash Runway, Debt Payback & Seasonal Liquidity
            cost_of_equity_ke_pct DOUBLE,
            croic_pct DOUBLE,
            reinvestment_efficiency_pct DOUBLE,
            fcf_to_debt_payback_years DOUBLE,
            cash_conversion_quality_score DOUBLE,
            seasonal_wc_buffer_ratio DOUBLE,
            sector_seasonal_cycle_tag VARCHAR,
            dcf_intrinsic_value DOUBLE,
            dcf_margin_of_safety_pct DOUBLE,
            reverse_dcf_implied_growth_pct DOUBLE,
            dcf_growth_cushion_pct DOUBLE,
            cash_sustainability_percentile DOUBLE,
            cash_debt_regime_tag VARCHAR,

            -- Sector Medians & Z-Scores
            sector_pe_median DOUBLE,
            sector_pb_median DOUBLE,
            sector_roe_median DOUBLE,
            val_zscore_sector DOUBLE,
            roe_zscore_sector DOUBLE,
            margin_zscore_sector DOUBLE,
            growth_zscore_sector DOUBLE,

            -- Fundamental Factor Percentiles (0 to 100)
            value_percentile DOUBLE,
            quality_percentile DOUBLE,
            growth_percentile DOUBLE,
            ownership_percentile DOUBLE,
            fundamental_composite_score DOUBLE,

            -- Time-Series Technical & Regime Statistics
            ret_1m_pct DOUBLE,
            ret_3m_pct DOUBLE,
            ret_6m_pct DOUBLE,
            ret_12m_pct DOUBLE,
            mom_12m_ex_1m_pct DOUBLE,
            realized_vol_20d_ann_pct DOUBLE,
            realized_vol_252d_ann_pct DOUBLE,
            sharpe_momentum_6m DOUBLE,
            sharpe_momentum_12m DOUBLE,
            sortino_ratio_1y DOUBLE,
            beta_252d DOUBLE,
            alpha_252d_ann_pct DOUBLE,
            rsi_14 DOUBLE,
            bollinger_zscore_20d DOUBLE,
            atr_14_pct DOUBLE,
            vol_squeeze_percentile DOUBLE,
            up_down_vol_ratio_20d DOUBLE,
            hurst_exponent_126d DOUBLE,
            dist_from_50dma_pct DOUBLE,
            dist_from_200dma_pct DOUBLE,
            dist_from_52w_high_pct DOUBLE,
            max_drawdown_1y_pct DOUBLE,
            cvar_95_daily_pct DOUBLE,
            technical_composite_score DOUBLE,
            technical_regime VARCHAR,

            -- Filing & Catalyst Quant Bridge
            extracted_yoy_growth_median_pct DOUBLE,
            bullish_catalyst_count INTEGER,
            risk_catalyst_count INTEGER,
            has_numerical_guidance BOOLEAN,
            filing_catalyst_score DOUBLE,

            -- Final Conviction Score & Signal Tag
            quant_conviction_score DOUBLE,
            quant_signal_tag VARCHAR,
            as_of VARCHAR
        )
    """)


def sync_daily_prices_yfinance(conn: duckdb.DuckDBPyConnection, period: str = "1y", batch_size: int = 100):
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; skipping OHLCV download and using cached prices.")
        return

    tickers = [r[0] for r in conn.execute("SELECT ticker FROM equities_master ORDER BY market_cap_rank ASC").fetchall()]
    all_symbols = ["^NSEI"] + tickers

    existing_cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices WHERE date >= CURRENT_DATE - INTERVAL 7 DAY").fetchone()[0]
    if existing_cnt >= 700:
        logger.info(f"`prices` table already has fresh OHLCV for {existing_cnt} tickers. Skipping download.")
        return

    logger.info(f"Downloading {period} daily OHLCV for {len(all_symbols)} symbols in batches of {batch_size}...")
    for i in tqdm(range(0, len(all_symbols), batch_size), desc="OHLCV Batch Sync"):
        batch = all_symbols[i : i + batch_size]
        try:
            df = yf.download(batch, period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if df.empty:
                continue
            records = []
            for sym in batch:
                sub = df[sym] if len(batch) > 1 and sym in df.columns.get_level_values(0) else df
                sub = sub.dropna(subset=["Close"])
                for dt_idx, r in sub.iterrows():
                    dt_str = pd.Timestamp(dt_idx).strftime("%Y-%m-%d")
                    records.append((
                        sym,
                        dt_str,
                        float(r.get("Open", np.nan)),
                        float(r.get("High", np.nan)),
                        float(r.get("Low", np.nan)),
                        float(r.get("Close", np.nan)),
                        float(r.get("Volume", 0.0)),
                    ))
            if records:
                conn.executemany("INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)", records)
                conn.execute("CHECKPOINT")
        except Exception as e:
            logger.warning(f"Batch OHLCV warning at index {i}: {e}")


# ==========================================================
# 3. DCF, REVERSE DCF, CROIC & SEASONAL CASH BUFFER ENGINE
# ==========================================================
SECTOR_SEASONAL_REQUIREMENTS: List[Tuple[re.Pattern, float, float, str]] = [
    # Put BFSI & Capital Markets FIRST with strict word boundaries (\b) so "Credit Services" never matches "it services"!
    (re.compile(r"\b(exchange|asset management|broking|capital market|wealth|depository|rating|financial data)\b", re.I), 1.15, 1.00, "CAPITAL_MARKETS_ASSET_LIGHT_FEE_ENGINE"),
    (re.compile(r"\b(bank|banks|nbfc|insurance|credit services|mortgage|housing finance)\b", re.I), 1.00, 1.00, "BFSI_CASA_ALM_LIQUIDITY_FRAMEWORK"),
    (re.compile(r"\b(auto|automobile|ancillar|tractor|tyre|commercial vehicle|two-wheeler)\b", re.I), 1.35, 0.90, "AUTO_H1_CAPEX_M&A_Q3_Q4_FESTIVE_HARVEST"),
    (re.compile(r"\b(construction|infra|infrastructure|epc|capital goods|defence|aerospace|engineering|cables)\b", re.I), 1.50, 1.05, "EPC_H1_WORKING_CAPITAL_LOCK_Q4_GOVT_BILLING_RELEASE"),
    (re.compile(r"\b(fertilizer|agro|sugar|seed|pesticide|agricultural)\b", re.I), 1.45, 0.85, "AGRI_MONSOON_INVENTORY_BUILD_SUBSIDY_LAG"),
    (re.compile(r"\b(consumer durable|appliance|air condition|electronics|jewellery|retailing|apparel|footwear)\b", re.I), 1.35, 0.75, "CONSUMER_Q2_INVENTORY_STOCKING_Q3_DIWALI_CASH_RELEASE"),
    (re.compile(r"\b(software|information technology|it services|fmcg|personal products|tobacco|beverages|packaged food)\b", re.I), 1.15, 0.95, "ASSET_LIGHT_NEGATIVE_WC_STEADY_CASH_CYCLE"),
    (re.compile(r"\b(pharma|drug|hospital|healthcare|chemical|specialty)\b", re.I), 1.40, 1.00, "PHARMA_CHEM_USFDA_EXPORT_RECEIVABLE_CYCLE"),
]


def resolve_sector_seasonal_profile(sector: str, industry: str) -> Tuple[float, float, str]:
    combined = f"{sector or ''} {industry or ''}"
    for pat, min_cr, min_qr, tag in SECTOR_SEASONAL_REQUIREMENTS:
        if pat.search(combined):
            return min_cr, min_qr, tag
    return 1.30, 0.90, "INDUSTRIAL_STANDARD_WORKING_CAPITAL_CYCLE"


def is_balance_sheet_lender_or_insurer(sector: str, industry: str) -> bool:
    """True ONLY for balance-sheet lenders/insurers (Banks, NBFCs, Credit Services, Life/General Insurance).
       False for asset-light Capital Markets, Exchanges (MCX, BSE), AMCs (HDFCAMC), and Brokers (MOTILALOFS, GROWW)."""
    ind_low = (industry or "").lower()
    sec_low = (sector or "").lower()
    if any(k in ind_low for k in ("exchange", "financial data", "asset management", "capital market", "broking", "broker", "depository", "rating")):
        return False
    if any(k in ind_low for k in ("bank", "credit", "nbfc", "insurance", "mortgage", "housing finance", "lending")):
        return True
    return "financial" in sec_low and not ind_low


def compute_dcf_and_cash_sustainability(row: dict) -> dict:
    """Computes:
       1. 2-Stage 10-Year Free-Cash-Flow DCF (for Industrials, IT, Auto, Capital Markets/Exchanges)
       2. 2-Stage 10-Year Book-Value & Earnings Compounding Model (for Balance-Sheet Banks & NBFCs like BAJFINANCE)
       3. Binary-Search Reverse DCF Implied 5Y Growth Rate
       4. CROIC, FCF Debt Payback Years, and Sector-Seasonal Working Capital Buffer."""
    price = float(row.get("current_price") or row.get("friday_close") or np.nan)
    mcap = float(row.get("market_cap") or np.nan)
    fcf = float(row.get("free_cash_flow") or np.nan)
    ocf = float(row.get("operating_cash_flow") or np.nan)
    pe = float(row.get("pe_ratio") or np.nan)
    pb = float(row.get("pb_ratio") or np.nan)
    roe = float(row.get("roe") or np.nan)
    de_raw = float(row.get("debt_to_equity") or 0.0)
    cur_ratio = float(row.get("current_ratio") or np.nan)
    quick_ratio = float(row.get("quick_ratio") or np.nan)
    rev_g = float(row.get("revenue_growth") or 10.0)
    eps_g = float(row.get("earnings_growth") or 12.0)
    beta = float(row.get("beta_252d") or row.get("beta") or 1.0)

    sector_str = str(row.get("sector") or "")
    ind_str = str(row.get("industry") or "")
    is_lender = is_balance_sheet_lender_or_insurer(sector_str, ind_str)

    # 1. Cost of Equity (Ke) via Indian Institutional CAPM: Rf = 6.85%, ERP = 5.25%
    # For regulated large/mid-cap lenders, cap fundamental Beta at 1.25 so short-term stock volatility doesn't push Ke > 13.5%
    max_beta = 1.25 if is_lender else 1.75
    beta_clamped = float(np.clip(beta if not np.isnan(beta) else 1.0, 0.65, max_beta))
    ke_dec = float(np.clip(0.0685 + beta_clamped * 0.0525, 0.105, 0.155))
    ke_pct = round(ke_dec * 100.0, 2)
    g_term = 0.065  # 6.5% Long-run nominal GDP/terminal growth floor for India

    # 2. Reconstruct Implied Book Equity, Total Debt, and Invested Capital
    book_equity = (mcap / pb) if (not np.isnan(mcap) and not np.isnan(pb) and pb > 0.05) else np.nan
    implied_pat = (mcap / pe) if (not np.isnan(mcap) and not np.isnan(pe) and pe > 0.5) else np.nan
    de_ratio = max(de_raw / 100.0, 0.0)
    total_debt = (book_equity * de_ratio) if not np.isnan(book_equity) else 0.0
    invested_capital = (book_equity + total_debt) if not np.isnan(book_equity) else np.nan

    # 3. Cash Return on Invested Capital (CROIC %) & Reinvestment Efficiency
    effective_fcf = fcf
    if np.isnan(effective_fcf) or effective_fcf <= 0.0:
        if not np.isnan(ocf) and ocf > 0:
            effective_fcf = 0.72 * ocf
        elif not np.isnan(implied_pat) and implied_pat > 0:
            effective_fcf = 0.78 * implied_pat
    elif not is_lender and not np.isnan(implied_pat) and implied_pat > 0 and effective_fcf < 0.45 * implied_pat and (roe or 0) >= 18.0 and de_ratio < 0.35:
        # Smooth captive-finance (e.g. BACL/MMFSL) or treasury working-capital distortions for low-debt high-ROE compounders
        effective_fcf = 0.50 * effective_fcf + 0.50 * (0.78 * implied_pat)

    if is_lender:
        # For Banks & NBFCs, ROE is the equity cash compounding return, and ~82% of PAT is retained to fund AUM growth
        croic_pct = float(np.clip(roe if not np.isnan(roe) else 15.0, -20.0, 45.0))
        reinvest_eff = 82.0  # Earnings Retention Ratio into Tier-1 Capital
    else:
        if not np.isnan(effective_fcf) and not np.isnan(invested_capital) and invested_capital > 0:
            croic_pct = float(np.clip((effective_fcf / invested_capital) * 100.0, -25.0, 60.0))
        elif not np.isnan(roe) and de_ratio >= 0:
            croic_pct = float(np.clip((roe / (1.0 + de_ratio)) * 0.85, -25.0, 55.0))
        else:
            croic_pct = 12.0

        if not np.isnan(fcf) and not np.isnan(ocf) and ocf > 0:
            reinvest_eff = float(np.clip((fcf / ocf) * 100.0, -50.0, 95.0))
        else:
            reinvest_eff = float(np.clip(croic_pct - ke_pct + 50.0, 10.0, 85.0))

    # 4. Debt Payback Horizon (Years of Free Cash Flow needed to retire 100% of Total Debt)
    if is_lender:
        fcf_debt_payback_yrs = 0.0
    elif total_debt <= 0 or de_ratio < 0.03:
        fcf_debt_payback_yrs = 0.0
    elif not np.isnan(effective_fcf) and effective_fcf > 0:
        fcf_debt_payback_yrs = float(np.clip(total_debt / effective_fcf, 0.1, 25.0))
    else:
        fcf_debt_payback_yrs = 25.0

    # 5. Sector-Seasonal Working Capital Buffer Ratio
    min_cr, min_qr, seasonal_tag = resolve_sector_seasonal_profile(sector_str, ind_str)
    if is_lender:
        seasonal_buffer = 1.25
    else:
        cr_part = (cur_ratio / min_cr) if (not np.isnan(cur_ratio) and cur_ratio > 0) else 1.15
        qr_part = (quick_ratio / min_qr) if (not np.isnan(quick_ratio) and quick_ratio > 0) else cr_part
        seasonal_buffer = float(np.clip(0.55 * qr_part + 0.45 * cr_part, 0.25, 3.50))

    # 6. 2-Stage 10-Year DCF & Reverse DCF
    hist_g = 0.55 * (eps_g if not np.isnan(eps_g) else 12.0) + 0.45 * (rev_g if not np.isnan(rev_g) else 10.0)
    sustainable_g = float(np.clip(croic_pct * 0.85, 5.0, 26.0))
    g_stage1_pct = float(np.clip(0.65 * hist_g + 0.35 * sustainable_g, 6.0, 26.0))
    # Cap Stage-1 growth for deep cyclicals trading at < 7x P/E (e.g. Shipping/Refineries at peak cycle)
    if not is_lender and not np.isnan(pe) and 0 < pe < 7.5:
        g_stage1_pct = min(g_stage1_pct, 8.0)
    g1 = g_stage1_pct / 100.0

    def pv_10yr_cash_flow_multiple(g_rate: float, disc_rate: float, term_rate: float) -> float:
        pv_sum = 0.0
        cf = 1.0
        for yr in range(1, 11):
            if yr <= 5:
                g_t = g_rate
            else:
                fade = (yr - 5) / 5.0
                g_t = g_rate * (1.0 - fade) + term_rate * fade
            cf *= (1.0 + g_t)
            pv_sum += cf / ((1.0 + disc_rate) ** yr)
        tv = (cf * (1.0 + term_rate)) / max(disc_rate - term_rate, 0.035)
        pv_sum += tv / ((1.0 + disc_rate) ** 10)
        return pv_sum

    dcf_iv = np.nan
    mos_pct = np.nan
    rev_dcf_g_pct = np.nan
    growth_cushion_pct = np.nan

    if not np.isnan(price) and price > 0 and not np.isnan(mcap) and mcap > 0:
        if is_lender and not np.isnan(pe) and pe > 0:
            # 2-Stage Distributable Equity Earnings / Franchise Compounding Model for Banks & NBFCs (e.g. BAJFINANCE, ICICIBANK)
            # In Stage 1 (Yrs 1-5), distributable equity capacity grows at g1; in Stage 2 (Yrs 6-10) it fades to g_term,
            # with terminal payout = 1 - (g_term / terminal_ROE) = ~68% of Year-10 earnings.
            roe_dec = float(np.clip((roe if not np.isnan(roe) else 16.0) / 100.0, 0.08, 0.28))
            payout_ratio = float(np.clip(1.0 - (g_term / max(roe_dec, 0.14)), 0.55, 0.78))
            base_distributable_earnings = implied_pat * payout_ratio
            mult_lender = pv_10yr_cash_flow_multiple(g1, ke_dec, g_term)
            intrinsic_mcap = base_distributable_earnings * mult_lender
            dcf_iv = float(price * (intrinsic_mcap / mcap))

            # Binary Search Reverse DCF for Lenders: What 5Y Earnings/AUM CAGR justifies current P/E?
            target_lender_mult = mcap / max(base_distributable_earnings, 1.0)
            lo_g, hi_g = -0.05, 0.42
            for _ in range(22):
                mid_g = 0.5 * (lo_g + hi_g)
                if pv_10yr_cash_flow_multiple(mid_g, ke_dec, g_term) < target_lender_mult:
                    lo_g = mid_g
                else:
                    hi_g = mid_g
            rev_dcf_g_pct = round(float(0.5 * (lo_g + hi_g) * 100.0), 1)
            growth_cushion_pct = round(float(g_stage1_pct - rev_dcf_g_pct), 1)
        else:
            base_cf = effective_fcf
            if not np.isnan(implied_pat) and implied_pat > 0:
                if np.isnan(base_cf) or base_cf <= 0:
                    base_cf = 0.72 * implied_pat
                else:
                    base_cf = 0.60 * base_cf + 0.40 * (0.82 * implied_pat)

            if not np.isnan(base_cf) and base_cf > 0:
                base_cf_yield = base_cf / mcap
                mult_intrinsic = pv_10yr_cash_flow_multiple(g1, ke_dec, g_term)
                intrinsic_mcap = max(0.1 * mcap, base_cf * mult_intrinsic - max(total_debt * 0.30, 0.0))
                dcf_iv = float(price * (intrinsic_mcap / mcap))

                target_mult = 1.0 / max(base_cf_yield, 0.002)
                lo_g, hi_g = -0.15, 0.48
                for _ in range(22):
                    mid_g = 0.5 * (lo_g + hi_g)
                    if pv_10yr_cash_flow_multiple(mid_g, ke_dec, g_term) < target_mult:
                        lo_g = mid_g
                    else:
                        hi_g = mid_g
                rev_dcf_g_pct = round(float(0.5 * (lo_g + hi_g) * 100.0), 1)
                growth_cushion_pct = round(float(g_stage1_pct - rev_dcf_g_pct), 1)

        if not np.isnan(dcf_iv) and dcf_iv > 0:
            mos_pct = round(float(((dcf_iv - price) / dcf_iv) * 100.0), 1)

    # 7. Composite Cash Conversion & Debt-Hedge Score
    ocf_pat = 1.15 if is_lender else float(row.get("ocf_to_pat_ratio") or 1.0)
    if np.isnan(ocf_pat):
        ocf_pat = 1.0
    cash_conv_score = float(np.clip(
        35.0 * np.clip(croic_pct / 20.0, -0.5, 1.5)
        + 25.0 * np.clip((4.0 - fcf_debt_payback_yrs) / 4.0, -0.8, 1.2)
        + 20.0 * np.clip(ocf_pat / 1.1, -0.5, 1.3)
        + 20.0 * np.clip(seasonal_buffer / 1.2, 0.2, 1.3),
        5.0,
        99.0
    ))

    # 8. Classify Cash & Debt Sustainability Regime
    if is_lender:
        cash_regime = "BFSI_CAPITAL_AND_ROE_FORTRESS" if croic_pct >= 14.0 else "BFSI_MODERATE_ROE"
    elif fcf_debt_payback_yrs <= 1.2 and croic_pct >= 15.0 and seasonal_buffer >= 1.0:
        cash_regime = "SELF_FUNDED_CASH_FORTRESS"
    elif fcf_debt_payback_yrs <= 3.5 and croic_pct >= 10.0 and seasonal_buffer >= 0.85:
        cash_regime = "HEALTHY_INTERNAL_ACCRUALS"
    elif fcf_debt_payback_yrs > 6.0 and croic_pct >= 10.0:
        cash_regime = "CAPEX_PHASE_EXTERNAL_DEBT_DEPENDENT"
    elif seasonal_buffer < 0.75 or fcf_debt_payback_yrs >= 12.0:
        cash_regime = "WORKING_CAPITAL_OR_DEBT_STRESS"
    else:
        cash_regime = "MODERATE_CASH_CONVERSION"

    return {
        "cost_of_equity_ke_pct": ke_pct,
        "croic_pct": round(croic_pct, 1),
        "reinvestment_efficiency_pct": round(reinvest_eff, 1),
        "fcf_to_debt_payback_years": round(fcf_debt_payback_yrs, 1),
        "cash_conversion_quality_score": round(cash_conv_score, 1),
        "seasonal_wc_buffer_ratio": round(seasonal_buffer, 2),
        "sector_seasonal_cycle_tag": seasonal_tag,
        "dcf_intrinsic_value": round(dcf_iv, 1) if not np.isnan(dcf_iv) else np.nan,
        "dcf_margin_of_safety_pct": mos_pct,
        "reverse_dcf_implied_growth_pct": rev_dcf_g_pct,
        "dcf_growth_cushion_pct": growth_cushion_pct,
        "cash_debt_regime_tag": cash_regime,
    }


# ==========================================================
# 4. ROBUST STATISTICAL HELPERS (Z-SCORE, HURST, RSI, ATR)
# ==========================================================
def robust_zscore_series(s: pd.Series, invert: bool = False) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 3:
        return pd.Series(0.0, index=s.index)
    med = valid.median()
    mad = (valid - med).abs().median()
    scale = 1.4826 * mad if mad > 1e-6 else valid.std()
    if not scale or np.isnan(scale) or scale < 1e-6:
        return pd.Series(0.0, index=s.index)
    z = ((s - med) / scale).clip(-4.0, 4.0).fillna(0.0)
    return -z if invert else z


def compute_hurst_exponent(price_series: np.ndarray, max_lag: int = 20) -> float:
    if len(price_series) < 60:
        return 0.50
    log_p = np.log(np.maximum(price_series, 1e-4))
    rets_1d = np.diff(log_p)
    var_1d = float(np.var(rets_1d, ddof=1)) if len(rets_1d) > 10 else 0.0
    if var_1d < 1e-10:
        return 0.50
    vr_horizons = []
    for k in (5, 10, 20):
        if len(log_p) > k * 3:
            rets_kd = log_p[k:] - log_p[:-k]
            vr = float(np.var(rets_kd, ddof=1)) / (k * var_1d)
            h_est = 0.5 + 0.5 * (math.log(max(vr, 0.1)) / math.log(k))
            vr_horizons.append(h_est)
    h_val = float(np.mean(vr_horizons)) if vr_horizons else 0.50
    return float(np.clip(h_val, 0.25, 0.85))


def compute_single_ticker_technicals(
    df_p: pd.DataFrame,
    bench_rets: pd.Series,
    master_row: dict
) -> dict:
    close_f = float(master_row.get("friday_close") or np.nan)
    d50_f = float(master_row.get("dist_from_50dma") or np.nan)
    d200_f = float(master_row.get("dist_from_200dma") or np.nan)
    hi52 = float(master_row.get("fifty_two_week_high") or np.nan)
    beta_f = float(master_row.get("beta") or 1.0)

    if not np.isnan(d50_f) and abs(d50_f) < 1.5:
        d50_f *= 100.0
    if not np.isnan(d200_f) and abs(d200_f) < 1.5:
        d200_f *= 100.0

    dist_52w_hi = ((close_f / hi52 - 1.0) * 100.0) if (close_f > 0 and hi52 > 0) else np.nan

    if df_p is None or len(df_p) < 25:
        return {
            "current_price": close_f,
            "ret_1m_pct": np.nan,
            "ret_3m_pct": np.nan,
            "ret_6m_pct": np.nan,
            "ret_12m_pct": np.nan,
            "mom_12m_ex_1m_pct": np.nan,
            "realized_vol_20d_ann_pct": np.nan,
            "realized_vol_252d_ann_pct": np.nan,
            "sharpe_momentum_6m": 0.0,
            "sharpe_momentum_12m": 0.0,
            "sortino_ratio_1y": 0.0,
            "beta_252d": beta_f,
            "alpha_252d_ann_pct": 0.0,
            "rsi_14": 50.0,
            "bollinger_zscore_20d": 0.0,
            "atr_14_pct": np.nan,
            "vol_squeeze_percentile": 50.0,
            "up_down_vol_ratio_20d": 1.0,
            "hurst_exponent_126d": 0.50,
            "dist_from_50dma_pct": d50_f,
            "dist_from_200dma_pct": d200_f,
            "dist_from_52w_high_pct": dist_52w_hi,
            "max_drawdown_1y_pct": np.nan,
            "cvar_95_daily_pct": np.nan,
        }

    df_p = df_p.sort_values("date")
    c = df_p["close"].to_numpy(dtype=float)
    h = df_p["high"].to_numpy(dtype=float)
    l = df_p["low"].to_numpy(dtype=float)
    v = df_p["volume"].to_numpy(dtype=float)
    n = len(c)
    cur_p = float(c[-1])

    def pct_ret(days: int) -> float:
        return float((c[-1] / c[-days - 1] - 1.0) * 100.0) if n > days and c[-days - 1] > 0 else np.nan

    ret_1m = pct_ret(21)
    ret_3m = pct_ret(63)
    ret_6m = pct_ret(126)
    ret_12m = pct_ret(min(252, n - 1)) if n > 180 else np.nan
    mom_12_1 = float((c[-22] / c[-min(252, n - 1) - 1] - 1.0) * 100.0) if n > 180 and c[-22] > 0 else ret_6m

    daily_rets = np.diff(c) / np.maximum(c[:-1], 1e-4)
    vol_20d = float(np.std(daily_rets[-20:], ddof=1) * math.sqrt(252) * 100.0) if len(daily_rets) >= 20 else np.nan
    vol_252d = float(np.std(daily_rets, ddof=1) * math.sqrt(252) * 100.0) if len(daily_rets) >= 20 else np.nan
    vol_6m = float(np.std(daily_rets[-126:], ddof=1) * math.sqrt(252) * 100.0) if len(daily_rets) >= 60 else vol_252d

    sharpe_6m = float((ret_6m * 2.0) / vol_6m) if (not np.isnan(ret_6m) and vol_6m and vol_6m > 1.0) else 0.0
    sharpe_12m = float(ret_12m / vol_252d) if (not np.isnan(ret_12m) and vol_252d and vol_252d > 1.0) else sharpe_6m

    downside = daily_rets[daily_rets < 0]
    down_vol = float(np.std(downside, ddof=1) * math.sqrt(252) * 100.0) if len(downside) >= 5 else vol_252d
    ann_ret = float(np.mean(daily_rets) * 252 * 100.0)
    sortino = float(ann_ret / down_vol) if (down_vol and down_vol > 1.0) else 0.0

    s_rets = pd.Series(daily_rets)
    roll_vol20 = s_rets.rolling(20).std().dropna()
    vol_squeeze_pct = float((roll_vol20 <= roll_vol20.iloc[-1]).mean() * 100.0) if len(roll_vol20) >= 20 else 50.0

    alpha_ann = 0.0
    beta_calc = beta_f
    if bench_rets is not None and not bench_rets.empty:
        stock_s = pd.Series(daily_rets, index=pd.to_datetime(df_p["date"].iloc[1:].values))
        aligned = pd.concat([stock_s.rename("s"), bench_rets.rename("b")], axis=1, sort=False).dropna()
        if len(aligned) >= 40:
            cov_m = np.cov(aligned["s"], aligned["b"])
            var_b = cov_m[1, 1]
            if var_b > 1e-8:
                beta_calc = float(np.clip(cov_m[0, 1] / var_b, 0.1, 3.0))
                alpha_daily = float(aligned["s"].mean() - beta_calc * aligned["b"].mean())
                alpha_ann = float(alpha_daily * 252.0 * 100.0)

    deltas = np.diff(c)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else 0.0
    avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else 0.0
    rsi_14 = 100.0 - (100.0 / (1.0 + avg_gain / max(avg_loss, 1e-6))) if avg_loss > 0 else 75.0

    ma20 = float(np.mean(c[-20:]))
    sd20 = float(np.std(c[-20:], ddof=1))
    bb_z = float((cur_p - ma20) / sd20) if sd20 > 1e-4 else 0.0

    prev_c = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
    atr_14_pct = float((np.mean(tr[-14:]) / cur_p) * 100.0) if len(tr) >= 14 and cur_p > 0 else np.nan

    recent_d = deltas[-20:]
    recent_v = v[-20:]
    up_vol = float(np.sum(recent_v[recent_d > 0]))
    dn_vol = float(np.sum(recent_v[recent_d < 0]))
    up_dn_vol_ratio = float(np.clip(up_vol / max(dn_vol, 1.0), 0.2, 5.0)) if dn_vol > 0 else 1.5

    ma50 = float(np.mean(c[-50:])) if n >= 50 else ma20
    ma200 = float(np.mean(c[-200:])) if n >= 200 else float(np.mean(c))
    d50 = float((cur_p / ma50 - 1.0) * 100.0) if ma50 > 0 else d50_f
    d200 = float((cur_p / ma200 - 1.0) * 100.0) if ma200 > 0 else d200_f
    hi_1y = float(np.max(h))
    dist_52w = float((cur_p / hi_1y - 1.0) * 100.0) if hi_1y > 0 else dist_52w_hi

    running_max = np.maximum.accumulate(c)
    drawdowns = (c / np.maximum(running_max, 1e-4) - 1.0) * 100.0
    max_dd_1y = float(np.min(drawdowns))

    var_5 = np.percentile(daily_rets, 5)
    tail_losses = daily_rets[daily_rets <= var_5]
    cvar_95 = float(np.mean(tail_losses) * 100.0) if len(tail_losses) > 0 else float(var_5 * 100.0)

    hurst = compute_hurst_exponent(c[-126:])

    return {
        "current_price": round(cur_p, 2),
        "ret_1m_pct": round(ret_1m, 2) if not np.isnan(ret_1m) else np.nan,
        "ret_3m_pct": round(ret_3m, 2) if not np.isnan(ret_3m) else np.nan,
        "ret_6m_pct": round(ret_6m, 2) if not np.isnan(ret_6m) else np.nan,
        "ret_12m_pct": round(ret_12m, 2) if not np.isnan(ret_12m) else np.nan,
        "mom_12m_ex_1m_pct": round(mom_12_1, 2) if not np.isnan(mom_12_1) else np.nan,
        "realized_vol_20d_ann_pct": round(vol_20d, 2) if not np.isnan(vol_20d) else np.nan,
        "realized_vol_252d_ann_pct": round(vol_252d, 2) if not np.isnan(vol_252d) else np.nan,
        "sharpe_momentum_6m": round(sharpe_6m, 2),
        "sharpe_momentum_12m": round(sharpe_12m, 2),
        "sortino_ratio_1y": round(sortino, 2),
        "beta_252d": round(beta_calc, 2),
        "alpha_252d_ann_pct": round(alpha_ann, 2),
        "rsi_14": round(rsi_14, 1),
        "bollinger_zscore_20d": round(bb_z, 2),
        "atr_14_pct": round(atr_14_pct, 2) if not np.isnan(atr_14_pct) else np.nan,
        "vol_squeeze_percentile": round(vol_squeeze_pct, 1),
        "up_down_vol_ratio_20d": round(up_dn_vol_ratio, 2),
        "hurst_exponent_126d": round(hurst, 3),
        "dist_from_50dma_pct": round(d50, 2) if not np.isnan(d50) else np.nan,
        "dist_from_200dma_pct": round(d200, 2) if not np.isnan(d200) else np.nan,
        "dist_from_52w_high_pct": round(dist_52w, 2) if not np.isnan(dist_52w) else np.nan,
        "max_drawdown_1y_pct": round(max_dd_1y, 2),
        "cvar_95_daily_pct": round(cvar_95, 2),
    }


def classify_technical_regime(row: pd.Series) -> str:
    d200 = float(row.get("dist_from_200dma_pct") or 0.0)
    d50 = float(row.get("dist_from_50dma_pct") or 0.0)
    rsi = float(row.get("rsi_14") or 50.0)
    squeeze = float(row.get("vol_squeeze_percentile") or 50.0)
    up_dn = float(row.get("up_down_vol_ratio_20d") or 1.0)
    hurst = float(row.get("hurst_exponent_126d") or 0.50)
    sh_6m = float(row.get("sharpe_momentum_6m") or 0.0)

    if d200 > 0 and d50 > -2.0 and squeeze <= 25.0 and up_dn >= 1.05:
        return "VOLATILITY_SQUEEZE_SETUP"
    if d200 > 5.0 and d50 > 1.5 and 52.0 <= rsi <= 76.0 and (up_dn >= 1.10 or sh_6m >= 1.0):
        return "STRONG_UPTREND_BREAKOUT"
    if d200 > 2.0 and d50 > 0.0 and sh_6m >= 0.5:
        return "STEADY_INSTITUTIONAL_UPTREND"
    if d200 > 0.0 and rsi <= 62.0 and (-8.0 <= d50 <= 1.5 or 35.0 <= rsi <= 55.0):
        return "UPTREND_PULLBACK_ACCUMULATE"
    if d200 > 2.0 and d50 >= -1.0 and sh_6m >= 0.4:
        return "STEADY_INSTITUTIONAL_UPTREND"
    if d200 < -4.0 and d50 < -1.5 and rsi < 45.0:
        return "DOWNTREND_AVOID"
    if abs(d50) <= 4.0 and hurst < 0.44:
        return "MEAN_REVERTING_RANGE"
    return "NEUTRAL_CONSOLIDATION"


# ==========================================================
# 5. MAIN QUANT ENGINE COMPUTATION PIPELINE
# ==========================================================
def build_quant_feature_store(conn: duckdb.DuckDBPyConnection, skip_price_download: bool = False):
    init_prices_and_quant_schema(conn)
    sync_point_in_time_tables_from_cache(conn)

    if not skip_price_download:
        sync_daily_prices_yfinance(conn, period="1y", batch_size=100)

    logger.info("Computing DCF, Cash-Flow Sustainability, Fundamental Z-Scores & Technicals for 750 stocks...")
    eq_df = conn.execute("""
        SELECT e.*, r.symbol
        FROM equities_master e
        LEFT JOIN nse_universe_ranks r ON e.ticker = r.ticker
        ORDER BY e.market_cap_rank ASC
    """).df()

    pct_cols = [
        "roe", "roa", "gross_margin", "operating_margin",
        "ebitda_margin", "net_profit_margin", "revenue_growth", "earnings_growth"
    ]
    for col in pct_cols:
        if col in eq_df.columns:
            med_abs = eq_df[col].dropna().abs().median()
            if med_abs and med_abs < 1.5:
                eq_df[col] = eq_df[col] * 100.0

    is_lender_series = eq_df.apply(
        lambda r: is_balance_sheet_lender_or_insurer(str(r.get("sector") or ""), str(r.get("industry") or "")),
        axis=1
    )

    mcap_raw = eq_df["market_cap"].replace(0, np.nan)
    eq_df["fcf_yield_pct"] = np.where(
        is_lender_series,
        np.nan,
        ((eq_df["free_cash_flow"] / mcap_raw) * 100.0).clip(-25.0, 25.0).round(2)
    )
    implied_pat = mcap_raw / eq_df["pe_ratio"].where(eq_df["pe_ratio"] > 0, np.nan)
    # For Lenders (Banks/NBFCs), GAAP OCF includes loan disbursements and is not a cash conversion metric
    eq_df["ocf_to_pat_ratio"] = np.where(
        is_lender_series,
        np.nan,
        (eq_df["operating_cash_flow"] / implied_pat).clip(-5.0, 5.0).round(2)
    )
    eq_df["pe_rerating_ratio"] = (
        eq_df["pe_ratio"] / eq_df["forward_pe"].where(eq_df["forward_pe"] > 0, np.nan)
    ).clip(0.3, 3.0)

    prices_df = conn.execute("SELECT ticker, date, open, high, low, close, volume FROM prices ORDER BY ticker, date").df()
    bench_rets = pd.Series(dtype=float)
    if not prices_df.empty and "^NSEI" in prices_df["ticker"].values:
        b_df = prices_df[prices_df["ticker"] == "^NSEI"].sort_values("date")
        b_c = b_df["close"].to_numpy(dtype=float)
        if len(b_c) > 2:
            bench_rets = pd.Series(
                np.diff(b_c) / np.maximum(b_c[:-1], 1e-4),
                index=pd.to_datetime(b_df["date"].iloc[1:].values),
            )

    grouped_prices = {t: grp for t, grp in prices_df.groupby("ticker")} if not prices_df.empty else {}

    tech_rows = []
    for _, m_row in eq_df.iterrows():
        tkr = m_row["ticker"]
        t_stats = compute_single_ticker_technicals(grouped_prices.get(tkr), bench_rets, m_row.to_dict())
        t_stats["ticker"] = tkr
        tech_rows.append(t_stats)
    tech_df = pd.DataFrame(tech_rows)

    merged = pd.merge(eq_df, tech_df, on="ticker", how="left")

    dcf_rows = []
    for _, r in merged.iterrows():
        d_stats = compute_dcf_and_cash_sustainability(r.to_dict())
        d_stats["ticker"] = r["ticker"]
        dcf_rows.append(d_stats)
    dcf_df = pd.DataFrame(dcf_rows)
    merged = pd.merge(merged, dcf_df, on="ticker", how="left")

    merged["cash_sustainability_percentile"] = (
        merged["cash_conversion_quality_score"].rank(pct=True) * 100.0
    ).round(1)

    merged["sector_clean"] = merged["sector"].fillna("Unclassified").replace("", "Unclassified")
    merged["sector_pe_median"] = merged.groupby("sector_clean")["pe_ratio"].transform("median").round(2)
    merged["sector_pb_median"] = merged.groupby("sector_clean")["pb_ratio"].transform("median").round(2)
    merged["sector_roe_median"] = merged.groupby("sector_clean")["roe"].transform("median").round(2)

    val_pe_z = merged.groupby("sector_clean")["pe_ratio"].transform(lambda s: robust_zscore_series(s.where(s > 0), invert=True))
    val_pb_z = merged.groupby("sector_clean")["pb_ratio"].transform(lambda s: robust_zscore_series(s.where(s > 0), invert=True))
    val_ev_z = merged.groupby("sector_clean")["ev_to_ebitda"].transform(lambda s: robust_zscore_series(s.where(s > 0), invert=True))
    val_fcf_z = merged.groupby("sector_clean")["fcf_yield_pct"].transform(lambda s: robust_zscore_series(s, invert=False))
    val_mos_z = merged.groupby("sector_clean")["dcf_margin_of_safety_pct"].transform(lambda s: robust_zscore_series(s, invert=False))

    is_lender_mask = merged.apply(lambda r: is_balance_sheet_lender_or_insurer(str(r.get("sector") or ""), str(r.get("industry") or "")), axis=1)
    merged["val_zscore_sector"] = np.where(
        is_lender_mask,
        0.45 * val_pb_z + 0.35 * val_pe_z + 0.20 * val_mos_z,
        0.30 * val_pe_z + 0.30 * val_ev_z + 0.20 * val_mos_z + 0.10 * val_fcf_z + 0.10 * val_pb_z,
    )

    roe_clean = merged["roe"].where((merged["roe"] >= -40.0) & (merged["roe"] <= 75.0), np.nan)
    merged["roe_zscore_sector"] = merged.groupby("sector_clean")[roe_clean.name].transform(
        lambda s: robust_zscore_series(roe_clean.loc[s.index])
    )
    roa_z = merged.groupby("sector_clean")["roa"].transform(lambda s: robust_zscore_series(s.clip(-20.0, 35.0)))
    opm_z = merged.groupby("sector_clean")["operating_margin"].transform(lambda s: robust_zscore_series(s.clip(-30.0, 60.0)))
    npm_z = merged.groupby("sector_clean")["net_profit_margin"].transform(lambda s: robust_zscore_series(s.clip(-30.0, 50.0)))
    de_z = merged.groupby("sector_clean")["debt_to_equity"].transform(lambda s: robust_zscore_series(s.clip(0.0, 500.0), invert=True))
    croic_z = merged.groupby("sector_clean")["croic_pct"].transform(robust_zscore_series)
    merged["margin_zscore_sector"] = 0.6 * opm_z + 0.4 * npm_z

    anomaly_penalty = np.where(
        (merged["roe"].fillna(0) > 80.0)
        | (merged["roe"].fillna(0) < 0.0)
        | ((~is_lender_mask) & (merged["debt_to_equity"].fillna(0) > 250.0)),
        -1.25,
        0.0,
    )

    qual_raw = np.where(
        is_lender_mask,
        0.55 * merged["roe_zscore_sector"] + 0.45 * roa_z + anomaly_penalty,
        0.35 * merged["roe_zscore_sector"] + 0.25 * croic_z + 0.20 * merged["margin_zscore_sector"] + 0.20 * de_z + anomaly_penalty,
    )

    rev_g_z = merged.groupby("sector_clean")["revenue_growth"].transform(lambda s: robust_zscore_series(s.clip(-40.0, 120.0)))
    eps_g_z = merged.groupby("sector_clean")["earnings_growth"].transform(lambda s: robust_zscore_series(s.clip(-50.0, 150.0)))
    rerate_z = merged.groupby("sector_clean")["pe_rerating_ratio"].transform(robust_zscore_series)
    merged["growth_zscore_sector"] = 0.45 * eps_g_z + 0.40 * rev_g_z + 0.15 * rerate_z

    own_raw = (
        0.65 * merged["held_by_institutions"].fillna(0.0)
        + 0.35 * merged["held_by_insiders"].fillna(0.0)
    )

    merged["value_percentile"] = (merged["val_zscore_sector"].rank(pct=True) * 100.0).round(1)
    merged["quality_percentile"] = (pd.Series(qual_raw, index=merged.index).rank(pct=True) * 100.0).round(1)
    merged["growth_percentile"] = (merged["growth_zscore_sector"].rank(pct=True) * 100.0).round(1)
    merged["ownership_percentile"] = (own_raw.rank(pct=True) * 100.0).round(1)

    extreme_pe_drag = np.where(
        (merged["pe_ratio"].fillna(0) > 80.0) & (merged["pe_ratio"].fillna(0) > 2.2 * merged["sector_pe_median"].fillna(30.0)),
        18.0,
        np.where(
            (merged["pe_ratio"].fillna(0) > 55.0) & (merged["pe_ratio"].fillna(0) > 1.8 * merged["sector_pe_median"].fillna(30.0)),
            8.0,
            0.0,
        ),
    )

    # Balanced institutional weights (Quality 34%, Growth 32%, Value 18%, Cash Sustainability 10%, Ownership 6%)
    merged["fundamental_composite_score"] = np.clip(
        0.34 * merged["quality_percentile"]
        + 0.32 * merged["growth_percentile"]
        + 0.18 * merged["value_percentile"]
        + 0.10 * merged["cash_sustainability_percentile"]
        + 0.06 * merged["ownership_percentile"]
        - extreme_pe_drag,
        1.0,
        99.5,
    ).round(1)

    mom_rank = merged["sharpe_momentum_6m"].fillna(0.0).rank(pct=True) * 100.0
    alpha_rank = merged["alpha_252d_ann_pct"].fillna(0.0).rank(pct=True) * 100.0
    trend_rank = (
        0.6 * merged["dist_from_200dma_pct"].fillna(0.0)
        + 0.4 * merged["dist_from_50dma_pct"].fillna(0.0)
    ).rank(pct=True) * 100.0
    vol_acc_rank = merged["up_down_vol_ratio_20d"].fillna(1.0).rank(pct=True) * 100.0

    merged["technical_composite_score"] = (
        0.35 * mom_rank + 0.25 * alpha_rank + 0.25 * trend_rank + 0.15 * vol_acc_rank
    ).round(1)
    merged["technical_regime"] = merged.apply(classify_technical_regime, axis=1)

    facts_df = conn.execute("SELECT ticker, value_str, change_str FROM facts").df()
    events_df = conn.execute("SELECT ticker, event_type FROM events").df()
    guid_df = conn.execute("SELECT ticker, modality FROM guidance").df()

    yoy_by_ticker = {}
    fact_count_by_ticker = {}
    if not facts_df.empty:
        for tkr, grp in facts_df.groupby("ticker"):
            fact_count_by_ticker[tkr] = len(grp)
            yoy_vals = []
            for txt in pd.concat([grp["change_str"], grp["value_str"]]).dropna():
                for m in re.finditer(r"(\(?[+-]?\d+(?:\.\d+)?\)?)\s*%\s*(?:yoy|y-o-y|year-on-year|from prior)", str(txt), re.I):
                    raw_num = m.group(1)
                    val = -float(raw_num.strip("()")) if raw_num.startswith("(") else float(raw_num)
                    if -80.0 <= val <= 250.0:
                        yoy_vals.append(val)
            if yoy_vals:
                yoy_by_ticker[tkr] = round(float(np.median(yoy_vals)), 2)

    event_weights = {
        "ORDER_BOOK": 4.2,
        "CAPEX": 3.6,
        "PARTNERSHIP_JV": 3.1,
        "PRICING_ACTION": 2.7,
        "POLICY_REGULATION": 2.0,
        "OPERATIONAL_MILESTONE": 1.4,
        "RISK_FACTOR": -5.5,
    }
    weighted_ev_score = {}
    bull_counts = {}
    risk_counts = {}
    if not events_df.empty:
        for tkr, grp in events_df.groupby("ticker"):
            w_sum = 0.0
            b_cnt = 0
            r_cnt = 0
            for ev_t in grp["event_type"]:
                w_sum += event_weights.get(ev_t, 1.0)
                if ev_t == "RISK_FACTOR":
                    r_cnt += 1
                else:
                    b_cnt += 1
            weighted_ev_score[tkr] = w_sum
            bull_counts[tkr] = b_cnt
            risk_counts[tkr] = r_cnt

    guid_count_map = (
        guid_df[guid_df["modality"].isin(["GUIDANCE", "QUALITATIVE_OUTLOOK"])].groupby("ticker").size().to_dict()
        if not guid_df.empty else {}
    )

    merged["extracted_yoy_growth_median_pct"] = merged["ticker"].map(yoy_by_ticker)
    merged["bullish_catalyst_count"] = merged["ticker"].map(lambda t: int(bull_counts.get(t, 0)))
    merged["risk_catalyst_count"] = merged["ticker"].map(lambda t: int(risk_counts.get(t, 0)))
    merged["has_numerical_guidance"] = merged["ticker"].map(lambda t: int(guid_count_map.get(t, 0)) > 0)

    raw_cat_score = (
        merged["extracted_yoy_growth_median_pct"].fillna(merged["revenue_growth"].fillna(8.0)).clip(-25.0, 60.0) * 0.90
        + merged["ticker"].map(lambda t: float(weighted_ev_score.get(t, 0.0))) * 2.2
        + merged["ticker"].map(lambda t: min(int(guid_count_map.get(t, 0)), 3) * 1.5)
        + merged["ticker"].map(lambda t: min(int(fact_count_by_ticker.get(t, 0)), 12) * 0.40)
    )
    merged["filing_catalyst_score"] = (raw_cat_score.rank(pct=True) * 100.0).round(1)

    merged["quant_conviction_score"] = (
        0.50 * merged["fundamental_composite_score"]
        + 0.30 * merged["technical_composite_score"]
        + 0.20 * merged["filing_catalyst_score"]
    ).round(1)

    def assign_signal_tag(r: pd.Series) -> str:
        conv = float(r["quant_conviction_score"] or 0.0)
        f_score = float(r["fundamental_composite_score"] or 0.0)
        t_score = float(r["technical_composite_score"] or 0.0)
        pe = float(r.get("pe_ratio") or 0.0)
        sec_pe = float(r.get("sector_pe_median") or 25.0)
        roe = float(r.get("roe") or 0.0)

        if (pe > 75.0 and pe > 2.2 * sec_pe) or roe > 80.0:
            return "MOMENTUM_HIGH_VALUATION_RISK" if t_score >= 65.0 else "AVOID_VALUATION_OR_ANOMALY"
        if conv >= 74.0 and f_score >= 68.0 and t_score >= 60.0 and 12.0 <= roe <= 75.0:
            return "HIGH_CONVICTION_COMPOUNDER"
        if f_score >= 72.0 and t_score < 55.0:
            return "FUNDAMENTAL_VALUE_PULLBACK"
        if t_score >= 75.0 and r["filing_catalyst_score"] >= 65.0:
            return "MOMENTUM_CATALYST_LEADER"
        if conv < 32.0:
            return "AVOID_WEAK_MULTI_FACTOR"
        return "CORE_HOLD_MONITOR"

    merged["quant_signal_tag"] = merged.apply(assign_signal_tag, axis=1)
    merged["as_of"] = datetime.now().strftime("%Y-%m-%d")

    out_cols = [
        "ticker", "symbol", "company_name", "sector", "industry",
        "market_cap_rank", "market_cap_cr", "market_cap_category", "current_price",
        "pe_ratio", "forward_pe", "pb_ratio", "ev_to_ebitda",
        "roe", "roa", "operating_margin", "net_profit_margin",
        "debt_to_equity", "fcf_yield_pct", "ocf_to_pat_ratio",
        "revenue_growth", "earnings_growth", "pe_rerating_ratio",
        "cost_of_equity_ke_pct", "croic_pct", "reinvestment_efficiency_pct",
        "fcf_to_debt_payback_years", "cash_conversion_quality_score",
        "seasonal_wc_buffer_ratio", "sector_seasonal_cycle_tag",
        "dcf_intrinsic_value", "dcf_margin_of_safety_pct",
        "reverse_dcf_implied_growth_pct", "dcf_growth_cushion_pct",
        "cash_sustainability_percentile", "cash_debt_regime_tag",
        "sector_pe_median", "sector_pb_median", "sector_roe_median",
        "val_zscore_sector", "roe_zscore_sector", "margin_zscore_sector", "growth_zscore_sector",
        "value_percentile", "quality_percentile", "growth_percentile", "ownership_percentile",
        "fundamental_composite_score",
        "ret_1m_pct", "ret_3m_pct", "ret_6m_pct", "ret_12m_pct", "mom_12m_ex_1m_pct",
        "realized_vol_20d_ann_pct", "realized_vol_252d_ann_pct",
        "sharpe_momentum_6m", "sharpe_momentum_12m", "sortino_ratio_1y",
        "beta_252d", "alpha_252d_ann_pct", "rsi_14", "bollinger_zscore_20d",
        "atr_14_pct", "vol_squeeze_percentile", "up_down_vol_ratio_20d",
        "hurst_exponent_126d", "dist_from_50dma_pct", "dist_from_200dma_pct",
        "dist_from_52w_high_pct", "max_drawdown_1y_pct", "cvar_95_daily_pct",
        "technical_composite_score", "technical_regime",
        "extracted_yoy_growth_median_pct", "bullish_catalyst_count", "risk_catalyst_count",
        "has_numerical_guidance", "filing_catalyst_score",
        "quant_conviction_score", "quant_signal_tag", "as_of"
    ]

    final_df = merged[out_cols].copy()
    conn.execute("DELETE FROM stock_quant_signals")
    conn.register("final_quant_df", final_df)
    conn.execute("INSERT INTO stock_quant_signals SELECT * FROM final_quant_df")
    conn.execute("CHECKPOINT")
    logger.info(f"Saved {len(final_df)} rows (with 70 DCF, Cash-Flow & Technical columns) to `stock_quant_signals`!")


def print_top_quant_leaders(conn: duckdb.DuckDBPyConnection, limit: int = 20):
    df = conn.execute(f"""
        SELECT market_cap_rank AS rank, symbol,
               ROUND(current_price, 1) AS price,
               ROUND(dcf_intrinsic_value, 1) AS dcf_iv,
               ROUND(reverse_dcf_implied_growth_pct, 1) AS rev_dcf_g,
               ROUND(croic_pct, 1) AS croic,
               ROUND(fcf_to_debt_payback_years, 1) AS debt_yrs,
               ROUND(seasonal_wc_buffer_ratio, 2) AS wc_buf,
               cash_sustainability_percentile AS cash_pct,
               fundamental_composite_score AS fund,
               technical_composite_score AS tech,
               quant_conviction_score AS conviction,
               cash_debt_regime_tag,
               technical_regime
        FROM stock_quant_signals
        WHERE quant_signal_tag = 'HIGH_CONVICTION_COMPOUNDER'
        ORDER BY quant_conviction_score DESC
        LIMIT {limit}
    """).df()
    print("\n=== TOP 20 INSTITUTIONAL COMPOUNDERS WITH DCF, CROIC & DEBT PAYBACK (NSE 750) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE 750 Quantitative DCF, Cash-Flow & Technical Statistics Engine")
    parser.add_argument("--skip-price-download", action="store_true", help="Skip yfinance 1Y OHLCV download and use cached prices")
    parser.add_argument("--top-k", type=int, default=20, help="Print top K stocks by multi-factor quant conviction")
    args = parser.parse_args()

    conn = duckdb.connect(str(DB_PATH))
    try:
        build_quant_feature_store(conn, skip_price_download=args.skip_price_download)
        print_top_quant_leaders(conn, limit=args.top_k)
    finally:
        conn.close()