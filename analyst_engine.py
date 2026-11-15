import argparse
import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import httpx
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

# ==========================================================
# 0. CONFIGURATION
# ==========================================================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

DB_CONFIG = next(db for db in config["databases"] if db["name"] == "nse_universe")
DB_PATH = BASE_DIR / DB_CONFIG["path"]

LLM_CFG = config.get("llm", {})
OLLAMA_URL = LLM_CFG.get("ollama_url", "http://localhost:11434/api/chat")
ANALYST_MODEL = LLM_CFG.get("judge_model", "qwen3.5:9b")


# ==========================================================
# 1. POINT-IN-TIME PREDICTIONS, HOLDINGS & COMMODITIES SCHEMA
# ==========================================================
def init_predictions_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "pred_id VARCHAR PRIMARY KEY, ticker VARCHAR, symbol VARCHAR, "
        "as_of DATE, price_at_call DOUBLE, horizon VARCHAR, "
        "directional_call VARCHAR, target_metric VARCHAR, "
        "quant_conviction_score DOUBLE, fact_ids_cited VARCHAR, "
        "rationale VARCHAR, model_version VARCHAR, "
        "realized_return_pct DOUBLE, benchmark_return_pct DOUBLE, "
        "excess_alpha_pct DOUBLE, evaluated_at VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS commodities_fx ("
        "series_id VARCHAR PRIMARY KEY, name VARCHAR, "
        "latest_value DOUBLE, ret_1m_pct DOUBLE, "
        "ret_3m_pct DOUBLE, ret_6m_pct DOUBLE, updated_at VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS holdings ("
        "symbol VARCHAR PRIMARY KEY, promoter_pct DOUBLE, "
        "fii_pct DOUBLE, dii_pct DOUBLE, public_pct DOUBLE, as_of VARCHAR)"
    )


def extract_clean_json_object(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    start = cleaned.find("{")
    if start == -1:
        return "{}"
    cleaned = cleaned[start:]
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    last_nl = cleaned.rfind("\n")
    if last_nl != -1 and not cleaned.rstrip().endswith("}"):
        cleaned = cleaned[:last_nl].rstrip()
    cleaned = re.sub(r"[,:\s]+$", "", cleaned)
    in_str, esc = False, False
    stack = []
    for ch in cleaned:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack and stack[-1] == ch:
                stack.pop()
    if in_str:
        cleaned += '"'
    cleaned = re.sub(r",\s*$", "", cleaned)
    while stack:
        cleaned += stack.pop()
    return cleaned

class StructuredAnalystVerdict(BaseModel):
    directional_call: str = "NEUTRAL_HOLD"
    horizon: str = "1Q-2Q (3-6 Months)"
    target_fundamental_metric: str = "Next-Quarter Earnings & Margin Trajectory"
    executive_thesis: str = ""
    dcf_and_cash_sustainability_view: str = ""
    fundamental_and_valuation_view: str = ""
    technical_and_regime_view: str = ""
    catalysts_and_guidance_view: str = ""
    macro_geopolitical_and_competitive_view: str = ""
    key_risks: List[str] = Field(default_factory=list)
    cited_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def repair_llm_verdict_dict(cls, data: Any) -> dict:
        if not isinstance(data, dict):
            return {}
        text_keys = (
            "directional_call",
            "horizon",
            "target_fundamental_metric",
            "executive_thesis",
            "dcf_and_cash_sustainability_view",
            "fundamental_and_valuation_view",
            "technical_and_regime_view",
            "catalysts_and_guidance_view",
            "macro_geopolitical_and_competitive_view",
        )
        for k in text_keys:
            v = data.get(k)
            if isinstance(v, dict):
                data[k] = "; ".join(f"{sk}: {sv}" for sk, sv in v.items())
            elif isinstance(v, list):
                data[k] = "; ".join(str(x) for x in v)
            elif v is None:
                data[k] = ""

        for lk in ("key_risks", "cited_ids"):
            v = data.get(lk)
            if isinstance(v, str):
                data[lk] = [s.strip() for s in re.split(r"[;\n]+", v) if s.strip()]
            elif not isinstance(v, list):
                data[lk] = []
            else:
                data[lk] = [str(x).strip() for x in v if x and str(x).strip()]

        all_text = " ".join(str(v) for v in data.values())
        raw_ids = [re.sub(r"[\[\]\s]", "", str(x)).upper() for x in data.get("cited_ids", [])]
        valid_raw = [x for x in raw_ids if re.match(r"^(?:F\d{3}|N\d{2}|G\d{2})$", x)]
        found_ids = sorted(set(re.findall(r"\b(?:F\d{3}|N\d{2}|G\d{2})\b", all_text)))
        data["cited_ids"] = list(dict.fromkeys(valid_raw + found_ids))
        return data


# ==========================================================
# 2. SECTOR-TO-COMMODITY/FX MAPPING (ZERO ENTITY NAMES)
# ==========================================================
SECTOR_COMMODITY_LINKS: List[Tuple[re.Pattern, List[str], List[str]]] = [
    (
        re.compile(r"\b(bank|credit|nbfc|housing finance|insurance)\b", re.I),
        ["USDINR", "GOLD", "US10Y"],
        [
            "RBI Repo Rate & Systemic Liquidity (Cost of Funds / NIM)",
            "Retail Unsecured & Gold-Loan Regulatory Risk Weights",
            "Domestic 10Y Sovereign Yield Trajectory",
        ],
    ),
    (
        re.compile(r"\b(auto|automobile|ancillar|tyre|tractor|two-wheeler|commercial vehicle)\b", re.I),
        ["COPPER", "ALUMINIUM", "BRENT", "USDINR"],
        [
            "Base Metal Input Costs (Steel, Aluminium, Copper)",
            "Crude Oil & Rubber Derivatives (Tyre & Freight impact)",
            "Export FX Realization (USD/INR & Emerging Market FX Liquidity)",
        ],
    ),
    (
        re.compile(r"\b(software|it services|information technology)\b", re.I),
        ["USDINR", "US10Y"],
        [
            "Global Enterprise Tech Spend & US Fed Rate Cycle (US 10Y)",
            "USD/INR Constant-Currency Realization",
            "Cross-Border Services & Visa Regulatory Policy",
        ],
    ),
    (
        re.compile(r"\b(oil|gas|refiner|petro|paint|aviation|chemical|fertilizer)\b", re.I),
        ["BRENT", "NATGAS", "USDINR"],
        [
            "Brent Crude & Refining/Petrochemical Crack Spreads",
            "Maritime Freight & Geopolitical Supply-Chain Routing",
            "Natural Gas / LNG Feedstock Pricing",
        ],
    ),
    (
        re.compile(r"\b(steel|metal|mining|aluminium|copper|zinc|cable|engineering|infra|cement)\b", re.I),
        ["COPPER", "ALUMINIUM", "BRENT"],
        [
            "LME Base Metal Realizations & Global Import Parity",
            "Petcoke / Thermal Coal & Power Input Costs",
            "Central & State Government Capex Execution Cycle",
        ],
    ),
    (
        re.compile(r"\b(jewellery|gold|retail|fmcg|food|consumer)\b", re.I),
        ["GOLD", "BRENT", "USDINR"],
        [
            "Bullion Price Volatility & Import Duty Structure",
            "Agri-Commodity & Packaging Input Inflation",
            "Rural Monsoon Demand vs Urban Discretionary Mix",
        ],
    ),
]


class EquitiesAnalystTools:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._idf_cache: Dict[str, float] = {}

    def _get_conn(self, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def sync_and_get_commodities_fx(self) -> Dict[str, dict]:
        conn = self._get_conn(read_only=False)
        try:
            init_predictions_schema(conn)
            today_str = datetime.now().strftime("%Y-%m-%d")
            existing = conn.execute(
                "SELECT * FROM commodities_fx WHERE updated_at >= ?", [today_str]
            ).df()
            if not existing.empty:
                return {r["series_id"]: r for r in existing.to_dict("records")}

            tickers_map = {
                "BRENT": ("BZ=F", "Brent Crude Oil ($/bbl)"),
                "GOLD": ("GC=F", "Gold Futures ($/oz)"),
                "COPPER": ("HG=F", "Copper Futures ($/lb)"),
                "ALUMINIUM": ("ALI=F", "Aluminium Futures ($/t)"),
                "NATGAS": ("NG=F", "Natural Gas ($/MMBtu)"),
                "USDINR": ("INR=X", "USD/INR Exchange Rate"),
                "US10Y": ("^TNX", "US 10Y Treasury Yield (%)"),
            }
            try:
                import yfinance as yf
                yf_list = [v[0] for v in tickers_map.values()]
                df = yf.download(
                    yf_list, period="6mo", interval="1d",
                    group_by="ticker", progress=False, threads=True
                )
                rows_to_save = []
                lvl0 = df.columns.get_level_values(0) if not df.empty else []
                for sid, (yf_tkr, label) in tickers_map.items():
                    sub = df[yf_tkr].dropna(subset=["Close"]) if yf_tkr in lvl0 else pd.DataFrame()
                    if len(sub) >= 5:
                        c = sub["Close"].to_numpy(dtype=float)
                        cur = round(float(c[-1]), 2)
                        r1m = round(float((c[-1] / c[-min(21, len(c) - 1)] - 1.0) * 100.0), 1)
                        r3m = round(float((c[-1] / c[-min(63, len(c) - 1)] - 1.0) * 100.0), 1)
                        r6m = round(float((c[-1] / c[0] - 1.0) * 100.0), 1)
                        rows_to_save.append((sid, label, cur, r1m, r3m, r6m, today_str))
                if rows_to_save:
                    conn.executemany(
                        "INSERT OR REPLACE INTO commodities_fx VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows_to_save
                    )
                    conn.execute("CHECKPOINT")
            except Exception:
                pass

            final_df = conn.execute("SELECT * FROM commodities_fx").df()
            if final_df.empty:
                return {}
            return {r["series_id"]: r for r in final_df.to_dict("records")}
        finally:
            conn.close()

    def get_quant_profile(self, symbol: str) -> dict:
        conn = self._get_conn(read_only=True)
        try:
            sql = (
                "SELECT sq.*, "
                "ROUND(em.held_by_insiders * 100.0, 1) AS promoter_holding_pct, "
                "ROUND(em.held_by_institutions * 100.0, 1) AS fii_dii_institutional_pct "
                "FROM stock_quant_signals sq "
                "LEFT JOIN equities_master em ON sq.ticker = em.ticker "
                "WHERE UPPER(sq.symbol) = UPPER(?)"
            )
            df = conn.execute(sql, [symbol.strip()]).df()
            if df.empty:
                return {}
            rec = df.to_dict("records")[0]
            for k_own in ("promoter_holding_pct", "fii_dii_institutional_pct"):
                val = rec.get(k_own)
                if val is not None and not math.isnan(val) and val > 100.0:
                    rec[k_own] = round(val / 100.0, 1)
            return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}
        finally:
            conn.close()

    def get_sql_industry_peers(self, symbol: str, industry: str, sector: str, limit: int = 4) -> List[str]:
        conn = self._get_conn(read_only=True)
        try:
            sql = (
                "SELECT symbol, market_cap_rank, "
                "ROUND(pe_ratio, 1) AS pe, ROUND(pb_ratio, 1) AS pb, "
                "ROUND(roe_pct, 1) AS roe, ROUND(croic_pct, 1) AS croic, "
                "quant_conviction_score AS conviction "
                "FROM stock_quant_signals "
                "WHERE UPPER(symbol) != UPPER(?) "
                "AND (UPPER(industry) = UPPER(?) OR UPPER(sector) = UPPER(?)) "
                "ORDER BY CASE WHEN UPPER(industry) = UPPER(?) THEN 0 ELSE 1 END, "
                "market_cap_rank ASC LIMIT ?"
            )
            peers_df = conn.execute(
                sql, [symbol, industry or "", sector or "", industry or "", limit]
            ).df()
            out = []
            for _, r in peers_df.iterrows():
                out.append(
                    f"{r['symbol']} (Rank #{r['market_cap_rank']}, P/E={r['pe']}x, "
                    f"P/B={r['pb']}x, ROE={r['roe']}%, CROIC={r['croic']}%, "
                    f"Conviction={r['conviction']})"
                )
            return out
        finally:
            conn.close()

    def extract_competitive_and_macro_intelligence(
        self, symbol: str, sector: str, industry: str,
        pres_txt: str, trans_txt: str, news_txt: str
    ) -> dict:
        comm_map = self.sync_and_get_commodities_fx()
        combined_sec = f"{sector or ''} {industry or ''}"
        linked_sids = ["USDINR", "BRENT"]
        baseline_sens = [
            "Domestic Interest Rate & Liquidity Cycle",
            "INR Exchange Rate & Input Cost Inflation",
        ]

        for pat, sids, sens_list in SECTOR_COMMODITY_LINKS:
            if pat.search(combined_sec):
                linked_sids = sids
                baseline_sens = list(sens_list)
                break

        commodity_lines = []
        for sid in linked_sids:
            if sid in comm_map:
                c = comm_map[sid]
                commodity_lines.append(
                    f"{c['name']}: {c['latest_value']} "
                    f"(1M: {c['ret_1m_pct']:+.1f}%, 3M: {c['ret_3m_pct']:+.1f}%, 6M: {c['ret_6m_pct']:+.1f}%)"
                )

        corpus = f"{pres_txt or ''}\n{trans_txt or ''}\n{news_txt or ''}"
        unwrapped_c = (corpus or "").replace("\r\n", "\n").replace("\r", "\n")
        unwrapped_c = re.sub(r"(?<!\n)\n(?!\n)", " ", unwrapped_c)
        unwrapped_c = re.sub(r"\b(Rs|Cr|Mr|Dr|Ms|vs)\.\s+", r"\1_DOT_ ", unwrapped_c, flags=re.I)
        clean_sents = [
            re.sub(r"\s+", " ", re.sub(r"\s*\|\s*", " ", s.replace("_DOT_ ", ". "))).strip(" -*|")
            for s in re.split(r"(?<=[\.\!])\s+(?=[A-Z0-9])|\n{2,}", unwrapped_c)
            if 45 <= len(s.strip()) <= 320 and "?" not in s
        ]

        unorg_pat = re.compile(
            r"\b(unorgani[sz]ed|informal sector|organi[sz]ed sector|organi[sz]ed players|"
            r"market share|regional players|fragmented|gst|bis norms|shift from)\b",
            re.I,
        )
        geo_pat = re.compile(
            r"\b(geopolit|red sea|west asia|middle east|tariff|anti-dumping|export|"
            r"import duty|usfda|rbi|sebi|freight|supply chain|headwind|currency)\b",
            re.I,
        )
        sub_pat = re.compile(
            r"\b(subsidiar(?:y|ies)|joint venture|\bjv\b|wholly[- ]owned|"
            r"associate compan|step-down|stake in|acquires?|merger|demerger)\b",
            re.I,
        )

        unorg_notes = [s for s in clean_sents if unorg_pat.search(s)][:3]
        geo_notes = [s for s in clean_sents if geo_pat.search(s)][:3]
        sub_notes = [s for s in clean_sents if sub_pat.search(s)][:3]

        return {
            "commodity_fx_benchmarks": commodity_lines,
            "sector_macro_sensitivities": baseline_sens,
            "geopolitical_and_commodity_filing_notes": geo_notes,
            "unorganized_sector_and_market_share_notes": unorg_notes,
            "subsidiaries_and_jv_notes": sub_notes,
        }

    def get_facts_events_guidance(self, symbol: str) -> dict:
        conn = self._get_conn(read_only=True)
        try:
            facts = conn.execute(
                "SELECT span_id, period_id, scope, metric_id, value_str, change_str, source_doc "
                "FROM facts WHERE UPPER(symbol) = UPPER(?) ORDER BY span_id ASC LIMIT 25",
                [symbol],
            ).df().to_dict("records")

            events = conn.execute(
                "SELECT span_id, event_date, fiscal_period, event_type, headline "
                "FROM events WHERE UPPER(symbol) = UPPER(?) ORDER BY span_id ASC LIMIT 8",
                [symbol],
            ).df().to_dict("records")

            guidance = conn.execute(
                "SELECT span_id, period_id, modality, statement, source_doc "
                "FROM guidance WHERE UPPER(symbol) = UPPER(?) ORDER BY span_id ASC LIMIT 6",
                [symbol],
            ).df().to_dict("records")

            ci_row = conn.execute(
                "SELECT core_business_summary, extracted_facts_json, "
                "cleaned_presentation_text, cleaned_transcript_text, latest_news_text "
                "FROM company_intelligence WHERE UPPER(symbol) = UPPER(?)",
                [symbol],
            ).fetchone()

            sensitivities, competitors, drivers = [], [], []
            core_model, pres_txt, trans_txt, news_txt = "", "", "", ""
            if ci_row:
                core_model = ci_row[0] or ""
                pres_txt = ci_row[2] or ""
                trans_txt = ci_row[3] or ""
                news_txt = ci_row[4] or ""
                if ci_row[1]:
                    try:
                        parsed = json.loads(ci_row[1])
                        sensitivities = parsed.get("raw_materials_and_macro_sensitivities", [])
                        competitors = parsed.get("explicitly_named_competitors", [])
                        for s in parsed.get("segment_facts", []):
                            if s.get("reported_driver"):
                                drivers.append(
                                    f"[{s.get('fact_id', '')}] {s.get('segment_name', '')}: {s.get('reported_driver', '')}"
                                )
                    except Exception:
                        pass

            return {
                "core_business_model": core_model,
                "facts": facts,
                "segment_drivers": drivers,
                "events": events,
                "guidance": guidance,
                "macro_sensitivities": sensitivities,
                "named_competitors": competitors,
                "pres_txt": pres_txt,
                "trans_txt": trans_txt,
                "news_txt": news_txt,
            }
        finally:
            conn.close()

    def _ensure_corpus_idf(self, conn: duckdb.DuckDBPyConnection):
        if self._idf_cache:
            return
        rows = conn.execute(
            "SELECT cleaned_presentation_text, cleaned_transcript_text FROM company_intelligence"
        ).fetchall()
        n_docs = max(len(rows), 1)
        df_counts = Counter()
        for pres, trans in rows:
            words = set(re.findall(r"[a-z0-9]{3,}", f"{pres or ''} {trans or ''}".lower()))
            for w in words:
                df_counts[w] += 1
        self._idf_cache = {
            w: math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5))
            for w, c in df_counts.items()
        }

    def search_filing_evidence_bm25(self, symbol: str, query: str, top_k: int = 5) -> List[str]:
        conn = self._get_conn(read_only=True)
        try:
            self._ensure_corpus_idf(conn)
            row = conn.execute(
                "SELECT cleaned_presentation_text, cleaned_transcript_text "
                "FROM company_intelligence WHERE UPPER(symbol) = UPPER(?)",
                [symbol],
            ).fetchone()
            if not row:
                return []

            chunks = []
            q_filter = re.compile(r"(\?|how do you see|how do you think|can you share|my question is)", re.I)
            for src_prefix, text in [("Deck", row[0] or ""), ("Concall", row[1] or "")]:
                unwrapped = (text or "").replace("\r\n", "\n").replace("\r", "\n")
                unwrapped = re.sub(r"(?<!\n)\n(?!\n)", " ", unwrapped)
                unwrapped = re.sub(r"\b(Rs|Cr|Mr|Dr|Ms|vs)\.\s+", r"\1_DOT_ ", unwrapped, flags=re.I)
                for raw_sent in re.split(r"(?<=[\.\!])\s+(?=[A-Z0-9])|\n{2,}", unwrapped):
                    clean_s = raw_sent.replace("_DOT_ ", ". ")
                    clean_s = re.sub(r"\s*\|\s*", " ", clean_s)
                    clean_s = re.sub(r"\s+", " ", clean_s).strip(" -*|")
                    if 45 <= len(clean_s) <= 320 and not q_filter.search(clean_s):
                        chunks.append((src_prefix, clean_s))

            if not chunks:
                return []

            q_terms = re.findall(r"[a-z0-9]{3,}", query.lower())
            if not q_terms:
                q_terms = ["growth", "margin", "order", "guidance", "demand", "capex", "cash", "debt"]

            avg_dl = sum(len(c[1].split()) for c in chunks) / len(chunks)
            k1, b = 1.5, 0.75
            scored = []
            for src_prefix, sent in chunks:
                tokens = re.findall(r"[a-z0-9]{3,}", sent.lower())
                if not tokens:
                    continue
                tf_map = Counter(tokens)
                dl = len(tokens)
                score = 0.0
                for term in q_terms:
                    if term in tf_map:
                        tf = tf_map[term]
                        idf = self._idf_cache.get(term, 2.5)
                        score += idf * ((tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avg_dl))))
                if re.search(r"\d+(\.\d+)?%|\u20b9|\bcr\b|\bbn\b", sent, re.I):
                    score *= 1.25
                if score > 0:
                    scored.append((score, f"[{src_prefix}] {sent}"))

            scored.sort(key=lambda x: x[0], reverse=True)
            dedup, seen = [], set()
            for _, snip in scored:
                key = snip.lower()[:55]
                if key not in seen:
                    seen.add(key)
                    dedup.append(snip)
                if len(dedup) >= top_k:
                    break
            return dedup
        finally:
            conn.close()

    def screen_stocks_sql(
        self,
        signal_tag: Optional[str] = None,
        sector: Optional[str] = None,
        regime: Optional[str] = None,
        max_pe: Optional[float] = None,
        min_roe: Optional[float] = None,
        limit: int = 15,
    ) -> pd.DataFrame:
        conn = self._get_conn(read_only=True)
        try:
            clauses = ["1=1"]
            params = []
            if signal_tag:
                clauses.append("UPPER(quant_signal_tag) LIKE UPPER(?)")
                params.append(f"%{signal_tag}%")
            if sector:
                clauses.append("UPPER(sector) LIKE UPPER(?)")
                params.append(f"%{sector}%")
            if regime:
                clauses.append("UPPER(technical_regime) LIKE UPPER(?)")
                params.append(f"%{regime}%")
            if max_pe is not None:
                clauses.append("pe_ratio > 0 AND pe_ratio <= ?")
                params.append(max_pe)
            if min_roe is not None:
                clauses.append("roe_pct >= ?")
                params.append(min_roe)

            where_sql = " AND ".join(clauses)
            sql = (
                "SELECT market_cap_rank AS rank, symbol, sector, "
                "ROUND(current_price, 1) AS price, "
                "ROUND(dcf_intrinsic_value, 1) AS dcf_iv, "
                "ROUND(reverse_dcf_implied_growth_pct, 1) AS rev_dcf_g, "
                "ROUND(croic_pct, 1) AS croic, "
                "ROUND(fcf_to_debt_payback_years, 1) AS debt_yrs, "
                "fundamental_composite_score AS fund, "
                "technical_composite_score AS tech, "
                "quant_conviction_score AS conviction, "
                "cash_debt_regime_tag, technical_regime "
                f"FROM stock_quant_signals WHERE {where_sql} "
                f"ORDER BY quant_conviction_score DESC LIMIT {int(limit)}"
            )
            return conn.execute(sql, params).df()
        finally:
            conn.close()


# ==========================================================
# 3. COMPACT CONTEXT PACKET BUILDER & ZERO-LEAKAGE PROMPT
# ==========================================================
def build_compact_analyst_packet(tools: EquitiesAnalystTools, symbol: str, user_query: str = "") -> Tuple[str, dict]:
    q = tools.get_quant_profile(symbol)
    if not q:
        raise ValueError(f"Symbol '{symbol}' not found in `stock_quant_signals`.")

    feg = tools.get_facts_events_guidance(symbol)
    sql_peers = tools.get_sql_industry_peers(symbol, q.get("industry") or "", q.get("sector") or "", limit=4)
    macro_intel = tools.extract_competitive_and_macro_intelligence(
        symbol, q.get("sector") or "", q.get("industry") or "",
        feg["pres_txt"], feg["trans_txt"], feg["news_txt"]
    )
    snippets = tools.search_filing_evidence_bm25(
        symbol,
        user_query or "growth margin order book guidance capex cash debt market share subsidiary export commodity",
        top_k=6,
    )

    fact_lines = [
        f"  [{f['span_id']}] ({f['period_id']}) {f['scope']} | {f['metric_id']}: {f['value_str']} ({f['change_str']}) [src={f['source_doc']}]"
        for f in feg["facts"][:15]
    ]
    event_lines = [
        f"  [{e['span_id']}] [{e['event_date']}] ({e['fiscal_period']}) {e['event_type']}: {e['headline']}"
        for e in feg["events"]
    ]
    guid_lines = [
        f"  [{g['span_id']}] ({g['period_id']}) [{g['modality']}]: {g['statement']}"
        for g in feg["guidance"]
    ]

    all_sensitivities = list(dict.fromkeys(feg["macro_sensitivities"] + macro_intel["sector_macro_sensitivities"]))
    val_z = q.get("val_zscore_sector") or 0.0
    roe_z = q.get("roe_zscore_sector") or 0.0
    mcap_cr = q.get("market_cap_cr") or 0.0

    lines = [
        f"=== 1. QUANTITATIVE, DCF, CASH-RUNWAY & OWNERSHIP PROFILE: {q['company_name']} (NSE: {q['symbol']}) ===",
        f"- Market Cap Rank: #{q['market_cap_rank']} (\u20b9{mcap_cr:,.0f} Cr | {q['market_cap_category']}) | Sector: {q['sector']} ({q['industry']})",
        f"- Current Price: \u20b9{q['current_price']} | Signal Tag: {q['quant_signal_tag']} | Overall Quant Conviction: {q['quant_conviction_score']} / 100",
        f"- Factor Scores (0-100): Fundamental={q['fundamental_composite_score']} (Quality={q['quality_percentile']}%, Growth={q['growth_percentile']}%, Cash Sustainability={q['cash_sustainability_percentile']}%, Value={q['value_percentile']}%, Ownership={q['ownership_percentile']}%) | Technical={q['technical_composite_score']} | Filing/Catalyst={q['filing_catalyst_score']}",
        f"- Shareholding & Institutional Interest: Promoter/Insider Holding={q.get('promoter_holding_pct')}% | Institutional (FII + DII) Holding={q.get('fii_dii_institutional_pct')}% (Ownership Percentile={q['ownership_percentile']}%)",
        f"- DCF & Cash-Flow Sustainability: DCF Intrinsic Value=\u20b9{q['dcf_intrinsic_value']} (Margin of Safety={q['dcf_margin_of_safety_pct']}%) | Reverse DCF Implied 5Y Growth={q['reverse_dcf_implied_growth_pct']}% (Growth Cushion={q['dcf_growth_cushion_pct']}%) | CAPM Cost of Equity (Ke)={q['cost_of_equity_ke_pct']}% | CROIC={q['croic_pct']}% | Reinvestment/Retention Efficiency={q['reinvestment_efficiency_pct']}% | FCF Debt Payback={q['fcf_to_debt_payback_years']} Years | Seasonal WC Buffer={q['seasonal_wc_buffer_ratio']}x ({q['sector_seasonal_cycle_tag']}) | Cash Regime={q['cash_debt_regime_tag']}",
        f"- Valuation vs Sector: P/E={q['pe_ratio']}x (Forward P/E={q['forward_pe']}x, Sector Median={q['sector_pe_median']}x) | P/B={q['pb_ratio']}x (Sector={q['sector_pb_median']}x) | EV/EBITDA={q['ev_to_ebitda']}x | FCF Yield={q['fcf_yield_pct']}% | Valuation Sector Z-Score={val_z:+.2f}",
        f"- Profitability & Balance Sheet: ROE={q['roe_pct']}% (Sector Median={q['sector_roe_median']}%, Z={roe_z:+.2f}) | ROA={q['roa_pct']}% | Operating Margin={q['operating_margin_pct']}% | Net Profit Margin={q['net_profit_margin_pct']}% | Debt/Equity={q['debt_to_equity']}% | OCF/PAT={q['ocf_to_pat_ratio']}x",
        f"- Growth Rates: TTM Revenue Growth={q['revenue_growth_pct']}% | Earnings Growth={q['earnings_growth_pct']}% | Median Filing YoY Growth={q['extracted_yoy_growth_median_pct']}%",
        f"- Technical & Risk Stats: Regime={q['technical_regime']} | Returns: 1M={q['ret_1m_pct']}%, 3M={q['ret_3m_pct']}%, 6M={q['ret_6m_pct']}%, 12M={q['ret_12m_pct']}% | 6M Sharpe Momentum={q['sharpe_momentum_6m']} | Sortino(1Y)={q['sortino_ratio_1y']} | OLS Alpha(1Y)={q['alpha_252d_ann_pct']}% | Beta={q['beta_252d']} | RSI(14)={q['rsi_14']} | Vol Squeeze Pct={q['vol_squeeze_percentile']}% | Up/Down Vol Ratio(20D)={q['up_down_vol_ratio_20d']} | Hurst(126D)={q['hurst_exponent_126d']} | Dist 50DMA={q['dist_from_50dma_pct']}% | Dist 200DMA={q['dist_from_200dma_pct']}% | Dist 52W High={q['dist_from_52w_high_pct']}% | Max DD(1Y)={q['max_drawdown_1y_pct']}% | 95% Daily CVaR={q['cvar_95_daily_pct']}%",
        "",
        "=== 2. DETERMINISTIC FILING FACTS ([Fxxx]) ===",
        "\n".join(fact_lines) if fact_lines else "  None",
        "",
        "=== 3. REPORTED SEGMENT DRIVERS ===",
        "\n".join("  - " + d for d in feg["segment_drivers"]) if feg["segment_drivers"] else "  None",
        "",
        "=== 4. RECENT CORPORATE EVENTS & NEWS ([Nxx]) ===",
        "\n".join(event_lines) if event_lines else "  None",
        "",
        "=== 5. MODALITY-TAGGED MANAGEMENT GUIDANCE ([Gxx]) ===",
        "\n".join(guid_lines) if guid_lines else "  None",
        "",
        "=== 6. BM25 RETRIEVED FILING EVIDENCE SNIPPETS ===",
        "\n".join("  - " + s for s in snippets) if snippets else "  None",
        "",
        "=== 7. MACRO, COMMODITY/FX, SUBSIDIARY & COMPETITIVE CONTEXT ===",
        "- Live Sector Commodity & FX Benchmarks: " + (" | ".join(macro_intel["commodity_fx_benchmarks"]) or "Standard INR Macro"),
        "- Sector Macro & Input Sensitivities: " + "; ".join(all_sensitivities),
        "- Same-Industry Quantitative Peer Benchmarks: " + ("; ".join(sql_peers) or "None"),
        "- Filing Subsidiary / JV / Associate Mentions: " + (" || ".join(macro_intel["subsidiaries_and_jv_notes"]) or "None explicitly highlighted in top snippets"),
        "- Filing Market-Share / Unorganized-to-Organized Shift Mentions: " + (" || ".join(macro_intel["unorganized_sector_and_market_share_notes"]) or "None explicitly highlighted in top snippets"),
        "- Filing Geopolitical / Export / Regulatory Mentions: " + (" || ".join(macro_intel["geopolitical_and_commodity_filing_notes"]) or "None explicitly highlighted in top snippets"),
    ]
    return "\n".join(lines), q


def generate_and_log_stock_verdict(
    symbol: str,
    user_query: str = "Provide a comprehensive institutional DCF, cash-flow sustainability, fundamental, technical, macro/commodity, and catalyst assessment.",
    model_name: str = ANALYST_MODEL,
    log_prediction: bool = True,
) -> Tuple[str, StructuredAnalystVerdict]:
    tools = EquitiesAnalystTools()
    packet, q = build_compact_analyst_packet(tools, symbol, user_query)

    prompt_lines = [
        f"You are a Senior Institutional Quant & Fundamental Equity Analyst covering the NSE 750 universe.",
        f"Analyze the deterministic Evidence Packet below for {q['company_name']} ({q['symbol']}) and answer: \"{user_query}\"",
        "",
        packet,
        "",
        "MANDATORY ANALYST RULES (ZERO HALLUCINATION & ZERO LEAKAGE):",
        "1. GROUNDED NUMBERS ONLY: Never invent or recalculate numbers. Cite exact figures from Section 1 (DCF Intrinsic Value, Reverse DCF Implied Growth, CROIC %, FCF Debt Payback Years, Seasonal WC Buffer, Promoter & Institutional Holding %) and cite [Fxxx], [Nxx], and [Gxx] IDs from Sections 2-5.",
        "2. MACRO, SUBSIDIARIES, INSTITUTIONAL OWNERSHIP & INFORMAL COMPETITION: In `macro_geopolitical_and_competitive_view`, evaluate:",
        "   - How the live Commodity & FX trends in Section 7 affect input costs, realizations, or margins for this specific industry.",
        "   - How Promoter and Institutional (FII + DII) ownership levels and any capital-raise events in [Nxx] reflect institutional conviction.",
        "   - Any subsidiary/JV contributions, export/geopolitical headwinds, or market-share shifts between organized and unorganized/informal industry players evidenced in Sections 4, 6, and 7, plus valuation comparison against the Same-Industry Quantitative Peer Benchmarks.",
        "3. CONCISE DENSITY & COMPLETE ARRAYS: Keep each text field to 2-3 tight, high-signal sentences (max 65 words per field) so the JSON completes cleanly. Populate `key_risks` with 2-4 specific risks and list every cited ID in `cited_ids`.",
        "",
        "Return ONLY a valid JSON object with these exact keys:",
        "{",
        '  "directional_call": "BULLISH_ACCUMULATE | NEUTRAL_HOLD | BEARISH_AVOID",',
        '  "horizon": "1Q-2Q (3-6 Months)",',
        '  "target_fundamental_metric": "string",',
        '  "executive_thesis": "string",',
        '  "dcf_and_cash_sustainability_view": "string",',
        '  "fundamental_and_valuation_view": "string",',
        '  "technical_and_regime_view": "string",',
        '  "catalysts_and_guidance_view": "string",',
        '  "macro_geopolitical_and_competitive_view": "string",',
        '  "key_risks": ["string"],',
        '  "cited_ids": ["string"]',
        "}",
    ]
    prompt = "\n".join(prompt_lines)

    res = None
    for attempt in range(2):
        try:
            res = httpx.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "think": False,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2200},
                },
                timeout=240.0,
            )
            res.raise_for_status()
            break
        except Exception as e:
            if attempt == 0:
                time.sleep(2.0)
            else:
                raise e

    raw_content = extract_clean_json_object(res.json().get("message", {}).get("content", ""))
    verdict = StructuredAnalystVerdict.model_validate_json(raw_content)

    if log_prediction:
        conn = duckdb.connect(str(DB_PATH))
        try:
            init_predictions_schema(conn)
            pred_id = f"PRED-{symbol.upper()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)",
                [
                    pred_id,
                    q["ticker"],
                    q["symbol"],
                    datetime.now().strftime("%Y-%m-%d"),
                    float(q["current_price"] or 0.0),
                    verdict.horizon,
                    verdict.directional_call,
                    verdict.target_fundamental_metric,
                    float(q["quant_conviction_score"] or 0.0),
                    ", ".join(verdict.cited_ids),
                    verdict.executive_thesis,
                    model_name,
                ],
            )
            conn.execute("CHECKPOINT")
        finally:
            conn.close()

    return packet, verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE 750 Tool-Calling Quant + DCF + Filing Analyst & Prediction Logger")
    parser.add_argument("--symbol", type=str, default="BAJAJ-AUTO", help="NSE symbol to analyze")
    parser.add_argument(
        "--query",
        type=str,
        default="Evaluate the DCF intrinsic value, cash-flow sustainability, technical regime, macro/commodity/competitive dynamics, and key filing catalysts.",
        help="Custom question",
    )
    parser.add_argument("--screen", type=str, default=None, help="Optional signal_tag or regime filter to screen")
    args = parser.parse_args()

    tools = EquitiesAnalystTools()
    if args.screen:
        is_reg = "SQUEEZE" in args.screen or "UPTREND" in args.screen
        df_s = tools.screen_stocks_sql(
            regime=args.screen if is_reg else None,
            signal_tag=None if is_reg else args.screen,
            limit=20,
        )
        print(f"\n=== SQL SCREEN RESULTS FOR: {args.screen} ===")
        print(df_s.to_string(index=False))
    else:
        pkt, v = generate_and_log_stock_verdict(args.symbol, args.query)
        print(pkt)
        print("\n=== STRUCTURED ANALYST VERDICT (LOGGED TO `predictions` TABLE) ===")
        print(json.dumps(v.model_dump(), indent=2, ensure_ascii=False))