import argparse
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, List, Literal

import duckdb
import httpx
import pymupdf as fitz
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from tqdm import tqdm

# ==========================================================
# IMPORT DETERMINISTIC ENGINES FROM NORMALIZER.PY
# ==========================================================
from normalizer import (
    JUNK_NEWS_PAT,
    extract_guidance_catalog,
    extract_news_catalog,
    extract_typed_facts_from_Sources,
    format_guidance_catalog,
    format_news_catalog,
    format_typed_fact_catalog,
)

# ==========================================================
# 0. LOAD CONFIG.YAML & SKILLS.YAML SAFELY
# ==========================================================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
SKILLS_PATH = BASE_DIR / "skills.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

if not SKILLS_PATH.exists():
    raise FileNotFoundError(f"Missing skills file: {SKILLS_PATH}. Please save skills.yaml in {BASE_DIR}")
with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    skills_cfg = yaml.safe_load(f)
if not skills_cfg or "sector_archetypes" not in skills_cfg:
    raise ValueError(f"{SKILLS_PATH} is empty or missing 'sector_archetypes'. Save the skills.yaml content first!")

log_cfg = config.get("logging", {})
logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO")),
    format=log_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DB_CONFIG = next(db for db in config["databases"] if db["name"] == "nse_universe")
DB_PATH = BASE_DIR / DB_CONFIG["path"]
DEFAULT_TOP_N = DB_CONFIG.get("top_n", 750)
MAX_RETRIES = DB_CONFIG.get("max_retries", 3)

EXT_CFG = config["intelligence_extraction"]
MAX_PRESENTATION_SLIDES = EXT_CFG["max_presentation_slides"]
MAX_PRESENTATION_CHARS = EXT_CFG["max_presentation_chars"]
MIN_SLIDE_CHARS = EXT_CFG["min_slide_chars"]
MIN_SLIDE_SCORE = EXT_CFG["min_slide_score"]
MAX_TRANSCRIPT_CHARS = EXT_CFG["max_transcript_chars"]
MIN_PARAGRAPH_CHARS = EXT_CFG["min_paragraph_chars"]
MAX_NEWS_ITEMS = EXT_CFG["max_news_items"]
PROMPT_PRES_CAP = EXT_CFG.get("prompt_pres_char_cap", 6000)
PROMPT_TRANS_CAP = EXT_CFG.get("prompt_trans_char_cap", 9000)
REQUEST_TIMEOUT = EXT_CFG["request_timeout"]
NSE_SLEEP = EXT_CFG["nse_rate_limit_sleep"]
ERROR_SLEEP = EXT_CFG["error_backoff_sleep"]

LLM_CFG = config["llm"]
OLLAMA_URL = LLM_CFG["ollama_url"]
EXTRACTOR_MODEL = LLM_CFG.get("extractor_model") or LLM_CFG.get("default_model", "nemotron-3-nano:4b")
ROUTER_MODEL = LLM_CFG.get("router_model", "qwen3.5:4b")
JUDGE_MODEL = LLM_CFG.get("judge_model", "qwen3.5:9b")
LLM_TEMP = LLM_CFG.get("temperature", 0.0)
LLM_CTX = LLM_CFG.get("num_ctx", 8192)
LLM_NUM_PREDICT = LLM_CFG.get("num_predict", 2500)
LLM_TIMEOUT = LLM_CFG.get("timeout", 180.0)
MAX_JSON_RETRIES = LLM_CFG.get("max_json_retries", 2)

SECTOR_ARCHETYPES = skills_cfg["sector_archetypes"]
SECTOR_EXTRACTION_RULES = {k: v["rules"].strip() for k, v in SECTOR_ARCHETYPES.items()}
SECTOR_NSE_KEYWORDS = {k: [w.lower() for w in v.get("nse_keywords", [])] for k, v in SECTOR_ARCHETYPES.items()}
SECTOR_REGEX_PATTERNS = {
    k: re.compile(v["regex"], re.IGNORECASE)
    for k, v in SECTOR_ARCHETYPES.items()
    if v.get("regex")
}
EXTRACTOR_PROMPT_TEMPLATE = skills_cfg["prompts"]["extractor_prompt"]
JUDGE_PROMPT_TEMPLATE = skills_cfg["prompts"]["judge_prompt"]


def resolve_sync_week() -> str:
    cfg_week = DB_CONFIG.get("sync_week")
    if cfg_week and str(cfg_week).strip().lower() != "auto":
        return str(cfg_week).strip()
    return datetime.now().strftime("%Y-W%V")


# ==========================================================
# 1. PYDANTIC SCHEMA & DETERMINISTIC FACT/NEWS BINDER
# ==========================================================
VALID_EVENT_TYPES = {
    "CAPEX", "PARTNERSHIP_JV", "POLICY_REGULATION",
    "PRICING_ACTION", "ORDER_BOOK", "OPERATIONAL_MILESTONE", "RISK_FACTOR"
}


class SegmentFact(BaseModel):
    fact_id: str = ""
    segment_name: str = "Consolidated"
    financial_performance: str = "Bound by Python"
    reported_driver: str = "Not explicitly reported"

    @model_validator(mode="before")
    @classmethod
    def coerce_segment_dict(cls, data: Any) -> dict:
        if isinstance(data, str):
            m_fid = re.search(r"\b(F\d{3})\b", data.upper())
            return {
                "fact_id": m_fid.group(1) if m_fid else "",
                "segment_name": "Consolidated",
                "financial_performance": data,
                "reported_driver": ""
            }
        if not isinstance(data, dict):
            return {}
        raw_fid = str(data.get("fact_id") or data.get("id") or "").strip().upper()
        m_fid = re.search(r"(F\d{3})", raw_fid)
        return {
            "fact_id": m_fid.group(1) if m_fid else raw_fid,
            "segment_name": str(data.get("segment_name") or data.get("segment") or data.get("name") or "Consolidated"),
            "financial_performance": str(
                data.get("financial_performance")
                or data.get("revenue_or_margin_trend")
                or data.get("trend")
                or data.get("performance")
                or ""
            ),
            "reported_driver": str(data.get("reported_driver") or data.get("driver") or data.get("drivers") or "")
        }


class ExtractedCatalyst(BaseModel):
    news_id: str = ""
    event_type: Literal[
        "CAPEX", "PARTNERSHIP_JV", "POLICY_REGULATION",
        "PRICING_ACTION", "ORDER_BOOK", "OPERATIONAL_MILESTONE", "RISK_FACTOR"
    ] = "OPERATIONAL_MILESTONE"
    description: str = ""
    timeline: str = "Reported Quarter"

    @model_validator(mode="before")
    @classmethod
    def coerce_catalyst_dict(cls, data: Any) -> dict:
        if isinstance(data, str):
            m_nid = re.search(r"\b(N\d{2})\b", data.upper())
            return {
                "news_id": m_nid.group(1) if m_nid else "",
                "event_type": "OPERATIONAL_MILESTONE",
                "description": data,
                "timeline": "Reported Quarter"
            }
        if not isinstance(data, dict):
            return {}
        raw_nid = str(data.get("news_id") or data.get("id") or "").strip().upper()
        m_nid = re.search(r"(N\d{2})", raw_nid)
        return {
            "news_id": m_nid.group(1) if m_nid else "",
            "event_type": data.get("event_type") or data.get("type") or data.get("category") or "OPERATIONAL_MILESTONE",
            "description": str(data.get("description") or data.get("event") or data.get("detail") or ""),
            "timeline": str(data.get("timeline") or data.get("date") or data.get("period") or "Reported Quarter")
        }

    @field_validator("event_type", mode="before")
    @classmethod
    def normalize_event_type(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "OPERATIONAL_MILESTONE"
        u = v.strip().upper()
        if u in VALID_EVENT_TYPES:
            return u
        if any(k in u for k in ("PRICE", "PRICING", "TARIFF", "RATE", "YIELD", "BILLING")):
            return "PRICING_ACTION"
        if any(k in u for k in ("CAPEX", "EXPANSION", "CAPACITY", "PLANT", "INFRA", "DATA CENTER", "BED", "STORE")):
            return "CAPEX"
        if any(k in u for k in ("JV", "PARTNER", "ACQUISITION", "MERGER", "ALLIANCE", "STAKE", "IPO", "LISTING", "CDMO", "RIGHTS")):
            return "PARTNERSHIP_JV"
        if any(k in u for k in ("POLICY", "REGULAT", "PLI", "GOV", "RBI", "SEBI", "TRAI", "USFDA", "483", "APPROVAL", "NLEM")):
            return "POLICY_REGULATION"
        if any(k in u for k in ("RISK", "HEADWIND", "LITIGATION", "ATTRITION", "CHURN", "NPA", "SLIPPAGE", "EROSION")):
            return "RISK_FACTOR"
        if any(k in u for k in ("ORDER", "TCV", "ACV", "BACKLOG", "CONTRACT", "DEAL", "BOOKING")):
            return "ORDER_BOOK"
        return "OPERATIONAL_MILESTONE"


class MasterCompanyFacts(BaseModel):
    core_business_model: str = ""
    segment_facts: List[SegmentFact] = Field(default_factory=list)
    key_catalysts_and_events: List[ExtractedCatalyst] = Field(default_factory=list)
    raw_materials_and_macro_sensitivities: List[str] = Field(default_factory=list)
    explicitly_named_competitors: List[str] = Field(default_factory=list)
    management_numerical_guidance: str = ""
    factual_news_digest: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_keys_and_types(cls, data: Any) -> dict:
        if not isinstance(data, dict):
            return {}

        key_aliases = {
            "core_business_model": ["business_model", "core_business", "summary", "company_overview"],
            "segment_facts": ["segments", "segment_performance", "business_segments"],
            "key_catalysts_and_events": ["catalysts", "key_catalysts", "events", "deal_wins", "contracts_and_events"],
            "raw_materials_and_macro_sensitivities": [
                "raw_materials", "macro_sensitivities", "raw_materials_and_macro",
                "sensitivities", "macro_and_raw_materials", "cost_drivers",
                "operational_sensitivities", "key_sensitivities", "talent_and_fx_sensitivities",
                "input_costs_and_macro_sensitivities", "risk_and_macro_sensitivities"
            ],
            "explicitly_named_competitors": ["competitors", "named_competitors", "peers"],
            "management_numerical_guidance": ["management_guidance", "numerical_guidance", "guidance", "outlook"],
            "factual_news_digest": ["news_digest", "recent_news", "news", "factual_news"]
        }
        for canonical, aliases in key_aliases.items():
            if canonical not in data or data.get(canonical) in (None, "", []):
                for alias in aliases:
                    if alias in data and data[alias] not in (None, "", []):
                        data[canonical] = data.pop(alias)
                        break

        str_list_fields = (
            "raw_materials_and_macro_sensitivities",
            "explicitly_named_competitors",
            "factual_news_digest"
        )
        for field_name in str_list_fields:
            val = data.get(field_name)
            if val is None:
                data[field_name] = []
            elif isinstance(val, str):
                data[field_name] = [s.strip() for s in val.split(";") if s.strip()] if val.strip() else []
            elif isinstance(val, list):
                data[field_name] = [
                    json.dumps(item) if isinstance(item, dict) else str(item).strip()
                    for item in val if item is not None and str(item).strip()
                ]

        bad_comp = {"none", "n/a", "not mentioned", "nearest competitor", "competitors", "peers", "null"}
        data["explicitly_named_competitors"] = [
            c for c in data.get("explicitly_named_competitors", [])
            if c.lower().strip() not in bad_comp
        ]

        for obj_list_field in ("segment_facts", "key_catalysts_and_events"):
            val = data.get(obj_list_field)
            if val is None:
                data[obj_list_field] = []
            elif isinstance(val, dict):
                data[obj_list_field] = [val]
            elif not isinstance(val, list):
                data[obj_list_field] = []

        for str_field in ("core_business_model", "management_numerical_guidance"):
            val = data.get(str_field)
            if val is None:
                data[str_field] = "Not explicitly stated in filing."
            elif isinstance(val, list):
                data[str_field] = "; ".join(str(x) for x in val if x)
            elif isinstance(val, dict):
                data[str_field] = "; ".join(f"{k}: {v}" for k, v in val.items())
            else:
                data[str_field] = str(val)

        return data

    @model_validator(mode="after")
    def clean_and_deduplicate_post(self) -> "MasterCompanyFacts":
        placeholder_comp = re.compile(r"^(player|peer|competitor|bank|company|operator)\s*\d*[a-z]?$", re.IGNORECASE)
        self.explicitly_named_competitors = [
            c for c in self.explicitly_named_competitors
            if not placeholder_comp.match(c.strip())
        ]

        junk_sensitivity_pat = re.compile(
            r"^(revenue_from_operations|raw_material_cost|employee_benefit_expense|finance_costs|"
            r"depreciation|profit_before_tax|net_profit_pat|basic_eps|derived_|total_income|"
            r"interest_earned|interest earned|net profit pat|employee benefit expense|"
            r"growth \d+\s*basis points|.*incremental revenue:|.*taken from trai|.*wireless matrix|"
            r".*\bq[1-4]\s+(pat|profit|results)\s+(rises|jumps|drops|falls)|.*capex for the quarter)",
            re.IGNORECASE
        )
        dedup_sens = []
        seen_sens = set()
        for s in self.raw_materials_and_macro_sensitivities:
            s_clean = s.strip()
            s_key = re.sub(r"\W+", " ", s_clean.lower()).strip()
            if (
                len(s_clean) >= 12
                and s_key not in seen_sens
                and not junk_sensitivity_pat.search(s_clean)
                and not re.search(r":\s*₹?\d[\d,]*\.\d+", s_clean)
                and not re.search(r"\b(share price|stock rises|stock falls|goldman sachs|morgan stanley|brokerage|upside|profit more than)\b", s_clean, re.I)
            ):
                seen_sens.add(s_key)
                dedup_sens.append(s_clean)
        self.raw_materials_and_macro_sensitivities = dedup_sens[:4]
        return self


def format_bound_fact_string(f: dict) -> str:
    chg_txt = f["change"]
    if chg_txt not in ("Slide Highlight", "Concall Highlight", "XBRL Reported", "Reported"):
        return f"[{f['id']}] {f['metric']}: {f['value']} ({chg_txt}, {f['period']})"
    return f"[{f['id']}] {f['metric']} — {f['value']} ({f['period']})"


def bind_and_ground_facts(
    validated: MasterCompanyFacts,
    typed_facts: List[dict],
    news_items: List[dict],
    guidance_items: List[dict],
    company_name: str,
    xbrl_dict: dict,
    pres_slice: str,
    trans_slice: str
) -> MasterCompanyFacts:
    """1. Binds segment_facts to [Fxxx]. 2. Binds events to [Nxx] by ID or word overlap. 3. Retains all N01..N08 news items."""
    fact_by_id = {f["id"]: f for f in typed_facts}
    fact_by_scope = {f["scope"].lower(): f for f in typed_facts}
    news_by_id = {n["id"]: n for n in news_items}
    default_period = typed_facts[0]["period"] if typed_facts else xbrl_dict.get("Period_End", "Q2 FY27")

    bound_segments = []
    seen_bound_ids = set()

    for seg in validated.segment_facts:
        matched_fact = None
        m_fid = re.search(r"(F\d{3})", f"{seg.fact_id} {seg.segment_name} {seg.financial_performance}".upper())
        if m_fid and m_fid.group(1) in fact_by_id:
            matched_fact = fact_by_id[m_fid.group(1)]
        elif seg.segment_name.lower().strip() in fact_by_scope:
            matched_fact = fact_by_scope[seg.segment_name.lower().strip()]

        if matched_fact and matched_fact["id"] not in seen_bound_ids:
            seen_bound_ids.add(matched_fact["id"])
            seg.fact_id = matched_fact["id"]
            seg.segment_name = matched_fact["scope"]
            seg.financial_performance = format_bound_fact_string(matched_fact)
            if not seg.reported_driver or seg.reported_driver == "Not explicitly reported":
                seg.reported_driver = f"Reported in {matched_fact['src']} ({matched_fact['period']})"
            bound_segments.append(seg)

    if len(bound_segments) < 3 and typed_facts:
        for f in typed_facts:
            if f["id"] in seen_bound_ids:
                continue
            seen_bound_ids.add(f["id"])
            bound_segments.append(
                SegmentFact(
                    fact_id=f["id"],
                    segment_name=f["scope"],
                    financial_performance=format_bound_fact_string(f),
                    reported_driver=f"Reported in {f['src']} ({f['period']})"
                )
            )
            if len(bound_segments) >= 4:
                break

    validated.segment_facts = bound_segments[:5]

    # Bind events to [N01..N08] by news_id or keyword match so dates & headlines are 100% deterministic
    clean_events = []
    seen_event_keys = set()
    stale_csr_pat = re.compile(r"\b(vaccine|pandemic|covid|fy21|fy22|defexpo 2022)\b", re.I)

    for ev in validated.key_catalysts_and_events:
        if stale_csr_pat.search(ev.description) or stale_csr_pat.search(ev.timeline):
            continue
        matched_news = None
        m_nid = re.search(r"(N\d{2})", f"{ev.news_id} {ev.description} {ev.timeline}".upper())
        if m_nid and m_nid.group(1) in news_by_id:
            matched_news = news_by_id[m_nid.group(1)]
        else:
            desc_words = set(re.findall(r"[a-z0-9]{4,}", ev.description.lower())) - {
                "india", "limited", "crore", "quarter", "company", "shares", "stock"
            }
            for n in news_items:
                h_words = set(re.findall(r"[a-z0-9]{4,}", n["headline"].lower()))
                if len(desc_words & h_words) >= 2:
                    matched_news = n
                    break

        if matched_news:
            ev.news_id = matched_news["id"]
            ev.event_type = matched_news["event_type"]
            ev.description = f"[{matched_news['id']}] {matched_news['headline']}"
            ev.timeline = matched_news["timeline"]
        else:
            ev.news_id = ""
            if re.search(r"\d{4}-\d{2}-\d{2}", ev.timeline):
                ev.timeline = default_period

        ev_key = ev.description.lower()[:45]
        if ev_key not in seen_event_keys:
            seen_event_keys.add(ev_key)
            clean_events.append(ev)

    # Ensure top [Nxx] corporate events are bound into key_catalysts_and_events
    if len(clean_events) < 4 and news_items:
        for n in news_items:
            ev_desc = f"[{n['id']}] {n['headline']}"
            ev_key = ev_desc.lower()[:45]
            if ev_key in seen_event_keys:
                continue
            seen_event_keys.add(ev_key)
            clean_events.append(
                ExtractedCatalyst(
                    news_id=n["id"],
                    event_type=n["event_type"],
                    description=ev_desc,
                    timeline=n["timeline"]
                )
            )
            if len(clean_events) >= 5:
                break

    validated.key_catalysts_and_events = clean_events[:5]

    # Bind non-hedged guidance from [G01..G06] if LLM left guidance empty
    non_hedged = [g for g in guidance_items if g["modality"] == "GUIDANCE"]
    if (
        not validated.management_numerical_guidance
        or "not explicitly stated" in validated.management_numerical_guidance.lower()
    ) and non_hedged:
        validated.management_numerical_guidance = "; ".join(
            f"[{g['id']}] {g['text']}" for g in non_hedged[:2]
        )

    # Remove company's own name/subsidiaries from competitors
    filing_lower = f"{pres_slice}\n{trans_slice}".lower()
    comp_words = set(re.findall(r"[a-z]{4,}", company_name.lower())) - {"limited", "india", "corporation", "enterprises", "industries"}
    own_subsidiaries = {"bccl", "ccl", "secl", "mcl", "ncl", "wcl", "eccl", "cmpdil", "vecv", "caratlane", "jio", "nelco"}
    clean_comps = []
    for c in validated.explicitly_named_competitors:
        c_low = c.lower().strip()
        if c_low in own_subsidiaries or any(w in c_low for w in comp_words):
            continue
        if c_low in filing_lower:
            clean_comps.append(c)
    validated.explicitly_named_competitors = clean_comps[:5]

    # Keep all N01..N08 news items in factual_news_digest so Judge never flags N06..N08 as missing
    validated.factual_news_digest = [n["raw_line"] for n in news_items]
    return validated


# ==========================================================
# 2. DETERMINISTIC FINANCIAL EXTRACTOR (Y-ALIGNED SLIDE PARSER)
# ==========================================================
class DeterministicFinancialExtractor:
    LEGAL_BOILERPLATE = re.compile(
        r"(safe harbor|forward-looking statements|within the meaning of applicable securities laws|"
        r"actual results may differ materially|no obligation to publicly update|"
        r"for information purposes only|does not constitute an offer|strictly confidential)",
        re.IGNORECASE
    )

    NON_CONTENT_SLIDE = re.compile(
        r"(table of contents|sections and navigation|\bagenda\b|\bglossary\b|"
        r"thank you|for further information please contact|investor relations contact)",
        re.IGNORECASE
    )

    PLEASANTRIES = re.compile(
        r"^(thank you.*|good morning.*|good evening.*|congratulations on a (good|great|strong) set.*|"
        r"moderator:.*|operator:.*|ladies and gentlemen.*|i now hand the conference over.*)$",
        re.IGNORECASE | re.MULTILINE
    )

    CORE_FINANCIAL_KEYWORDS = re.compile(
        r"\b(margin|ebitda|ebit|pat|pbt|roce|roe|capex|guidance|outlook|utilization|capacity|"
        r"tailwind|headwind|pli|subsidy|export|import|duty|tariff|joint venture|jv|acquisition|"
        r"r&d|market share|raw material|realization|volume|bps|crore|cr|fy\d{2}|q[1-4]|yoy|qoq)\b",
        re.IGNORECASE
    )

    @classmethod
    def count_signal_hits(cls, text: str) -> int:
        hits = len(cls.CORE_FINANCIAL_KEYWORDS.findall(text))
        for pat in SECTOR_REGEX_PATTERNS.values():
            hits += len(pat.findall(text)) * 2
        return hits

    @classmethod
    def resolve_sector_archetypes(cls, nse_profile: dict, filing_text: str) -> List[str]:
        ind_str = " ".join([
            str(nse_profile.get("macro_sector", "")),
            str(nse_profile.get("sector", "")),
            str(nse_profile.get("industry", "")),
            str(nse_profile.get("basic_industry", ""))
        ]).lower()

        matched = []
        if ind_str.strip():
            for archetype, kw_list in SECTOR_NSE_KEYWORDS.items():
                if any(kw in ind_str for kw in kw_list):
                    matched.append(archetype)

        if matched and matched[0] in ("BFSI_BANK_NBFC", "INSURANCE", "CAPITAL_MARKETS_AMC", "TELECOM_DIGITAL"):
            return matched[:1]

        scores = {}
        for sector, pat in SECTOR_REGEX_PATTERNS.items():
            matches = len(pat.findall(filing_text))
            if matches >= 4:
                scores[sector] = matches
        for top_regex_sec, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]:
            if top_regex_sec not in matched:
                matched.append(top_regex_sec)

        return matched[:2] if matched else ["GENERAL_CORPORATE"]

    @staticmethod
    def parse_nse_xbrl_bytes(xml_bytes: bytes) -> dict:
        if not xml_bytes:
            return {}
        monetary_tags = {
            "RevenueFromOperations": "Revenue_From_Operations_Cr",
            "TotalIncome": "Total_Income_Cr",
            "ProfitBeforeTax": "Profit_Before_Tax_Cr",
            "ProfitLossForPeriod": "Net_Profit_PAT_Cr",
            "InterestEarned": "Interest_Earned_Cr",
            "InterestIncome": "Interest_Income_Cr",
            "NetInterestIncome": "Net_Interest_Income_Cr",
        }
        ratio_or_date_tags = {
            "BasicEarningsLossPerShareFromContinuingOperations": "Basic_EPS_INR",
            "PercentageOfGrossNPA": "Gross_NPA_Pct",
            "PercentageOfNetNPA": "Net_NPA_Pct",
            "CapitalAdequacyRatio": "Capital_Adequacy_Ratio_CRAR_Pct",
            "DateOfStartOfReportingPeriod": "Period_Start",
            "DateOfEndOfReportingPeriod": "Period_End"
        }
        contexts = {}
        global_dates = {}
        try:
            root = ET.fromstring(xml_bytes)
            for elem in root.iter():
                clean_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if not elem.text or not elem.text.strip():
                    continue
                ctx_ref = elem.attrib.get("contextRef", "default")
                ctx_bucket = contexts.setdefault(ctx_ref, {})
                val = elem.text.strip()

                if clean_tag in ("DateOfStartOfReportingPeriod", "DateOfEndOfReportingPeriod"):
                    out_k = ratio_or_date_tags[clean_tag]
                    if out_k not in global_dates or val > global_dates[out_k]:
                        global_dates[out_k] = val

                if clean_tag in monetary_tags:
                    out_key = monetary_tags[clean_tag]
                    if out_key not in ctx_bucket:
                        try:
                            num_val = float(val)
                            if num_val != 0.0:
                                if abs(num_val) >= 1e7:
                                    num_val = round(num_val / 1e7, 2)
                                ctx_bucket[out_key] = f"{num_val:,.2f}"
                                ctx_bucket[f"_{out_key}_raw"] = num_val
                        except ValueError:
                            pass
                elif clean_tag in ratio_or_date_tags:
                    out_key = ratio_or_date_tags[clean_tag]
                    if out_key not in ctx_bucket:
                        try:
                            num_val = float(val)
                            if num_val != 0.0:
                                ctx_bucket[out_key] = round(num_val, 2)
                        except ValueError:
                            ctx_bucket[out_key] = val

            valid_ctxs = [
                (c_id, data) for c_id, data in contexts.items()
                if any(k.endswith("_Cr") for k in data.keys())
            ]
            if not valid_ctxs:
                return {}

            valid_ctxs.sort(
                key=lambda item: (
                    str(item[1].get("Period_End") or global_dates.get("Period_End") or ""),
                    1 if any(tag in item[0] for tag in ("OneD", "OneI", "Current")) else 0
                ),
                reverse=True
            )
            best = {k: v for k, v in valid_ctxs[0][1].items() if not k.startswith("_")}
            for dk, dv in global_dates.items():
                best.setdefault(dk, dv)
            raw_best = valid_ctxs[0][1]

            period_end = str(best.get("Period_End", ""))
            if period_end and period_end < "2026-01-01":
                return {}

            rev = (
                raw_best.get("_Revenue_From_Operations_Cr_raw")
                or raw_best.get("_Interest_Earned_Cr_raw")
                or raw_best.get("_Interest_Income_Cr_raw")
            )
            pat = raw_best.get("_Net_Profit_PAT_Cr_raw")
            if isinstance(rev, (int, float)) and rev > 0 and isinstance(pat, (int, float)) and abs(pat) > 0:
                best["Derived_PAT_Margin_Pct"] = round((pat / rev) * 100, 2)

            return best
        except Exception as e:
            logger.debug(f"XBRL parse warning: {e}")
            return {}

    def clean_investor_presentation(self, pdf_bytes: bytes) -> str:
        if not pdf_bytes:
            return ""
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            raw_slides = []
            line_counts = Counter()
            num_pages = len(doc)

            for page_idx in range(num_pages):
                page = doc[page_idx]
                words = page.get_text("words")
                rows_by_y = defaultdict(list)
                for w in words:
                    y_bucket = int(round(w[1] / 4.0))
                    rows_by_y[y_bucket].append((w[0], str(w[4]).strip()))

                lines = []
                for y_bucket in sorted(rows_by_y.keys()):
                    row_words = sorted(rows_by_y[y_bucket], key=lambda item: item[0])
                    row_parts = []
                    prev_x = None
                    for x0, word_txt in row_words:
                        if not word_txt:
                            continue
                        if prev_x is not None and (x0 - prev_x) > 35:
                            row_parts.append("|")
                        row_parts.append(word_txt)
                        prev_x = x0 + (len(word_txt) * 4.5)
                    line_str = " ".join(row_parts).strip()
                    if line_str:
                        lines.append(line_str)

                for line in set(lines):
                    if len(line) < 85:
                        line_counts[line] += 1
                raw_slides.append((page_idx + 1, lines))

            repeat_threshold = max(3, int(num_pages * 0.25))
            repeating_headers = {l for l, c in line_counts.items() if c >= repeat_threshold}

            scored_slides = []
            for slide_num, lines in raw_slides:
                clean_lines = [l for l in lines if l not in repeating_headers]
                slide_text = "\n".join(clean_lines)

                if (
                    len(slide_text) < MIN_SLIDE_CHARS
                    or self.LEGAL_BOILERPLATE.search(slide_text)
                    or self.NON_CONTENT_SLIDE.search(slide_text)
                ):
                    continue

                kw_hits = self.count_signal_hits(slide_text)
                num_hits = len(re.findall(r"\b\d+(\.\d+)?%\b|\b\d{1,3}(,\d{2,3})+(\.\d+)?\b", slide_text))
                score = kw_hits + (num_hits * 2)

                if score >= MIN_SLIDE_SCORE:
                    scored_slides.append((score, slide_num, slide_text))

            scored_slides.sort(key=lambda x: x[0], reverse=True)
            selected_slides = []
            current_chars = 0
            for score, s_num, txt in scored_slides[:MAX_PRESENTATION_SLIDES]:
                chunk = f"[Slide {s_num}]\n{txt}"
                if current_chars + len(chunk) > PROMPT_PRES_CAP:
                    continue
                selected_slides.append((s_num, chunk))
                current_chars += len(chunk)

            selected_slides.sort(key=lambda x: x[0])
            return "\n\n".join(chunk for _, chunk in selected_slides)
        except Exception as e:
            logger.debug(f"Presentation clean warning: {e}")
            return ""
        finally:
            if doc is not None:
                doc.close()

    def clean_concall_transcript(self, pdf_bytes: bytes) -> str:
        if not pdf_bytes:
            return ""
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for idx, page in enumerate(doc):
                txt = page.get_text("text", sort=True)
                if idx == 0 and self.LEGAL_BOILERPLATE.search(txt):
                    continue
                pages.append(txt)

            raw_text = "\n".join(pages)
            cleaned = self.PLEASANTRIES.sub("", raw_text)
            cleaned = re.sub(r"\n\s*Page \d+ of \d+\s*\n", "\n", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\n([A-Z][a-zA-Z\s\.]{2,35}:)", r"\n\n\1", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

            paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) >= MIN_PARAGRAPH_CHARS]
            scored_paragraphs = []

            for idx, p in enumerate(paragraphs):
                if self.LEGAL_BOILERPLATE.search(p):
                    continue
                kw_score = self.count_signal_hits(p)
                pct_hits = len(re.findall(r"\b\d+(\.\d+)?%\b|\b\d+\s*bps\b", p, flags=re.IGNORECASE))
                total_score = kw_score + (pct_hits * 3)

                if total_score >= 3:
                    compact_p = re.sub(r"\s+", " ", p)
                    scored_paragraphs.append((total_score, idx, compact_p))

            scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
            selected = []
            current_chars = 0
            for score, idx, compact_p in scored_paragraphs:
                if current_chars + len(compact_p) > PROMPT_TRANS_CAP:
                    continue
                selected.append((idx, compact_p))
                current_chars += len(compact_p)

            selected.sort(key=lambda x: x[0])
            return "\n\n".join(p for _, p in selected)
        except Exception as e:
            logger.debug(f"Transcript clean warning: {e}")
            return ""
        finally:
            if doc is not None:
                doc.close()


# ==========================================================
# 3. NSE NETWORK CLIENT (WITH OFFICIAL INDUSTRY TAXONOMY)
# ==========================================================
class NSECorporateClient:
    LISTICLE_NOISE = re.compile(
        r"(stocks to (buy|watch|trade)|top (picks|gainers|losers)|share price live|"
        r"dividend.*stocks|d-st|sensex|nifty|market live|q[1-4] results today|"
        r"buy, sell or hold|brokerage radar|stock forecast|analyst prediction|price target|"
        r"stock price, news, quote and history|share price today|step-by-step guide|"
        r"outlook for the week|option chain)",
        re.IGNORECASE
    )

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
        }
        self.client = httpx.Client(headers=self.headers, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        self.refresh_cookies()

    def refresh_cookies(self):
        try:
            self.client.get("https://www.nseindia.com")
            time.sleep(0.4)
        except Exception:
            pass

    def fetch_nse_industry_profile(self, symbol: str) -> dict:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        try:
            res = self.client.get(url)
            if res.status_code in (401, 403):
                self.refresh_cookies()
                res = self.client.get(url)
            if res.status_code == 200:
                data = res.json()
                ind = data.get("industryInfo") or {}
                return {
                    "macro_sector": ind.get("macro", ""),
                    "sector": ind.get("sector", ""),
                    "industry": ind.get("industry", ""),
                    "basic_industry": ind.get("basicIndustry", "")
                }
        except Exception:
            pass
        return {}

    def get_filing_urls(self, symbol: str) -> dict:
        urls = {"transcript_url": None, "presentation_url": None, "outcome_url": None, "xbrl_url": None}
        ann_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"

        for _ in range(MAX_RETRIES):
            try:
                res = self.client.get(ann_url)
                if res.status_code in (401, 403):
                    self.refresh_cookies()
                    res = self.client.get(ann_url)
                if res.status_code == 200:
                    for item in res.json():
                        desc = (item.get("desc") or "").lower()
                        text = (item.get("attchmntText") or "").lower()
                        pdf_url = item.get("attchmntFile")
                        if not pdf_url or not pdf_url.endswith(".pdf"):
                            continue
                        if not urls["transcript_url"] and ("transcript" in desc or "transcript" in text or "concall" in text):
                            urls["transcript_url"] = pdf_url
                        elif not urls["presentation_url"] and ("presentation" in desc or "investor presentation" in text):
                            urls["presentation_url"] = pdf_url
                        elif not urls["outcome_url"] and ("outcome of board meeting" in desc or "financial results" in desc):
                            urls["outcome_url"] = pdf_url
                    break
            except Exception:
                time.sleep(0.5)

        xbrl_api = f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}&period=Quarterly"
        try:
            x_res = self.client.get(xbrl_api)
            if x_res.status_code == 200 and isinstance(x_res.json(), list):
                for row in x_res.json():
                    x_link = row.get("xbrl")
                    if x_link and x_link != "-" and x_link.endswith(".xml"):
                        urls["xbrl_url"] = x_link
                        break
        except Exception:
            pass

        return urls

    def fetch_bytes(self, url: str) -> bytes:
        if not url:
            return b""
        for _ in range(MAX_RETRIES):
            try:
                res = self.client.get(url)
                if res.status_code == 200:
                    return res.content
            except Exception:
                time.sleep(0.4)
        return b""

    def fetch_news_rss(self, symbol: str, company_name: str, limit: int = MAX_NEWS_ITEMS) -> str:
        short_name = re.sub(r"\b(limited|ltd\.?|india|corporation|corp\.?)\b", "", company_name, flags=re.IGNORECASE).strip()
        short_name = re.sub(r"\s+", " ", short_name)

        rss_url = "https://news.google.com/rss/search"
        query_term = f'("{short_name}" OR "{symbol}") when:90d' if len(short_name) >= 4 else f'"{symbol}" NSE when:90d'
        params = {"q": query_term, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}

        try:
            res = self.client.get(rss_url, params=params)
            if res.status_code != 200:
                return ""
            root = ET.fromstring(res.content)
            items = []

            patterns = [rf"\b{re.escape(symbol)}\b"]
            if len(short_name) >= 4:
                patterns.append(rf"\b{re.escape(short_name)}\b")
            must_match = re.compile("|".join(patterns), re.IGNORECASE)
            now_utc = datetime.now(timezone.utc)
            seen_stems = set()

            for item in root.findall(".//item"):
                raw_title = item.findtext("title", "")
                clean_title = raw_title.rsplit(" - ", 1)[0].strip()

                if not must_match.search(clean_title):
                    continue
                if self.LISTICLE_NOISE.search(clean_title) or clean_title.count(",") >= 3:
                    continue

                stem = re.sub(r"\W+", " ", clean_title.lower()).strip()[:40]
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)

                pub_raw = item.findtext("pubDate", "")
                date_label = pub_raw[:16]
                try:
                    pub_dt = parsedate_to_datetime(pub_raw)
                    if (now_utc - pub_dt).days > 120:
                        continue
                    date_label = pub_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

                items.append(f"[{date_label}] {clean_title}")
                if len(items) >= limit:
                    break
            return "\n".join(items)
        except Exception:
            return ""


# ==========================================================
# 4. POINT-IN-TIME RELATIONAL DATABASE SCHEMA & DUAL-WRITE
# ==========================================================
def init_intelligence_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS company_intelligence (
            ticker VARCHAR PRIMARY KEY,
            symbol VARCHAR,
            company_name VARCHAR,
            latest_transcript_url VARCHAR,
            latest_presentation_url VARCHAR,
            latest_xbrl_url VARCHAR,
            xbrl_numbers_json JSON,
            cleaned_presentation_text VARCHAR,
            cleaned_transcript_text VARCHAR,
            latest_news_text VARCHAR,
            core_business_summary VARCHAR,
            management_guidance VARCHAR,
            extracted_facts_json JSON,
            audit_flags_json JSON,
            sync_week VARCHAR,
            updated_at VARCHAR
        )
    """)
    cols = [c[0] for c in conn.execute("DESCRIBE company_intelligence").fetchall()]
    if "audit_flags_json" not in cols:
        conn.execute("ALTER TABLE company_intelligence ADD COLUMN audit_flags_json JSON")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            ticker VARCHAR,
            symbol VARCHAR,
            period_id VARCHAR,
            scope VARCHAR,
            metric_id VARCHAR,
            value_str VARCHAR,
            change_str VARCHAR,
            source_doc VARCHAR,
            span_id VARCHAR,
            as_of VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            ticker VARCHAR,
            symbol VARCHAR,
            event_date VARCHAR,
            fiscal_period VARCHAR,
            event_type VARCHAR,
            headline VARCHAR,
            span_id VARCHAR,
            as_of VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guidance (
            ticker VARCHAR,
            symbol VARCHAR,
            period_id VARCHAR,
            modality VARCHAR,
            statement VARCHAR,
            source_doc VARCHAR,
            span_id VARCHAR,
            as_of VARCHAR
        )
    """)


def persist_point_in_time_tables(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    symbol: str,
    typed_facts: List[dict],
    news_items: List[dict],
    guidance_items: List[dict],
    as_of_date: str
):
    conn.execute("DELETE FROM facts WHERE ticker = ? AND as_of = ?", [ticker, as_of_date])
    for f in typed_facts:
        conn.execute(
            "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [ticker, symbol, f["period"], f["scope"], f["metric"], f["value"], f["change"], f["src"], f["id"], as_of_date]
        )

    conn.execute("DELETE FROM events WHERE ticker = ? AND as_of = ?", [ticker, as_of_date])
    for n in news_items:
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [ticker, symbol, n["date"], n["fiscal_period"], n["event_type"], n["headline"], n["id"], as_of_date]
        )

    conn.execute("DELETE FROM guidance WHERE ticker = ? AND as_of = ?", [ticker, as_of_date])
    for g in guidance_items:
        conn.execute(
            "INSERT INTO guidance VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [ticker, symbol, g["period"], g["modality"], g["text"], g["src"], g["id"], as_of_date]
        )


def audit_extraction_grounding(
    facts_json_str: str,
    typed_fact_catalog: str,
    news_catalog: str,
    guidance_catalog: str,
    xbrl_str: str,
    pres_txt: str,
    trans_txt: str,
    news_txt: str
) -> dict:
    parsed = json.loads(facts_json_str)
    source_corpus = (
        f"{typed_fact_catalog}\n{news_catalog}\n{guidance_catalog}\n"
        f"{xbrl_str}\n{pres_txt}\n{trans_txt}\n{news_txt}"
    ).replace(",", "")
    source_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", source_corpus))

    check_text = json.dumps({
        "segments": parsed.get("segment_facts", []),
        "events": parsed.get("key_catalysts_and_events", []),
        "guidance": parsed.get("management_numerical_guidance", "")
    }).replace(",", "")
    emitted_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", check_text))

    ignore = {"0", "1", "2", "3", "4", "5", "24", "25", "26", "27", "28", "2024", "2025", "2026", "2027", "2028"}
    ungrounded = sorted([n for n in emitted_nums if n not in source_nums and n not in ignore])

    return {
        "ungrounded_numbers": ungrounded,
        "ungrounded_count": len(ungrounded),
        "status": "PASS" if len(ungrounded) == 0 else "WARN_UNGROUNDED_NUMS"
    }


def extract_clean_json_object(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    start = cleaned.find("{")
    if start == -1:
        return cleaned
    cleaned = cleaned[start:]

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    last_newline = cleaned.rfind("\n")
    if last_newline != -1 and not cleaned.rstrip().endswith("}"):
        cleaned = cleaned[:last_newline].rstrip()

    cleaned = re.sub(r"[,:\s]+$", "", cleaned)

    in_string = False
    escape = False
    stack = []
    for ch in cleaned:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        cleaned += '"'
    cleaned = re.sub(r",\s*$", "", cleaned)
    while stack:
        cleaned += stack.pop()

    return cleaned


# ==========================================================
# 5. LLM CALLING (WITH FULL FACT, NEWS & GUIDANCE BINDING)
# ==========================================================
def call_local_llm(
    model_name: str,
    symbol: str,
    company_name: str,
    nse_profile: dict,
    xbrl_dict: dict,
    news: str,
    pres: str,
    trans: str
) -> tuple[str, str, str, str, List[dict], List[dict], List[dict]]:
    pres_slice = pres[:PROMPT_PRES_CAP]
    trans_slice = trans[:PROMPT_TRANS_CAP]
    news_slice = (news or "").strip()

    typed_facts = extract_typed_facts_from_Sources(xbrl_dict, pres_slice, trans_slice)
    typed_fact_catalog = format_typed_fact_catalog(typed_facts)

    news_items = extract_news_catalog(news_slice)
    news_catalog = format_news_catalog(news_items)

    primary_period = typed_facts[0]["period"] if typed_facts else xbrl_dict.get("Period_End", "Q2 FY27")
    guidance_items = extract_guidance_catalog(pres_slice, trans_slice, doc_period=primary_period)
    guidance_catalog = format_guidance_catalog(guidance_items)

    detected_sectors = DeterministicFinancialExtractor.resolve_sector_archetypes(nse_profile, f"{pres_slice}\n{trans_slice}")
    sector_specific_block = "\n".join(SECTOR_EXTRACTION_RULES.get(s, "") for s in detected_sectors)
    nse_industry_str = " > ".join(
        filter(None, [
            nse_profile.get("macro_sector"),
            nse_profile.get("sector"),
            nse_profile.get("industry"),
            nse_profile.get("basic_industry")
        ])
    ) or "Not classified"

    prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
        company_name=company_name,
        symbol=symbol,
        typed_fact_catalog=typed_fact_catalog,
        news_catalog=news_catalog,
        guidance_catalog=guidance_catalog,
        nse_industry_str=nse_industry_str,
        detected_sectors=", ".join(detected_sectors),
        sector_specific_block=sector_specific_block,
        xbrl_json=json.dumps(xbrl_dict),
        news=news_slice or "None",
        pres=pres_slice or "None",
        trans=trans_slice or "None"
    )

    last_err = None
    for attempt in range(MAX_JSON_RETRIES):
        res = httpx.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "think": False,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMP if attempt == 0 else 0.1,
                    "repeat_penalty": 1.1,
                    "num_ctx": LLM_CTX,
                    "num_predict": LLM_NUM_PREDICT if attempt == 0 else LLM_NUM_PREDICT + 1000
                }
            },
            timeout=LLM_TIMEOUT
        )
        res.raise_for_status()
        msg = res.json().get("message", {})
        raw_text = extract_clean_json_object(msg.get("content") or "")

        try:
            validated = MasterCompanyFacts.model_validate_json(raw_text)
            validated = bind_and_ground_facts(
                validated, typed_facts, news_items, guidance_items,
                company_name, xbrl_dict, pres_slice, trans_slice
            )
            return (
                validated.model_dump_json(),
                typed_fact_catalog,
                news_catalog,
                guidance_catalog,
                typed_facts,
                news_items,
                guidance_items,
            )
        except Exception as e:
            last_err = e
            logger.warning(
                f"JSON schema retry ({attempt + 1}/{MAX_JSON_RETRIES}) for {symbol} "
                f"(content_len={len(raw_text)}): {e}"
            )

    raise ValueError(f"Failed valid JSON schema for {symbol} after {MAX_JSON_RETRIES} attempts: {last_err}")


# ==========================================================
# 6. ALIGNED LLM-AS-A-JUDGE SPOT AUDIT (WITH 500 RETRY)
# ==========================================================
def run_llm_judge_sample(conn: duckdb.DuckDBPyConnection, sample_size: int = 10, judge_model: str = JUDGE_MODEL) -> float:
    sample_df = conn.execute(f"""
        SELECT ci.symbol, ci.company_name, ci.xbrl_numbers_json, ci.latest_news_text,
               ci.cleaned_presentation_text, ci.cleaned_transcript_text, ci.extracted_facts_json
        FROM company_intelligence ci
        JOIN nse_universe_ranks r ON ci.ticker = r.ticker
        WHERE ci.extracted_facts_json IS NOT NULL
        ORDER BY r.market_cap_rank ASC
        LIMIT {sample_size}
    """).df()

    if sample_df.empty:
        logger.warning("No extracted_facts_json rows available to audit.")
        return 0.0

    logger.info(f"Running aligned LLM-as-a-Judge audit on top {len(sample_df)} companies using {judge_model}...")
    scores = []
    for _, row in sample_df.iterrows():
        symbol = row["symbol"]
        try:
            pres_slice = (row["cleaned_presentation_text"] or "")[:PROMPT_PRES_CAP]
            trans_slice = (row["cleaned_transcript_text"] or "")[:PROMPT_TRANS_CAP]
            news_slice = (row["latest_news_text"] or "").strip() or "None"
            xbrl_slice = row["xbrl_numbers_json"] or "{}"

            typed_facts = extract_typed_facts_from_Sources(json.loads(xbrl_slice), pres_slice, trans_slice)
            typed_fact_catalog = format_typed_fact_catalog(typed_facts)
            news_items = extract_news_catalog(news_slice if news_slice != "None" else "")
            news_catalog = format_news_catalog(news_items)
            primary_period = typed_facts[0]["period"] if typed_facts else "Q2 FY27"
            guidance_items = extract_guidance_catalog(pres_slice, trans_slice, doc_period=primary_period)
            guidance_catalog = format_guidance_catalog(guidance_items)

            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                company_name=row["company_name"],
                symbol=symbol,
                typed_fact_catalog=typed_fact_catalog,
                news_catalog=news_catalog,
                guidance_catalog=guidance_catalog,
                xbrl_slice=xbrl_slice,
                news_slice=news_slice,
                pres_slice=pres_slice,
                trans_slice=trans_slice,
                extracted_json=row["extracted_facts_json"]
            )

            res = None
            for judge_attempt in range(2):
                try:
                    res = httpx.post(
                        OLLAMA_URL,
                        json={
                            "model": judge_model,
                            "messages": [{"role": "user", "content": judge_prompt}],
                            "format": "json",
                            "think": False,
                            "stream": False,
                            "options": {"temperature": 0.0, "num_ctx": 10240, "num_predict": 600}
                        },
                        timeout=300.0
                    )
                    res.raise_for_status()
                    break
                except Exception as http_err:
                    if judge_attempt == 0:
                        time.sleep(3.0)
                    else:
                        raise http_err

            msg = res.json().get("message") or {}
            verdict_raw = msg.get("content") or ""
            if not verdict_raw:
                logger.warning(f"[AUDIT {symbol}] Empty response from Judge: {res.text[:200]}")
                continue

            verdict_clean = extract_clean_json_object(verdict_raw)
            logger.info(f"[AUDIT {symbol}]: {verdict_clean}")
            score_val = json.loads(verdict_clean).get("faithfulness_score_1_to_5", 0)
            if isinstance(score_val, (int, float)) and 1 <= score_val <= 5:
                scores.append(score_val)
        except Exception as e:
            logger.warning(f"[AUDIT {symbol}] Judge call skipped due to error: {e}")
            time.sleep(2.0)

    if scores:
        avg_score = sum(scores) / len(scores)
        logger.info(f"=== AVERAGE FAITHFULNESS SCORE ACROSS {len(scores)} STOCKS: {avg_score:.2f} / 5.0 ===")
        return avg_score
    return 0.0


# ==========================================================
# 7. MAIN PIPELINE EXECUTION
# ==========================================================
def run_pipeline(top_n: int = DEFAULT_TOP_N, skip_llm: bool = False, model_name: str = EXTRACTOR_MODEL, force_reclean: bool = False):
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("PRAGMA wal_autocheckpoint='1MB'")
    init_intelligence_schema(conn)

    limit_clause = f"LIMIT {top_n}" if top_n > 0 else ""
    targets_df = conn.execute(f"""
        SELECT ticker, symbol, company_name
        FROM nse_universe_ranks
        ORDER BY market_cap_rank ASC
        {limit_clause}
    """).df()

    nse = NSECorporateClient()
    extractor = DeterministicFinancialExtractor()
    current_week = resolve_sync_week()
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Processing {len(targets_df)} ranked companies into {DB_PATH.name} (model={model_name}, skip_llm={skip_llm})...")

    pbar = tqdm(targets_df.iterrows(), total=len(targets_df), desc="Corporate Intel Sync")
    try:
        for idx, (_, row) in enumerate(pbar, start=1):
            ticker = row["ticker"]
            symbol = row["symbol"]
            company_name = row["company_name"]
            pbar.set_postfix_str(f"Now: {symbol}")

            try:
                urls = nse.get_filing_urls(symbol)
                news_text = nse.fetch_news_rss(symbol, company_name)
                nse_profile = nse.fetch_nse_industry_profile(symbol)

                existing = conn.execute("""
                    SELECT latest_transcript_url, latest_presentation_url, latest_xbrl_url,
                           xbrl_numbers_json, cleaned_presentation_text, cleaned_transcript_text,
                           core_business_summary, management_guidance, extracted_facts_json, audit_flags_json
                    FROM company_intelligence
                    WHERE ticker = ?
                """, [ticker]).fetchone()

                urls_unchanged = (
                    not force_reclean
                    and existing is not None
                    and existing[0] == urls["transcript_url"]
                    and existing[1] == urls["presentation_url"]
                    and existing[2] == urls["xbrl_url"]
                    and bool(existing[8])
                )

                core_summary = existing[6] if existing else None
                guidance = existing[7] if existing else None
                facts_json_str = existing[8] if existing else None
                audit_json_str = existing[9] if existing else None

                if urls_unchanged:
                    xbrl_json_str = existing[3] or "{}"
                    cleaned_pres = existing[4] or ""
                    cleaned_trans = existing[5] or ""
                else:
                    xbrl_bytes = nse.fetch_bytes(urls["xbrl_url"])
                    xbrl_dict = extractor.parse_nse_xbrl_bytes(xbrl_bytes)
                    xbrl_json_str = json.dumps(xbrl_dict)

                    pres_bytes = nse.fetch_bytes(urls["presentation_url"])
                    cleaned_pres = extractor.clean_investor_presentation(pres_bytes)

                    trans_bytes = nse.fetch_bytes(urls["transcript_url"] or urls["outcome_url"])
                    cleaned_trans = extractor.clean_concall_transcript(trans_bytes)

                if not skip_llm and (not urls_unchanged or not facts_json_str):
                    if cleaned_pres or cleaned_trans or news_text or xbrl_json_str != "{}":
                        try:
                            (
                                facts_json_str,
                                typed_fact_catalog,
                                news_catalog,
                                guidance_catalog,
                                typed_facts,
                                news_items,
                                guidance_items,
                            ) = call_local_llm(
                                model_name, symbol, company_name, nse_profile,
                                json.loads(xbrl_json_str), news_text, cleaned_pres, cleaned_trans
                            )
                            parsed_facts = json.loads(facts_json_str)
                            core_summary = parsed_facts.get("core_business_model", "")
                            guidance = parsed_facts.get("management_numerical_guidance", "")

                            persist_point_in_time_tables(
                                conn, ticker, symbol, typed_facts, news_items, guidance_items, as_of_date
                            )

                            audit_result = audit_extraction_grounding(
                                facts_json_str,
                                typed_fact_catalog,
                                news_catalog,
                                guidance_catalog,
                                xbrl_json_str,
                                cleaned_pres,
                                cleaned_trans,
                                news_text,
                            )
                            audit_json_str = json.dumps(audit_result)
                            if audit_result["ungrounded_count"] > 0:
                                logger.warning(f"[{symbol}] Ungrounded numbers flagged: {audit_result['ungrounded_numbers']}")
                        except Exception as llm_err:
                            logger.error(f"LLM extraction failed for {symbol} (saving deterministic data anyway): {llm_err}")

                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("""
                    INSERT OR REPLACE INTO company_intelligence (
                        ticker, symbol, company_name,
                        latest_transcript_url, latest_presentation_url, latest_xbrl_url,
                        xbrl_numbers_json, cleaned_presentation_text, cleaned_transcript_text, latest_news_text,
                        core_business_summary, management_guidance, extracted_facts_json, audit_flags_json,
                        sync_week, updated_at
                    ) VALUES (
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?
                    )
                """, (
                    ticker, symbol, company_name,
                    urls["transcript_url"], urls["presentation_url"], urls["xbrl_url"],
                    xbrl_json_str, cleaned_pres, cleaned_trans, news_text,
                    core_summary, guidance, facts_json_str, audit_json_str,
                    current_week, now_str
                ))

                if idx % 5 == 0:
                    conn.execute("CHECKPOINT")

                time.sleep(NSE_SLEEP)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                time.sleep(ERROR_SLEEP)
    finally:
        conn.execute("CHECKPOINT")
        conn.close()
        logger.info("Corporate Intelligence sync complete and WAL checkpointed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Multi-Sector PDF/XBRL + Extract-Then-Bind Fact Pipeline")
    parser.add_argument("--top-n", type=int, default=None, help="Number of top ranked stocks to process (0 = all)")
    parser.add_argument("--skip-llm", action="store_true", help="Only run deterministic PDF/XBRL/RSS extraction")
    parser.add_argument("--force-reclean", action="store_true", help="Force re-cleaning PDFs and re-running LLM even if URLs are unchanged")
    parser.add_argument("--model", type=str, default=EXTRACTOR_MODEL, help="Local Ollama extraction model")
    parser.add_argument("--audit-sample", type=int, default=0, help="Run LLM-as-a-Judge audit on top N rows after extraction")
    parser.add_argument("--auto-full-if-pass", type=float, default=0.0, help="Auto-run all 750 stocks if audit average >= threshold (e.g. 4.2)")
    args = parser.parse_args()

    if args.top_n is not None or args.audit_sample == 0:
        target_n = args.top_n if args.top_n is not None else DEFAULT_TOP_N
        run_pipeline(
            top_n=target_n,
            skip_llm=args.skip_llm,
            model_name=args.model,
            force_reclean=args.force_reclean
        )

    if args.audit_sample > 0:
        audit_conn = duckdb.connect(str(DB_PATH), read_only=True)
        avg_score = run_llm_judge_sample(audit_conn, sample_size=args.audit_sample)
        audit_conn.close()

        if args.auto_full_if_pass > 0 and avg_score >= args.auto_full_if_pass:
            logger.info(
                f"Quality gate PASSED ({avg_score:.2f} >= {args.auto_full_if_pass:.2f})! "
                f"Launching full {DEFAULT_TOP_N}-stock run (skipping already-extracted top {args.top_n or 20})..."
            )
            run_pipeline(
                top_n=DEFAULT_TOP_N,
                skip_llm=args.skip_llm,
                model_name=args.model,
                force_reclean=False
            )
        elif args.auto_full_if_pass > 0:
            logger.warning(
                f"Quality gate NOT met ({avg_score:.2f} < {args.auto_full_if_pass:.2f}). "
                f"Stopped before full {DEFAULT_TOP_N}-stock batch."
            )