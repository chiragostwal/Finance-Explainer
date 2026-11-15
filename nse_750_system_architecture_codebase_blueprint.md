# NSE 750 Automated Financial & Quantitative Intelligence System
## End-to-End Architecture, Package Glossary, Method-by-Method Reference & Calibration Ledger

---

## 0. Plain-English Glossary of Every Package & Concept Used

Every package in your codebase has one specific, deterministic job:

### 1. What is `pydantic` (`BaseModel`, `Field`, `model_validator`, `field_validator`)?
* **The Problem:** Small local LLMs (`nemotron-3-nano:4b` and `qwen3.5:9b`) suffer from "schema drift." You ask for `{"segment_facts": [{"fact_id": "F001"}]}` and the model occasionally emits `{"segments": {"id": "f001"}}` or leaves trailing arrays like `cited_ids` with brackets `["[F001]", "F001"]`. Standard `json.loads()` parses the text and then crashes your downstream code with `KeyError` or `TypeError`.
* **What `pydantic` does:** Think of `pydantic` as an **automated customs officer and data-repair bouncer** sitting between the LLM and DuckDB:
  * **`BaseModel`:** Defines the strict data contract (`MasterCompanyFacts`, `SegmentFact`, `ExtractedCatalyst`, `StructuredAnalystVerdict`).
  * **`@model_validator(mode="before")`:** Runs *before* type checking. It catches wrong key names (`"segments"` $\rightarrow$ `"segment_facts"`), coerces strings/dicts into lists, and in `analyst_engine.py`, automatically harvests `[Fxxx]`, `[Nxx]`, and `[Gxx]` IDs from the generated text into `cited_ids`.
  * **`@field_validator("event_type", mode="before")`:** Maps invented LLM category strings (like `"DEAL_WIN"`) to your strict 7-value `Literal` enum (`"ORDER_BOOK"`).
  * **`@model_validator(mode="after")`:** Runs *after* validation to deduplicate entries and strip out noise.

### 2. Every Other Package in Your Codebase Explained

| Package (`import ...`) | Type of Work | Plain-English Role in Your System |
| :--- | :--- | :--- |
| **`duckdb`** | Embedded Analytical SQL Database (OLAP) | Single-file columnar SQL engine (`NSE-universe-raw.duckdb`). Reads entire columns in C++ at once, executing cross-sectional percentile ranks and joins across 750 stocks + 200,000 daily price rows in milliseconds. |
| **`pymupdf` (`import pymupdf as fitz`)** | C++ PDF Geometry & Coordinate Reader | Opens binary PDFs from NSE. `page.get_text("words")` returns exact `(x0, y0, x1, y1)` pixel coordinates for every word on a slide so we group words by vertical `y0` (`int(round(y0 / 4.0))`) and sort left-to-right by `x0` to reconstruct tables with `\|` separators. |
| **`httpx`** | HTTP Network Client | Maintains persistent anti-bot cookies with `www.nseindia.com` and sends long-timeout (`240s`) JSON requests to your local Ollama server (`http://localhost:11434/api/chat`). |
| **`xml.etree.ElementTree as ET`** | XML Tree Parser | Parses NSE's XBRL financial filings (`.xml`) and Google News RSS feeds (`<item><title>...`). |
| **`re` (Regular Expressions)** | Deterministic Pattern Matching | Scans text in microseconds using character patterns (`\u20b9` for `₹` so Windows PowerShell never corrupts Unicode into `?`). Powers `normalizer.py` to lock units, right-anchored table cells, and `F001`/`N01`/`G01` tokens. |
| **`numpy` (`import numpy as np`)** | Vectorized Array Math | Computes standard deviations, log returns, covariance matrices (`np.cov`), linear regressions (`np.polyfit`), 10-year DCF discounting, and binary-search Reverse DCF across 750 stocks. |
| **`pandas` (`import pandas as pd`)** | Tabular GroupBy & Ranking Engine | Computes within-sector medians (`groupby("sector_clean")`), percentile ranks (`.rank(pct=True)`), and rolling 20-day volatility windows (`.rolling(20).std()`). |
| **`yfinance`** | Market & Macro Data Downloader | Downloads 1-year daily OHLCV for the 750 `.NS` stocks, `^NSEI` (Nifty 50 benchmark), and 6-month daily series for the 7 live Commodity & FX benchmarks (`BZ=F`, `GC=F`, `HG=F`, `ALI=F`, `NG=F`, `INR=X`, `^TNX`). |
| **`yaml` (`PyYAML`)** | Configuration Loader | Loads `config.yaml` (paths, model names, caps) and `skills.yaml` (sector rules and prompts). |

---

## 1. Master System Flowchart (End-to-End Architecture)

> **CORE ARCHITECTURAL RULE:** **`quant_stats_engine.py` uses ZERO LLMs and ZERO prompts.** It is 100% deterministic math in Python/SQL. It writes 70 quantitative, DCF, cash-runway, and technical columns into the `stock_quant_signals` table in DuckDB. Later, **`analyst_engine.py`** queries those pre-computed numbers via SQL and places them into **Section 1 of the Analyst Prompt** so the 9B LLM never does arithmetic on its own.

```
+---------------------------------------------------------------------------------------------------+
| STAGE 1: INGESTION & COORDINATE CLEANING (competitive_inteligence.py)                             |
|                                                                                                   |
|  [NSE APIs: Announcements, XBRL, Quote-Equity] + [Google News RSS 90d]                            |
|         |                                                                                         |
|         +--> parse_nse_xbrl_bytes() ---------> Global Period_End + OneD Crore ground-truth dict   |
|         +--> clean_investor_presentation() --> (y0, x0) pixel row bucketing + Score-first 6k cap  |
|         +--> clean_concall_transcript() -----> Pleasantry stripping + Scored paragraph 9k cap     |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| STAGE 2: DETERMINISTIC TOKENIZATION (normalizer.py — ZERO LLM)                                    |
|                                                                                                   |
|  • extract_typed_facts_from_Sources() --> [F001..F045] (Rightmost table col + Unit inheritance)   |
|  • extract_news_catalog() --------------> [N01..N08]   (YYYY-MM-DD -> Qx FYxx + Event Type)       |
|  • extract_guidance_catalog() ----------> [G01..G06]   (Unwraps PDF lines, stitches <65 char      |
|                                                         fragments, drops '?' Q&A, tags            |
|                                                         GUIDANCE vs QUALITATIVE_OUTLOOK vs HEDGED)|
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| STAGE 3: EXTRACT-THEN-BIND LLM PIPELINE (competitive_inteligence.py)                              |
|                                                                                                   |
|  [nemotron-3-nano:4b] --> Picks fact_id ("F015"), news_id ("N01") & writes reported_driver        |
|         |                                                                                         |
|         v                                                                                         |
|  [MasterCompanyFacts (Pydantic)] + bind_and_ground_facts()                                        |
|  • Overwrites segment names, numbers, units & dates from Python's [Fxxx] & [Nxx] catalogs         |
|  • Saves to DuckDB: `company_intelligence`, `facts`, `events`, `guidance`                         |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| STAGE 4: DCF, CASH SUSTAINABILITY & TECHNICAL QUANT ENGINE (quant_stats_engine.py — ZERO LLM)     |
|                                                                                                   |
|  1. sync_point_in_time_tables_from_cache():                                                       |
|     Merges Tier 1 (normalizer) + Tier 2 (all JSON segments) + Tier 3 (TTM baseline facts)         |
|  2. compute_dcf_and_cash_sustainability():                                                        |
|     • Distinguishes Balance-Sheet Lenders (Banks/NBFCs) vs Asset-Light Capital Markets (MCX/AMCs) |
|     • Fixes Debt/Equity scale (`de_ratio = de_raw / 100.0`)                                       |
|     • Smooths Captive-Finance OCF drag (e.g. Bajaj Auto Credit) for high-ROE low-debt firms       |
|     • Computes 2-Stage 10Y FCF DCF (Non-Lenders) & 10Y Distributable Earnings DCF (Lenders)       |
|     • Solves 22-step Binary Search Reverse DCF Implied 5Y Growth Rate                             |
|     • Computes CROIC (%), FCF Debt Payback Years & Sector-Seasonal Working Capital Buffer         |
|  3. compute_single_ticker_technicals() + Cross-Sectional Sector MAD Z-Scores:                     |
|     • 1M/3M/6M/12M Sharpe Momentum, OLS Alpha/Beta vs ^NSEI, Hurst(126D), RSI(14), Squeeze, CVaR  |
|     • Saves 750 rows x 70 columns into DuckDB `stock_quant_signals`                               |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| STAGE 5: ZERO-LEAKAGE TOOL-CALLING ANALYST & LOGGER (analyst_engine.py)                           |
|                                                                                                   |
|  EquitiesAnalystTools queries DuckDB + yfinance:                                                  |
|  • Tool 1: get_quant_profile() -------------> 70 Quant/DCF cols + Promoter & FII/DII Holding %    |
|  • Tool 2: get_sql_industry_peers() --------> Top 4 direct SQL industry peers with P/E, ROE, CROIC|
|  • Tool 3: sync_and_get_commodities_fx() ---> Live 1M/3M/6M returns for Brent, Gold, Copper,      |
|                                               Aluminium, NatGas, USD/INR, US 10Y                  |
|  • Tool 4: extract_competitive_and_macro_intelligence() --> Generic structural grammar scanner    |
|                                               (Subsidiaries/JVs, Unorganized-to-Organized shift,  |
|                                               Geopolitical/Export notes — ZERO hardcoded names)   |
|  • Tool 5: search_filing_evidence_bm25() ---> Unwraps PDF lines, drops '?' Q&A, ranks top 6       |
|                                               sentences via 750-stock Corpus-IDF BM25             |
|         |                                                                                         |
|         v                                                                                         |
|  [qwen3.5:9b (format="json", num_predict=2200)] --> Validated by StructuredAnalystVerdict         |
|  • Auto-closes truncated JSON strings & deduplicates cited_ids (["F001", "N01", "G01"])           |
|  • Logs point-in-time call to DuckDB `predictions` table                                          |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Method-by-Method Reference Across All 4 Python Files

### File 1: `normalizer.py` (Deterministic Financial Tokenizer)
*Note: Uses pure ASCII Unicode escapes (`\u20b9` for `₹`, `\u20ac` for `€`, `\u2022` for `•`) so Windows PowerShell never corrupts currency symbols into regex `?` quantifiers.*

| Method Name | Packages Used | Type of Work Performed | Exact Input $\rightarrow$ Output & Logic |
| :--- | :--- | :--- | :--- |
| `classify_metric_label(row_label, slide_heading)` | `re` | Lexicon Classification | Tests `row_label` first, then `slide_heading`, against 23 compiled regex patterns (`METRIC_LEXICON`) so bank loan rows inherit `"Loan Book (Advances)"` and aviation/hospital rows inherit `"Aviation / Hospital / Hotel Unit Metrics"`. |
| `date_to_indian_fiscal_quarter(date_str)` | `re` | Fiscal Calendar Math | Converts `YYYY-MM-DD` to Indian Fiscal Quarters (`Apr–Jun = Q1`, `Jul–Sep = Q2`, `Oct–Dec = Q3`, `Jan–Mar = Q4`). |
| `detect_slide_unit_scope(slide_text)` | `re` | Unit Header Inheritance | Scans the top 450 characters of a slide for parenthesized unit headers (`(₹ billion)`, `(Rs. Crore)`, `(US$ Million)`) and locks that unit to every number extracted from the slide. |
| `extract_news_catalog(news_text)` | `re` | News Deduplication & Event Typing | Filters out stock-price listicles (`JUNK_NEWS_PAT`), deduplicates headlines by 45-char stem, classifies `event_type` (`CAPEX`, `ORDER_BOOK`, `PARTNERSHIP_JV`, `PRICING_ACTION`, `POLICY_REGULATION`, `RISK_FACTOR`, `OPERATIONAL_MILESTONE`), and numbers them `[N01..N08]`. |
| `extract_guidance_catalog(pres_text, trans_text, doc_period)` | `re` | Unwrapped Context Stitching & Modality Tagging | 1. Unwraps single `\n` PDF line breaks while protecting abbreviations (`Rs.`, `Cr.`, `Mr.`, `Dr.`) with `_DOT_`.<br>2. Drops analyst questions (`?`, `"how do you see"`).<br>3. **Context Stitching:** If a guidance sentence is `< 65` chars (e.g., `"our guidance of 16% to 18%."`), it automatically prepends the preceding sentence so the metric name is never lost.<br>4. Tags `modality` as `"HEDGED"`, `"GUIDANCE"` (if numerical), or `"QUALITATIVE_OUTLOOK"` (if qualitative forward commentary with a metric anchor). |
| `extract_typed_facts_from_Sources(xbrl_dict, pres_text, trans_text)` | `re` | Multi-Pass Fact Extractor | Extracts up to 45 `[F001..F045]` facts from: (1) XBRL ground truth, (2) Right-anchored slide tables (`TABLE_ROW_PAT`, picking the rightmost current-period value column `v3 or v2` and converting `₹ bn * 100` to `₹ Cr`), and (3) Concall & slide clauses (`ATOMIC_FACT_CLAUSE_PAT`, matching `₹`, `Rs.`, `INR`, `%`, and physical volume units like `lakh units` across verbs like `grew by`, `delivered`, `clocked`, `reported`, `jumped`). |

---

### File 2: `competitive_inteligence.py` (NSE Ingestion, 2D PDF Geometry & Extract-Then-Bind)

| Method Name | Packages Used | Type of Work Performed | Exact Input $\rightarrow$ Output & Logic |
| :--- | :--- | :--- | :--- |
| `SegmentFact`, `ExtractedCatalyst`, `MasterCompanyFacts` | `pydantic`, `re`, `json` | Schema Validation & Auto-Repair | Repairs drifted LLM JSON keys (`mode="before"`), enforces valid `event_type` enums (`field_validator`), strips unit leaks and stock-price noise from sensitivities (`mode="after"`). |
| `bind_and_ground_facts(...)` | `re` | Deterministic Fact & News Binding | **The Hallucination Firewall:** Looks up the `fact_id` (`F015`) and `news_id` (`N01`) selected by the 4B LLM and **overwrites** the LLM's text with Python's exact string from `normalizer.py`. Retains all `N01..N08` news items in `factual_news_digest`. |
| `clean_investor_presentation(pdf_bytes)` | `pymupdf` (`fitz`), `Counter`, `re` | 2D Coordinate Table Reconstruction | Reads `(x0, y0)` word coordinates, buckets words within `4pt` vertical bands into horizontal rows, inserts `\|` when horizontal gap `> 35pt`, removes repeating headers/footers, **scores slides by financial/number density, truncates to `6,000` chars by highest score first**, and sorts kept slides by slide number. |
| `clean_concall_transcript(pdf_bytes)` | `pymupdf` (`fitz`), `re` | Transcript Paragraph Scorer | Strips operator pleasantries and safe-harbor boilerplate, scores paragraphs by `%` / `bps` / keyword density, and keeps top paragraphs up to `9,000` chars in chronological order. |

---

### File 3: `quant_stats_engine.py` (DCF, Reverse DCF, Cash Sustainability & Technical Engine)

#### Mathematical Formulas & Institutional Calibrations Built Into `quant_stats_engine.py`

1. **Distinguishing Balance-Sheet Lenders vs. Asset-Light Capital Markets (`is_balance_sheet_lender_or_insurer`):**
   * Yahoo Finance puts **Exchanges (`MCX`, `BSE`), AMCs (`HDFCAMC`, `NAM-INDIA`), and Brokers (`MOTILALOFS`, `GROWW`, `ANGELONE`)** in `sector = "Financial Services"` alongside commercial banks and NBFCs (`BAJFINANCE`, `ICICIBANK`, `SBIN`).
   * `is_balance_sheet_lender_or_insurer()` returns `False` for Exchanges, AMCs, Brokers, Depositories, and Rating Agencies (routing them to the **2-Stage Free Cash Flow DCF**), and returns `True` *only* for Banks, NBFCs, Credit Services, Housing Finance, and Insurers.

2. **Debt-to-Equity Scale Normalization (`de_ratio = max(de_raw / 100.0, 0.0)`):**
   * In `equities_master`, `debt_to_equity` from `yfinance` is always reported on a percentage scale (`0.7` = `0.7%` = `0.007x` Book Equity for `BAJAJ-AUTO`; `45.0` = `45.0%` = `0.45x`).
   * Unconditionally dividing `de_raw / 100.0` ensures low-debt companies (`0.7%` D/E) are never misread as `70%` D/E.

3. **Captive-Finance & Treasury OCF Smoothing for High-ROE Compounders (`effective_fcf`):**
   * Industrial/Auto compounders with captive NBFC subsidiaries (such as *Bajaj Auto Credit Ltd* inside `BAJAJ-AUTO` or *MMFSL* inside `M&M`) record captive loan-book expansion as an operating cash outflow on their consolidated cash-flow statement, artificially depressing single-year consolidated `free_cash_flow` in `yfinance`.
   * Rule: When a non-lender has `ROE >= 18%`, `Debt/Equity < 0.35x`, and reported `FCF < 0.45 * Implied PAT`, Python blends `50% Reported FCF + 50% Normalized NOPAT Cash Flow (0.78 * Implied PAT)`:
     $$\text{Effective FCF} = 0.50 \cdot \text{FCF}_{\text{reported}} + 0.50 \cdot (0.78 \cdot \text{PAT}_{\text{implied}})$$
   * This prevents captive-finance accounting from producing a false `2.2% CROIC` or `18.5 Years Debt Payback` on a debt-free auto manufacturer.

4. **Cash Return on Invested Capital ($\text{CROIC}\%$) & FCF Debt Payback Horizon:**
   $$\text{Book Equity} = \frac{\text{Market Cap}}{P/B}, \qquad \text{Total Debt} = \text{Book Equity} \times \left(\frac{\text{Debt/Equity}}{100}\right), \qquad \text{Invested Capital} = \text{Book Equity} + \text{Total Debt}$$
   $$\text{CROIC (\%)} = \frac{\text{Effective FCF}}{\text{Invested Capital}} \times 100, \qquad \text{FCF Debt Payback (Years)} = \frac{\text{Total Debt}}{\text{Effective FCF}}$$

5. **Sector-Seasonal Working Capital Buffer (`resolve_sector_seasonal_profile`):**
   Uses strict word boundaries (`\b`) and evaluates specific industry liquidity cycles against each company's `Current Ratio` and `Quick Ratio`:
   $$\text{Seasonal WC Buffer Ratio} = 0.55 \left(\frac{\text{Quick Ratio}}{\text{Sector Min QR}}\right) + 0.45 \left(\frac{\text{Current Ratio}}{\text{Sector Min CR}}\right)$$
   * `CAPITAL_MARKETS_ASSET_LIGHT_FEE_ENGINE` (`Min CR = 1.15`, `Min QR = 1.00`)
   * `BFSI_CASA_ALM_LIQUIDITY_FRAMEWORK` (`Min CR = 1.00`, `Min QR = 1.00`)
   * `AUTO_H1_CAPEX_M&A_Q3_Q4_FESTIVE_HARVEST` (`Min CR = 1.35`, `Min QR = 0.90` — accounts for H1 tooling/M&A outlays and Q3/Q4 festive cash harvest)
   * `EPC_H1_WORKING_CAPITAL_LOCK_Q4_GOVT_BILLING_RELEASE` (`Min CR = 1.50`, `Min QR = 1.05`)
   * `AGRI_MONSOON_INVENTORY_BUILD_SUBSIDY_LAG` (`Min CR = 1.45`, `Min QR = 0.85`)
   * `CONSUMER_Q2_INVENTORY_STOCKING_Q3_DIWALI_CASH_RELEASE` (`Min CR = 1.35`, `Min QR = 0.75`)
   * `ASSET_LIGHT_NEGATIVE_WC_STEADY_CASH_CYCLE` (`Min CR = 1.15`, `Min QR = 0.95`)
   * `PHARMA_CHEM_USFDA_EXPORT_RECEIVABLE_CYCLE` (`Min CR = 1.40`, `Min QR = 1.00`)

6. **2-Stage 10-Year DCF & 22-Step Binary-Search Reverse DCF:**
   * **CAPM Cost of Equity ($K_e$):** $R_f = 6.85\%$ (India 10Y G-Sec), $\text{ERP} = 5.25\%$, $\beta$ clamped to $[0.65, 1.25]$ for regulated lenders and $[0.65, 1.75]$ for non-lenders.
   * **For Non-Lenders:** Projects $\text{Base CF}$ growing at $g_1$ (Years 1–5), fading linearly to $g_{\text{term}} = 6.5\%$ (Years 6–10) + Gordon Terminal Value, minus $30\%$ of Total Debt. (Deep cyclicals trading at $P/E < 7.5\times$ have $g_1$ capped at $8.0\%$ to avoid peak-cycle distortion.)
   * **For Balance-Sheet Lenders (`BAJFINANCE`, `ICICIBANK`):** Uses a **2-Stage Distributable Equity Earnings Model** where $\text{Distributable Base} = \text{Implied PAT} \times \left(1 - \frac{g_{\text{term}}}{\text{ROE}}\right)$, compounded over 10 years and discounted at $K_e$. Also sets `ocf_to_pat_ratio = NULL` so loan-book disbursements are never misread by the LLM as negative operating cash flow.
   * **Reverse DCF (`reverse_dcf_implied_growth_pct`):** Uses 22-step binary search over $[-15\%, +48\%]$ to find the exact 5-year CAGR required to justify the current share price.

7. **Technical Regime Classification (`classify_technical_regime`):**
   * `VOLATILITY_SQUEEZE_SETUP`: `d200 > 0`, `d50 > -2%`, `vol_squeeze_percentile <= 25%`, `up_down_vol_ratio >= 1.05`.
   * `STRONG_UPTREND_BREAKOUT`: `d200 > 5%`, `d50 > 1.5%`, `52 <= RSI <= 76`, and (`up_down_vol_ratio >= 1.10` or `sharpe_6m >= 1.0`).
   * `UPTREND_PULLBACK_ACCUMULATE`: Requires `d200 > 0`, **`RSI <= 62.0`** (prevents overbought `RSI = 76.8` stocks from being labeled pullbacks!), and (`-8% <= d50 <= 1.5%` or `35 <= RSI <= 55`).
   * `STEADY_INSTITUTIONAL_UPTREND`: `d200 > 2%`, `d50 >= -1.0%`, `sharpe_6m >= 0.4`.

---

### File 4: `analyst_engine.py` (Zero-Leakage SQL/BM25 Tool Router & Prediction Logger)

| Method Name | Packages Used | Type of Work Performed | How It Works & Prevents Prompt Leakage |
| :--- | :--- | :--- | :--- |
| `sync_and_get_commodities_fx()` | `yfinance`, `duckdb`, `numpy` | Live Commodity & FX Feed | Downloads 6-month daily series for `BZ=F` (Brent), `GC=F` (Gold), `HG=F` (Copper), `ALI=F` (Aluminium), `NG=F` (NatGas), `INR=X` (USD/INR), and `^TNX` (US 10Y), computes 1M/3M/6M returns, caches them in `commodities_fx` for the day, and maps relevant series to each sector via `SECTOR_COMMODITY_LINKS`. |
| `get_quant_profile(symbol)` | `duckdb` | SQL Quant + Ownership Join | Queries `stock_quant_signals` joined with `equities_master` (`held_by_insiders * 100` as `promoter_holding_pct`, `held_by_institutions * 100` as `fii_dii_institutional_pct`). |
| `get_sql_industry_peers(symbol, industry, sector)` | `duckdb` | Dynamic SQL Peer Matrix | Queries the top 4 peer companies in the exact same `industry` from `stock_quant_signals` with their `Rank`, `P/E`, `P/B`, `ROE`, `CROIC`, and `Conviction` scores. |
| `extract_competitive_and_macro_intelligence(...)` | `re` | Zero-Entity Structural Scanner | Scans unwrapped filing sentences using **only generic legal/economic terms** (`subsidiary`, `joint venture`, `wholly-owned`, `unorganised sector`, `informal sector`, `market share`, `export`, `tariff`, `red sea`, `west asia`)—contains **zero hardcoded company or brand names** to guarantee zero prompt leakage across stocks. |
| `search_filing_evidence_bm25(symbol, query)` | `duckdb`, `re`, `Counter` | Unwrapped Corpus-IDF BM25 | Unwraps broken PDF lines first, filters out analyst questions (`?`, `"how do you see"`, `"how do you think"`), and scores sentences using 750-stock Corpus-IDF BM25 with a $1.25\times$ boost for sentences containing numbers/units. |
| `generate_and_log_stock_verdict(...)` | `httpx`, `pydantic`, `duckdb` | Fast JSON Synthesis & Logging | Calls `qwen3.5:9b` with `format="json"` and `num_predict=2200`, repairs any truncated output via `extract_clean_json_object()`, deduplicates `cited_ids` (`["F001", "N01"]`), and logs the point-in-time call into the `predictions` table. |

---

## 3. Summary of the 6 Institutional Audit Fixes Applied

| # | Issue Discovered During Testing | Root Cause in Code | Exact Fix Applied |
| :--- | :--- | :--- | :--- |
| **1** | `BAJAJ-AUTO` (`0.7%` D/E) showed `18.5 Years` Debt Payback & `WORKING_CAPITAL_OR_DEBT_STRESS` | `de_ratio = (de_raw / 100.0) if de_raw > 3.0 else de_raw` treated `0.7%` as `0.70x` (70% D/E = ₹25,530 Cr imaginary debt). | Changed to `de_ratio = max(de_raw / 100.0, 0.0)` since `yfinance` `debtToEquity` is always in percentage points. Debt Payback dropped to `0.0 Years`. |
| **2** | `BAJAJ-AUTO` (`35.6%` ROE) showed `2.2% CROIC` & `6.7% Cash Sustainability` | Captive-finance subsidiary (*Bajaj Auto Credit*) loan-book additions depressed consolidated GAAP OCF in `yfinance`. | Blended `50% reported FCF + 50% (0.78 * Implied PAT)` when `ROE >= 18%` and `D/E < 0.35x`. `CROIC` rose to `~20%` (`SELF_FUNDED_CASH_FORTRESS`). |
| **3** | `MCX`, `MOTILALOFS`, `GROWW` were valued on 1x Bank Book Value (`P/B`) | `sector = "Financial Services"` matched `is_bfsi` for asset-light exchanges/brokers. | Added `is_balance_sheet_lender_or_insurer()` so Exchanges/AMCs/Brokers use 2-Stage FCF DCF while Banks/NBFCs (`BAJFINANCE`) use 10Y Distributable Earnings DCF. |
| **4** | `BAJFINANCE` had `OCF/PAT = -4.66x` & matched `it services` inside `Cred**it** Services` | Growing a loan book is a GAAP OCF outflow for NBFCs; `it services` regex lacked `\b` word boundaries. | Set `ocf_to_pat_ratio = None` for lenders and added `\b` word boundaries to `SECTOR_SEASONAL_REQUIREMENTS`. |
| **5** | `BAJAJ-AUTO` (`RSI = 76.8`) was labeled `UPTREND_PULLBACK_ACCUMULATE` | Pullback condition checked `-8.0 <= d50 <= 1.5` without checking if RSI was overbought (`> 62`). | Added `rsi <= 62.0` requirement to `UPTREND_PULLBACK_ACCUMULATE`; routed `RSI > 62` uptrends to `STEADY_INSTITUTIONAL_UPTREND`. |
| **6** | BM25 snippets had analyst questions (`?`) & `cited_ids` had `["[F001]", "F001"]` | PDF lines weren't unwrapped before sentence splitting; brackets weren't stripped before ID deduplication. | Unwrapped single `\n` breaks, filtered out `?` / `"how do you see"`, and normalized `cited_ids` via `^(?:F\d{3}\|N\d{2}\|G\d{2})$`. |