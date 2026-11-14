# Equities Intelligence Pipeline: Audit Findings and Research-Backed Roadmap

Nov 14, 2026 · @Chirag

## Purpose and scope

The 3.50 / 5 audit average comes mostly from numbers detached from their labels, units and periods, and the fix is to bind those in code before the small model sees them, not to prompt harder. This document records where the equities pipeline stands after the 14 Nov 2026 audit, why it fails where it does, and a roadmap grounded in published arXiv work.

Scope is equities only: the NSE top-750 corporate intelligence pipeline that feeds a chat-in-a-box analyst. That analyst will combine live news, corporate actions, investor holdings, price, geopolitics and commodity prices to judge where a stock may move fundamentally. IPO and mutual-fund pipelines are deferred; most of the normalizers below carry over unchanged.

- **Biggest lever:** extract-then-bind. Code builds a typed fact table; the LLM picks fact IDs and writes only driver text.
- **Cheapest fix:** remove real numbers and segment names from the prompts, which the 4B model copies (50,615.5 appeared in three unrelated companies).
- **Prediction caveat:** published LLM stock-forecasting results are vulnerable to look-ahead bias, so the predictive layer needs a logged, point-in-time evaluation before any trust.

Source note: the papers cited here were reviewed through their arXiv abstracts and search summaries, not read end to end, and nothing in this document has yet been tested on your data.

## Where the pipeline stands today

The pipeline already separates deterministic cleaning from LLM extraction, which is the right shape; the weak point is that the LLM still writes every number. It runs per ranked company, reads filings from NSE, and stores one latest-quarter record in `company_intelligence` (DuckDB).

| Stage | What it does now | Assessment |
| --- | --- | --- |
| Filing discovery | NSE announcements API for transcript, presentation, outcome PDF; financial-results API for XBRL | Works; no history kept |
| XBRL parse | Groups facts by `contextRef`, picks latest `Period_End`, normalizes to crore, drops pre-2026 | Good base; only 7 monetary tags and no prior-period or YoY facts |
| Presentation cleaning | PyMuPDF text blocks joined with a pipe character, slides scored by keyword and number hits, top 18 kept | Loses table structure and unit captions |
| Transcript cleaning | Paragraph scoring, pleasantries stripped, 18k char cap | No speaker or prepared-vs-Q&A tagging |
| Sector routing | NSE industry keywords plus regex hit counts, 2 archetypes max, rules from `skills.yaml` | Useful; rules contain leaking examples |
| LLM extraction | `nemotron-3-nano:4b`, JSON mode, temperature 0, schema repair in Pydantic | Writes numbers and labels freely |
| Post-checks | Pydantic dedup and junk filters; grounding check that a number appears somewhere in sources | Cannot catch a real number with the wrong label |
| Audit | `qwen3.5:9b` judge on top-N by market-cap rank | Noisy; see the audit section |

Two things work well and should be kept: the Pydantic repair layer (it absorbs the small model's schema drift) and the news handling, where `factual_news_digest` is overwritten verbatim from fetched headlines so the model cannot rewrite them.

## What the 50-stock audit shows

The judge averaged 3.50 / 5 across 50 stocks: 2 scored 5, 28 scored 4, 14 scored 3, 5 scored 2 and 1 scored 1 (ICICI Bank). Twenty stocks, 40%, scored 3 or lower. The errors fall into six repeating classes rather than fifty unrelated ones.

&#91;embedded content: qwen3.5:9b judge scores · top 50 stocks by market-cap rank · audit run 14 Nov 2026\]

| Error class | Examples from the log | Pattern |
| --- | --- | --- |
| Wrong metric labelled as Revenue | ICICI Bank (PAT, core operating profit), UltraTech (EBITDA), ITC (PBT), Coal India (PAT), HAL (operating profit, EPS), Reliance, Adani Power, Grasim | A number is bound to the wrong row label |
| Magnitude, currency or unit scope | TCS (US$ billion read as crore), Sun Pharma and HCLTech (USD shown as ₹), Nestlé (₹ billion as ₹ Cr), Tata Steel (€ million as Cr), IOC and DMart (MMT and mn sq ft given a ₹ sign) | Slide or table unit caption lost |
| Decimal shift on correct digits | Power Grid (12,055 became 12.05), Kotak (5,968 became 5.968), Maruti (506,155 became 506.15) | Comma read as a decimal point |
| Wrong segment, column or period | SBI and Axis (whole-bank advances under SME), Asian Paints, Eicher (consolidated split across two segments), Adani Ports, Hindalco, Shriram, Bajaj Finserv, JPL | Table column or row misread |
| Period or basis confusion | HUL, Infosys, Adani Green, DMart, DLF, Sun Pharma (YoY written as QoQ), Grasim | Q2 vs H1 vs FY, YoY vs QoQ |
| Prompt leakage | TVS Motor (`₹X Cr` placeholder), Eicher and Shriram ("credit growth" for a non-lender in Eicher's case), Bajaj Finance, HUL and Wipro (the same 50,615.5) | Model copies prompt text |

Two further patterns are smaller but recurring. Competitors were taken from news or from the company's own subsidiary (Divi's got TVS Motor; Eicher got VECV). Hedged or analyst statements were stored as facts (HDFC Bank, Sun Pharma's "difficult to predict").

Treat the judge as a triage signal, not ground truth. Its Adani Ports and Axis Bank entries contain its own working-out ("Wait, re-evaluating"), the Adani Ports verdict has a duplicated `errors_found` key, and the Sun Pharma entry cites "$192.9Bn" for India Formulations, which is implausible for a single segment. A 9B model grading a 4B model is still useful for finding bad rows, but not for tracking a score to one decimal.

## Root causes in the current code

Six causes in the code and prompts account for most of the logged errors, and each maps to a specific function or template.

| # | Cause | Where | Errors it explains |
| --- | --- | --- | --- |
| 1 | Concrete values and names inside prompts: `₹X Cr`, `+Y% YoY`, the 506,155 Million example, "Retail / Corporate / SME", "Enterprise/B2B", "credit growth %" | `extractor_prompt` and the BFSI, TELECOM and AUTO rules in `skills.yaml` | Prompt leakage; invented segments for banks and JPL |
| 2 | Tables flattened to a stream of cells with no row or column header | `clean_investor_presentation` (block text joined with a pipe) | Wrong-metric-as-Revenue; wrong segment or column |
| 3 | Unit captions such as "₹ Cr", "US$ Mn" or "bn" are not attached to the numbers they govern | same function; nothing parses units | Magnitude, currency and decimal-shift errors |
| 4 | Quarter and basis are left for the 4B model to infer from a calendar paragraph | both prompts | Q2 vs H1 vs FY; YoY vs QoQ; judge confusion on Oct to Dec dates |
| 5 | Best-scoring slides are re-sorted by slide number, then cut at 6,000 of the 12,000 characters kept | `top_slides.sort(...)` then `pres[:PROMPT_PRES_CAP]` | Segment tables on later slides silently dropped; same for transcript paragraphs (18,000 kept, 9,000 sent) |
| 6 | The grounding check tests only that a number appears somewhere in the sources, and only logs a warning | `audit_extraction_grounding` and the `run_pipeline` write path | Real numbers with wrong labels pass; bad rows are still stored |

One further suspicion, not yet verified: the presentation, transcript and XBRL may sometimes come from different quarters. The judge noted FY25 figures in the Adani Green source, Q1 FY27 figures in the DLF source, and a Q1 call for Eicher dated November 2026. `get_filing_urls` takes the first matching attachment per category and never checks that the three documents share a reporting period. A cheap test is to read the period off each PDF's first page and compare it with the XBRL `Period_End`.

## What the arXiv research says

The published work converges on one design: keep structure and units in code, give the model classification-style jobs, and verify each claim against a cited span. The rows below give each paper's finding as reported in its abstract or summary, then the change it suggests here. Entries marked inference are my reading of how a result applies to your setup, not something the paper tested.

| Paper | Finding | Change for this pipeline |
| --- | --- | --- |
| [Financial Report Chunking for Effective RAG](https://arxiv.org/abs/2402.05131) (2402.05131) | Chunking by document element (titles, paragraphs, tables) beat fixed token chunks on FinanceBench retrieval and answer quality | Chunk filings by element, with slide title and table caption as metadata, instead of ranking whole slides by keyword count |
| [GTE / FinTabNet](https://arxiv.org/abs/2005.00589) (2005.00589) and [TableFormer](https://arxiv.org/abs/2203.01017) (2203.01017) | Financial table structure can be recovered as rows, columns and spans; TableFormer outputs structure but not cell text, and the authors pair it with the PDF's own text layer | Use the PDF text layer plus PyMuPDF table detection first; call a structure model only for image-only slides |
| [TAT-QA](https://arxiv.org/abs/2105.07624) (2105.07624) | Finance QA over table plus text; the TagOp model extracts cells, predicts the scale from the table header, then applies symbolic operators (58.0 F1 vs 90.8 for humans) | Read the scale from the caption or header into every cell; compute derived values with operators in code |
| [FinQA](https://arxiv.org/abs/2109.00122) (2109.00122) | 8,281 expert questions with annotated reasoning programs; the baseline model reached 61.24% vs 91.16% for experts | Have the model emit a short program over fact IDs; use FinQA-style examples as unit tests for the normalizer |
| [Program of Thoughts](https://arxiv.org/abs/2211.12588) (2211.12588) | Delegating arithmetic to an interpreter gave about 12% average gain over chain-of-thought across math and finance datasets | Growth rates, margins and currency conversion happen in Python or SQL, never in the LLM |
| [Tokenization counts](https://arxiv.org/abs/2402.14903) (2402.14903) | Forcing right-to-left 3-digit grouping with commas lifted GPT-3.5 addition accuracy from 75.6% to 97.8% and GPT-4 from 84.4% to 98.9%; the gap narrows with model size | Inference: Indian grouping such as 52,08,344 misaligns 3-digit groups, and a 4B model is likely more sensitive. Show plain digits or western grouping plus an explicit unit, and A/B test on 200 numbers |
| [FinTagging](https://arxiv.org/abs/2505.20650) (2505.20650) | XBRL tagging framed as two steps, numeric identification then concept alignment; LLMs remain limited at full-taxonomy scale | Same two steps, but align to a small closed metric lexicon; score the alignment against your NSE XBRL facts |
| [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) (2408.02442) | Strict format constraints degraded reasoning, helped classification, and a free-text-then-convert step preserved quality | Make the LLM pick IDs from enums (classification); for driver text let it write freely, then convert |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) (2307.03172) | Accuracy is highest when relevant information is at the start or end of the context and lower in the middle | Put the fact table first and the instruction last; at chat time retrieve a few facts rather than a 15k-character blob |
| [SelfCheckGPT](https://arxiv.org/abs/2303.08896) (2303.08896) | Facts that vary across sampled answers are more likely hallucinated | Sample the extractor several times at temperature above 0 and flag fields that disagree. It will not catch a consistent mislabel, so binding stays the main control |
| [Chain-of-Verification](https://arxiv.org/abs/2309.11495) (2309.11495) | Draft, plan verification questions, answer them independently, then revise, reduced hallucination | Replace the holistic judge score with per-claim checks: does the cited span support this claim? |
| [FinanceBench](https://arxiv.org/abs/2311.11944) (2311.11944) | 10,231 questions on US company filings showed clear limits for LLMs, including hallucination | Build an India equivalent: a few hundred hand-verified questions on top-50 filings as a regression set |

## Enhancement roadmap

&#91;embedded content: roadmap · 4 phases with gates · effort is a rough single-developer estimate\]

Phase 3 is highlighted because binding facts in code is what removes label, unit and period errors by construction; phases 1 and 2 are prerequisites and run mostly in code. Each gate is checked against the same 50 stocks so the numbers are comparable between runs.

## Target data model for the chat product

The chat layer should read numbers through SQL tools and qualitative context through search, with every fact carrying a period, unit and as-of date. This replaces the single overwritten row per company with history that can be queried and backtested.

&#91;embedded content: target architecture for the equities chat product · 8 components\]

| Table | Key columns | Filled from | Notes |
| --- | --- | --- | --- |
| `facts` | ticker, period\_id, scope, metric\_id, value, currency, scale, basis, source\_doc, span\_id, as\_of | XBRL and parsed presentations | One row per metric per period; history kept |
| `events` | ticker, event\_date, fiscal\_period, event\_type, headline, url, as\_of | News and NSE announcements | Fiscal period computed in code |
| `guidance` | ticker, period\_id, metric\_id, low, high, modality, speaker, span\_id | Transcripts and decks | Modality: actual, guidance or hedged |
| `holdings` | ticker, holder, holder\_type, pct, quarter, as\_of | NSE shareholding filings (not yet in the pipeline) | Needed for the portfolio signal |
| `prices` | ticker, date, close, volume | Price feed (to be chosen) | Adjusted for corporate actions |
| `commodities_fx` | series\_id, date, value | Feed to be chosen | Linked to sectors through the archetype |
| `predictions` | pred\_id, ticker, as\_of, horizon, call, rationale, fact\_ids | Chat product | Scored later against realized returns |

- **Point in time.** Never overwrite a fact; insert a new row with its own `as_of`, so any past date can be reconstructed.
- **Tools, not blobs.** The chat model calls something like `get_facts(ticker, metrics, periods)`, `get_events(...)` and `search_evidence(query, ticker)`, and answers from a handful of returned facts. This also follows the lost-in-the-middle finding: short contexts with the key facts first.
- **Hybrid retrieval.** Keyword (BM25) plus embedding search over evidence snippets, with IDF computed across all filings so boilerplate scores low.
- **Constrained output.** Ollama accepts a JSON schema in `format`, so metric IDs, event types and modality can be enums rather than free strings.

## Evaluation and backtesting discipline

Treat any LLM-derived return signal as unproven until it survives point-in-time, forward-tested scoring; extraction quality is necessary but not sufficient for prediction. Two separate questions need two separate tests.

**1. Is the extraction correct?** Replace the judge's mean score with a hand-checked gold set.

- Hand-verify the top 50 stocks once: per segment, the metric label, value, unit, period and basis. Score each field separately, so a run reports label accuracy, unit accuracy and period accuracy rather than one average.
- Keep the 9B judge only as a triage tool to find suspicious rows, and make it check one claim at a time against its cited span, in the spirit of Chain-of-Verification.
- Build a few hundred India-specific questions on those filings as a regression set, modelled on FinanceBench, and re-run it after every pipeline change.

**2. Does the signal predict anything?** The literature is encouraging but carries a specific trap.

- [Lopez-Lira and Tang](https://arxiv.org/abs/2304.07619) found that GPT-4 scores of news headlines predicted next-day returns, more so for smaller stocks and after negative news, and that strategy returns decline as LLM adoption rises. Expect any edge from public news to decay.
- [A Test of Lookahead Bias in LLM Forecasts](https://arxiv.org/abs/2512.23847) reports that memorization of training data accounted for roughly 37% of the apparent LLM effect in a stock-return task and about 19% in an earnings-call capex task. An LLM backtested on events it saw in training looks better than it is.
- [Look-Ahead-Bench](https://arxiv.org/abs/2601.13770) and [FinCAD](https://arxiv.org/abs/2605.24564) are a benchmark and a decoding-time mitigation for the same problem.

Practical rules that follow:

1. Do not trust a backtest over any period your local models may have been trained on. Evaluate on a forward test: log each prediction at the time it is made and score it later.
2. Store every prediction with its `as_of` date, horizon, the fact IDs it used and the model version, then compare against realized sector-adjusted returns and against a simple baseline such as sector momentum.
3. Define the target before building: for example the direction of next-quarter EBITDA versus the prior quarter, not "where the stock moves". Price moves mix fundamentals with flows and sentiment.
4. To test for leakage in a historical run, one idea worth trying (my suggestion, not tested in the papers above) is to hide company names and dates in the prompt and see whether accuracy drops.

## References

All papers were checked on 14 Nov 2026 through arXiv listing pages and search summaries; none was read in full, so details such as exact figures should be confirmed against the paper before you rely on them.

- [Financial Report Chunking for Effective Retrieval Augmented Generation](https://arxiv.org/abs/2402.05131), arXiv:2402.05131
- [Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context](https://arxiv.org/abs/2005.00589), arXiv:2005.00589
- [TableFormer: Table Structure Understanding with Transformers](https://arxiv.org/abs/2203.01017), arXiv:2203.01017
- [TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance](https://arxiv.org/abs/2105.07624), arXiv:2105.07624
- [FinQA: A Dataset of Numerical Reasoning over Financial Data](https://arxiv.org/abs/2109.00122), arXiv:2109.00122
- [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/abs/2211.12588), arXiv:2211.12588
- [Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs](https://arxiv.org/abs/2402.14903), arXiv:2402.14903
- [FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information](https://arxiv.org/abs/2505.20650), arXiv:2505.20650
- [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/abs/2408.02442), arXiv:2408.02442
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), arXiv:2307.03172
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896), arXiv:2303.08896
- [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495), arXiv:2309.11495
- [FinanceBench: A New Benchmark for Financial Question Answering](https://arxiv.org/abs/2311.11944), arXiv:2311.11944
- [Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models](https://arxiv.org/abs/2304.07619), arXiv:2304.07619
- [A Test of Lookahead Bias in LLM Forecasts](https://arxiv.org/abs/2512.23847), arXiv:2512.23847
- [Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance](https://arxiv.org/abs/2601.13770), arXiv:2601.13770
- [Summoning the Oracle to Slay It: Mitigating Look-Ahead Bias in Financial Backtesting with Large Language Models](https://arxiv.org/abs/2605.24564), arXiv:2605.24564
