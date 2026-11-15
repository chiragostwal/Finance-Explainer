import re
from typing import Dict, List, Tuple

# ==========================================================
# 1. CANONICAL METRIC LEXICON (ALIGNED WITH SKILLS.YAML V3)
# ==========================================================
METRIC_LEXICON: List[Tuple[str, re.Pattern]] = [
    ("Net Interest Income (NII)", re.compile(r"\b(net interest income|nii|interest earned)\b", re.I)),
    ("Net Interest Margin (NIM)", re.compile(r"\b(net interest margin|nim)\b", re.I)),
    ("Fee / Broking Income", re.compile(r"\b(fee income|broking revenue|broking income)\b", re.I)),
    ("Operating Expenses", re.compile(r"\b(operating expenses|opex|employee benefit)\b", re.I)),
    ("Core Operating Profit", re.compile(r"\b(core operating profit|operating profit before provisions|ppop)\b", re.I)),
    ("Operating Profit (EBIT/EBITDA)", re.compile(r"\b(op\.?\s*ebitda|ebitdaal|ebitda\s*per\s*ton(?:ne)?|ebitda|operating profit|ebit)\b", re.I)),
    ("Profit Before Tax (PBT)", re.compile(r"\b(profit before tax|pbt)\b", re.I)),
    ("Profit After Tax (PAT)", re.compile(r"\b(profit after tax|net profit|\bpat\b|profit for the period)\b", re.I)),
    ("Standalone / Consolidated RoE", re.compile(r"\b(roe|return on equity|roa|return on assets|roce)\b", re.I)),
    ("Revenue from Operations", re.compile(
        r"\b(revenue from operations|net sales|advertising revenue|ad revenue|subscription revenue|revenue|total income)\b", re.I
    )),
    ("Gross Written Premium (GWP/NWP)", re.compile(r"\b(gross written premium|net written premium|gwp|nwp|nbp)\b", re.I)),
    ("Annualized Premium Equivalent (APE)", re.compile(r"\b(annualized premium equivalent|\bape\b)\b", re.I)),
    ("Value of New Business (VNB)", re.compile(r"\b(value of new business|\bvnb\b)\b", re.I)),
    ("Loan Book (Advances)", re.compile(
        r"\b(loan portfolio|advances|domestic book|overseas book|retail and rural|business banking|"
        r"domestic corporate|mortgages|vehicle loans|auto finance|commercial vehicle|personal loans|"
        r"credit cards|jewel loans|farmer finance|disbursement)\b", re.I
    )),
    ("Deposits", re.compile(r"\b(outstanding deposits|average deposits|total deposits|casa|current|savings|term)\b", re.I)),
    ("Assets Under Management (AUM/SIP)", re.compile(r"\b(assets under management|\baum\b|sip book|adtv|daily turnover|tpv|payment volume)\b", re.I)),
    ("Asset Quality (NPA/Provisions)", re.compile(
        r"\b(net npa|gross npa|gnpa|nnpa|slippages|provision coverage|contingency provisions|provisions|"
        r"recoveries and upgrades|written off|credit cost)\b", re.I
    )),
    ("Capital Adequacy (CET1/CAR)", re.compile(r"\b(cet\s*-?\s*1|common equity tier|capital adequacy|tier i\b|tier ii\b|risk weighted assets|net worth|solvency ratio)\b", re.I)),
    ("Order Book / Pre-Sales / TCV", re.compile(r"\b(order book|order backlog|order inflow|tcv|acv|deal wins|pre-sales|presales|collections|gdv|gross booking value)\b", re.I)),
    ("Aviation / Hospital / Hotel Unit Metrics", re.compile(r"\b(rpk|ask|cask|rask|load factor|arpob|alos|bed occupancy|arr|revpar|average room rate|occupancy)\b", re.I)),
    ("Refining / Power / Physical Volume", re.compile(r"\b(grm|refining margin|plf|plant load factor|installed capacity|cargo volume|saleable steel|crude steel|throughput)\b", re.I)),
    ("Capex", re.compile(r"\b(capex|capital expenditure)\b", re.I)),
    ("ARPU", re.compile(r"\b(arpu)\b", re.I)),
]


def classify_metric_label(row_label: str, slide_heading: str = "") -> str:
    for canonical, pat in METRIC_LEXICON:
        if pat.search(row_label):
            return canonical
    for canonical, pat in METRIC_LEXICON:
        if pat.search(slide_heading):
            return canonical
    return "Reported Metric"


# ==========================================================
# 2. DETERMINISTIC INDIAN FISCAL CALENDAR & PERIOD NORMALIZER
# ==========================================================
# Uses \u20b9 for INR Rupee symbol and \u20ac for Euro symbol so Windows PowerShell never corrupts them into '?'
UNIT_HEADER_PAT = re.compile(
    r"\(\s*(\u20b9\s*billion|\u20b9\s*crore|\u20b9\s*cr|rs\.?\s*crore|rs\.?\s*cr|inr\s*million|inr\s*cr|us\$\s*million|us\$\s*mn|us\$\s*billion|\$\s*mn|\$\s*bn|\u20ac\s*million|\u20ac\s*mn)\s*\)",
    re.I
)

PERIOD_TOKEN_PAT = re.compile(
    r"\b(Q[1-4][-\s]?FY\d{2,4}|Q[1-4][-\s]?202\d|[1-4]Q\s*FY\d{2,4}|H[12][-\s]?FY\d{2,4}|9M[-\s]?FY\d{2,4}|FY\d{2,4}|"
    r"(?:Sep|September|Jun|June|Mar|March|Dec|December)\s+\d{1,2},?\s+202\d)",
    re.I
)

JUNK_NEWS_PAT = re.compile(
    r"(stock price, news, quote and history|share price today|stock price live|step-by-step guide|"
    r"how to add money|changes net banking url|stock forecast|analyst prediction|outlook for the week|"
    r"option chain|nifty|sensex)",
    re.I
)


def date_to_indian_fiscal_quarter(date_str: str) -> str:
    """Deterministically converts YYYY-MM-DD into Indian Fiscal Quarter (e.g. 2026-10-17 -> Q3 FY27)."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str.strip())
    if not m:
        return "Reported Period"
    year, month = int(m.group(1)), int(m.group(2))
    if month in (4, 5, 6):
        return f"Q1 FY{str(year + 1)[-2:]}"
    if month in (7, 8, 9):
        return f"Q2 FY{str(year + 1)[-2:]}"
    if month in (10, 11, 12):
        return f"Q3 FY{str(year + 1)[-2:]}"
    return f"Q4 FY{str(year)[-2:]}"


def canonicalize_period_token(raw_period: str) -> str:
    s = re.sub(r"\s+", " ", raw_period).strip()
    u = s.upper()
    m_q = re.match(r"^(?:Q([1-4])[-\s]?(?:FY)?(\d{2,4})|([1-4])Q\s*FY(\d{2,4}))$", u)
    if m_q:
        q_num = m_q.group(1) or m_q.group(3)
        yr = (m_q.group(2) or m_q.group(4))[-2:]
        return f"Q{q_num} FY{yr}"
    if re.search(r"(SEP|SEPTEMBER)\s+30,?\s+2026", u):
        return "Q2 FY27 (Sep 30, 2026)"
    if re.search(r"(JUN|JUNE)\s+30,?\s+2026", u):
        return "Q1 FY27 (Jun 30, 2026)"
    if re.search(r"(DEC|DECEMBER)\s+31,?\s+2026", u):
        return "Q3 FY27 (Dec 31, 2026)"
    return s


def detect_document_primary_period(text: str, default_period: str = "Q2 FY27") -> str:
    if not text:
        return default_period
    tokens = PERIOD_TOKEN_PAT.findall(text[:1500])
    current = [t for t in tokens if any(y in t for y in ("2026", "2027", "27"))]
    for p in current:
        if any(k in p.lower() for k in ("q2", "sep", "h1", "q1", "jun", "q3", "dec")):
            return canonicalize_period_token(p)
    if current:
        return canonicalize_period_token(current[0])
    return default_period


def detect_slide_unit_scope(slide_text: str) -> str:
    m = UNIT_HEADER_PAT.search(slide_text[:450])
    if not m:
        return ""
    raw = m.group(1).lower().replace(" ", "")
    if "billion" in raw and ("\u20b9" in raw or "rs" in raw or "inr" in raw):
        return "\u20b9 bn"
    if "crore" in raw or "cr" in raw:
        return "\u20b9 Cr"
    if "inrmillion" in raw:
        return "\u20b9 Mn"
    if "us$million" in raw or "$mn" in raw:
        return "US$ Mn"
    if "us$billion" in raw or "$bn" in raw:
        return "US$ Bn"
    if "\u20ac" in raw:
        return "\u20ac Mn"
    return m.group(1).strip()


def detect_slide_latest_period(slide_text: str, fallback_period: str = "Q2 FY27") -> str:
    clean_header = re.sub(r"\s+", " ", slide_text[:450])
    periods = [re.sub(r"\s+", " ", p).strip() for p in PERIOD_TOKEN_PAT.findall(clean_header)]
    current_periods = [p for p in periods if any(y in p for y in ("2026", "2027", "27"))]
    for p in reversed(current_periods):
        if any(k in p.lower() for k in ("sep", "q2", "h1", "dec", "q3")):
            return canonicalize_period_token(p)
    return fallback_period


# ==========================================================
# 3. NEWS-BY-ID CATALOG & EVENT CLASSIFIER ([N01..N08])
# ==========================================================
def classify_headline_event_type(text: str) -> str:
    u = text.upper()
    if any(k in u for k in ("52-WEEK LOW", "FALLS", "SLIPS", "TANKS", "CRASH", "STRIKE", "BREACH", "NOTICE", "PENALTY", "HEADWIND", "LOSS", "MISSES")):
        return "RISK_FACTOR"
    if any(k in u for k in ("CAPEX", "EXPANSION", "CAPACITY", "PLANT", "GW", "GREENFIELD", "BROWNFIELD", "INVEST")):
        return "CAPEX"
    if any(k in u for k in ("ORDER", "TCV", "CONTRACT", "EPC", "EPCIC", "DEAL", "WINS", "BACKLOG")):
        return "ORDER_BOOK"
    if any(k in u for k in ("JV", "PARTNER", "ACQUISITION", "MERGER", "STAKE", "BUYOUT", "SUBSIDIARY", "RIGHTS TO OPERATE", "TAKE OVER", "SALE")):
        return "PARTNERSHIP_JV"
    if any(k in u for k in ("PRICE HIKE", "TARIFF", "PRICING", "MARGIN", "NIM")):
        return "PRICING_ACTION"
    if any(k in u for k in ("RBI", "SEBI", "TRAI", "USFDA", "FSSAI", "GOVT", "MINISTRY", "APPROVAL", "REGULAT", "CARE RATINGS", "UPGRADE")):
        return "POLICY_REGULATION"
    return "OPERATIONAL_MILESTONE"


def extract_news_catalog(news_text: str) -> List[Dict[str, str]]:
    """Numbers every valid news item [N01..N08] with its exact [YYYY-MM-DD] date and event_type."""
    news_items: List[Dict[str, str]] = []
    seen_stems = set()
    for raw_line in (news_text or "").splitlines():
        line = raw_line.strip()
        if not line or JUNK_NEWS_PAT.search(line):
            continue
        m = re.match(r"^\[(\d{4}-\d{2}-\d{2})\]\s*(.+)$", line)
        if not m:
            continue
        dt_str, headline = m.group(1), m.group(2).strip()
        stem = re.sub(r"\W+", " ", headline.lower()).strip()[:45]
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        nid = f"N{len(news_items) + 1:02d}"
        fiscal_q = date_to_indian_fiscal_quarter(dt_str)
        ev_type = classify_headline_event_type(headline)
        news_items.append({
            "id": nid,
            "date": dt_str,
            "fiscal_period": fiscal_q,
            "timeline": f"[{dt_str}]",
            "event_type": ev_type,
            "headline": headline,
            "raw_line": f"[{dt_str}] {headline}"
        })
        if len(news_items) >= 8:
            break
    return news_items


def format_news_catalog(news_items: List[Dict[str, str]]) -> str:
    if not news_items:
        return "None"
    return "\n".join(
        f"[{n['id']} | date={n['date']} | type={n['event_type']} | headline={n['headline']}]"
        for n in news_items
    )


# ==========================================================
# 4. SPEAKER & MODALITY TAGGING FOR GUIDANCE ([G01..G06])
# ==========================================================
ANALYST_QUESTION_PAT = re.compile(
    r"(\?|quick question|two questions|three questions|my question is|can you share|how do you see|"
    r"where are you seeing|if you could add|congratulations on)",
    re.I
)

GUIDANCE_SIGNAL_PAT = re.compile(
    r"\b(guidance|outlook|target|expect|aim|ambition|going forward|full year|fy\d{2}|"
    r"margin to|margins to|credit cost|credit growth|capex plan|pipeline|trajectory)\b",
    re.I
)

HEDGED_MODALITY_PAT = re.compile(
    r"\b(difficult to predict|too early to|uncertain|cannot quantify|hard to say|not giving guidance|"
    r"depends on|subject to|may vary|volatile)\b",
    re.I
)


def extract_guidance_catalog(pres_text: str, trans_text: str, doc_period: str = "Q2 FY27") -> List[Dict[str, str]]:
    """Extracts full-context management guidance & forward outlook [G01..G06], preserving multi-line context."""
    guidance_items: List[Dict[str, str]] = []
    seen = set()

    metric_anchor_pat = re.compile(
        r"\b(revenue|margin|ebitda|pat|roe|roa|roce|aum|nim|credit cost|credit growth|loan|capex|order|"
        r"volume|export|demand|realization|utilization|fy27|fy28|h2|q3|q4|full year|long term|medium term)\b",
        re.I
    )

    for src_label, text in [("concall", trans_text or ""), ("pres", pres_text or "")]:
        clean_block = re.sub(r"\s*\|\s*", " ", text)
        clean_block = clean_block.replace("\r\n", "\n").replace("\r", "\n")
        clean_block = re.sub(r"(?<!\n)\n(?!\n)", " ", clean_block)
        clean_block = re.sub(r"\b(Rs|Cr|Mr|Dr|Ms|St|vs|approx)\.\s+", r"\1_DOT_ ", clean_block, flags=re.I)

        raw_sents = re.split(r"(?<=[\.\!])\s+(?=[A-Z0-9])|\n{2,}", clean_block)
        restored_sents = [s.replace("_DOT_ ", ". ").strip(" -*|") for s in raw_sents if s.strip()]

        for idx, sent in enumerate(restored_sents):
            sent = re.sub(r"\s+", " ", sent).strip()
            if len(sent) < 20 or len(sent) > 340:
                continue
            if ANALYST_QUESTION_PAT.search(sent):
                continue
            if not GUIDANCE_SIGNAL_PAT.search(sent):
                continue

            if len(sent) < 65 and idx > 0 and not ANALYST_QUESTION_PAT.search(restored_sents[idx - 1]):
                prev_s = re.sub(r"\s+", " ", restored_sents[idx - 1]).strip()
                if 20 <= len(prev_s) <= 220:
                    sent = f"{prev_s} {sent}"

            if len(sent) < 42 or not metric_anchor_pat.search(sent):
                continue

            has_num = bool(re.search(r"\b\d+(\.\d+)?%|\b\d+\s*bps|\u20b9|\$|crore|cr\b|bn\b|gw\b", sent, re.I))
            has_strong_kw = bool(re.search(r"\b(guidance|outlook|target|expect|trajectory|going forward|h2|q3|q4|fy27)\b", sent, re.I))
            if not has_num and not has_strong_kw:
                continue

            if HEDGED_MODALITY_PAT.search(sent):
                modality = "HEDGED"
            elif has_num:
                modality = "GUIDANCE"
            else:
                modality = "QUALITATIVE_OUTLOOK"

            stem = re.sub(r"\W+", " ", sent.lower())[:55]
            if stem in seen:
                continue
            seen.add(stem)
            gid = f"G{len(guidance_items) + 1:02d}"
            guidance_items.append({
                "id": gid,
                "period": doc_period,
                "modality": modality,
                "text": sent[:320],
                "src": src_label
            })
            if len(guidance_items) >= 6:
                break
    return guidance_items


def format_guidance_catalog(guidance_items: List[Dict[str, str]]) -> str:
    if not guidance_items:
        return "None"
    return "\n".join(
        f"[{g['id']} | period={g['period']} | modality={g['modality']} | src={g['src']} | statement=\"{g['text']}\"]"
        for g in guidance_items
    )


# ==========================================================
# 5. MULTI-ROW TABLE SCANNER & GREEDY CLAUSE EXTRACTOR
# ==========================================================
TABLE_ROW_PAT = re.compile(
    r"(?:^|\|\s*)(-?\s*[A-Za-z][A-Za-z0-9\s\-\(\)/,&\.]{2,52}?)\s*\|\s*"
    r"([\d,]+\.\d+|\(\d[\d,]*\.\d+\))"
    r"(?:\s*\|\s*([\d,]+\.\d+|\(\d[\d,]*\.\d+\)))?"
    r"(?:\s*\|\s*|\s+)([\d,]+\.\d+|\(\d[\d,]*\.\d+\))?"
    r"\s*\|\s*([\d\.]+%|\([\d\.]+%?\)?|-)"
    r"(?:\s*\|\s*([\d\.]+%|\([\d\.]+%?\)?|-))?"
    r"(?:\s*\|\s*([\d\.]+%|\([\d\.]+%?\)?|-))?"
)

ATOMIC_FACT_CLAUSE_PAT = re.compile(
    r"^([A-Z][A-Za-z0-9\s\-\(\),/]{2,65}?)\s+"
    r"(?:grew by|increased by|rose by|rose|was|were|stood at|delivered|clocked|reported|crossed|touched|jumped|registered|approached)\s+"
    r"(.{3,140}?(?:(?:\u20b9|rs\.?|inr)\s*[\d,]+(?:\.\d+)?\s*(?:lakh\s+crore|crore|cr|bn|billion)|[\d,]+\s*(?:lakh\s+units|units)|[\d\.]+%)(?:\s+in\s+[A-Za-z0-9\-]+)?)",
    re.I
)

BAD_SUBJECT_PREFIX = re.compile(
    r"^(during|there|at\s+sep|at\s+jun|excluding|including|as\s+we|as\s+i|if\s+i|so\s+where|also|that\s+is|"
    r"the\s+total\s+provisions\s+during|.*as\s+you\s+can\s+see|.*you\s+would\s+have)",
    re.I
)


def split_into_atomic_clauses(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    norm = re.sub(r"\s*\|\s*\u2022\s*|\s*\u2022\s*|\s+-\s*\|\s*", "\n", text)
    norm = re.sub(r"(?:^|\s+)\d{1,2}\.\s+", "\n", norm)
    norm = re.sub(r"\.\s+(?=[A-Z])", "\n", norm)
    clauses = []
    for raw_line in norm.splitlines():
        clean_line = re.sub(r"\s*\|\s*", " ", raw_line)
        clean_line = re.sub(r"\s+", " ", clean_line).strip(" -\u2022|.")
        if len(clean_line) >= 15 and not ANALYST_QUESTION_PAT.search(clean_line):
            clauses.append(clean_line)
    return clauses


def extract_typed_facts_from_Sources(
    xbrl_dict: dict,
    pres_text: str,
    trans_text: str
) -> List[Dict[str, str]]:
    """Builds a deterministic list of typed, unit-locked facts [F001..F045] with independent document-period scope."""
    facts: List[Dict[str, str]] = []
    seen_keys = set()

    pres_doc_period = detect_document_primary_period(pres_text, default_period="Q2 FY27")
    trans_doc_period = detect_document_primary_period(trans_text, default_period=pres_doc_period)

    def add_fact(period: str, scope: str, metric: str, value: str, change: str, src: str):
        period_clean = canonicalize_period_token(period)
        scope_clean = re.sub(r"\s+", " ", scope).strip(" -\u2022|:,")
        val_clean = re.sub(r"\s+", " ", value).strip(" -\u2022|:,")
        if not re.search(r"\d", val_clean) or val_clean.endswith((" to", " by", " at", " of")):
            return
        if not scope_clean or len(scope_clean) < 2 or scope_clean.lower() in ("and", "of", "the", "growth"):
            return
        key = (scope_clean.lower(), metric.lower(), val_clean.lower()[:35])
        if key in seen_keys:
            return
        seen_keys.add(key)
        fid = f"F{len(facts) + 1:03d}"
        facts.append({
            "id": fid,
            "period": period_clean,
            "scope": scope_clean,
            "metric": metric,
            "value": val_clean,
            "change": change.strip(),
            "src": src
        })

    # 1. XBRL Ground-Truth Facts
    p_end = xbrl_dict.get("Period_End", pres_doc_period)
    for k, label in [
        ("Revenue_From_Operations_Cr", "Revenue from Operations"),
        ("Interest_Earned_Cr", "Net Interest / Interest Earned"),
        ("Total_Income_Cr", "Total Income"),
        ("Profit_Before_Tax_Cr", "Profit Before Tax (PBT)"),
        ("Net_Profit_PAT_Cr", "Profit After Tax (PAT)")
    ]:
        if k in xbrl_dict:
            add_fact(p_end, "Consolidated", label, f"\u20b9{xbrl_dict[k]} Cr", "XBRL Reported", "xbrl")

    # 2. Slide Table & Bullet Facts
    slide_chunks = re.split(r"(?=\[Slide \d+\])", pres_text or "")
    for chunk in slide_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = [l.strip() for l in chunk.splitlines() if l.strip()]
        if not lines:
            continue
        slide_tag = lines[0].strip("[]").lower().replace(" ", "") if lines[0].startswith("[Slide") else "slide"
        slide_heading = " ".join(lines[1:3]) if len(lines) > 1 else ""
        unit_scope = detect_slide_unit_scope(chunk)
        slide_period = detect_slide_latest_period(chunk, fallback_period=pres_doc_period)

        # Pass A: Structured Table Rows
        for raw_line in lines[1:]:
            line = re.sub(r"([a-z\(])\s*\|\s*([a-z])", r"\1 \2", raw_line)
            row_matches = list(TABLE_ROW_PAT.finditer(line))
            if row_matches:
                for m_row in row_matches:
                    row_label = re.sub(r"^\d{4}\s*\|\s*", "", m_row.group(1)).strip(" -\u2022|")
                    if row_label.lower().startswith(("sep ", "jun ", "q1", "q2", "% share", "total capital")):
                        continue
                    v1, v2, v3 = m_row.group(2), m_row.group(3), m_row.group(4)
                    p1, p2, p3 = m_row.group(5), m_row.group(6), m_row.group(7)
                    if not v2 and not v3:
                        continue
                    latest_val = v3 or v2
                    is_negative = latest_val.startswith("(") and latest_val.endswith(")")
                    unit_suffix = f" {unit_scope}" if unit_scope else ""
                    val_str = f"{latest_val}{unit_suffix}".strip()
                    if is_negative:
                        val_str = f"Negative / Loss {val_str}"
                    elif unit_scope == "\u20b9 bn" and latest_val:
                        try:
                            cr_val = round(float(latest_val.replace(",", "")) * 100, 1)
                            val_str = f"\u20b9{latest_val} bn (\u20b9{cr_val:,.1f} Cr)"
                        except ValueError:
                            pass

                    chg_parts = []
                    if p1 and p1 != "-":
                        chg_parts.append(f"{p1 if '%' in p1 else p1 + '%'} YoY")
                    if p2 and p2 != "-":
                        chg_parts.append(f"{p2 if '%' in p2 else p2 + '%'} QoQ")
                    if p3 and p3 != "-":
                        chg_parts.append(f"{p3} share")

                    metric_type = classify_metric_label(row_label, slide_heading)
                    add_fact(slide_period, row_label, metric_type, val_str, ", ".join(chg_parts) or "Reported", slide_tag)

        # Pass B: Slide Bullet Highlights
        for clause in split_into_atomic_clauses("\n".join(lines[1:])):
            m_cl = ATOMIC_FACT_CLAUSE_PAT.match(clause)
            if m_cl:
                subj = m_cl.group(1).strip(" -\u20220123456789.,")
                if BAD_SUBJECT_PREFIX.search(subj):
                    continue
                rest = clause[m_cl.start(2):].strip()
                metric_type = classify_metric_label(subj, slide_heading)
                if metric_type == "Reported Metric":
                    continue
                p_match = PERIOD_TOKEN_PAT.search(rest) or PERIOD_TOKEN_PAT.search(clause)
                period_str = p_match.group(1) if p_match else slide_period
                add_fact(period_str, "Consolidated", metric_type, f"{subj}: {rest}", "Slide Highlight", slide_tag)

    # 3. Concall Opening Remarks Highlights
    for clause in split_into_atomic_clauses(trans_text or ""):
        m_cl = ATOMIC_FACT_CLAUSE_PAT.match(clause)
        if not m_cl:
            continue
        subj = m_cl.group(1).strip(" -\u20220123456789.,")
        if BAD_SUBJECT_PREFIX.search(subj) or len(subj) < 3:
            continue
        rest = clause[m_cl.start(2):].strip()
        metric_type = classify_metric_label(subj, "Concall Highlights")
        if metric_type == "Reported Metric":
            continue
        p_match = PERIOD_TOKEN_PAT.search(rest) or PERIOD_TOKEN_PAT.search(clause)
        period_str = p_match.group(1) if p_match else trans_doc_period
        add_fact(period_str, "Consolidated", metric_type, f"{subj}: {rest}", "Concall Highlight", "concall")
        if len(facts) >= 45:
            break

    return facts[:45]


def format_typed_fact_catalog(facts: List[Dict[str, str]]) -> str:
    if not facts:
        return "None"
    return "\n".join(
        f"[{f['id']} | period={f['period']} | scope={f['scope']} | metric={f['metric']} | "
        f"value={f['value']} | change={f['change']} | src={f['src']}]"
        for f in facts
    )