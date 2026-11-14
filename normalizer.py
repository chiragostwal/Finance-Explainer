import re
from typing import Dict, List, Tuple

# ==========================================================
# 1. CANONICAL METRIC LEXICON (PREVENTS "REVENUE" MISLABELING)
# ==========================================================
METRIC_LEXICON: List[Tuple[str, re.Pattern]] = [
    ("Net Interest Income (NII)", re.compile(r"\b(net interest income|nii|interest earned)\b", re.I)),
    ("Net Interest Margin (NIM)", re.compile(r"\b(net interest margin|nim)\b", re.I)),
    ("Fee Income", re.compile(r"\b(fee income)\b", re.I)),
    ("Operating Expenses", re.compile(r"\b(operating expenses|opex|employee benefit)\b", re.I)),
    ("Core Operating Profit", re.compile(r"\b(core operating profit|operating profit before provisions|ppop)\b", re.I)),
    ("Operating Profit (EBIT/EBITDA)", re.compile(r"\b(op\.?\s*ebitda|ebitdaal|ebitda|operating profit|ebit)\b", re.I)),
    ("Profit Before Tax (PBT)", re.compile(r"\b(profit before tax|pbt)\b", re.I)),
    ("Profit After Tax (PAT)", re.compile(r"\b(profit after tax|net profit|\bpat\b|profit for the period)\b", re.I)),
    ("Standalone / Consolidated RoE", re.compile(r"\b(roe|return on equity|roa|return on assets)\b", re.I)),
    ("Revenue from Operations", re.compile(r"\b(revenue from operations|net sales|revenue|total income)\b", re.I)),
    ("Gross Written Premium (GWP)", re.compile(r"\b(gross written premium|gwp)\b", re.I)),
    ("Annualized Premium Equivalent (APE)", re.compile(r"\b(annualized premium equivalent|\bape\b)\b", re.I)),
    ("Value of New Business (VNB)", re.compile(r"\b(value of new business|\bvnb\b)\b", re.I)),
    ("Loan Book (Advances)", re.compile(
        r"\b(loan portfolio|advances|domestic book|overseas book|retail and rural|business banking|"
        r"domestic corporate|mortgages|vehicle loans|auto finance|commercial vehicle|personal loans|"
        r"credit cards|jewel loans|farmer finance)\b", re.I
    )),
    ("Deposits", re.compile(r"\b(outstanding deposits|average deposits|total deposits|casa|current|savings|term)\b", re.I)),
    ("Assets Under Management (AUM)", re.compile(r"\b(assets under management|\baum\b)\b", re.I)),
    ("Asset Quality (NPA/Provisions)", re.compile(
        r"\b(net npa|gross npa|gnpa|nnpa|slippages|provision coverage|contingency provisions|provisions|"
        r"recoveries and upgrades|written off)\b", re.I
    )),
    ("Capital Adequacy (CET1/CAR)", re.compile(r"\b(cet\s*-?\s*1|common equity tier|capital adequacy|tier i\b|tier ii\b|risk weighted assets|net worth)\b", re.I)),
    ("Order Book / TCV", re.compile(r"\b(order book|order backlog|order inflow|tcv|deal wins)\b", re.I)),
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
# 2. SLIDE UNIT & PERIOD SCOPE DETECTOR
# ==========================================================
UNIT_HEADER_PAT = re.compile(
    r"\(\s*(₹\s*billion|₹\s*crore|₹\s*cr|rs\.?\s*crore|rs\.?\s*cr|inr\s*million|inr\s*cr|us\$\s*million|us\$\s*mn|us\$\s*billion|\$\s*mn|\$\s*bn|€\s*million|€\s*mn)\s*\)",
    re.I
)

PERIOD_TOKEN_PAT = re.compile(
    r"\b(Q[1-4][-\s]?FY\d{2,4}|Q[1-4][-\s]?202\d|[1-4]Q\s*FY\d{2,4}|H[12][-\s]?FY\d{2,4}|FY\d{2,4}|"
    r"(?:Sep|September|Jun|June|Mar|March|Dec|December)\s+\d{1,2},?\s+202\d)",
    re.I
)


def detect_slide_unit_scope(slide_text: str) -> str:
    m = UNIT_HEADER_PAT.search(slide_text[:450])
    if not m:
        return ""
    raw = m.group(1).lower().replace(" ", "")
    if "billion" in raw and ("₹" in raw or "rs" in raw or "inr" in raw):
        return "₹ bn"
    if "crore" in raw or "cr" in raw:
        return "₹ Cr"
    if "inrmillion" in raw:
        return "₹ Mn"
    if "us$million" in raw or "$mn" in raw:
        return "US$ Mn"
    if "us$billion" in raw or "$bn" in raw:
        return "US$ Bn"
    if "€" in raw:
        return "€ Mn"
    return m.group(1).strip()


def detect_slide_latest_period(slide_text: str, fallback_period: str = "Q2-2027") -> str:
    clean_header = re.sub(r"\s+", " ", slide_text[:450])
    periods = [re.sub(r"\s+", " ", p).strip() for p in PERIOD_TOKEN_PAT.findall(clean_header)]
    current_periods = [p for p in periods if any(y in p for y in ("2026", "2027", "27"))]
    for p in reversed(current_periods):
        if any(k in p.lower() for k in ("sep", "q2", "h1", "dec", "q3")):
            return p
    return fallback_period


# ==========================================================
# 3. MULTI-ROW TABLE SCANNER & GREEDY CLAUSE EXTRACTOR
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
    r"(?:grew by|increased by|rose by|rose|was|were|stood at)\s+"
    r"(.{3,130}?(?:₹\s*[\d,]+(?:\.\d+)?\s*(?:lakh\s+crore|crore|cr|bn|billion)|[\d\.]+%)(?:\s+in\s+[A-Za-z0-9\-]+)?)",
    re.I
)

BAD_SUBJECT_PREFIX = re.compile(
    r"^(during|there|at\s+sep|at\s+jun|excluding|including|as\s+we|as\s+i|if\s+i|so\s+where|also|that\s+is|the\s+total\s+provisions\s+during)\b",
    re.I
)


def split_into_atomic_clauses(text: str) -> List[str]:
    norm = re.sub(r"\s*\|\s*•\s*|\s*•\s*|\s+-\s*\|\s*", "\n", text)
    norm = re.sub(r"(?:^|\s+)\d{1,2}\.\s+", "\n", norm)
    norm = re.sub(r"\.\s+(?=[A-Z])", "\n", norm)
    clauses = []
    for raw_line in norm.splitlines():
        clean_line = re.sub(r"\s*\|\s*", " ", raw_line)
        clean_line = re.sub(r"\s+", " ", clean_line).strip(" -•|.")
        if len(clean_line) >= 15:
            clauses.append(clean_line)
    return clauses


def extract_typed_facts_from_Sources(
    xbrl_dict: dict,
    pres_text: str,
    trans_text: str
) -> List[Dict[str, str]]:
    """Builds a deterministic list of typed, unit-locked facts [F001..F045] from XBRL, Slides, and Concall."""
    facts: List[Dict[str, str]] = []
    seen_keys = set()

    all_periods = PERIOD_TOKEN_PAT.findall(f"{(pres_text or '')[:1000]} {(trans_text or '')[:1000]}")
    doc_period = next((p for p in all_periods if any(k in p.lower() for k in ("q2-2027", "q2 fy27", "sep 30, 2026"))), "Q2-2027")

    def add_fact(period: str, scope: str, metric: str, value: str, change: str, src: str):
        period_clean = re.sub(r"\s+", " ", period).strip()
        scope_clean = re.sub(r"\s+", " ", scope).strip(" -•|:")
        if not scope_clean or len(scope_clean) < 2 or scope_clean.lower() in ("and", "of", "the", "growth"):
            return
        key = (scope_clean.lower(), metric.lower(), value.lower()[:35])
        if key in seen_keys:
            return
        seen_keys.add(key)
        fid = f"F{len(facts) + 1:03d}"
        facts.append({
            "id": fid,
            "period": period_clean,
            "scope": scope_clean,
            "metric": metric,
            "value": value.strip(),
            "change": change.strip(),
            "src": src
        })

    # 1. XBRL Ground-Truth Facts
    p_end = xbrl_dict.get("Period_End", doc_period)
    for k, label in [
        ("Revenue_From_Operations_Cr", "Revenue from Operations"),
        ("Interest_Earned_Cr", "Net Interest / Interest Earned"),
        ("Total_Income_Cr", "Total Income"),
        ("Profit_Before_Tax_Cr", "Profit Before Tax (PBT)"),
        ("Net_Profit_PAT_Cr", "Profit After Tax (PAT)")
    ]:
        if k in xbrl_dict:
            add_fact(p_end, "Consolidated", label, f"₹{xbrl_dict[k]} Cr", "XBRL Reported", "xbrl")

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
        slide_period = detect_slide_latest_period(chunk, fallback_period=doc_period)

        # Pass A: Structured Table Rows
        for raw_line in lines[1:]:
            line = re.sub(r"([a-z\(])\s*\|\s*([a-z])", r"\1 \2", raw_line)
            row_matches = list(TABLE_ROW_PAT.finditer(line))
            if row_matches:
                for m_row in row_matches:
                    row_label = re.sub(r"^\d{4}\s*\|\s*", "", m_row.group(1)).strip(" -•|")
                    if row_label.lower().startswith(("sep ", "jun ", "q1", "q2", "% share", "total capital")):
                        continue
                    v1, v2, v3 = m_row.group(2), m_row.group(3), m_row.group(4)
                    p1, p2, p3 = m_row.group(5), m_row.group(6), m_row.group(7)
                    if not v2 and not v3:
                        continue
                    latest_val = v3 or v2
                    unit_suffix = f" {unit_scope}" if unit_scope else ""
                    val_str = f"{latest_val}{unit_suffix}".strip()
                    if unit_scope == "₹ bn" and latest_val and not latest_val.startswith("("):
                        try:
                            cr_val = round(float(latest_val.replace(",", "")) * 100, 1)
                            val_str = f"₹{latest_val} bn (₹{cr_val:,.1f} Cr)"
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
                subj = m_cl.group(1).strip(" -•0123456789.")
                if BAD_SUBJECT_PREFIX.search(subj):
                    continue
                rest = clause[m_cl.start(2):].strip()
                metric_type = classify_metric_label(subj, slide_heading)
                if metric_type == "Reported Metric":
                    continue
                p_match = PERIOD_TOKEN_PAT.search(rest) or PERIOD_TOKEN_PAT.search(clause)
                period_str = p_match.group(1) if p_match else slide_period
                add_fact(period_str, "Consolidated", metric_type, f"{subj}: {rest}", "Slide Highlight", slide_tag)

    # 3. Concall Opening Remarks Numbered Highlights
    for clause in split_into_atomic_clauses(trans_text or ""):
        m_cl = ATOMIC_FACT_CLAUSE_PAT.match(clause)
        if not m_cl:
            continue
        subj = m_cl.group(1).strip(" -•0123456789.")
        if BAD_SUBJECT_PREFIX.search(subj) or len(subj) < 3:
            continue
        rest = clause[m_cl.start(2):].strip()
        metric_type = classify_metric_label(subj, "Concall Highlights")
        if metric_type == "Reported Metric":
            continue
        p_match = PERIOD_TOKEN_PAT.search(rest) or PERIOD_TOKEN_PAT.search(clause)
        period_str = p_match.group(1) if p_match else doc_period
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