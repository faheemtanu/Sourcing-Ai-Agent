from __future__ import annotations
import io
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from transformers import pipeline

# ---------------------------
# Page config & styles
# ---------------------------
st.set_page_config(page_title="AI Sourcing Agent – Live", page_icon="📦", layout="wide")
st.markdown(
    """
    <style>
      body {background: #f5f7fb;}
      .card {background:#fff;border-radius:12px;padding:12px;box-shadow:0 6px 18px rgba(2,6,23,0.04);border:1px solid rgba(2,6,23,0.04);}
      div[data-testid="stMetricValue"]{font-size:1.4rem;}
      div[data-testid="stMetricLabel"]{color:#64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Template & columns
# ---------------------------
TEMPLATE_COLUMNS = [
    "Supplier Name",
    "Product Category",
    "HS_Code",
    "Country",
    "Location",
    "Minimum Order Quantity",
    "Lead Time (days)",
    "Rating",
    "Verified",
    "Verified Source",
    "Contact Email",
]

COLUMN_ALIASES = {
    "supplier": "Supplier Name",
    "supplier name": "Supplier Name",
    "product": "Product Category",
    "category": "Product Category",
    "hs code": "HS_Code",
    "hscode": "HS_Code",
    "country": "Country",
    "city": "Location",
    "location": "Location",
    "moq": "Minimum Order Quantity",
    "minimum order": "Minimum Order Quantity",
    "leadtime": "Lead Time (days)",
    "lead time": "Lead Time (days)",
    "delivery days": "Lead Time (days)",
    "rating": "Rating",
    "verified": "Verified",
    "verified source": "Verified Source",
    "email": "Contact Email",
    "contact": "Contact Email",
}

# ---------------------------
# Helpers
# ---------------------------
def guess_verified_source(email: Optional[str]) -> str:
    if pd.isna(email) or not isinstance(email, str) or "@" not in email:
        return "Unknown"
    email = email.lower()
    # Known platforms
    if email.endswith(".gov.in"):
        return "DGFT / Govt"
    if "indiamart" in email:
        return "IndiaMART"
    if "tradeindia" in email:
        return "TradeIndia"
    if "exportersindia" in email:
        return "ExportersIndia"
    if email.endswith(".org") or email.endswith(".ngo"):
        return "NGO / Association"
    # Domain mapping
    if email.endswith(".in"):
        return "Indian Private Company"
    if email.endswith(".com"):
        return "Global Trader / Exporter"
    if email.endswith(".net"):
        return "Technology/Service Provider"
    if email.endswith(".co"):
        return "Company / Startup"
    if email.endswith(".edu") or email.endswith(".ac.in"):
        return "Educational Institution"
    return "Unknown"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
        else:
            # fuzzy match
            key2 = re.sub(r"[^\w\s]", "", key).replace("_", " ").strip()
            if key2 in COLUMN_ALIASES:
                renamed[col] = COLUMN_ALIASES[key2]
    if renamed:
        df = df.rename(columns=renamed)
    for col in TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # try to coerce numeric fields
    if "Lead Time (days)" in df.columns:
        df["Lead Time (days)"] = pd.to_numeric(df["Lead Time (days)"], errors="coerce")
    if "Minimum Order Quantity" in df.columns:
        df["Minimum Order Quantity"] = pd.to_numeric(df["Minimum Order Quantity"], errors="coerce")
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    # fill Verified Source if missing
    if "Verified Source" not in df.columns or df["Verified Source"].isna().all():
        if "Contact Email" in df.columns:
            df["Verified Source"] = df["Contact Email"].apply(guess_verified_source)
        else:
            df["Verified Source"] = "Unknown"
    # normalize Verified values
    df["Verified"] = df["Verified"].astype(str).str.strip().str.lower().map(
        {"yes": "Yes", "y": "Yes", "true": "Yes", "no": "No", "n": "No", "false": "No"}
    ).fillna("No")
    # sensible defaults
    if "Lead Time (days)" in df.columns:
        median_lead = df["Lead Time (days)"].median(skipna=True)
        if pd.isna(median_lead):
            median_lead = 7
        df["Lead Time (days)"] = df["Lead Time (days)"].fillna(median_lead)
    if "Minimum Order Quantity" in df.columns:
        df["Minimum Order Quantity"] = df["Minimum Order Quantity"].fillna(0)
    return df[TEMPLATE_COLUMNS].copy()

def load_sample_dataframe() -> pd.DataFrame:
    data = [
        ["BrightLite Industries", "LED Bulbs", "940540", "India", "Delhi", 100, 15, 4.5, "Yes", "Indian Private Company", "sales@brightlite.in"],
        ["Shakti Exports", "Basmati Rice", "100630", "India", "Karnal", 500, 12, 4.2, "Yes", "IndiaMART", "export@shaktigroup.indiamart.com"],
        ["GlobalTech Pharma", "Pharmaceuticals", "300490", "India", "Mumbai", 200, 20, 4.8, "No", "Global Trader / Exporter", "bd@globaltechpharma.com"],
    ]
    df = pd.DataFrame(data, columns=TEMPLATE_COLUMNS)
    df['HS_Code'] = df['HS_Code'].astype(str)
    return df

# ---------------------------
# LLM QA pipeline (cached)
# ---------------------------
@st.cache_resource
def get_qa_pipeline():
    try:
        return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    except Exception:
        return None

qa_pipeline = get_qa_pipeline()

# ---------------------------
# UN Comtrade helper (live fetch)
# ---------------------------
# Small mapping for common reporters (ISO numeric codes used by Comtrade API)
REPORTER_MAPPING = {
    "All": "all",
    "India": "356",
    "China": "156",
    "United States": "842"
}

def fetch_comtrade_preview(hs_code: str = "0901", reporter_code: str = "356", year: int = None, max_rows: int = 250) -> pd.DataFrame:
    """
    Fetch trade data from UN Comtrade legacy API (public endpoint).
    This function uses the comtrade 'get' endpoint and maps results to supplier-like rows.
    reporter_code can be numeric code as string (e.g., '356' for India) or 'all'.
    """
    if year is None:
        year = datetime.now().year - 1
    base = "https://comtrade.un.org/api/get"
    params = {
        "max": max_rows,
        "type": "C",
        "freq": "A",
        "px": "HS",
        "ps": year,
        "r": reporter_code,  # reporter
        "p": "all",          # partner
        "cc": hs_code,       # commodity code
        "fmt": "json"
    }
    try:
        resp = requests.get(base, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        dataset = payload.get("dataset", []) if isinstance(payload, dict) else []
        rows = []
        for item in dataset:
            # item fields differ; use partner (ptTitle) and trade flow
            partner = item.get("ptTitle") or item.get("ptCode") or ""
            reporter = item.get("rtTitle") or item.get("rtCode") or ""
            trade_value = item.get("TradeValue") or item.get("TradeValue (US$)") or item.get("TradeValue (US$)") or None
            # Create supplier-like row: use partner as Country, supplier name as "Exporters in <partner>"
            supplier_name = f"Exporters in {partner}" if partner else f"Exporter {len(rows)+1}"
            rows.append({
                "Supplier Name": supplier_name,
                "Product Category": f"HS {hs_code}",
                "HS_Code": str(hs_code),
                "Country": partner or "Unknown",
                "Location": partner or "Unknown",
                "Minimum Order Quantity": 0,
                "Lead Time (days)": None,
                "Rating": None,
                "Verified": "Yes",
                "Verified Source": "UN Comtrade",
                "Contact Email": None
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Comtrade fetch error: {e}")
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)

# ---------------------------
# Session init
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = normalize_columns(load_sample_dataframe())

# ---------------------------
# Sidebar: Upload + Live data loader
# ---------------------------
st.sidebar.title("AI Sourcing Agent")
uploaded = st.sidebar.file_uploader("Upload supplier CSV / Excel", type=["csv", "xlsx"], help="Upload your supplier list")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Load Live Data (UN Comtrade)")
with st.sidebar.form("live_fetch_form"):
    live_hs = st.text_input("HS Code (e.g., 0901 for coffee)", value="0901")
    reporter_choice = st.selectbox("Reporter (exporter country)", options=["All", "India", "China", "United States"], index=1)
    year_choice = st.number_input("Year", min_value=2000, max_value=datetime.now().year, value=datetime.now().year - 1)
    fetch_btn = st.form_submit_button("Load live Comtrade data")
if fetch_btn:
    reporter_code = REPORTER_MAPPING.get(reporter_choice, "all")
    st.sidebar.info("Fetching live trade data from UN Comtrade...")
    df_live = fetch_comtrade_preview(hs_code=live_hs, reporter_code=reporter_code, year=year_choice, max_rows=250)
    if not df_live.empty:
        # Normalize and append to session dataframe
        df_live = normalize_columns(df_live)
        # Keep source as UN Comtrade
        df_live["Verified Source"] = "UN Comtrade"
        # Append (optional: you could replace instead)
        st.session_state.df = pd.concat([st.session_state.df, df_live], ignore_index=True)
        st.sidebar.success(f"Loaded {len(df_live)} rows from UN Comtrade and appended to dataset")
    else:
        st.sidebar.error("No data returned from Comtrade or fetch failed.")

# handle file upload
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded, dtype=str)
        else:
            df_upload = pd.read_excel(uploaded, dtype=str)
        df_upload = normalize_columns(df_upload)
        # Auto-fill Verified Source from email if missing
        if "Contact Email" in df_upload.columns:
            df_upload["Verified Source"] = df_upload["Contact Email"].apply(guess_verified_source)
        else:
            df_upload["Verified Source"] = "Unknown"
        st.session_state.df = df_upload
        st.sidebar.success("Uploaded and normalized")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

df = st.session_state.df.copy()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters & Search")
q = st.sidebar.text_input("Search Supplier / Product / Location")
sel_product = st.sidebar.multiselect("Product Category", sorted(df["Product Category"].dropna().unique()))
sel_country = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
sel_location = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()))
sel_verified = st.sidebar.selectbox("Verified filter", options=["All", "Yes", "No"], index=0)
sel_source = st.sidebar.multiselect("Verified Source", sorted(df["Verified Source"].dropna().unique()))
min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), 0.1)
hs_query = st.sidebar.text_input("HS Code contains")
sort_by = st.sidebar.multiselect("Sort by", options=["Rating", "Lead Time (days)", "Minimum Order Quantity", "Supplier Name"], default=["Rating"])

mask = pd.Series(True, index=df.index)
if q:
    qm = q.strip().lower()
    mask &= (
        df["Supplier Name"].fillna("").str.lower().str.contains(qm)
        | df["Product Category"].fillna("").str.lower().str.contains(qm)
        | df["Location"].fillna("").str.lower().str.contains(qm)
        | df["Country"].fillna("").str.lower().str.contains(qm)
    )
if sel_product:
    mask &= df["Product Category"].isin(sel_product)
if sel_country:
    mask &= df["Country"].isin(sel_country)
if sel_location:
    mask &= df["Location"].isin(sel_location)
if sel_verified != "All":
    mask &= df["Verified"].astype(str).str.lower().eq(sel_verified.lower())
if sel_source:
    mask &= df["Verified Source"].isin(sel_source)
if hs_query:
    mask &= df["HS_Code"].astype(str).str.contains(hs_query, na=False, case=False)
mask &= df["Rating"].fillna(0).between(min_rating, max_rating)
filtered_df = df[mask].copy().reset_index(drop=True)
if sort_by:
    sort_cols = [c for c in sort_by if c in filtered_df.columns]
    if sort_cols:
        filtered_df = filtered_df.sort_values(by=sort_cols, ascending=[False]*len(sort_cols), na_position="last").reset_index(drop=True)

# ---------------------------
# Header & KPIs
# ---------------------------
st.markdown(f"## 📊 Supplier Intelligence Dashboard\n_Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}_")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Suppliers", len(filtered_df))
pct_verified = filtered_df["Verified"].astype(str).str.lower().eq("yes").mean() * 100 if len(filtered_df) else 0
c2.metric("Verified %", f"{pct_verified:.1f}%")
avg_rating = filtered_df["Rating"].dropna().mean() if len(filtered_df) else 0
c3.metric("Avg Rating", f"{(avg_rating if pd.notna(avg_rating) else 0):.2f}")
avg_lead = filtered_df["Lead Time (days)"].mean() if len(filtered_df) else 0
c4.metric("Avg Lead Time", f"{(avg_lead if pd.notna(avg_lead) else 0):.0f} days")

# ---------------------------
# Tabs
# ---------------------------
TAB_DASH, TAB_TABLE, TAB_EDIT, TAB_QA, TAB_IO, TAB_ABOUT = st.tabs([
    "📈 Dashboard", "📋 Suppliers", "✏️ Editor", "🤖 Ask AI (Table-aware)", "⬆️⬇️ Import & Export", "ℹ️ About"
])

with TAB_DASH:
    if filtered_df.empty:
        st.info("No data to display. Upload data, load live data, or relax filters.")
    else:
        r1, r2 = st.columns(2)
        with r1:
            grp = filtered_df.groupby(["Location", "Product Category"], as_index=False)["Supplier Name"].count()
            fig = px.bar(grp, x="Location", y="Supplier Name", color="Product Category", title="Suppliers by Location & Category")
            st.plotly_chart(fig, use_container_width=True)
            fig3 = px.histogram(filtered_df, x="Rating", nbins=10, title="Rating Distribution")
            st.plotly_chart(fig3, use_container_width=True)
        with r2:
            fig2 = px.pie(filtered_df, names="Product Category", title="Category Share")
            st.plotly_chart(fig2, use_container_width=True)
            if "Lead Time (days)" in filtered_df.columns:
                fig4 = px.box(filtered_df, y="Lead Time (days)", title="Lead Time Distribution")
                st.plotly_chart(fig4, use_container_width=True)
            top_hs = filtered_df.groupby("HS_Code", as_index=False)["Supplier Name"].count().rename(columns={"Supplier Name": "Count"}).sort_values("Count", ascending=False).head(15)
            fig5 = px.bar(top_hs, x="HS_Code", y="Count", title="Top HS Codes")
            st.plotly_chart(fig5, use_container_width=True)

with TAB_TABLE:
    st.subheader("Supplier Data")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    col_csv, col_json, col_xlsx = st.columns(3)
    with col_csv:
        st.download_button("⬇️ Download Filtered CSV", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name="suppliers_filtered.csv", mime="text/csv")
    with col_json:
        st.download_button("⬇️ Download Filtered JSON", data=filtered_df.to_json(orient="records", force_ascii=False).encode('utf-8'), file_name="suppliers_filtered.json", mime="application/json")
    with col_xlsx:
        to_write = io.BytesIO()
        with pd.ExcelWriter(to_write, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="suppliers")
            writer.save()
        to_write.seek(0)
        st.download_button("⬇️ Download Filtered Excel", data=to_write.getvalue(), file_name="suppliers_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with TAB_EDIT:
    st.subheader("Edit / Add Suppliers (session only)")
    if st.button("➕ Add Blank Row"):
        new_row = pd.DataFrame([{c: "" for c in TEMPLATE_COLUMNS}])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.experimental_rerun()
    try:
        edited = st.data_editor(st.session_state.df, use_container_width=True, num_rows="dynamic", hide_index=True, key="editor")
        st.session_state["edited_df"] = edited
    except Exception:
        st.info("Data editor not available in this Streamlit version. Download CSV, edit locally, re-upload.")
    if st.button("💾 Save Edits to Session"):
        if "edited_df" in st.session_state:
            st.session_state.df = normalize_columns(pd.DataFrame(st.session_state["edited_df"]))
            # ensure Verified Source is recalculated for any new/edited emails
            if "Contact Email" in st.session_state.df.columns:
                st.session_state.df["Verified Source"] = st.session_state.df["Contact Email"].apply(guess_verified_source)
            st.success("Saved to session")
        else:
            st.warning("No edits detected.")

# ---------------------------
# Simple natural-language -> dataframe filter parser
# ---------------------------
def parse_query_to_filters(query: str, df_sample: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[Tuple[str, Dict[str, Any]]]]:
    q = query.lower()
    filters = {}
    intent = None
    # categories
    categories = [str(x).lower() for x in df_sample["Product Category"].dropna().unique()]
    for cat in categories:
        if cat and cat in q:
            filters["Product Category"] = ("==", cat)
            break
    # location, country detection
    for loc in [str(x).lower() for x in df_sample["Location"].dropna().unique()]:
        if loc and loc in q:
            filters["Location"] = ("==", loc)
            break
    for ctry in [str(x).lower() for x in df_sample["Country"].dropna().unique()]:
        if ctry and ctry in q:
            filters["Country"] = ("==", ctry)
            break
    # verified source keywords
    if "dgft" in q or "govt" in q or "government" in q:
        filters["Verified Source"] = ("==", "DGFT / Govt")
    if "indiamart" in q:
        filters["Verified Source"] = ("==", "IndiaMART")
    if "tradeindia" in q:
        filters["Verified Source"] = ("==", "TradeIndia")
    # numeric patterns
    inline_num = re.findall(r"(moq|minimum order|lead time|rating)\s*(?:<|>|<=|>=|=)?\s*([0-9]+)", q)
    for field, val in inline_num:
        v = float(val)
        if "moq" in field:
            if "<" in q or "under" in q or "less" in q:
                filters["Minimum Order Quantity"] = ("<=", v)
            else:
                filters["Minimum Order Quantity"] = (">=", v)
        if "lead" in field:
            if "<" in q or "under" in q or "less" in q:
                filters["Lead Time (days)"] = ("<=", v)
            else:
                filters["Lead Time (days)"] = (">=", v)
        if "rating" in field:
            filters["Rating"] = (">=", v)
    # intent detection
    if any(k in q for k in ["fastest", "shortest", "lowest lead"]):
        intent = ("top_by_lead", {"n": 5})
    nmatch = re.search(r"top\s*(\d+)", q)
    if nmatch:
        intent = ("top_n", {"n": int(nmatch.group(1))})
    if any(q.strip().startswith(k) for k in ("list", "show", "give", "get")):
        intent = intent or ("list", {})
    return filters, intent

def apply_filters_dict(df_in: pd.DataFrame, filters: Dict[str, Tuple[str, Any]]) -> pd.DataFrame:
    df_out = df_in.copy()
    for col, (op, val) in filters.items():
        if col not in df_out.columns:
            continue
        if op == "==":
            df_out = df_out[df_out[col].astype(str).str.lower() == str(val).lower()]
        elif op == "<=":
            df_out = df_out[pd.to_numeric(df_out[col], errors="coerce") <= float(val)]
        elif op == ">=":
            df_out = df_out[pd.to_numeric(df_out[col], errors="coerce") >= float(val)]
        elif op == "<":
            df_out = df_out[pd.to_numeric(df_out[col], errors="coerce") < float(val)]
        elif op == ">":
            df_out = df_out[pd.to_numeric(df_out[col], errors="coerce") > float(val)]
    return df_out

with TAB_QA:
    st.subheader("🤖 Ask AI about your suppliers (table-aware)")
    user_q = st.text_area("Type a question (e.g., 'List all coffee suppliers with MOQ < 500 and lead time < 14 days')", height=120)
    if st.button("Run Query"):
        if not user_q or user_q.strip() == "":
            st.error("Please enter a question.")
        else:
            filters, intent = parse_query_to_filters(user_q, df)
            if filters or intent:
                df_candidate = apply_filters_dict(filtered_df, filters) if filters else filtered_df.copy()
                if intent and intent[0] == "top_by_lead":
                    n = intent[1].get("n", 5)
                    df_candidate = df_candidate.sort_values("Lead Time (days)", ascending=True, na_position="last").head(n)
                    st.success(f"Showing top {n} by shortest lead time.")
                    st.dataframe(df_candidate.reset_index(drop=True), use_container_width=True)
                elif intent and intent[0] == "top_n":
                    n = intent[1].get("n", 5)
                    st.success(f"Showing top {n} matching suppliers.")
                    st.dataframe(df_candidate.head(n).reset_index(drop=True), use_container_width=True)
                else:
                    if not df_candidate.empty:
                        st.success(f"Found {len(df_candidate)} suppliers matching your query.")
                        st.dataframe(df_candidate.reset_index(drop=True), use_container_width=True)
                        st.download_button("⬇️ Download results (CSV)", data=df_candidate.to_csv(index=False).encode('utf-8'), file_name="qa_results.csv", mime="text/csv")
                    else:
                        st.warning("Deterministic parsing found no matches — falling back to semantic QA.")
            # fallback to semantic QA
            if qa_pipeline is None:
                st.error("LLM QA model not available; deterministic parsing did not return results.")
            else:
                # build context from top-k relevant rows
                q_tokens = set(re.findall(r"\w+", user_q.lower()))
                scores = []
                texts = []
                for _, row in filtered_df.iterrows():
                    text = " ".join([str(row.get(c, "")) for c in ["Supplier Name", "Product Category", "Location", "Country", "HS_Code"]]).lower()
                    texts.append(text)
                    scores.append(len(q_tokens & set(re.findall(r"\w+", text))))
                filtered_df_local = filtered_df.copy()
                filtered_df_local["_score"] = scores
                top_rows = filtered_df_local.sort_values("_score", ascending=False).head(12)
                context_parts = []
                for i, r in top_rows.reset_index(drop=True).iterrows():
                    context_parts.append(f"[{i}] {r['Supplier Name']} | {r['Product Category']} | {r['Location']} | {r['Country']} | HS:{r['HS_Code']} | MOQ:{r['Minimum Order Quantity']} | Lead:{r['Lead Time (days)']} | Rating:{r['Rating']} | Verified:{r['Verified']} | Source:{r['Verified Source']}")
                context_text = "\n".join(context_parts) if context_parts else "No supplier context available."
                try:
                    res = qa_pipeline(question=user_q, context=context_text)
                    answer = res.get("answer", "")
                    st.markdown("### LLM Answer (fallback)")
                    st.write(answer)
                    st.markdown("#### Context rows used")
                    st.dataframe(top_rows.drop(columns=["_score"], errors="ignore").reset_index(drop=True), use_container_width=True)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

with TAB_IO:
    st.subheader("Import & Export")
    st.markdown("Upload new data in the sidebar, or download the dataset below.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download All (CSV)", data=st.session_state.df.to_csv(index=False).encode('utf-8'), file_name="suppliers_all.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Download All (JSON)", data=st.session_state.df.to_json(orient="records", force_ascii=False).encode('utf-8'), file_name="suppliers_all.json", mime="application/json")
    with c3:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            st.session_state.df.to_excel(writer, index=False, sheet_name="suppliers")
            writer.save()
        buf.seek(0)
        st.download_button("⬇️ Download All (Excel)", data=buf.getvalue(), file_name="suppliers_all.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with TAB_ABOUT:
    st.markdown("""
    ### About this app
    - Upload CSV/Excel or load live trade data (UN Comtrade) into the dashboard.
    - Auto-normalizes columns, auto-detects Verified Source from email domains.
    - Table-aware deterministic query parsing for accurate supplier filtering; semantic QA fallback using Hugging Face DistilBERT.
    - Charts, editor, CSV/JSON/Excel export.
    """)
