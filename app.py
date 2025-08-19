from __future__ import annotations
import io
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------
# Config / Hugging Face (Inference API)
# ---------------------------
HF_MODEL = "distilbert-base-uncased-distilled-squad"
HF_API_KEY = st.secrets.get("HF_API_KEY", None)  # set in Streamlit Cloud secrets

def ask_hf(question: str, context: str) -> str:
    """Ask Hugging Face Inference API (returns answer string)."""
    if not HF_API_KEY:
        return "⚠️ Hugging Face API key not set in Streamlit secrets."
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": {"question": question, "context": context}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Some HF models return a dict, some return a list
        if isinstance(data, dict):
            return data.get("answer") or data.get("label") or str(data)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("answer") or str(data[0])
        return "No answer found."
    except Exception as e:
        return f"Error calling Hugging Face API: {e}"

# ---------------------------
# Template Columns & Aliases
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
    "suppliername": "Supplier Name",
    "product": "Product Category",
    "category": "Product Category",
    "hs code": "HS_Code",
    "hscode": "HS_Code",
    "hs_code": "HS_Code",
    "country": "Country",
    "city": "Location",
    "location": "Location",
    "moq": "Minimum Order Quantity",
    "minimum order": "Minimum Order Quantity",
    "minimum_order": "Minimum Order Quantity",
    "leadtime": "Lead Time (days)",
    "lead time": "Lead Time (days)",
    "delivery days": "Lead Time (days)",
    "rating": "Rating",
    "verified": "Verified",
    "verified source": "Verified Source",
    "verified_source": "Verified Source",
    "email": "Contact Email",
    "contact": "Contact Email",
}

# ---------------------------
# Helpers: Verified Source detection & normalization
# ---------------------------
def detect_verified_source(email: Optional[str]) -> str:
    if not isinstance(email, str) or "@" not in email:
        return "Unknown"
    domain = email.split("@")[-1].lower()
    if domain.endswith(".gov.in") or domain.endswith(".gov"):
        return "DGFT / Govt"
    if "indiamart" in domain:
        return "IndiaMART"
    if "tradeindia" in domain:
        return "TradeIndia"
    if "exportersindia" in domain:
        return "ExportersIndia"
    if domain.endswith(".org") or domain.endswith(".ngo"):
        return "NGO / Association"
    if domain.endswith(".in"):
        return "Indian Private Company"
    if domain.endswith(".com"):
        return "Global Trader / Exporter"
    if domain.endswith(".co"):
        return "Company / Startup"
    if domain.endswith(".edu") or domain.endswith(".ac.in"):
        return "Educational Institution"
    return "Unknown"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # rename using aliases
    renamed = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in COLUMN_ALIASES:
            renamed[c] = COLUMN_ALIASES[key]
        else:
            # fuzzy check
            k2 = re.sub(r"[^\w\s]", "", key).replace("_", " ").strip()
            if k2 in COLUMN_ALIASES:
                renamed[c] = COLUMN_ALIASES[k2]
            else:
                for alias, target in COLUMN_ALIASES.items():
                    if alias in key:
                        renamed[c] = target
                        break
    if renamed:
        df = df.rename(columns=renamed)
    # ensure columns exist
    for col in TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # coerce numeric
    if "Lead Time (days)" in df.columns:
        df["Lead Time (days)"] = pd.to_numeric(df["Lead Time (days)"], errors="coerce")
    if "Minimum Order Quantity" in df.columns:
        df["Minimum Order Quantity"] = pd.to_numeric(df["Minimum Order Quantity"], errors="coerce")
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    # fill verified source if missing
    if "Verified Source" not in df.columns or df["Verified Source"].isna().all():
        if "Contact Email" in df.columns:
            df["Verified Source"] = df["Contact Email"].apply(detect_verified_source)
        else:
            df["Verified Source"] = "Unknown"
    else:
        # fill blanks using email
        df["Verified Source"] = df.apply(
            lambda r: r["Verified Source"] if pd.notna(r["Verified Source"]) and str(r["Verified Source"]).strip() != "" else detect_verified_source(r.get("Contact Email", "")),
            axis=1
        )
    # normalize Verified column to Yes/No
    df["Verified"] = df["Verified"].astype(str).str.strip().str.lower().map(
        {"yes": "Yes", "y": "Yes", "true": "Yes", "no": "No", "n": "No", "false": "No"}
    ).fillna("No")
    # sensible defaults for missing leads
    if "Lead Time (days)" in df.columns:
        median_lead = df["Lead Time (days)"].median(skipna=True)
        if pd.isna(median_lead):
            median_lead = 7
        df["Lead Time (days)"] = df["Lead Time (days)"].fillna(median_lead)
    if "Minimum Order Quantity" in df.columns:
        df["Minimum Order Quantity"] = df["Minimum Order Quantity"].fillna(0)
    return df[TEMPLATE_COLUMNS].copy()

# ---------------------------
# Sample Data (used if no upload)
# ---------------------------
def load_sample_dataframe() -> pd.DataFrame:
    data = [
        ["BrightLite Industries", "LED Bulbs", "940540", "India", "Delhi", 100, 15, 4.5, "Yes", "IndiaMART", "sales@brightlite.in"],
        ["Shakti Exports", "Basmati Rice", "100630", "India", "Karnal", 500, 12, 4.2, "Yes", "DGFT / Govt", "export@shaktigroup.in"],
        ["BeanCraft Coffee", "Coffee", "0901", "India", "Bengaluru", 200, 10, 4.6, "Yes", "IndiaMART", "sales@beancraft.indiamart.com"],
        ["SpiceHouse", "Spices", "0904", "India", "Kochi", 1000, 7, 4.4, "No", "Unknown", "contact@spicehouse.com"],
    ]
    return pd.DataFrame(data, columns=TEMPLATE_COLUMNS)

# initialize session df
if "df" not in st.session_state:
    st.session_state.df = normalize_columns(load_sample_dataframe())

# ---------------------------
# UN Comtrade preview loader (v1 public preview endpoint) - fixed
# Endpoint format:
# https://comtradeapi.un.org/public/v1/preview/{flow}/{reporter}/{period}/{partner}/{cmdCode}
# flow: 1=import, 2=export ; partner 0 = World
# ---------------------------
def fetch_comtrade_preview(hs_code: str = "0901", reporter_code: str = "356", year: int = None, flow: int = 2) -> pd.DataFrame:
    """
    Fetch preview rows from UN Comtrade (preview endpoint). Returns supplier-like rows.
    flow: 1=Import, 2=Export (we use 2 for exports)
    reporter_code: numeric code as string (e.g., "356" for India) or "all"
    partner: using 0 means "World"
    """
    if year is None:
        year = datetime.now().year - 1
    # Build url
    url = f"https://comtradeapi.un.org/public/v1/preview/{flow}/{reporter_code}/{year}/0/{hs_code}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", []) if isinstance(payload, dict) else []
        rows = []
        for i, rec in enumerate(records):
            partner_country = rec.get("ptTitle") or rec.get("partnerDesc") or rec.get("partnerTitle") or "World"
            rows.append({
                "Supplier Name": f"Exporters to {partner_country}",
                "Product Category": rec.get("cmdDescE") or f"HS {hs_code}",
                "HS_Code": str(rec.get("cmdCode", hs_code)),
                "Country": partner_country,
                "Location": partner_country,
                "Minimum Order Quantity": None,
                "Lead Time (days)": None,
                "Rating": None,
                "Verified": "Yes",
                "Verified Source": "UN Comtrade",
                "Contact Email": None
            })
        if rows:
            df_live = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)
            return normalize_columns(df_live)
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    except requests.HTTPError as he:
        st.warning(f"Comtrade fetch HTTP error: {he}")
    except Exception as e:
        st.warning(f"Comtrade fetch error: {e}")
    return pd.DataFrame(columns=TEMPLATE_COLUMNS)

# ---------------------------
# Streamlit UI: Sidebar (upload, live fetch, filters)
# ---------------------------
st.set_page_config(page_title="AI Sourcing Agent", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("📦 AI Sourcing Agent")

# Upload support
uploaded = st.sidebar.file_uploader("Upload supplier CSV / Excel", type=["csv", "xlsx"], help="Supported: .csv, .xlsx")
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_upload = pd.read_csv(uploaded, dtype=str)
        else:
            df_upload = pd.read_excel(uploaded, dtype=str)
        st.session_state.df = normalize_columns(df_upload)
        st.sidebar.success("✅ Uploaded and normalized")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Live Data (UN Comtrade preview)")

# HS code & year inputs
hs_input = st.sidebar.text_input("HS Code (e.g., 0901 for coffee)", value="0901")
year_input = st.sidebar.number_input("Year", min_value=2000, max_value=datetime.now().year, value=datetime.now().year - 1)
reporter_choice = st.sidebar.selectbox("Reporter (country)", options=["India (356)", "China (156)", "United States (842)", "All (all)"], index=0)
reporter_code_map = {"India (356)": "356", "China (156)": "156", "United States (842)": "842", "All (all)": "all"}

if st.sidebar.button("Load live Comtrade preview"):
    rc = reporter_code_map.get(reporter_choice, "356")
    with st.spinner("Fetching live Comtrade preview..."):
        df_live = fetch_comtrade_preview(hs_code=hs_input.strip(), reporter_code=rc, year=int(year_input), flow=2)
    if not df_live.empty:
        # append to existing data (you can change to replace if desired)
        st.session_state.df = pd.concat([st.session_state.df, df_live], ignore_index=True)
        st.sidebar.success(f"Loaded {len(df_live)} preview rows from Comtrade (HS {hs_input}).")
    else:
        st.sidebar.warning("No preview rows returned from Comtrade for those parameters.")

# Quick template download
with io.StringIO() as buf:
    pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(buf, index=False)
    st.sidebar.download_button("Download blank template (CSV)", data=buf.getvalue(), file_name="supplier_template.csv", mime="text/csv")

# Reset to sample
if st.sidebar.button("Reset to sample data"):
    st.session_state.df = normalize_columns(load_sample_dataframe())
    st.sidebar.success("Reset dataset to sample rows")

# ---------------------------
# Prepare dataframe + Filters (main)
# ---------------------------
df = st.session_state.df.copy()

st.sidebar.markdown("---")
st.sidebar.header("Filters & Search")
q = st.sidebar.text_input("Search Supplier / Product / Location")
sel_product = st.sidebar.multiselect("Product Category", sorted(df["Product Category"].dropna().unique()))
sel_country = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
sel_location = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()))
sel_verified = st.sidebar.selectbox("Verified filter", options=["All", "Yes", "No"], index=0)
sel_verified_source = st.sidebar.multiselect("Verified Source", sorted(df["Verified Source"].dropna().unique()))
min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), 0.1)
hs_query = st.sidebar.text_input("HS Code contains", placeholder="e.g., 0901")

mask = pd.Series(True, index=df.index)
if q:
    qm = q.strip().lower()
    mask &= (
        df["Supplier Name"].fillna("").str.lower().str.contains(qm)
        | df["Product Category"].fillna("").str.lower().str.contains(qm)
        | df["Location"].fillna("").str.lower().str.contains(qm)
    )
if sel_product:
    mask &= df["Product Category"].isin(sel_product)
if sel_country:
    mask &= df["Country"].isin(sel_country)
if sel_location:
    mask &= df["Location"].isin(sel_location)
if sel_verified != "All":
    mask &= df["Verified"].astype(str).str.lower().eq(sel_verified.lower())
if sel_verified_source:
    mask &= df["Verified Source"].isin(sel_verified_source)
if hs_query:
    mask &= df["HS_Code"].astype(str).str.contains(hs_query, na=False, case=False)
mask &= df["Rating"].fillna(0).between(min_rating, max_rating)
filtered_df = df[mask].reset_index(drop=True)

# ---------------------------
# Header / KPIs
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
# Tabs: Dashboard / Table / Editor / QA / IO / About
# ---------------------------
TAB_DASH, TAB_TABLE, TAB_EDIT, TAB_QA, TAB_IO, TAB_ABOUT = st.tabs([
    "📈 Dashboard", "📋 Suppliers", "✏️ Editor", "🤖 Ask AI", "⬆️⬇️ Import & Export", "ℹ️ About"
])

with TAB_DASH:
    if filtered_df.empty:
        st.info("No data to display. Upload data, load live data, or relax filters.")
    else:
        left_col, right_col = st.columns(2)
        with left_col:
            grp = filtered_df.groupby(["Location", "Product Category"], as_index=False)["Supplier Name"].count()
            fig = px.bar(grp, x="Location", y="Supplier Name", color="Product Category", title="Suppliers by Location & Category")
            st.plotly_chart(fig, use_container_width=True)
            fig_hist = px.histogram(filtered_df, x="Rating", nbins=10, title="Rating Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
        with right_col:
            fig_pie = px.pie(filtered_df, names="Product Category", title="Category Share")
            st.plotly_chart(fig_pie, use_container_width=True)
            if "Lead Time (days)" in filtered_df.columns:
                fig_box = px.box(filtered_df, y="Lead Time (days)", title="Lead Time Distribution")
                st.plotly_chart(fig_box, use_container_width=True)
            top_hs = filtered_df.groupby("HS_Code", as_index=False)["Supplier Name"].count().rename(columns={"Supplier Name": "Count"}).sort_values("Count", ascending=False).head(15)
            fig_hs = px.bar(top_hs, x="HS_Code", y="Count", title="Top HS Codes")
            st.plotly_chart(fig_hs, use_container_width=True)

with TAB_TABLE:
    st.subheader("Supplier Data (filtered)")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Filtered CSV", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name="suppliers_filtered.csv", mime="text/csv")
    st.download_button("⬇️ Download Filtered JSON", data=filtered_df.to_json(orient="records").encode("utf-8"), file_name="suppliers_filtered.json", mime="application/json")

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
        st.info("Data editor unavailable in this Streamlit version. Download CSV and re-upload to edit.")
    if st.button("💾 Save Edits to Session"):
        if "edited_df" in st.session_state:
            st.session_state.df = normalize_columns(pd.DataFrame(st.session_state["edited_df"]))
            st.success("Saved to session")
        else:
            st.warning("No edits detected.")

with TAB_QA:
    st.subheader("🤖 Ask AI about your suppliers")
    user_q = st.text_input("Enter a natural-language question (e.g., 'Show me coffee suppliers with lead time < 15 days')")
    if user_q:
        if filtered_df.empty:
            st.warning("No data to answer with.")
        else:
            # Build context from top matching rows (simple)
            context_parts = []
            for _, r in filtered_df.head(40).iterrows():
                context_parts.append(
                    f"{r['Supplier Name']} | {r['Product Category']} | {r['Location']} | HS:{r['HS_Code']} | Lead:{r['Lead Time (days)']} | Rating:{r['Rating']} | Verified:{r['Verified Source']}"
                )
            context = "\n".join(context_parts) if context_parts else "No supplier data."
            answer = ask_hf(user_q, context)
            st.markdown("**Answer:**")
            st.write(answer)
            st.markdown("**Context rows used (top 40):**")
            st.dataframe(filtered_df.head(40).reset_index(drop=True), use_container_width=True)

with TAB_IO:
    st.subheader("Import & Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download All (CSV)", data=st.session_state.df.to_csv(index=False).encode("utf-8"), file_name="suppliers_all.csv", mime="text/csv")
    with col2:
        # Excel export
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            st.session_state.df.to_excel(writer, index=False, sheet_name="suppliers")
        buf.seek(0)
        st.download_button("⬇️ Download All (Excel)", data=buf.getvalue(), file_name="suppliers_all.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with TAB_ABOUT:
    st.markdown("""
    ### About
    - Supplier Intelligence dashboard with upload, normalization, filters, charts, inline editor, and exports.
    - Auto-detects **Verified Source** from email domains when missing.
    - AI Q&A uses Hugging Face Inference API (set `HF_API_KEY` in Streamlit secrets).
    - Live preview fetch from UN Comtrade (preview endpoint) — choose HS code & year in sidebar.
    """)

# End of app.py
