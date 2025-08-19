from __future__ import annotations
import io
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline

# ---------------------------
# Page Setup & Global Styles
# ---------------------------
st.set_page_config(
    page_title="AI Sourcing Agent – Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Theme
st.markdown(
    """
    <style>
      body {background: #f5f7fb;}    
      .block-container {padding-top: 1.5rem; padding-bottom: 4rem;}
      header {visibility: hidden;}
      .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 6px 24px rgba(2,6,23,0.06);
        border: 1px solid rgba(2,6,23,0.06);
      }
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stMetricLabel"] {color: #64748b;}  
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Template Columns
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

def guess_verified_source(email: str) -> str:
    """Auto-guess verification source based on email domain"""
    if not isinstance(email, str) or "@" not in email:
        return "Unknown"
    domain = email.lower().split("@")[-1]
    if domain.endswith("gov.in"):
        return "DGFT / Govt"
    elif "indiamart.com" in domain:
        return "IndiaMART"
    elif "tradeindia.com" in domain:
        return "TradeIndia"
    return "Unknown"

def load_sample_dataframe() -> pd.DataFrame:
    data = [
        ["BrightLite Industries", "LED Bulbs", "940540", "India", "Delhi", 100, 15, 4.5, "Yes", "DGFT / Govt", "sales@brightlite.in"],
        ["Shakti Exports", "Basmati Rice", "100630", "India", "Karnal", 500, 12, 4.2, "Yes", "IndiaMART", "export@shaktigroup.indiamart.com"],
        ["GlobalTech Pharma", "Pharmaceuticals", "300490", "India", "Mumbai", 200, 20, 4.8, "No", "Unknown", "bd@globaltechpharma.com"],
    ]
    df = pd.DataFrame(data, columns=TEMPLATE_COLUMNS)
    return df

# Initialize session
if "df" not in st.session_state:
    st.session_state.df = load_sample_dataframe()

# ---------------------------
# Sidebar Upload
# ---------------------------
st.sidebar.title("AI Sourcing Agent")
uploaded = st.sidebar.file_uploader(
    "Upload supplier CSV/Excel", type=["csv", "xlsx"], help="Supports CSV or Excel"
)

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded, dtype={"HS_Code": str})
        else:
            df_upload = pd.read_excel(uploaded, dtype={"HS_Code": str})

        # Normalize columns
        rename_map = {c.lower().strip(): c for c in TEMPLATE_COLUMNS}
        df_upload.columns = [col.strip() for col in df_upload.columns]

        # Add missing columns
        for col in TEMPLATE_COLUMNS:
            if col not in df_upload.columns:
                if col == "Verified Source":
                    df_upload[col] = df_upload.get("Contact Email", "").apply(guess_verified_source)
                else:
                    df_upload[col] = ""

        st.session_state.df = df_upload[TEMPLATE_COLUMNS]
        st.sidebar.success("Data uploaded ✔")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

df = st.session_state.df.copy()

# ---------------------------
# Filters
# ---------------------------
q_supplier = st.sidebar.text_input("🔎 Search Supplier / Product")
sel_product = st.sidebar.multiselect("Product Category", sorted(df["Product Category"].dropna().unique()))
sel_location = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()))
sel_country = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
sel_source = st.sidebar.multiselect("Verified Source", sorted(df["Verified Source"].dropna().unique()))

min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), 0.1)
verified_only = st.sidebar.checkbox("Verified suppliers only", value=False)

hs_query = st.sidebar.text_input("HS Code contains", placeholder="e.g., 940540")

mask = pd.Series(True, index=df.index)
if q_supplier:
    q = q_supplier.lower()
    mask &= df["Supplier Name"].str.lower().str.contains(q) | df["Product Category"].str.lower().str.contains(q)
if sel_product: mask &= df["Product Category"].isin(sel_product)
if sel_location: mask &= df["Location"].isin(sel_location)
if sel_country: mask &= df["Country"].isin(sel_country)
if sel_source: mask &= df["Verified Source"].isin(sel_source)
if verified_only: mask &= df["Verified"].astype(str).str.lower().eq("yes")
if hs_query: mask &= df["HS_Code"].astype(str).str.contains(hs_query)

mask &= df["Rating"].fillna(0).between(min_rating, max_rating)
filtered_df = df[mask].reset_index(drop=True)

# ---------------------------
# Header & KPIs
# ---------------------------
st.markdown(
    f"""
    <div class="card">
      <div style="display:flex; align-items:center; gap:14px;">
        <div style="font-size: 1.8rem;">📊 <b>Supplier Intelligence Dashboard</b></div>
        <div style="margin-left:auto; color: #64748b;">Updated: {datetime.now().strftime("%d %b %Y, %I:%M %p")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Suppliers", len(filtered_df))
col2.metric("Verified %", f"{(filtered_df['Verified'].astype(str).str.lower().eq('yes').mean()*100):.1f}%")
col3.metric("Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
col4.metric("Avg Lead Time", f"{filtered_df['Lead Time (days)'].mean():.0f} days")

# ---------------------------
# Tabs
# ---------------------------
TAB_DASH, TAB_TABLE, TAB_QA = st.tabs(["📈 Dashboard", "📋 Suppliers", "🤖 Ask AI about Suppliers"])

with TAB_DASH:
    if not filtered_df.empty:
        fig = px.bar(filtered_df.groupby("Location")["Supplier Name"].count().reset_index(),
                     x="Location", y="Supplier Name", title="Suppliers by Location")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to show.")

with TAB_TABLE:
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download CSV", filtered_df.to_csv(index=False), "suppliers_filtered.csv", "text/csv")

with TAB_QA:
    st.subheader("🤖 Ask AI about your suppliers")
    query = st.text_input("Ask a question (e.g., 'Show me coffee suppliers in Delhi')")
    if query:
        # convert DataFrame to text
        context = "\n".join(filtered_df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist())
        qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        try:
            ans = qa(question=query, context=context)
            st.success(ans["answer"])
        except Exception as e:
            st.error(f"LLM Error: {e}")
