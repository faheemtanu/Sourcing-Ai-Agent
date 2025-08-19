from __future__ import annotations
import re
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(
    page_title="AI Sourcing Agent – Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Load QA model (Hugging Face)
# ---------------------------
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

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
    "Verified Source",   # 👈 added new
    "Contact Email",
]

COLUMN_ALIASES = {
    "supplier": "Supplier Name",
    "suppliername": "Supplier Name",
    "product": "Product Category",
    "category": "Product Category",
    "hs code": "HS_Code",
    "hscode": "HS_Code",
    "country": "Country",
    "location": "Location",
    "moq": "Minimum Order Quantity",
    "minimum order": "Minimum Order Quantity",
    "leadtime": "Lead Time (days)",
    "delivery days": "Lead Time (days)",
    "rating": "Rating",
    "verified": "Verified",
    "verified source": "Verified Source",
    "email": "Contact Email",
    "contact": "Contact Email",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
        else:
            for alias, target in COLUMN_ALIASES.items():
                if alias in key:
                    renamed[col] = target
                    break
    df = df.rename(columns=renamed)
    for col in TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[TEMPLATE_COLUMNS]
    if "Lead Time (days)" in df.columns:
        df["Lead Time (days)"] = pd.to_numeric(df["Lead Time (days)"], errors="coerce")
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    return df

# ---------------------------
# Verified Source Detector
# ---------------------------
def detect_verified_source(email: str) -> str:
    if pd.isna(email):
        return "Unknown"
    email = str(email).lower()

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

# ---------------------------
# Sample Data (default)
# ---------------------------
def load_sample_dataframe() -> pd.DataFrame:
    data = [
        ["BrightLite Industries", "LED Bulbs", "940540", "India", "Delhi", 100, 15, 4.5, "Yes", "Indian Private Company", "sales@brightlite.in"],
        ["Shakti Exports", "Basmati Rice", "100630", "India", "Karnal", 500, 12, 4.2, "Yes", "Indian Private Company", "export@shaktigroup.in"],
        ["GlobalTech Pharma", "Pharmaceuticals", "300490", "India", "Mumbai", 200, 20, 4.8, "No", "Global Trader / Exporter", "bd@globaltechpharma.com"],
        ["AquaSteel Ltd", "Stainless Steel", "721934", "India", "Ahmedabad", 50, 18, 4.1, "Yes", "Company / Startup", "info@aquasteel.co"],
        ["Suryan Solar", "Solar Panels", "854140", "India", "Hyderabad", 25, 30, 4.6, "Yes", "Indian Private Company", "hello@suryansolar.in"],
        ["Veda Botanicals", "Herbal Extracts", "130219", "India", "Bengaluru", 80, 10, 4.3, "No", "Indian Private Company", "contact@vedabotanicals.in"],
    ]
    df = pd.DataFrame(data, columns=TEMPLATE_COLUMNS)
    df['HS_Code'] = df['HS_Code'].astype(str)
    return df

if "df" not in st.session_state:
    st.session_state.df = load_sample_dataframe()

# ---------------------------
# Sidebar – Upload + Filters
# ---------------------------
st.sidebar.title("📦 AI Sourcing Agent")
uploaded = st.sidebar.file_uploader("Upload supplier CSV/Excel", type=["csv", "xlsx"])

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded, dtype={"HS_Code": str})
        else:
            df_upload = pd.read_excel(uploaded, dtype={"HS_Code": str})
        st.session_state.df = normalize_columns(df_upload)

        # Auto-detect Verified Source
        if "Contact Email" in st.session_state.df.columns:
            st.session_state.df["Verified Source"] = st.session_state.df["Contact Email"].apply(detect_verified_source)
        else:
            st.session_state.df["Verified Source"] = "Unknown"

        st.sidebar.success("✅ Data uploaded & normalized")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

df = st.session_state.df.copy()

# ---------------------------
# Filters
# ---------------------------
q_supplier = st.sidebar.text_input("🔎 Search Supplier / Product")
sel_product = st.sidebar.multiselect("Product Category", sorted(df["Product Category"].dropna().unique()))
sel_location = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()))
sel_country = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), 0.1)
verified_only = st.sidebar.checkbox("Verified only", value=False)
hs_query = st.sidebar.text_input("HS Code contains")

mask = pd.Series(True, index=df.index)
if q_supplier:
    q = q_supplier.lower().strip()
    mask &= (
        df["Supplier Name"].fillna('').str.lower().str.contains(q)
        | df["Product Category"].fillna('').str.lower().str.contains(q)
        | df["Location"].fillna('').str.lower().str.contains(q)
    )
if sel_product:
    mask &= df["Product Category"].isin(sel_product)
if sel_location:
    mask &= df["Location"].isin(sel_location)
if sel_country:
    mask &= df["Country"].isin(sel_country)
if verified_only:
    mask &= df["Verified"].astype(str).str.lower().eq("yes")
if hs_query:
    mask &= df["HS_Code"].astype(str).str.contains(hs_query, na=False)

mask &= df["Rating"].fillna(0).between(min_rating, max_rating)
filtered_df = df[mask].reset_index(drop=True)

# ---------------------------
# Header / Metrics
# ---------------------------
left, right = st.columns([4,1])
with left:
    st.markdown(f"## 📊 Supplier Intelligence Dashboard\n_Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}_")
with right:
    st.metric("Rows Loaded", len(df))
    st.metric("Rows After Filters", len(filtered_df))

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total Suppliers", int(filtered_df.shape[0]))
with col2:
    pct_verified = filtered_df["Verified"].astype(str).str.lower().eq("yes").mean() * 100 if len(filtered_df)>0 else 0
    st.metric("Verified %", f"{pct_verified:.1f}%")
with col3:
    avg_rating = filtered_df["Rating"].mean() if len(filtered_df) else 0
    st.metric("Avg Rating", f"{avg_rating:.2f}")
with col4:
    avg_lead = filtered_df["Lead Time (days)"].mean() if len(filtered_df) else 0
    st.metric("Avg Lead Time", f"{avg_lead:.0f} days")

# ---------------------------
# Tabs
# ---------------------------
TAB_DASH, TAB_TABLE, TAB_EDIT, TAB_QA, TAB_IO, TAB_ABOUT = st.tabs([
    "📈 Dashboard", "📋 Suppliers", "✏️ Editor", "🤖 Ask AI", "⬆️⬇️ Import & Export", "ℹ️ About",
])

with TAB_DASH:
    if not filtered_df.empty:
        c1,c2 = st.columns(2)
        with c1:
            fig1 = px.bar(
                filtered_df.groupby(["Location","Product Category"], as_index=False)["Supplier Name"].count(),
                x="Location", y="Supplier Name", color="Product Category", title="Suppliers by Location & Category"
            )
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.pie(filtered_df, names="Product Category", title="Category Share")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for charts.")

with TAB_TABLE:
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button("Download Filtered CSV", data=filtered_df.to_csv(index=False).encode('utf-8'),
                       file_name="suppliers_filtered.csv", mime="text/csv")

with TAB_EDIT:
    st.subheader("Edit / Add Suppliers")
    if st.button("➕ Add Blank Row"):
        new_row = pd.DataFrame([{c: "" for c in TEMPLATE_COLUMNS}])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.rerun()

    edited_df = st.data_editor(st.session_state.df, use_container_width=True, num_rows="dynamic", hide_index=True)
    if st.button("💾 Save Changes"):
        st.session_state.df = normalize_columns(edited_df)
        if "Contact Email" in st.session_state.df.columns:
            st.session_state.df["Verified Source"] = st.session_state.df["Contact Email"].apply(detect_verified_source)
        st.success("Saved!")

with TAB_QA:
    st.subheader("🤖 Ask AI about your suppliers")
    question = st.text_input("Ask a question (e.g., 'Which supplier has the fastest lead time?')")
    if question and not df.empty:
        context = "\n".join(
            f"{row['Supplier Name']} in {row['Location']} ({row['Product Category']}) - Lead time {row['Lead Time (days)']} days, Rating {row['Rating']}, Source {row['Verified Source']}"
            for _, row in filtered_df.head(30).iterrows()
        )
        result = qa_pipeline(question=question, context=context)
        st.write("**Answer:**", result['answer'])
    elif question:
        st.warning("No data available to answer.")

with TAB_IO:
    st.download_button("⬇️ Download All (CSV)",
        data=st.session_state.df.to_csv(index=False).encode('utf-8'),
        file_name="suppliers_all.csv", mime="text/csv")

with TAB_ABOUT:
    st.markdown("""
    ### About
    - Free Streamlit dashboard for supplier intelligence  
    - Features: filters, charts, editable table, import/export  
    - Auto-detects **Verified Source** from email domain  
    - Added 🤖 Q&A with Hugging Face (DistilBERT QA)  
    """)
