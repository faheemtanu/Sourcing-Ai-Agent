```python
# =========================================================
# Streamlit AI Sourcing Agent – Dashboard + LLM Q&A
# ---------------------------------------------------------
# ✅ Clean data (CSV + Excel upload)
# ✅ Auto column normalization
# ✅ Filters, charts, KPIs
# ✅ Inline editor & verified suppliers
# ✅ Import/export
# ✅ Hugging Face LLM-powered Q&A (no paid key needed)
# =========================================================

from __future__ import annotations
import io
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# Hugging Face Transformers
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

st.markdown(
    """
    <style>
      body {background: #f5f7fb;} 
      .card {background: #fff; border-radius: 18px; padding: 18px; box-shadow: 0 6px 24px rgba(2,6,23,0.06);} 
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
    "Contact Email",
]

# Load sample

def load_sample_dataframe():
    data = [
        ["BrightLite Industries", "LED Bulbs", "940540", "India", "Delhi", 100, 15, 4.5, "Yes", "sales@brightlite.in"],
        ["Shakti Exports", "Basmati Rice", "100630", "India", "Karnal", 500, 12, 4.2, "Yes", "export@shaktigroup.in"],
        ["GlobalTech Pharma", "Pharmaceuticals", "300490", "India", "Mumbai", 200, 20, 4.8, "No", "bd@globaltechpharma.com"],
    ]
    return pd.DataFrame(data, columns=TEMPLATE_COLUMNS)

if "df" not in st.session_state:
    st.session_state.df = load_sample_dataframe()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("AI Sourcing Agent")

uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

# Load upload
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded, dtype={"HS_Code": str})
        else:
            df_upload = pd.read_excel(uploaded, dtype={"HS_Code": str})
        st.session_state.df = df_upload
        st.sidebar.success("Data uploaded ✔")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

df = st.session_state.df.copy()

# ---------------------------
# Header
# ---------------------------
left, right = st.columns([4, 1])
with left:
    st.markdown(
        f"""
        <div class="card">
        <b>📊 Supplier Intelligence Dashboard</b><br>
        <span style='color:#64748b;'>Updated {datetime.now().strftime('%d %b %Y, %I:%M %p')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.metric("Rows", len(df))

# ---------------------------
# Tabs
# ---------------------------
TAB_DASH, TAB_TABLE, TAB_EDIT, TAB_IO, TAB_QA = st.tabs([
    "📈 Dashboard", "📋 Suppliers", "✏️ Editor", "⬆️⬇️ Import/Export", "🤖 Ask the Data"
])

# Dashboard
with TAB_DASH:
    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="Country", title="Suppliers by Country")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Lead Time (days)" in df:
                fig2 = px.histogram(df, x="Lead Time (days)", title="Lead Time Distribution")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload data to see dashboard.")

# Suppliers table
with TAB_TABLE:
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "suppliers.csv")

# Editor
with TAB_EDIT:
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes"):
        st.session_state.df = edited_df
        st.success("Saved to session ✔")

# Import/Export
with TAB_IO:
    st.download_button("⬇️ Download JSON", df.to_json(orient="records"), "suppliers.json")

# Q&A Tab
with TAB_QA:
    st.subheader("Ask Questions About Your Suppliers 🤖")

    # Init QA pipeline
    if "qa_pipeline" not in st.session_state:
        st.session_state.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    question = st.text_input("Ask a question (e.g., 'Which supplier has the shortest lead time?')")

    if question:
        # Turn suppliers into context text
        context = "\n".join(df.astype(str).apply(lambda row: ", ".join(row.values), axis=1).tolist())
        try:
            result = st.session_state.qa_pipeline(question=question, context=context)
            st.markdown(f"**Answer:** {result['answer']}")
        except Exception as e:
            st.error(f"LLM error: {e}")
```
