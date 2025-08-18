# =========================================================
# Streamlit AI Sourcing Agent – Full Dashboard (Single File)
# ---------------------------------------------------------
# ✅ Drop-in replacement for your old app.py
# ✅ Modern UI, filters, charts, editable table, CSV export
# ✅ 100% free to develop, deploy (Streamlit Cloud/HF Spaces)
#
# How to run locally:
#   pip install streamlit pandas plotly
#   streamlit run app.py
#
# Optional (but nice to have):
#   pip install openpyxl  # if you later add Excel export
# =========================================================

from __future__ import annotations
import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Page Setup & Global Styles
# ---------------------------
st.set_page_config(
    page_title="AI Sourcing Agent – Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Soft theme via CSS (works even without .streamlit/config.toml)
st.markdown(
    """
    <style>
      :root {
        --primary: #3b82f6; /* Sky blue */
        --secondary: #10b981; /* Emerald */
        --bg: #f5f7fb;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --ring: rgba(59,130,246,.35);
      }
      .block-container {padding-top: 1.5rem; padding-bottom: 4rem;}
      body {background: var(--bg);}    
      header {visibility: hidden;} /* Hide default header */

      /* Card */
      .card {
        background: var(--card);
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 6px 24px rgba(2,6,23,0.06);
        border: 1px solid rgba(2,6,23,0.06);
      }
      /* KPI metric tweaks */
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stMetricLabel"] {color: var(--muted);}  

      /* Buttons */
      .stButton>button {
        border-radius: 12px;
        padding: 0.55rem 0.9rem;
        border: 1px solid rgba(2,6,23,0.08);
        box-shadow: 0 2px 8px rgba(2,6,23,0.06);
      }
      .stDownloadButton>button {border-radius: 12px;}

      /* Inputs focus ring */
      .stTextInput>div>div>input:focus,
      .stSelectbox>div>div>select:focus,
      .stMultiselect>div>div>div>input:focus {
        box-shadow: 0 0 0 3px var(--ring) !important;
        border-color: var(--primary) !important;
      }

      /* Sidebar title */
      section[data-testid="stSidebar"] {background: #ffffff;}
      
      /* Responsive layout for smaller screens */
      @media (max-width: 992px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
            flex-wrap: wrap;
        }
        div[data-testid="stHorizontalBlock"] > div {
            flex: 1;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Sample / Template Data
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


def load_sample_dataframe() -> pd.DataFrame:
    data = [
        ["BrightLite Industries", "LED Bulbs", "940540", "India", "Delhi", 100, 15, 4.5, "Yes", "sales@brightlite.in"],
        ["Shakti Exports", "Basmati Rice", "100630", "India", "Karnal", 500, 12, 4.2, "Yes", "export@shaktigroup.in"],
        ["GlobalTech Pharma", "Pharmaceuticals", "300490", "India", "Mumbai", 200, 20, 4.8, "No", "bd@globaltechpharma.com"],
        ["AquaSteel Ltd", "Stainless Steel", "721934", "India", "Ahmedabad", 50, 18, 4.1, "Yes", "info@aquasteel.co"],
        ["Suryan Solar", "Solar Panels", "854140", "India", "Hyderabad", 25, 30, 4.6, "Yes", "hello@suryansolar.in"],
        ["Veda Botanicals", "Herbal Extracts", "130219", "India", "Bengaluru", 80, 10, 4.3, "No", "contact@vedabotanicals.in"],
    ]
    df = pd.DataFrame(data, columns=TEMPLATE_COLUMNS)
    df['HS_Code'] = df['HS_Code'].astype(str)
    return df


# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = load_sample_dataframe()


# ---------------------------
# Sidebar – Data & Filters
# ---------------------------
st.sidebar.image(
    "https://static.streamlit.io/examples/dice.jpg", width=60
)
st.sidebar.title("AI Sourcing Agent")
st.sidebar.caption("Modern dashboard • Free deploy")

uploaded = st.sidebar.file_uploader(
    "Upload supplier CSV", type=["csv"], help="Use the template below if needed"
)

# Download a clean template
with io.StringIO() as buffer:
    pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    st.sidebar.download_button(
        label="Download CSV Template",
        data=buffer.getvalue(),
        file_name="supplier_template.csv",
        mime="text/csv",
        help="Start from a blank template",
    )

# Load/upload data
if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded, dtype={"HS_Code": str})
        missing = [c for c in TEMPLATE_COLUMNS if c not in df_upload.columns]
        if missing:
            st.sidebar.error(f"Missing columns in upload: {', '.join(missing)}")
        else:
            st.session_state.df = df_upload[TEMPLATE_COLUMNS]
            st.sidebar.success("Data uploaded ✔")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

df = st.session_state.df.copy()

st.sidebar.markdown("---")

# Quick Filters
q_supplier = st.sidebar.text_input("🔎 Search Supplier / Product")
sel_product = st.sidebar.multiselect("Product Category", sorted(df["Product Category"].dropna().unique()))
sel_location = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()))
sel_country = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))

min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), 0.1)
verified_only = st.sidebar.checkbox("Verified suppliers only", value=False)

# HS code filter
hs_query = st.sidebar.text_input("HS Code contains", placeholder="e.g., 940540")

# Apply filters
mask = pd.Series(True, index=df.index)

if q_supplier:
    q = q_supplier.lower().strip()
    mask &= (
        df["Supplier Name"].fillna('').str.lower().str.contains(q, na=False)
        | df["Product Category"].fillna('').str.lower().str.contains(q, na=False)
        | df["Location"].fillna('').str.lower().str.contains(q, na=False)
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
# Header / Hero
# ---------------------------
left, right = st.columns([4, 1])
with left:
    st.markdown(
        """
        <div class="card">
          <div style="display:flex; align-items:center; gap:14px;">
            <div style="font-size: 1.8rem;">📊 <b>Supplier Intelligence Dashboard</b></div>
            <div style="margin-left:auto; color: var(--muted);">Updated: {}</div>
          </div>
          <div style="color:var(--muted); margin-top:6px;">Track suppliers, verify status, explore HS codes, and export data for outreach.</div>
        </div>
        """.format(datetime.now().strftime("%d %b %Y, %I:%M %p")),
        unsafe_allow_html=True,
    )
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Rows Loaded", len(df))
    st.metric("Rows After Filters", len(filtered_df))
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------
# KPI Row
# ---------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Suppliers", int(filtered_df.shape[0]))
with col2:
    pct_verified = (
        filtered_df["Verified"].astype(str).str.lower().eq("yes").mean() * 100
        if len(filtered_df) > 0 else 0
    )
    st.metric("Verified %", f"{pct_verified:.1f}%")
with col3:
    avg_rating = filtered_df["Rating"].mean() if len(filtered_df) else 0
    st.metric("Avg Rating", f"{avg_rating:.2f}")
with col4:
    avg_lead = filtered_df["Lead Time (days)"].mean() if len(filtered_df) else 0
    st.metric("Avg Lead Time", f"{avg_lead:.0f} days")

st.write("")

# ---------------------------
# Tabs: Dashboard • Suppliers • Editor • Import/Export • About
# ---------------------------
TAB_DASH, TAB_TABLE, TAB_EDIT, TAB_IO, TAB_ABOUT = st.tabs([
    "📈 Dashboard", "📋 Suppliers", "✏️ Editor", "⬆️⬇️ Import & Export", "ℹ️ About",
])

with TAB_DASH:
    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        if not filtered_df.empty:
            fig1 = px.bar(
                filtered_df.groupby(["Location", "Product Category"], as_index=False)["Supplier Name"].count(),
                x="Location",
                y="Supplier Name",
                color="Product Category",
                title="Suppliers by Location & Category",
            )
            fig1.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No data to display for this chart.")

    with c2:
        if not filtered_df.empty:
            fig2 = px.pie(
                filtered_df,
                names="Product Category",
                title="Category Share",
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data to display for this chart.")

    st.write("")
    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        if not filtered_df.empty:
            fig3 = px.histogram(
                filtered_df,
                x="Rating",
                nbins=10,
                title="Rating Distribution",
            )
            fig3.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data to display for this chart.")

    with c4:
        if not filtered_df.empty:
            top_hs = (
                filtered_df.groupby("HS_Code", as_index=False)["Supplier Name"].count()
                .rename(columns={"Supplier Name": "Count"})
                .sort_values("Count", ascending=False)
                .head(15)
            )
            fig4 = px.bar(top_hs, x="HS_Code", y="Count", title="Top HS Codes")
            fig4.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No data to display for this chart.")


with TAB_TABLE:
    st.subheader("Supplier Data")
    st.caption("Sortable & filterable. Use sidebar for advanced filters.")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rating": st.column_config.NumberColumn("Rating", help="0–5", format="%.1f"),
            "Minimum Order Quantity": st.column_config.NumberColumn("MOQ"),
            "Lead Time (days)": st.column_config.NumberColumn("Lead Time (days)"),
            "Contact Email": st.column_config.TextColumn("Contact Email"),
        },
    )

    st.download_button(
        "Download Filtered CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="suppliers_filtered.csv",
        mime="text/csv",
        help="Exports rows after current filters",
    )


with TAB_EDIT:
    st.subheader("Edit / Add Suppliers")
    st.caption("Make changes below, then click **Save Changes** to keep them for this session.")
    
    if st.button("➕ Add Blank Row"):
        new_row_df = pd.DataFrame([{c: "" for c in TEMPLATE_COLUMNS}])
        st.session_state.df = pd.concat([st.session_state.df, new_row_df], ignore_index=True)
        st.toast("Blank row added to the bottom of the table.")
        st.rerun()

    edited_df = st.data_editor(
        st.session_state.df,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Verified": st.column_config.SelectboxColumn(
                "Verified", options=["Yes", "No"], default="No"
            ),
            "Rating": st.column_config.NumberColumn("Rating", min_value=0.0, max_value=5.0, step=0.1, format="%.1f"),
            "HS_Code": st.column_config.TextColumn("HS Code"),
            "Minimum Order Quantity": st.column_config.NumberColumn("Minimum Order Quantity", min_value=0),
            "Lead Time (days)": st.column_config.NumberColumn("Lead Time (days)", min_value=0),
        },
        key="editor",
    )

    save_col, reset_col = st.columns([1,1])
    with save_col:
        if st.button("💾 Save Changes"):
            missing = [c for c in TEMPLATE_COLUMNS if c not in edited_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.session_state.df = edited_df[TEMPLATE_COLUMNS]
                st.success("Saved to session ✔ Use Export to download.")
                st.toast("Changes have been saved!")

    with reset_col:
        if st.button("↩️ Reset to Sample Data"):
            st.session_state.df = load_sample_dataframe()
            st.rerun()


with TAB_IO:
    st.subheader("Import & Export")
    st.markdown("Upload a CSV in the sidebar. Export your current dataset below.")

    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            "⬇️ Download All (CSV)",
            data=st.session_state.df.to_csv(index=False).encode('utf-8'),
            file_name="suppliers_all.csv",
            mime="text/csv",
        )
    with col_json:
        st.download_button(
            "⬇️ Download All (JSON)",
            data=st.session_state.df.to_json(orient="records").encode('utf-8'),
            file_name="suppliers_all.json",
            mime="application/json",
        )


with TAB_ABOUT:
    st.markdown(
        """
        ### About this App
        - **Tech**: Streamlit, Pandas, Plotly (all free)
        - **What you get**: KPI cards, advanced filters, interactive charts, editable table, CSV export
        - **Deploy free**: Streamlit Community Cloud or Hugging Face Spaces
        
        #### Tips
        - Use the **Editor** tab to add or update suppliers; click **Save Changes**.
        - Use **Import & Export** to move data between CSV/JSON.
        - Customize colors by editing the CSS variables at the top of this file.

        ⭐ If you later want a login, multi-user database, or API integrations (e.g., IndiaTrade/Indiamart scraping via their allowed endpoints), we can extend this without breaking your current flow.
        """
    )
