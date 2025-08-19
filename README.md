# 📦 AI Sourcing Agent – Streamlit Dashboard

A powerful **supplier intelligence dashboard** built with **Streamlit**, now with:

- ✅ CSV/Excel upload & column auto-normalization
- ✅ Auto-detect **Verified Source** (IndiaMART, DGFT, TradeIndia, etc.)
- ✅ Search + Filters (supplier/product/location/country/HS/rating/verified)
- ✅ Charts (bar, pie, metrics dashboard)
- ✅ Editable suppliers table with save
- ✅ Export (CSV, Excel, JSON)
- ✅ 🤖 AI Q&A (powered by Hugging Face DistilBERT)
- ✅ 📡 Live Data loader (UN Comtrade integration)

---

## 🚀 Features

### File Upload
- Upload supplier data in **CSV** or **Excel**.
- Automatically cleans & normalizes column names.
- Fills in **Verified Source** based on supplier email domain.

### Verified Source Detection
- `.gov.in` → **DGFT / Govt**  
- `indiamart.com` → **IndiaMART**  
- `tradeindia.com` → **TradeIndia**  
- `exportersindia.com` → **ExportersIndia**  
- `.in` → **Indian Private Company**  
- `.com` → **Global Trader / Exporter**  
- `.org` → **NGO / Association**  

### Dashboard
- Supplier overview with metrics:
  - Total Suppliers
  - Verified %
  - Average Rating
  - Average Lead Time
- Interactive charts (location vs category, category share).

### 🤖 Ask AI
- Ask natural-language questions like:
  - *Which supplier has the fastest lead time?*
  - *List verified coffee exporters.*
- Uses Hugging Face **DistilBERT QA** pipeline.

### 📡 Live Data Integration
- Fetch **real trade data** from **UN Comtrade API**.
- Example: Indian exports of **coffee (HS 0901)**.
- App merges live data with uploaded suppliers.

### Import/Export
- Export **filtered data** to CSV.
- Export **all suppliers** to CSV/Excel/JSON.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
