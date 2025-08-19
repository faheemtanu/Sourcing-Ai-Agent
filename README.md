# 📦 AI Sourcing Agent Dashboard

An AI-powered **supplier intelligence dashboard** built with **Streamlit**.  
Upload supplier data (CSV/Excel) or fetch live trade stats from **UN Comtrade**, explore insights, and ask natural language questions with Hugging Face 🤖.

---

## 🚀 Features
- **File Uploads** (CSV + Excel) → auto-normalizes messy column names
- **Verified & Verified Source** detection (guesses source from email domain if missing)
- **Filters**: supplier, category, HS code, country, location, rating, verified
- **Charts**: suppliers by location/category, product share
- **Editable Table**: add/remove suppliers inline
- **Import/Export**: download CSV or JSON
- **🤖 Ask AI**: natural language Q&A with Hugging Face Inference API
- **📡 Live Data**: pull sample exports (e.g., Coffee HS 0901) from **UN Comtrade**

---

## 🔧 Installation

Clone the repo and install requirements:

```bash
pip install -r requirements.txt
