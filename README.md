# 📦 AI Sourcing Agent – Dashboard

An interactive Streamlit app for supplier intelligence.

## 🚀 Features
- Upload supplier CSV/Excel (auto-normalized)
- Filters: product, location, country, HS Code, verified only
- Inline editor + add suppliers
- Charts (bar + pie)
- Export filtered / all data (CSV)
- 🤖 Q&A using Hugging Face Inference API (DistilBERT QA)
- Auto-detect **Verified Source** based on email domain
- 📡 Live data fetch from UN Comtrade API (e.g., Coffee HS 0901)

## 🔧 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
